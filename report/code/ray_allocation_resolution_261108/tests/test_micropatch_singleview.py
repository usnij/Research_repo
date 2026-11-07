"""CPU checks of the exact ray budget, physical SSIM support and patch objective."""
from dataclasses import replace
from contextlib import ExitStack, redirect_stdout
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sparse_grt.config import MethodConfig
from sparse_grt.contracts import candidate_radius, ray_geometry
from sparse_grt.inputs import SelectedInputs
from sparse_grt.sampling import StratifiedGrid, MicropatchGrid
from sparse_grt.losses import micropatch_coordinate as C
from sparse_grt.losses import coordinate_ssim_all as original_coordinate
from sparse_grt.losses.patch_ssim import loss as patch_loss
from sparse_grt.micropatch import PatchInputs, patch_objective
from sparse_grt import micropatch_protocol as P
from sparse_grt.growth import NativeGrowthAudit


def config(**kw):
    base = MethodConfig(views=1, block=3, sigma=2., l1_weight=.8,
                        boundary='all', sampler='micropatch', cap=P.BASE_CAPS['garden'])
    return replace(base, **kw)


def dense_coordinate_weights(y, x, sigma):
    coordinates = torch.stack((y.reshape(-1), x.reshape(-1)), -1).double()
    delta = coordinates[:, None]-coordinates[None, :]
    weight = torch.exp(-delta.square().sum(-1)/(2*sigma*sigma))
    weight *= (delta.abs() <= 3*sigma).all(-1)
    return weight/weight.sum(1, keepdim=True)


def unpack_weights(weight, k, shape):
    h, w = shape; n = h*w
    indices = F.unfold((torch.arange(n, dtype=torch.float64)+1).reshape(1,1,h,w),
                       k, padding=k//2)[0].long()-1
    valid = indices >= 0
    rows = torch.arange(n)[None].expand(k*k, n)
    dense = torch.zeros(n, n, dtype=weight.dtype)
    dense[rows[valid], indices[valid]] = weight[0][valid]
    return dense


def dense_ssim_reference(pred, gt, weight):
    x, y = pred.reshape(-1,3), gt.reshape(-1,3)
    mx, my = weight@x, weight@y
    # Independent conventional SSIM using residual population moments avoids
    # cancellation for constant/near-constant signals, without reusing a+b-ab.
    dx, dy = x[None]-mx[:,None], y[None]-my[:,None]
    vx = (weight[:,:,None]*dx.square()).sum(1)
    vy = (weight[:,:,None]*dy.square()).sum(1)
    covariance = (weight[:,:,None]*dx*dy).sum(1)
    similarity = ((2*mx*my+.0001)*(2*covariance+.0009)/
                  ((mx.square()+my.square()+.0001)*(vx+vy+.0009)))
    return (1-similarity).mean()


class Dataset:
    def __init__(self, h=19, w=25):
        generator = torch.Generator().manual_seed(123)
        self.rgb = torch.randint(256, (1,h,w,3), generator=generator, dtype=torch.uint8)
        self.ro = torch.zeros(1,h,w,3)
        self.rd = torch.rand(1,h,w,3,generator=generator)
    def __len__(self): return 2
    def __getitem__(self, i): return dict(data=self.rgb, pose=torch.eye(4)[None], intr=0)
    def _lazy_worker_intrinsics_cache(self): return {0:({},self.ro,self.rd,'test')}


class MicropatchSingleview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exhaustive_small_shape_counts_bounds_uniqueness_and_boundary_pairing(self):
        for h in range(1,20):
            for w in range(1,20):
                grid = MicropatchGrid(h,w)
                for seed in (777,910000000):
                    actual = grid.sample(torch.Generator().manual_seed(seed))
                    old = StratifiedGrid(h,w,3).sample(torch.Generator().manual_seed(seed))
                    n = math.ceil(h/3)*math.ceil(w/3)
                    self.assertEqual(actual.y.numel(), n)
                    self.assertEqual(grid.shape, (math.ceil(h/3),math.ceil(w/3)))
                    self.assertEqual(torch.unique(actual.y*w+actual.x).numel(), n)
                    self.assertTrue(bool(((actual.y>=0)&(actual.y<h)&(actual.x>=0)&(actual.x<w)).all()))
                    for key in ('y','x'):
                        torch.testing.assert_close(getattr(actual,key)[grid.boundary_mask],
                                                   getattr(old,key)[grid.boundary_mask],atol=0,rtol=0)
                    self.assertEqual(grid.patch_indices.shape, ((h//6)*(w//6),4))
                    py,px=actual.y[grid.patch_indices],actual.x[grid.patch_indices]
                    torch.testing.assert_close(py-py[:,:1],grid.local_y.expand_as(py),atol=0,rtol=0)
                    torch.testing.assert_close(px-px[:,:1],grid.local_x.expand_as(px),atol=0,rtol=0)
                    self.assertTrue(bool(((py[:,:1]//6)==grid.tile_y[:,None]//6).all()))
                    self.assertTrue(bool(((px[:,:1]//6)==grid.tile_x[:,None]//6).all()))

    def test_nine_anchors_give_exact_uniform_pixel_marginals(self):
        grid=MicropatchGrid(6,6);counts=torch.zeros(6,6,dtype=torch.long)
        for ay in range(3):
            for ax in range(3):
                with patch('torch.randint',side_effect=[torch.tensor([ay]),torch.tensor([ax])]):
                    sample=grid.sample(torch.Generator().manual_seed(7))
                counts[sample.y,sample.x]+=1
        torch.testing.assert_close(counts,torch.ones_like(counts),atol=0,rtol=0)

    def test_private_sampler_generator_and_scene_budgets(self):
        torch.manual_seed(99);before=torch.get_rng_state().clone()
        MicropatchGrid(19,25).sample(torch.Generator().manual_seed(777))
        torch.testing.assert_close(torch.get_rng_state(),before,atol=0,rtol=0)
        for h,w,n,edge in ((840,1297,121240,280),(822,1237,113162,274),
                           (1039,1559,180440,1212),(1039,1558,180440,1212)):
            geometry=ray_geometry(h,w,config())
            self.assertEqual(geometry['rays_per_view'],n)
            self.assertEqual(geometry['boundary_fallback_rays'],edge)
            self.assertEqual(geometry['patch_rays']+edge,n)
        self.assertEqual(config().nominal_total_ray_fraction(),1/9)

    def test_all_arms_gather_exactly_the_same_observed_rays(self):
        data=Dataset();a=SelectedInputs(data,SimpleNamespace,config(sigma=1.5),device='cpu',weight_fn=C.reference_weights)
        b=SelectedInputs(data,SimpleNamespace,config(),device='cpu',weight_fn=C.reference_weights)
        c=PatchInputs(data,SimpleNamespace,config(),device='cpu')
        for seed in (777,910000000):
            batches=[];plans=[]
            for adapter in (a,b,c):
                batch,_,plan=adapter.make([1],seed);batches.append(batch);plans.append(plan[0])
            for key in ('rays_ori','rays_dir','rgb_gt'):
                for other in batches[1:]:
                    torch.testing.assert_close(getattr(batches[0],key),getattr(other,key),atol=0,rtol=0)
            self.assertEqual([p['k'] for p in plans],[7,7,1])
            for other in plans[1:]:
                torch.testing.assert_close(plans[0]['patch_indices'],other['patch_indices'],atol=0,rtol=0)
            p=plans[0]
            expected=(data.rgb[:,p['y'],p['x']]/255.).reshape_as(batches[0].rgb_gt)
            torch.testing.assert_close(batches[0].rgb_gt,expected,atol=0,rtol=0)
        self.assertEqual(c.geometry[0]['structural_boundary_policy'],'exclude_all_fallback_rays')
        default=SelectedInputs(data,SimpleNamespace,config(),device='cpu')
        self.assertIs(default.weight_fn,C.weights)
        with self.assertRaises(ValueError):PatchInputs(Dataset(5,6),SimpleNamespace,config(),device='cpu')

    def test_candidate_envelope_covers_every_possible_small_image_pair(self):
        for sigma,expected in ((1.5,3),(2.,3),(5.,7)):
            radius=candidate_radius(config(sigma=sigma));self.assertEqual(radius,expected)
            # Exhaust each axis. The physical support is a Cartesian product,
            # so this covers every y/x pairing, including partial-tile fallback.
            for h in range(1,49):
                possible=[]
                for i in range(math.ceil(h/3)):
                    macro,local=divmod(i,2);origin=6*macro
                    positions=set(range(origin+3*local,min(origin+3*local+3,h)))
                    if h-origin>=6:positions.update(origin+a+local for a in (0,2,4))
                    possible.append(positions)
                for i,left in enumerate(possible):
                    for j,right in enumerate(possible):
                        if abs(i-j)>radius:
                            self.assertGreater(min(abs(a-b) for a in left for b in right),3*sigma)

    def test_coordinate_weights_loss_and_gradients_match_brute_force(self):
        generator=torch.Generator().manual_seed(42)
        for sigma in (1.5,2.,5.):
            for height,width in ((19,25),(30,31)):
                grid=MicropatchGrid(height,width)
                sample=grid.sample(generator)
                y,x=sample.y.reshape(grid.shape).double(),sample.x.reshape(grid.shape).double()
                weight,k=C.reference_weights(y,x,3,sigma,candidate_radius=candidate_radius(config(sigma=sigma)))
                exact=dense_coordinate_weights(y,x,sigma)
                torch.testing.assert_close(unpack_weights(weight,k,grid.shape),exact,atol=2e-16,rtol=2e-14)
                for case in ('random','identical','flat','constant_residual','near_flat'):
                    gt=torch.rand(*grid.shape,3,dtype=torch.float64,generator=generator)
                    pred=torch.rand(*grid.shape,3,dtype=torch.float64,generator=generator)
                    if case=='identical':pred=gt.clone()
                    elif case=='flat':pred.fill_(.4);gt.fill_(.5)
                    elif case=='constant_residual':pred=gt+.1
                    elif case=='near_flat':pred=.5+pred*1e-7;gt=.5+gt*1e-7
                    pred.requires_grad_()
                    actual=C.reference_loss(pred,gt,weight,k)
                    reference=dense_ssim_reference(pred,gt,exact)
                    torch.testing.assert_close(actual,reference,atol=2e-13,rtol=2e-11)
                    ga,=torch.autograd.grad(actual,pred,retain_graph=True)
                    gb,=torch.autograd.grad(reference,pred)
                    torch.testing.assert_close(ga,gb,atol=2e-12,rtol=2e-10)
                    self.assertTrue(bool(torch.isfinite(ga).all()))

    def test_sigma1p5_fractional_cutoff_and_required_radius_three(self):
        # Both arrangements are attainable micropatches in complete 6x6 tiles.
        # The axis-aligned cutoff is 4.5 pixels, including (4,4), excluding 5.
        axis=torch.tensor([2,3,6,7],dtype=torch.float64)
        y,x=torch.meshgrid(axis,axis,indexing='ij')
        weight,k=C.reference_weights(y,x,3,1.5)
        full=unpack_weights(weight,k,(4,4))
        self.assertEqual(k,7)
        self.assertGreater(full[0,10],0)  # (2,2) -> (6,6)
        self.assertEqual(full[0,3],0)     # (2,2) -> (2,7)
        self.assertEqual(full[0,12],0)    # (2,2) -> (7,2)
        torch.testing.assert_close(full,dense_coordinate_weights(y,x,1.5),atol=2e-16,rtol=2e-14)
        axis=torch.tensor([4,5,6,7],dtype=torch.float64)
        y,x=torch.meshgrid(axis,axis,indexing='ij')
        weight,k=C.reference_weights(y,x,3,1.5)
        full=unpack_weights(weight,k,(4,4))
        old,oldk=original_coordinate.reference_weights(y,x,3,1.5)
        self.assertGreater(full[0,15],0)  # virtual delta (3,3), physical delta (3,3)
        self.assertEqual(unpack_weights(old,oldk,(4,4))[0,15],0)
        with self.assertRaises(ValueError):C.reference_weights(y,x,3,1.5,candidate_radius=2)
        torch.testing.assert_close(full,dense_coordinate_weights(y,x,1.5),atol=2e-16,rtol=2e-14)
        metadata=P.record(config(sigma=1.5),'coordinate')
        self.assertEqual(metadata['coordinate_physical_support'],4.5)
        self.assertEqual(metadata['coordinate_candidate_radius'],3)
        geometry=ray_geometry(19,25,config(sigma=1.5))
        self.assertEqual(geometry['physical_support_radius_pixels'],4.5)
        self.assertEqual(geometry['candidate_grid'],7)

    def test_sigma5_cutoff_pair_needs_radius_seven(self):
        y=torch.tensor([4,5,6,7,12,13,18,19],dtype=torch.float64)[:,None].expand(8,2)
        x=torch.tensor([0,1],dtype=torch.float64)[None].expand(8,2)
        full,k=C.reference_weights(y,x,3,5.,candidate_radius=7)
        old,oldk=original_coordinate.reference_weights(y,x,3,5.)
        self.assertGreater(unpack_weights(full,k,(8,2))[0,14],0)
        self.assertEqual(unpack_weights(old,oldk,(8,2))[0,14],0)
        torch.testing.assert_close(unpack_weights(full,k,(8,2)),dense_coordinate_weights(y,x,5.),
                                   atol=2e-16,rtol=2e-14)

    def test_patch_loss_matches_independent_ssim_and_gradient_including_constants(self):
        generator=torch.Generator().manual_seed(8)
        indices=torch.arange(20).reshape(5,4)
        for case in ('random','identical','flat','constant_residual','near_flat'):
            gt=torch.rand(23,3,dtype=torch.float64,generator=generator)
            pred=torch.rand(23,3,dtype=torch.float64,generator=generator)
            if case=='identical':pred=gt.clone()
            elif case=='flat':pred.fill_(.4);gt.fill_(.5)
            elif case=='constant_residual':pred=gt+.1
            elif case=='near_flat':pred=.5+pred*1e-7;gt=.5+gt*1e-7
            pred.requires_grad_()
            actual=patch_loss(pred,gt,indices)
            # Four independent observed pixels, never neighbors in virtual grid.
            reference=torch.stack([dense_ssim_reference(pred[ix],gt[ix],
                torch.full((4,4),.25,dtype=torch.float64)) for ix in indices]).mean()
            torch.testing.assert_close(actual,reference,atol=2e-13,rtol=2e-11)
            ga,=torch.autograd.grad(actual,pred,retain_graph=True)
            gb,=torch.autograd.grad(reference,pred)
            torch.testing.assert_close(ga,gb,atol=2e-12,rtol=2e-10)
            self.assertEqual(torch.count_nonzero(ga[20:]),0)
            if case=='identical':self.assertEqual(actual.item(),0)
        pred=torch.rand(8,3,dtype=torch.float64,requires_grad=True)
        gt=torch.rand_like(pred)
        self.assertTrue(torch.autograd.gradcheck(lambda x:patch_loss(x,gt,indices[:2]),(pred,),fast_mode=True))

    def test_patch_objective_keeps_all_l1_rays_and_existing_coefficients(self):
        adapter=PatchInputs(Dataset(),SimpleNamespace,config(),device='cpu')
        batch,_,plans=adapter.make([0],777)
        pred=(batch.rgb_gt+.1).requires_grad_();density=torch.full((4,1),.3);scale=torch.full((4,3),.2)
        total,parts=patch_objective(pred,batch.rgb_gt,plans,density,scale,config())
        reg=.01*density.abs().mean()+.01*scale.abs().mean()
        torch.testing.assert_close(total,.8*(pred-batch.rgb_gt).abs().mean()+.2*parts['patch_SSIM']+reg)
        grad,=torch.autograd.grad(total,pred)
        boundary=adapter.grids[(19,25)].boundary_mask
        torch.testing.assert_close(grad.reshape(-1,3)[boundary],
            torch.full_like(grad.reshape(-1,3)[boundary],.8/pred.numel()),atol=0,rtol=0)

    def test_protocol_and_native_growth_are_explicit_without_forced_cap_arrival(self):
        for scene,cap in P.BASE_CAPS.items():
            args=SimpleNamespace(scene=Path('/data')/scene,downsample=P.DOWNSAMPLE[scene],
                                 growth='native_5percent',growth_end=None,budget='fixed',structural_loss='coordinate')
            for sigma in (1.5,2.):P.validate(config(cap=cap,sigma=sigma),args)
            with self.assertRaises(ValueError):P.validate(config(cap=cap,sigma=5.),args)
            args.structural_loss='patch';P.validate(config(cap=cap),args)
            self.assertTrue(P.record(config(cap=cap),'patch')['coordinate_sigma_config_unused']==2)
            for bad in (config(cap=cap,sigma=5),config(cap=cap+1),config(cap=cap,steps=60000),
                        config(cap=cap,sampler='point'),config(cap=cap,views=2)):
                with self.assertRaises(ValueError):P.validate(bad,args)
        cfg=SimpleNamespace(strategy=SimpleNamespace(add=SimpleNamespace()))
        P.configure_native_add(cfg)
        self.assertEqual(vars(cfg.strategy.add),dict(start_iteration=500,end_iteration=25000,
                                                     frequency=100,placement='opacity'))
        audit=NativeGrowthAudit(1,1000000)
        self.assertEqual((audit.events[0],audit.events[-1]),(600,24900))
        self.assertEqual(audit.count_after(29999),1)  # int(1.05*1) stays one; no forced fill.
        self.assertFalse(P.record(config(),'coordinate')['forced_cap_arrival'])

    def _fake_run(self, root, structural_loss='patch', failure_step=None):
        from sparse_grt import training as T, standalone_backend as B, micropatch as M
        method=config();(root/'config.json').write_text(json.dumps(vars(method)))
        args=SimpleNamespace(output=root/'run',scene=root/'garden',config=root/'config.json',
            manifest=None,protocol='micropatch_singleview',structural_loss=structural_loss,
            keep_checkpoints='7000,15000,30000',budget='fixed',budget_views='all',
            growth='native_5percent',growth_end=None,cache_mib=0,downsample=4,mode='train')
        runtime=B.load();runtime.S.KEYS=['positions'];runtime.B.CachedDataset=lambda ds:ds
        runtime.M.clear=lambda runner:None
        model=SimpleNamespace(num_gaussians=method.cap,positions=torch.zeros(2,3),
                              optimizer=SimpleNamespace(state={}),train=lambda:None)
        strategy=SimpleNamespace(paired_step=0);runner=SimpleNamespace(native=lambda:None)
        batch=SimpleNamespace(**{key:torch.ones(1,1,1,3) for key in ('rays_ori','rays_dir','rgb_gt')})
        inputs=SimpleNamespace(resident=False,resident_bytes=0,geometry={},make=lambda *a:(batch,None,[]))
        def initialize(rt,cfg,m,data,growth_factory):
            self.assertIsNone(growth_factory)
            self.assertEqual((cfg.strategy.add.start_iteration,cfg.strategy.add.end_iteration,
                              cfg.strategy.add.frequency),(500,25000,100))
            return model,strategy,runner,None
        def update(m,s,data,batch,plans,step,method,**kw):
            self.assertIs(kw['loss_backend'],C.loss)
            self.assertIs(kw['objective_backend'],M.patch_objective if structural_loss=='patch' else None)
            if step==failure_step:
                m.num_gaussians-=10
                raise torch.cuda.OutOfMemoryError('CUDA out of memory test')
            return dict(N=method.cap,delta_N=0,L1=.1,patch_SSIM=.2)
        saved=[]
        def save(rt,m,s,out,step,times,mh,keep):
            saved.append((step,times[0]));(out/f'ckpt_{step}.pt').write_bytes(b'CPU fixture')
        ticks=iter(range(1000000))
        with ExitStack() as stack:
            for target,name,value in ((T.backend,'load',runtime),(T.backend,'dataset',[0,1]),
                                      (T,'SelectedInputs',inputs),(M,'PatchInputs',inputs),
                                      (T,'loaded_runtime',{})):
                stack.enter_context(patch.object(target,name,return_value=value))
            stack.enter_context(patch.object(T.backend,'initialize',side_effect=initialize))
            stack.enter_context(patch.object(T.backend,'evaluate',side_effect=AssertionError('Unexpected evaluation')))
            stack.enter_context(patch.object(T,'update',side_effect=update))
            stack.enter_context(patch.object(T,'save',side_effect=save))
            stack.enter_context(patch.object(P,'verify_checkpoint',return_value={'load_verified':True}))
            stack.enter_context(patch.object(T.time,'perf_counter',side_effect=lambda:next(ticks)))
            stack.enter_context(patch.object(torch.cuda,'synchronize'))
            stack.enter_context(patch.object(torch.cuda,'current_device',return_value=0))
            stack.enter_context(patch.object(torch.cuda,'get_device_name',return_value='NVIDIA GeForce RTX 4090'))
            for name in ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved'):
                stack.enter_context(patch.object(torch.cuda,name,return_value=0))
            stack.enter_context(redirect_stdout(io.StringIO()))
            if failure_step is None:T.main(args)
            else:
                with self.assertRaises(torch.cuda.OutOfMemoryError):T.main(args)
        return args.output,saved

    def test_runner_keeps_30k_timing_native_growth_and_seven_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            out,saved=self._fake_run(Path(temporary))
            self.assertEqual([step for step,_ in saved],list(P.CHECKPOINTS))
            self.assertEqual([seconds for _,seconds in saved],list(map(float,P.CHECKPOINTS)))
            complete=json.loads((out/'training_complete.json').read_text())
            self.assertEqual(complete['steps'],30000)
            self.assertEqual(complete['training_seconds'],30000)
            self.assertEqual(complete['evaluation_steps'],list(P.CHECKPOINTS))
            self.assertTrue(complete['evaluation_pending'])
            protocol=json.loads((out/'protocol.json').read_text())
            self.assertEqual(protocol['ssim_backend'],'torch_observed_patch_population_moments')
            self.assertEqual(protocol['native_add_schedule']['last_event'],24900)
            self.assertFalse(protocol['forced_cap_arrival'])
            self.assertTrue(protocol['full_model_regularization'])
            self.assertEqual(len((out/'step_times.csv').read_text().splitlines()),301)

    def test_runner_records_micropatch_oom_without_claiming_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            out,saved=self._fake_run(Path(temporary),'coordinate',2100)
            failure=json.loads((out/'failure.json').read_text())
            self.assertEqual(failure['error_kind'],'cuda_out_of_memory')
            self.assertEqual((failure['step'],failure['attempted_step']),(2100,2101))
            self.assertEqual(failure['last_completed_G'],config().cap)
            self.assertEqual(failure['G'],config().cap-10)
            self.assertEqual(failure['final_checkpoint_saved_step'],2000)
            self.assertFalse((out/'training_complete.json').exists())


if __name__=='__main__':unittest.main()
