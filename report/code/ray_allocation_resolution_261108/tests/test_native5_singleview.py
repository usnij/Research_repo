"""CPU mathematics/recording tests. GPU loss timing/parity remain separate."""
import ast
from contextlib import redirect_stdout
from dataclasses import replace
import csv
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sparse_grt.config import MethodConfig
from sparse_grt.controls import GridInputs, grid_objective
from sparse_grt.growth import NativeGrowthAudit
from sparse_grt.losses.grid_ssim import gaussian_weights, valid_ssim
from sparse_grt.losses.coordinate_ssim_all import reference_weights, reference_loss, radius_for
from sparse_grt import native5_protocol as P


def full_kernel_reference(x, y, window, sigma):
    w = gaussian_weights(window, sigma, dtype=x.dtype, device=x.device)
    kernel = (w[:, None]*w[None, :])[None, None].expand(x.shape[1], 1, window, window)
    conv = lambda v: F.conv2d(v, kernel, groups=x.shape[1])
    mx, my = conv(x), conv(y)
    vx, vy, cov = conv(x*x)-mx*mx, conv(y*y)-my*my, conv(x*y)-mx*my
    return (((2*mx*my+.0001)*(2*cov+.0009))/
            ((mx*mx+my*my+.0001)*(vx+vy+.0009))).mean()


class Native5SingleView(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_even_window_is_half_pixel_symmetric_and_normalized(self):
        w = gaussian_weights(4, 1.5, dtype=torch.float64, device='cpu')
        torch.testing.assert_close(w, w.flip(0), atol=0, rtol=0)
        self.assertAlmostEqual(float(w.sum()), 1.)
        expected = torch.exp(-torch.tensor([2.25,.25,.25,2.25], dtype=torch.float64)/4.5)
        torch.testing.assert_close(w, expected/expected.sum(), atol=0, rtol=0)

    def test_separable_4x4_matches_full_2d_moments_and_gradients(self):
        generator = torch.Generator().manual_seed(777)
        x = torch.rand(2,3,7,8, dtype=torch.float64, generator=generator, requires_grad=True)
        y = torch.rand(2,3,7,8, dtype=torch.float64, generator=generator)
        actual = valid_ssim(x,y)
        reference = full_kernel_reference(x,y,4,1.5)
        torch.testing.assert_close(actual, reference, atol=2e-14, rtol=1e-12)
        ga, = torch.autograd.grad(actual,x,retain_graph=True)
        gb, = torch.autograd.grad(reference,x)
        torch.testing.assert_close(ga,gb,atol=2e-14,rtol=1e-11)
        self.assertTrue(torch.all(ga.abs().sum(dim=(0,1)) > 0))
        self.assertAlmostEqual(float(valid_ssim(x,x)),1.,places=12)
        self.assertAlmostEqual(float(actual),float(valid_ssim(x.flip(-1),y.flip(-1))),places=12)

    def test_4x4_finite_difference_gradient(self):
        x=torch.rand(1,1,5,5,dtype=torch.float64,requires_grad=True)
        y=torch.rand_like(x)
        self.assertTrue(torch.autograd.gradcheck(lambda a: valid_ssim(a,y),(x,),fast_mode=True))

    def test_grid11_keeps_identical_fused_call_and_grid4_uses_no_fused(self):
        x=torch.rand(1,1,13*17,3,dtype=torch.float64,requires_grad=True)
        y=torch.rand_like(x);density=torch.rand(4,1);scale=torch.rand(4,3)
        config=MethodConfig(views=1,block=3,l1_weight=.8,boundary='all')
        reference=lambda a,b,padding: full_kernel_reference(a,b,11,1.5)
        with patch.dict('sys.modules',fused_ssim=SimpleNamespace(fused_ssim=reference)):
            total,_=grid_objective(x,y,[dict(shape=(13,17))],density,scale,config)
            xp=x.reshape(1,13,17,3).permute(0,3,1,2).contiguous()
            yp=y.reshape(1,13,17,3).permute(0,3,1,2).contiguous()
            reg=.01*density.abs().mean()+.01*scale.abs().mean()
            expected=.8*(x-y).abs().mean()+.2*(1-reference(xp,yp,'valid'))+reg
            torch.testing.assert_close(total,expected,atol=0,rtol=0)
            a,=torch.autograd.grad(total,x,retain_graph=True)
            b,=torch.autograd.grad(expected,x)
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        def forbidden(*a,**k):raise AssertionError('Grid4 called fused11')
        with patch.dict('sys.modules',fused_ssim=SimpleNamespace(fused_ssim=forbidden)):
            four,_=grid_objective(x,y,[dict(shape=(13,17))],density,scale,replace(config,grid_window=4))
            self.assertTrue(torch.isfinite(four))

    def test_grid_window_does_not_change_sampled_rays(self):
        from test_cvpr_protocol import Dataset
        config=MethodConfig(views=1,block=3,l1_weight=.8,boundary='all')
        data=Dataset()
        a=GridInputs(data,SimpleNamespace,config,device='cpu')
        b=GridInputs(data,SimpleNamespace,replace(config,grid_window=4),device='cpu')
        for seed in (777,910000000):
            ba,_,pa=a.make([2],seed);bb,_,pb=b.make([2],seed)
            for key in ('rays_ori','rays_dir','rgb_gt'):
                torch.testing.assert_close(getattr(ba,key),getattr(bb,key),atol=0,rtol=0)
            self.assertEqual(pa[0]['k'],11);self.assertEqual(pb[0]['k'],4)
        self.assertEqual(b.geometry[0]['window_centre_offset_from_top_left'],1.5)

    def test_coordinate_support_uses_each_requested_sigma(self):
        yy,xx=torch.meshgrid(torch.arange(12)*3,torch.arange(13)*3,indexing='ij')
        for sigma in (1.,1.5,2.,3.,4.):
            weights,k=reference_weights(yy,xx,3,sigma)
            self.assertEqual(k,2*radius_for(3,sigma)+1)
            torch.testing.assert_close(weights.sum(1),torch.ones_like(weights.sum(1)))
            self.assertEqual(k,{1.:3,1.5:5,2.:5,3.:7,4.:9}[sigma])

    def test_small_sigma_boundary_weights_and_gradients_are_finite(self):
        for sigma in (1.,1.5):
            for h,w in ((1,1),(1,4),(4,1),(2,2),(12,13)):
                for seed in range(5):
                    generator=torch.Generator().manual_seed(seed)
                    yy,xx=torch.meshgrid(torch.arange(h)*3,torch.arange(w)*3,indexing='ij')
                    yy=yy+torch.randint(3,(h,w),generator=generator)
                    xx=xx+torch.randint(3,(h,w),generator=generator)
                    weights,k=reference_weights(yy,xx,3,sigma)
                    self.assertTrue(torch.isfinite(weights).all())
                    self.assertTrue((weights[0,k*k//2]>0).all())
                    torch.testing.assert_close(weights.sum(1),torch.ones_like(weights.sum(1)))
                    prediction=torch.rand(h,w,3,generator=generator,requires_grad=True)
                    target=torch.rand(h,w,3,generator=generator)
                    loss=reference_loss(prediction,target,weights,k)
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_native_schedule_matches_actual_native_integer_target(self):
        root=Path(__file__).resolve().parents[1]
        source=root/'src/sparse_grt_backend/strategy/mcmc.py'
        tree=ast.parse(source.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MCMCStrategy')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='add_new_gaussians')
        scope={'torch':torch,'logger':SimpleNamespace(info=lambda *a:None)}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),scope)
        for initial,target in ((206613,1209886),(138766,2044672),(54275,3021939)):
            plan=NativeGrowthAudit(initial,target);n=initial
            self.assertEqual(plan.events[0],600);self.assertEqual(plan.events[-1],24900)
            self.assertEqual(len(plan.events),244)
            self.assertEqual(plan.count_after(599),initial)
            for step in plan.events:
                n=min(target,int(1.05*n));self.assertEqual(plan.count_after(step),n)
            self.assertEqual(n,target)
        # Exercise the untouched native method itself, with small CPU tensors.
        for n,cap in ((41,100),(97,100),(100,100)):
            sampled=[];model=SimpleNamespace(num_gaussians=n,get_density=lambda:torch.ones(n,1))
            fake=SimpleNamespace(model=model,_audit_every=0,conf=SimpleNamespace(strategy=SimpleNamespace(
                add=SimpleNamespace(max_n_gaussians=cap),opacity_threshold=.005,print_stats=False)))
            def sample(count,alive):
                sampled.append(count);idx=alive[:count]
                return idx,torch.ones(count,1),torch.ones(count,3)
            fake.sample_new_gaussians=sample
            fake._update_param_with_optimizer=lambda param,opt:None
            scope['add_new_gaussians'](fake)
            self.assertEqual(sampled,[] if n==cap else [min(cap,int(1.05*n))-n])

    def test_new_protocol_guards_and_restores_native_cutoff(self):
        config=MethodConfig(views=1,block=3,sigma=2,l1_weight=.8,boundary='all')
        args=SimpleNamespace(growth='native_5percent',growth_end=None,structural_loss='coordinate',budget='fixed')
        for sigma in (1,1.5,2,3,4):P.validate(replace(config,sigma=sigma),args)
        for bad in (replace(config,views=4),replace(config,sigma=.5),
                    replace(config,sigma=5),replace(config,sigma=6),replace(config,block=4)):
            with self.assertRaises(ValueError):P.validate(bad,args)
        cfg=SimpleNamespace(strategy=SimpleNamespace(add=SimpleNamespace(start_iteration=599,end_iteration=5001)))
        P.configure_native_add(cfg)
        self.assertEqual(vars(cfg.strategy.add),dict(start_iteration=500,end_iteration=25000,frequency=100,placement='opacity'))

    def test_native_runner_keeps_seven_optimizer_checkpoints_and_no_growth_override(self):
        from sparse_grt import training as T
        from sparse_grt import standalone_backend as B
        config=MethodConfig(views=1,block=3,sigma=2,cap=2,l1_weight=.8,boundary='all')
        model=SimpleNamespace(num_gaussians=2,positions=torch.zeros(2,3),
                              optimizer=SimpleNamespace(state={}),train=lambda:None)
        def state():
            return dict(positions=model.positions,optimizer=dict(state={0:dict(
                step=torch.tensor(float(model.positions[0,0])),exp_avg=torch.ones(2,3),
                exp_avg_sq=torch.ones(2,3))},param_groups=[dict(params=[0],name='positions')]))
        model.get_model_parameters=state
        runtime=B.load();runtime.S.KEYS=['positions'];runtime.B.CachedDataset=lambda ds:ds
        runtime.M.clear=lambda runner:None
        model_strategy=SimpleNamespace(paired_step=0);runner=SimpleNamespace(native=lambda:None)
        fake_inputs=SimpleNamespace(resident=False,resident_bytes=0,geometry={})
        batch=SimpleNamespace(**{key:torch.ones(1,1,1,3) for key in ('rays_ori','rays_dir','rgb_gt')})
        fake_inputs.make=lambda views,seed:(batch,None,[])
        def initialize(rt,cfg,method,data,growth_factory):
            self.assertIsNone(growth_factory)
            self.assertEqual((cfg.strategy.add.start_iteration,cfg.strategy.add.end_iteration),
                             (500,25000))
            return model,model_strategy,runner,None
        def update(model,strategy,data,batch,plans,step,method,**kw):
            model.positions.fill_(step+1)
            return dict(N=2,delta_N=0,L1=.1,coordinate_SSIM=.2)
        ticks=iter(range(1000000))
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'config.json').write_text(json.dumps(vars(config)))
            args=SimpleNamespace(output=root/'run',scene=root/'scene',config=root/'config.json',
                manifest=None,protocol='native5_singleview',structural_loss='coordinate',
                keep_checkpoints='7000,15000,30000',budget='fixed',budget_views='all',
                growth='native_5percent',growth_end=None,cache_mib=0,downsample=2,mode='train')
            with patch.object(T.backend,'load',return_value=runtime), \
                 patch.object(T.backend,'dataset',return_value=[0,1]), \
                 patch.object(T.backend,'initialize',side_effect=initialize), \
                 patch.object(T.backend,'evaluate',side_effect=AssertionError('Unexpected eval')), \
                 patch.object(T,'SelectedInputs',return_value=fake_inputs), \
                 patch.object(T,'update',side_effect=update), \
                 patch.object(T,'loaded_runtime',return_value={}), \
                 patch.object(T.time,'perf_counter',side_effect=lambda:next(ticks)), \
                 patch.object(torch.cuda,'synchronize'), \
                 patch.object(torch.cuda,'current_device',return_value=0), \
                 patch.object(torch.cuda,'get_device_name',return_value='NVIDIA GeForce RTX 4090'), \
                 patch.object(torch.cuda,'get_rng_state_all',return_value=[]),redirect_stdout(io.StringIO()):
                T.main(args)
            self.assertEqual(sorted(int(p.stem.split('_')[1]) for p in args.output.glob('ckpt_*.pt')),list(P.CHECKPOINTS))
            self.assertFalse(list(args.output.glob('evaluation_*.json')))
            complete=json.loads((args.output/'training_complete.json').read_text())
            self.assertTrue(complete['final_optimizer_included']);self.assertTrue(complete['evaluation_pending'])
            self.assertEqual(complete['training_seconds'],30000)
            protocol=json.loads((args.output/'protocol.json').read_text())
            self.assertEqual(protocol['execution_protocol'],'native5_singleview')
            self.assertTrue(protocol['native_add_schedule']['original_add_method_preserved'])
            self.assertEqual(protocol['native_count_audit']['first_cap_completed_step'],0)
            for step in P.CHECKPOINTS:
                receipt=json.loads((args.output/f'checkpoint_verified_{step}.json').read_text())
                self.assertTrue(receipt['optimizer_moment_shapes_verified'])
            with (args.output/'step_times.csv').open() as log:
                self.assertEqual(len(list(csv.DictReader(log))),300)


if __name__=='__main__':unittest.main()
