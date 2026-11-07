import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from sparse_grt.config import MethodConfig
from sparse_grt.inputs import SelectedInputs
from sparse_grt.engine import update
from sparse_grt.losses import coordinate_ssim as C
from sparse_grt.objective import objective
from sparse_grt.sampling import StratifiedGrid, view_sequence

ROOT=Path(__file__).resolve().parents[1]


def original_sample(h,w,b,generator):
    by,bx=torch.meshgrid(torch.arange(0,h,b),torch.arange(0,w,b),indexing="ij")
    by,bx=by.reshape(-1),bx.reshape(-1)
    vh,vw=(h-by).clamp(max=b),(w-bx).clamp(max=b)
    ry=torch.rand(by.shape,generator=generator)
    rx=torch.rand(bx.shape,generator=generator)
    return by+torch.floor(ry*vh).long(),bx+torch.floor(rx*vw).long(),(vh*vw).double()/(h*w)


class Components(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_default_sources_exactly_preserved(self):
        for name,entry in json.loads((ROOT/"provenance/source_manifest.json").read_text()).items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),entry["sha256"])

    def test_sampling_and_rng_state_exact(self):
        for h,w,b in [(1038,1558,8),(19,23,4),(3,2,8),(24,24,8)]:
            grid=StratifiedGrid(h,w,b)
            for seed in range(4):
                old_g=torch.Generator().manual_seed(seed);new_g=torch.Generator().manual_seed(seed)
                ref=original_sample(h,w,b,old_g);got=grid.sample(new_g)
                for a,z in zip(ref,(got.y,got.x,got.weights)):self.assertTrue(torch.equal(a,z))
                self.assertTrue(torch.equal(old_g.get_state(),new_g.get_state()))

    def test_mv4_sequence_preserved(self):
        rng=np.random.default_rng(2610297100);primary=[]
        while len(primary)<30000:primary.extend(rng.permutation(210).tolist())
        expected=[]
        for first in primary[:30000]:
            expected.append([first]+rng.choice([v for v in range(210) if v!=first],3,replace=False).tolist())
        self.assertEqual(view_sequence(210,30000,2610297100),expected)

    def test_loss_identity_derivative_and_overlapping_support(self):
        torch.manual_seed(8)
        h,w=13,15;grid=StratifiedGrid(h*8,w*8)
        samples=grid.sample(torch.Generator().manual_seed(3))
        weight,k=C.reference_weights(samples.y.reshape(h,w).double(),samples.x.reshape(h,w).double())
        gt=torch.rand(h,w,3,dtype=torch.float64)
        for case in ("random","identical","flat","constant_residual","near_flat"):
            pred=torch.rand_like(gt)
            target=gt.clone()
            if case=="identical":pred=gt.clone()
            elif case=="flat":pred.fill_(.4);target.fill_(.5)
            elif case=="constant_residual":pred=target+.1
            elif case=="near_flat":pred=.5+pred*1e-7;target=.5+target*1e-7
            pred.requires_grad_()
            stable=C.reference_loss(pred,target,weight,k)
            raw=C.reference_loss(pred,target,weight,k,False)
            gs,=torch.autograd.grad(stable,pred,retain_graph=True)
            gr,=torch.autograd.grad(raw,pred)
            torch.testing.assert_close(stable,raw,atol=1e-12,rtol=1e-10)
            torch.testing.assert_close(gs,gr,atol=1e-12,rtol=1e-10)
            if case=="identical":
                self.assertEqual(stable.item(),0);self.assertEqual(torch.count_nonzero(gs),0)
            elif case=="random":
                # A non-centre border sample must still receive neighbor gradients.
                self.assertGreater(torch.count_nonzero(gs[:k//2]),0)

    def test_coefficients_and_regularization(self):
        p=torch.full((4,1,2,3),.4,requires_grad=True);g=torch.full_like(p,.5)
        density=torch.full((5,1),.3,requires_grad=True);scale=torch.full((5,3),.2,requires_grad=True)
        cfg=MethodConfig(l1_weight=.8,structure_weight=0)
        total,parts=objective(p,g,[],density,scale,cfg)
        expected=.8*(p-g).abs().mean()+(.01*density.abs().mean()+.01*scale.abs().mean())
        torch.testing.assert_close(total,expected,atol=0,rtol=0)
        grad,=torch.autograd.grad(total,p)
        torch.testing.assert_close(grad,torch.full_like(p,-.8/p.numel()))

    def test_selected_uint8_pipeline_and_exact_geometry(self):
        class Dataset:
            def __init__(self):
                h,w=72,80
                self.rgb=[torch.randint(256,(1,h,w,3),dtype=torch.uint8) for _ in range(4)]
                self.poses=[torch.eye(4)[None] for _ in range(4)]
                self.ro=torch.zeros(1,h,w,3);self.rd=torch.randn(1,h,w,3)
            def __len__(self):return 4
            def __getitem__(self,i):return dict(data=self.rgb[i],pose=self.poses[i],intr=0)
            def _lazy_worker_intrinsics_cache(self):return {0:({},self.ro,self.rd,"test")}
        torch.manual_seed(19);ds=Dataset();cfg=MethodConfig()
        for budget in (0,10**8):
            inputs=SelectedInputs(ds,SimpleNamespace,cfg,device="cpu",resident_budget_bytes=budget,
                                  weight_fn=C.reference_weights)
            batch,_,plans=inputs.make([0,1,2,3],910000000)
            for j in range(4):
                y,x,_=original_sample(72,80,8,torch.Generator().manual_seed(910000000+104729*j))
                expected=(ds.rgb[j]/255.)[:,y,x].reshape(1,1,-1,3)
                self.assertTrue(torch.equal(batch.rgb_gt[j:j+1],expected))
                self.assertTrue(torch.equal(plans[j]["y"],y))
                self.assertTrue(torch.equal(plans[j]["x"],x))
                self.assertTrue(torch.equal(batch.rays_dir[j:j+1],ds.rd[:,y,x].reshape(1,1,-1,3)))
        all_values=torch.arange(256,dtype=torch.uint8).reshape(16,16)
        ix=torch.tensor([1,8,15]);iy=torch.tensor([2,10,15])
        self.assertTrue(torch.equal((all_values/255.)[iy,ix],all_values[iy,ix]/255.))

    def test_configuration_does_not_renormalize_weights(self):
        cfg=MethodConfig(l1_weight=1.,structure_weight=.4)
        self.assertEqual(cfg.l1_weight,1.)
        self.assertEqual(cfg.nominal_total_ray_fraction(),1/16)

    def test_update_order_and_adam_match_direct_loop(self):
        # A CPU backend contract check, not a substitute for native GRT/MCMC parity.
        import copy
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                for key in ("positions","rotation","scale","density","features_albedo","features_specular"):
                    setattr(self,key,torch.nn.Parameter(torch.full((5,3),.1)))
                self.optimizer=torch.optim.Adam(self.parameters(),lr=.001)
                self.scene_extent=1.;self.progressive_training=False
                self.num_gaussians=5;self.n_active_features=0;self.calls=[]
            def forward(self,batch,train,frame_id):
                self.calls.append("forward")
                return dict(pred_rgb=self.positions[0].reshape(1,1,1,3).expand_as(batch.rgb_gt))
            def get_density(self):return self.density.sigmoid()
            def get_scale(self):return self.scale.exp()
            def scheduler_step(self,t):self.calls.append("scheduler")
            def build_acc(self,rebuild):self.calls.append("BVH")
        class Strategy:
            def __init__(self,model):self.model=model
            def pre_backward(self,*a,**k):self.model.calls.append("pre")
            def post_backward(self,*a,**k):self.model.calls.append("post")
            @torch.no_grad()
            def post_optimizer_step(self,*a,**k):
                self.model.calls.append("MCMC")
                # A state-mutating hook must happen after Adam/scheduler.
                self.model.positions[0].add_(.01)
        a,b=Model(),Model();sa,sb=Strategy(a),Strategy(b)
        cfg=MethodConfig(l1_weight=.8,structure_weight=0)
        batch=SimpleNamespace(rgb_gt=torch.full((4,1,3,3),.3))
        for t in (0,1):
            a.optimizer.zero_grad(set_to_none=True)
            pred=a(batch,train=True,frame_id=t)["pred_rgb"]
            reg=.01*a.get_density().abs().mean()+.01*a.get_scale().abs().mean()
            loss=.8*(pred-batch.rgb_gt).abs().mean()+reg
            sa.pre_backward(t,1.,None,batch=batch);loss.backward()
            sa.post_backward(t,1.,None,batch=batch)
            a.optimizer.step();a.optimizer.zero_grad(set_to_none=True);a.scheduler_step(t)
            sa.paired_step=t;sa.post_optimizer_step(t,1.,None,batch=batch);a.build_acc(rebuild=True)
            update(b,sb,None,batch,[],t,cfg,loss_backend=C.reference_loss,
                   feature_step_condition=lambda *args:False)
            self.assertEqual(a.calls,b.calls)
            for pa,pb in zip(a.parameters(),b.parameters()):
                torch.testing.assert_close(pa,pb,atol=0,rtol=0)
                for key,value in a.optimizer.state[pa].items():
                    torch.testing.assert_close(value,b.optimizer.state[pb][key],atol=0,rtol=0)


if __name__=="__main__":unittest.main()
