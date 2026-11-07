import argparse,importlib.util,sys,unittest
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import torch
from sparse_grt import allocation_protocol as P
from sparse_grt.config import MethodConfig
from sparse_grt.allocation_inputs import allocation_budget

class AllocationProtocol(unittest.TestCase):
 def args(self,scene='counter',arm='uniform',ds=2,eds=2,loss='coordinate'):
  return SimpleNamespace(scene=Path('/data')/scene,allocation_arm=arm,downsample=ds,eval_downsample=eds,
                         structural_loss=loss,growth='native_5percent',growth_end=None,budget='fixed')
 def method(self,scene='counter',**kw):
  return replace(MethodConfig(views=4,block=6,sigma=6.,l1_weight=.8,boundary='all',cap=P.CAPS[scene]),**kw)
 def test_all_25_authorized_arms(self):
  for scene in ('counter','garden','bicycle'):
   for arm in ('uniform','schedule','residual','combined'):
    P.validate(self.method(scene),self.args(scene,arm,P.STANDARD_DS[scene],P.STANDARD_DS[scene]))
  for scene in ('counter','garden','bonsai'):
   eds=2 if scene=='garden' else 1
   for ds in ((2,8) if scene=='garden' else (1,4,8)):
    P.validate(self.method(scene,views=1,block=1),self.args(scene,'none',ds,eds,'dense'))
   for block,sigma in (((8,6.),) if scene=='garden' else ((8,6.),(16,12.))):
    P.validate(self.method(scene,block=block,sigma=sigma),self.args(scene,'none',eds,eds))
 def test_rejects_confounded_recipes(self):
  for kwargs in (dict(views=1),dict(cap=2000000),dict(l1_weight=1.),dict(sigma=12.),dict(boundary='valid')):
   with self.assertRaises(ValueError):P.validate(self.method(**kwargs),self.args())
  a=self.args();a.growth='log_horizon'
  with self.assertRaises(ValueError):P.validate(self.method(),a)
  with self.assertRaises(ValueError):P.validate(self.method('counter',views=1,block=1),self.args('counter','none',4,4,'dense'))
 def test_exact_total_equality_real_shapes(self):
  for h,w in ((1038,1558),(840,1297),(822,1237)):
   totals=[]
   for arm in ('uniform','schedule','residual','combined'):
    totals.append(15000*4*sum(allocation_budget(h,w,arm,s)['target_count'] for s in (1,15001)))
   self.assertEqual(len(set(totals)),1)
 def test_point_objective_reduction_and_gradients(self):
  torch.manual_seed(8);p=torch.rand(8,3,dtype=torch.float64,requires_grad=True);gt=torch.rand_like(p)
  coords=torch.tensor([[0,0],[0,1],[1,0],[1,1]])
  plans=[dict(start=i,end=i+4,point_mode=True,y=coords[:,0],x=coords[:,1],image_shape=(2,2)) for i in (0,4)]
  cfg=self.method();d=torch.tensor([.2,.3]);s=torch.tensor([.1,.7])
  loss,parts=P.point_objective(p,gt,plans,d,s,cfg)
  self.assertAlmostEqual(float(loss),.8*float(parts['L1'])+.2*float(parts['coordinate_SSIM'])+.01*float(d.mean()+s.mean()),places=7)
  loss.backward();self.assertTrue(torch.isfinite(p.grad).all());self.assertTrue((p.grad.abs().sum(1)>0).all())
 def test_eval_cli_exposes_independent_target(self):
  path=Path(__file__).resolve().parents[1]/'eval_full.py'
  spec=importlib.util.spec_from_file_location('eval_allocation_cli',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  args=module.parser().parse_args(['--checkpoint','/c','--data-root','/d','--output','/o','--eval-downsample','1'])
  self.assertEqual(args.eval_downsample,1)
if __name__=='__main__':unittest.main()
