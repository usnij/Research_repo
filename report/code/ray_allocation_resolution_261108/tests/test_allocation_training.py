"""Exercise the real trainer's timing/checkpoint/callback routing without a GPU."""
from contextlib import ExitStack,redirect_stdout
import csv,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from sparse_grt.config import MethodConfig
from sparse_grt import allocation_protocol as P

class TrainingContract(unittest.TestCase):
 def test_seven_optimizer_and_input_state_checkpoints_exclude_save_and_eval_time(self):
  from sparse_grt import training as T,standalone_backend as B,allocation_inputs as A
  config=MethodConfig(views=4,block=6,sigma=6.,cap=2,l1_weight=.8,boundary='all')
  model=SimpleNamespace(num_gaussians=2,positions=torch.zeros(2,3),optimizer=SimpleNamespace(state={}),train=lambda:None)
  def state():
   return dict(positions=model.positions,optimizer=dict(state={0:dict(step=model.positions[0,0].clone(),
      exp_avg=torch.ones(2,3),exp_avg_sq=torch.ones(2,3))},param_groups=[dict(params=[0],name='positions')]))
  model.get_model_parameters=state
  rt=B.load();rt.S.KEYS=['positions'];rt.B.CachedDataset=lambda ds:ds;rt.M.clear=lambda r:None
  strategy=SimpleNamespace(paired_step=0);runner=SimpleNamespace(native=lambda:None)
  class FakeInputs:
   resident=False;resident_bytes=0;geometry={0:dict(height=24,width=24)};step=0;callbacks=0
   def set_step(self,t):self.step=t
   def make(self,views,seed):
    batch=SimpleNamespace(**{k:torch.ones(4,1,1,3) for k in ('rays_ori','rays_dir','rgb_gt')})
    plans=[dict(start=i,end=i+1,base_count=1,bonus_count=0,unique_ray_count=1) for i in range(4)]
    return batch,None,plans
   def update_residual(self,*a):self.callbacks+=1
   def state_dict(self):return dict(step=self.step,callbacks=self.callbacks)
  inputs=FakeInputs()
  def initialize(rt,cfg,method,data,growth_factory):
   self.assertIsNone(growth_factory);self.assertEqual(cfg.strategy.add.end_iteration,25000)
   return model,strategy,runner,None
  def update(model,strategy,data,batch,plans,t,method,**kw):
   self.assertEqual(inputs.step,t+1);kw['observation_callback'](batch.rgb_gt,batch.rgb_gt,plans)
   model.positions.fill_(t+1);return dict(N=2,delta_N=0,L1=.1,coordinate_SSIM=.2)
  ticks=iter(range(1000000))
  with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
   root=Path(tmp);(root/'config.json').write_text(json.dumps(vars(config)))
   args=SimpleNamespace(output=root/'run',scene=root/'counter',config=root/'config.json',manifest=None,
      protocol='allocation_resolution',structural_loss='coordinate',allocation_arm='uniform',
      eval_downsample=2,downsample=2,keep_checkpoints='',budget='fixed',budget_views='all',growth='native_5percent',
      growth_end=None,cache_mib=0,mode='train')
   for obj,name,opts in [(T.backend,'load',dict(return_value=rt)),(T.backend,'dataset',dict(return_value=[0,1,2,3])),
     (T.backend,'initialize',dict(side_effect=initialize)),(T.backend,'evaluate',dict(side_effect=AssertionError('Unexpected eval'))),
     (A,'AllocationInputs',dict(return_value=inputs)),(T,'update',dict(side_effect=update)),
     (T,'loaded_runtime',dict(return_value={})),(T.time,'perf_counter',dict(side_effect=lambda:next(ticks)))]:
    stack.enter_context(patch.object(obj,name,**opts))
   stack.enter_context(patch.dict(P.CAPS,{'counter':2}))
   for name in ('synchronize','reset_peak_memory_stats'):stack.enter_context(patch.object(torch.cuda,name))
   for name in ('current_device','max_memory_allocated','max_memory_reserved'):stack.enter_context(patch.object(torch.cuda,name,return_value=0))
   stack.enter_context(patch.object(torch.cuda,'get_device_name',return_value='NVIDIA GeForce RTX 4090'))
   stack.enter_context(patch.object(torch.cuda,'get_rng_state_all',return_value=[]))
   with redirect_stdout(io.StringIO()):T.main(args)
   completed=json.loads((args.output/'training_complete.json').read_text())
   self.assertEqual(completed['training_seconds'],30000);self.assertEqual(completed['cumulative_rays'],120000)
   self.assertEqual(inputs.callbacks,30000);self.assertEqual(completed['evaluation_seconds'],0)
   self.assertEqual(completed['evaluation_steps'],list(P.CHECKPOINTS))
   self.assertFalse(list(args.output.glob('evaluation_*.json')))
   for step in P.CHECKPOINTS:
    checkpoint=torch.load(args.output/f'ckpt_{step}.pt',weights_only=False)
    self.assertEqual(checkpoint['allocation_input_state'],dict(step=step,callbacks=step))
    proof=json.loads((args.output/f'checkpoint_verified_{step}.json').read_text())
    self.assertTrue(proof['optimizer_moment_shapes_verified']);self.assertEqual(proof['training_seconds'],step)
   with (args.output/'step_times.csv').open() as log:
    self.assertEqual(len(list(csv.DictReader(log))),300)
if __name__=='__main__':unittest.main()
