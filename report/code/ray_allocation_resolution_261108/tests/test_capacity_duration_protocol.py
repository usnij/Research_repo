"""CPU checks for exact schedule changes, checkpoint contract and OOM evidence."""
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

from sparse_grt.config import MethodConfig
from sparse_grt.growth import NativeGrowthAudit
from sparse_grt import capacity_duration_protocol as P
from sparse_grt import native5_protocol as N5


ROOT = Path(__file__).resolve().parents[1]


def config(steps=60000, cap=None):
    return MethodConfig(views=1, block=3, sigma=4., l1_weight=.8, boundary='all',
                        steps=steps, cap=cap or (3021939 if steps == 60000 else 6026079))


def args(root):
    return SimpleNamespace(output=root/'run', scene=root/'bicycle', config=root/'config.json',
        manifest=None, protocol='native5_capacity_duration', structural_loss='coordinate',
        keep_checkpoints='7000,15000,30000', budget='fixed', budget_views='all',
        growth='native_5percent', growth_end=None, cache_mib=0, downsample=4, mode='train')


class CapacityDuration(unittest.TestCase):
    def test_only_registered_eight_arms_are_accepted(self):
        a=args(Path('/tmp'))
        for scene in ('bicycle', 'garden'):
            a.scene=Path('/data')/scene
            base=config(30000, P.HIGH_CAPS[scene])
            for sigma in (2.,4.):
                a.structural_loss='coordinate';P.validate(replace(base,sigma=sigma),a)
            a.structural_loss='dense';P.validate(replace(base,block=1),a)
            a.structural_loss='coordinate';P.validate(config(60000,P.BASE_CAPS[scene]),a)
        a=args(Path('/tmp'))
        for bad in (replace(config(),sigma=2), replace(config(),cap=6026079),
                    replace(config(),views=4), replace(config(),block=4),
                    replace(config(),steps=45000), replace(config(),l1_weight=1.)):
            with self.assertRaises(ValueError):P.validate(bad,a)
        a.structural_loss='dense'
        with self.assertRaises(ValueError):P.validate(replace(config(),block=1),a)

    def test_highcap_schedule_is_exactly_unchanged_from_n5(self):
        path=ROOT/'src/sparse_grt/backend_config.json'
        old=OmegaConf.create(json.loads(path.read_text()));new=OmegaConf.create(old)
        N5.configure_native_add(old)
        report=P.configure(new,config(30000))
        self.assertEqual(OmegaConf.to_container(new),OmegaConf.to_container(old))
        self.assertEqual(report['schedule_scale'],1)
        self.assertTrue(report['n5_30k_direct_schedule_match'])
        for initial,target in ((54275,6026079),(138766,5709543)):
            old_audit=NativeGrowthAudit(initial,target)
            audit=P.CountAudit(initial,target,500,25000)
            self.assertEqual(old_audit.record(),audit.record())

    def test_longrun_scales_lr_add_relocation_noise_but_not_frequency_or_sh(self):
        cfg=OmegaConf.create(json.loads((ROOT/'src/sparse_grt/backend_config.json').read_text()))
        report=P.configure(cfg,config())
        self.assertEqual(cfg.scheduler.positions.max_steps,60000)
        self.assertEqual(cfg.scheduler.positions.lr_init,.00016)
        self.assertEqual(cfg.scheduler.positions.lr_final,.0000016)
        for name in ('add','relocate'):
            self.assertEqual((cfg.strategy[name].start_iteration,cfg.strategy[name].end_iteration,
                              cfg.strategy[name].frequency),(500,50000,100))
        self.assertEqual((cfg.strategy.perturb.start_iteration,cfg.strategy.perturb.end_iteration,
                          cfg.strategy.perturb.frequency,cfg.strategy.perturb.noise_lr),(0,55000,1,500000.))
        self.assertEqual(cfg.model.progressive_training.increase_frequency,1000)
        self.assertFalse(report['n5_30k_direct_schedule_match'])
        for initial,target in ((54275,3021939),(138766,2044672)):
            audit=P.CountAudit(initial,target,500,50000)
            self.assertEqual((audit.events[0],audit.events[-1]),(600,49900))
            expected=initial
            for step in range(60000):
                if 500 < step < 50000 and step%100 == 0:
                    expected=min(target,int(1.05*expected))
                if step%100 == 0 or step in (1099,1101,49999,50000,59999):
                    self.assertEqual(audit.count_after(step),expected)
            self.assertEqual(expected,target)

    def _fake_train(self, root, failure_step=None):
        from sparse_grt import training as T
        from sparse_grt import standalone_backend as B
        method=config();a=args(root);a.config.write_text(json.dumps(vars(method)))
        model=SimpleNamespace(num_gaussians=method.cap,positions=torch.zeros(2,3),
                              optimizer=SimpleNamespace(state={}),train=lambda:None)
        runtime=B.load();runtime.S.KEYS=['positions'];runtime.B.CachedDataset=lambda ds:ds
        runtime.M.clear=lambda runner:None
        strategy=SimpleNamespace(paired_step=0);runner=SimpleNamespace(native=lambda:None)
        fake_inputs=SimpleNamespace(resident=False,resident_bytes=0,geometry={})
        batch=SimpleNamespace(**{key:torch.ones(1,1,1,3) for key in ('rays_ori','rays_dir','rgb_gt')})
        fake_inputs.make=lambda views,seed:(batch,None,[])
        def initialize(rt,cfg,m,data,growth_factory):
            self.assertIsNone(growth_factory)
            self.assertEqual(cfg.scheduler.positions.max_steps,60000)
            self.assertEqual(cfg.strategy.perturb.end_iteration,55000)
            self.assertEqual(cfg.strategy.add.start_iteration,500)
            return model,strategy,runner,None
        def update(m,s,data,batch,plans,step,method,**kw):
            if step == failure_step:
                model.num_gaussians-=10  # Simulate a partial mutation before failure.
                raise torch.cuda.OutOfMemoryError('CUDA out of memory test')
            return dict(N=method.cap,delta_N=0,L1=.1,coordinate_SSIM=.2)
        saved=[]
        def save(rt,m,s,out,step,times,mh,keep):
            saved.append((step,times[0]));(out/f'ckpt_{step}.pt').write_bytes(b'test')
        ticks=iter(range(1000000))
        with ExitStack() as stack:
            for target,name,value in ((T.backend,'load',runtime),(T.backend,'dataset',[0,1]),
                                     (T,'SelectedInputs',fake_inputs),(T,'loaded_runtime',{})):
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
            stack.enter_context(redirect_stdout(io.StringIO()))
            if failure_step is None:T.main(a)
            else:
                with self.assertRaises(torch.cuda.OutOfMemoryError):T.main(a)
        return a.output,saved

    def test_60k_retains_2k_gate_and_evaluates_only_30_45_60k(self):
        with tempfile.TemporaryDirectory() as temporary:
            out,saved=self._fake_train(Path(temporary))
            self.assertEqual([s for s,t in saved],list(P.checkpoint_steps(config())))
            self.assertEqual([t for s,t in saved],[float(s) for s,t in saved])
            complete=json.loads((out/'training_complete.json').read_text())
            self.assertEqual(complete['steps'],60000)
            self.assertEqual(complete['evaluation_steps'],[30000,45000,60000])
            protocol=json.loads((out/'protocol.json').read_text())
            self.assertEqual(protocol['schedule_scale'],2)
            self.assertEqual(protocol['native_add_schedule']['first_event'],600)
            self.assertFalse((out/'failure.json').exists())

    def test_oom_distinguishes_completed_from_attempted_and_partial_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            out,saved=self._fake_train(Path(temporary),failure_step=2100)
            failure=json.loads((out/'failure.json').read_text())
            self.assertEqual(failure['error_kind'],'cuda_out_of_memory')
            self.assertEqual(failure['step'],2100)
            self.assertEqual(failure['attempted_step'],2101)
            self.assertEqual(failure['last_completed_G'],3021939)
            self.assertEqual(failure['G'],3021929)
            self.assertEqual(failure['final_checkpoint_saved_step'],2000)
            self.assertEqual(failure['training_seconds'],2100.)
            self.assertFalse((out/'training_complete.json').exists())
            self.assertFalse((out/'complete.json').exists())


if __name__ == '__main__':unittest.main()
