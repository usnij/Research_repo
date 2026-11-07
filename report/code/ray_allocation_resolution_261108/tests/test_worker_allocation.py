"""CPU checks of fail-closed ownership, exact timing, and planned result export."""
import copy
import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools.ray_allocation_261108 import common as C
from tools.ray_allocation_261108 import worker as W
from tools.ray_allocation_261108 import export_results as E


def run_fixture():
    return dict(run_id='one', experiment_id='E1', scene='bicycle', arm='uniform',
        allocation_arm='uniform', train_ds=4, eval_ds=4, train_shape=[1237, 822], eval_shape=[1237, 822],
        train_views=169, test_views=2, initial_N=54275,
        expected_G={str(s): C.count(54275, 3021939, s) for s in C.STEPS},
        rays_first_second=[113436, 113436], total_rays=113436*30000,
        config_path='configs/one.json', config_sha256='abc', execution_order=1,
        method=dict(views=4, block=6, sigma=6., cap=3021939, initialization_seed=777),
        checkpoint_steps=C.STEPS, loss='coordinate')


class WorkerAllocation(unittest.TestCase):
    def test_gate_only_first_node_first_run_can_bypass(self):
        plan = dict(first_run_id='one', first_node_id='node04', plan_signature='xyz')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertTrue(W.gate_ready(root, plan, dict(node_id='node04'), dict(run_id='one')))
            self.assertFalse(W.gate_ready(root, plan, dict(node_id='node01'), dict(run_id='one')))
            self.assertFalse(W.gate_ready(root, plan, dict(node_id='node04'), dict(run_id='two')))
            C.write(root/'.first_2k_gate.json', dict(plan_signature='wrong', first_run_id='one',
                checkpoint_verified=True, step=2000))
            with self.assertRaises(RuntimeError):
                W.gate_ready(root, plan, dict(node_id='node01'), dict(run_id='two'))
            C.write(root/'.first_2k_gate.json', dict(plan_signature='xyz', first_run_id='one',
                checkpoint_verified=True, step=2000))
            self.assertTrue(W.gate_ready(root, plan, dict(node_id='node01'), dict(run_id='two')))

    def test_instance_identity_is_not_just_gpu_model(self):
        node = dict(node_id='node04', instance_id=44)
        hardware = dict(uuid='GPU-123')
        identity = dict(node_id='node04', instance_id=44, gpu_uuid='GPU-123')
        W.validate_identity(identity, node, hardware, {})
        for changed in (dict(identity, instance_id=45), dict(identity, gpu_uuid='GPU-other')):
            with self.assertRaises(RuntimeError):
                W.validate_identity(changed, node, hardware, {})
        with self.assertRaises(RuntimeError):
            W.validate_identity(identity, node, hardware, dict(CONTAINER_ID='45'))

    def test_timing_sum_and_native_growth_are_verified(self):
        run = run_fixture()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            with (out/'step_times.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(E.LOG_COLUMNS)
                for step in range(100, 30001, 100):
                    writer.writerow([step, .25, C.count(run['initial_N'], run['method']['cap'], step)])
            W.verify_timing(out, run, dict(training_seconds=7500.))
            with self.assertRaises(RuntimeError):
                W.verify_timing(out, run, dict(training_seconds=7501.))
            rows = (out/'step_times.csv').read_text().splitlines()
            rows[-1] = '30000,0.25,1'
            (out/'step_times.csv').write_text('\n'.join(rows)+'\n')
            with self.assertRaises(RuntimeError):
                W.verify_timing(out, run, dict(training_seconds=7500.))

    def test_evaluation_cannot_silently_use_training_resolution(self):
        run = dict(run_fixture(), train_ds=8, eval_ds=1, eval_shape=[3115, 2076])
        value = dict(run_id='one', step=30000, views=2, train_downsample=8, eval_downsample=1,
            N=run['expected_G']['30000'], PSNR=25., SSIM=.8, LPIPS=.3,
            per_view=[dict(view=i, image=str(i), width=3115, height=2076,
                           PSNR=25., SSIM=.8, LPIPS=.3) for i in range(2)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'evaluation.json'
            C.write(path, value)
            E.checked_evaluation(path, run, 30000)
            value['per_view'][0]['width'] = 389
            C.write(path, value)
            with self.assertRaises(ValueError):
                E.checked_evaluation(path, run, 30000)

    def test_planned_export_has_all_25_rows_and_no_measured_values(self):
        runs = [dict(run_fixture(), run_id=f'run{i}', execution_order=i+1) for i in range(25)]
        plan = dict(campaign='test', plan_signature='draft', frozen_commit=None,
                    remote_root='/workspace/test', python='/env/python', renderer_settings={},
                    nodes=[dict(node_id='node04', instance_id=44, runs=runs)], changes_from_order=[])
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            out = Path(folder)
            E.export(plan, {}, out)
            with (out/'runs.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 25)
            self.assertTrue(all(r['status']=='planned' and r['training_seconds']=='NA' and
                                r['PSNR']=='NA' and r['final_G']=='NA' for r in rows))
            self.assertEqual(len(C.read(out/'summary.json')['rows']), 25)
            self.assertEqual((out/'commands.txt').read_text().count('# run'), 25)

    def test_oom_requires_explicit_cuda_evidence_and_records_evaluation_step(self):
        run = run_fixture(); plan = dict(plan_signature='p', frozen_commit='c')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); out = root/'runs/one'; out.mkdir(parents=True)
            log = root/'eval.log'; memory = root/'memory.json'
            C.write(out/'training_complete.json', dict(passed=True, steps=30000,
                N=run['expected_G']['30000'], training_seconds=10.))
            C.write(memory, dict(peak_used_mib=23000.))
            log.write_text('RuntimeError: disk is full')
            with self.assertRaises(RuntimeError):
                W.cuda_oom_record(root, plan, run, {}, 'evaluation', log, memory, 2000)
            log.write_text('torch.OutOfMemoryError: CUDA out of memory')
            value = W.cuda_oom_record(root, plan, run, {}, 'evaluation', log, memory, 2000)
            self.assertEqual(value['step'], 2000)
            self.assertEqual(value['G'], run['expected_G']['2000'])
            self.assertEqual(value['peak_device_used_mib'], 23000.)

    def test_cuda_oom_before_structured_failure_keeps_unknown_step_and_G(self):
        run = run_fixture(); plan = dict(plan_signature='p', frozen_commit='c')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            log = root/'train.log'; memory = root/'memory.json'
            log.write_text('torch.OutOfMemoryError: CUDA out of memory during initialization')
            C.write(memory, dict(peak_used_mib=23880.))
            value = W.cuda_oom_record(root, plan, run, {}, 'training', log, memory)
            self.assertEqual(value['stage'], 'init_unknown')
            self.assertIsNone(value['step'])
            self.assertIsNone(value['G'])
            self.assertTrue(value['terminal'])
            self.assertEqual(value['peak_device_used_mib'], 23880.)


if __name__ == '__main__':
    unittest.main()
