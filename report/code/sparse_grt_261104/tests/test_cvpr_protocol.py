"""CPU contract tests; native CUDA loss/renderer parity is a separate check."""
from contextlib import redirect_stdout
from dataclasses import replace
import csv
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sparse_grt import protocol as P
from sparse_grt.config import MethodConfig
from sparse_grt.controls import DenseInputs, GridInputs, dense_objective, grid_objective
from sparse_grt.inputs import SelectedInputs


def cpu_ssim(x, y, padding='valid'):
    """Differentiable CPU stand-in to test layout, weighting, and control routing."""
    assert padding == 'valid'
    coords = torch.arange(11, dtype=x.dtype, device=x.device)-5
    gaussian = torch.exp(-coords.square()/(2*1.5**2))
    gaussian /= gaussian.sum()
    kernel = (gaussian[:, None]*gaussian[None, :])[None, None].expand(3, 1, 11, 11)
    conv = lambda image: F.conv2d(image, kernel, groups=3)
    mx, my = conv(x), conv(y)
    vx, vy = conv(x*x)-mx*mx, conv(y*y)-my*my
    cov = conv(x*y)-mx*my
    return (((2*mx*my+.01**2)*(2*cov+.03**2))/
            ((mx.square()+my.square()+.01**2)*(vx+vy+.03**2))).mean()


class Dataset:
    def __init__(self):
        h, w = 57, 61
        self.rgb = [torch.randint(256, (1, h, w, 3), dtype=torch.uint8) for _ in range(4)]
        self.ro, self.rd = torch.randn(1, h, w, 3), torch.randn(1, h, w, 3)
        self.pose = torch.eye(4)[None]
        self.pose[0, :3, 3] = torch.tensor([.3, .2, .1])

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return dict(data=self.rgb[index], pose=self.pose, intr=0)

    def _lazy_worker_intrinsics_cache(self):
        return {0: ({}, self.ro, self.rd, 'test')}


class CVPRProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fixed_method_guard(self):
        args = SimpleNamespace(growth='log_horizon', growth_end=5000, structural_loss='coordinate')
        config = MethodConfig(l1_weight=.8, boundary='all')
        P.validate(config, args)
        for changed in (replace(config, steps=7000), replace(config, l1_weight=1.),
                        replace(config, sigma=1.5), replace(config, boundary='valid')):
            with self.assertRaises(ValueError):
                P.validate(changed, args)
        with self.assertRaises(RuntimeError):
            P.validate_gpu(SimpleNamespace(cuda=SimpleNamespace(
                current_device=lambda: 0, get_device_name=lambda _: 'RTX 4070 SUPER')))

    def test_grid_preserves_sampled_rays_and_gt_and_dense_is_full_image(self):
        ds = Dataset()
        config = MethodConfig(views=4, block=4, boundary='all')
        base = SelectedInputs(ds, SimpleNamespace, config, device='cpu',
                              weight_fn=lambda *a: (None, 11))
        grid = GridInputs(ds, SimpleNamespace, config, device='cpu')
        for seed in (1, 777, 910000001):
            expected, _, plans = base.make([0, 1, 2, 3], seed)
            actual, _, grid_plans = grid.make([0, 1, 2, 3], seed)
            for key in ('rays_ori', 'rays_dir', 'rgb_gt'):
                torch.testing.assert_close(getattr(expected, key), getattr(actual, key), rtol=0, atol=0)
            for x, y in zip(plans, grid_plans):
                self.assertTrue(torch.equal(x['x'], y['x']))
                self.assertTrue(torch.equal(x['y'], y['y']))
        dense = DenseInputs(ds, SimpleNamespace, replace(config, views=1, block=1), device='cpu')
        actual, _, plans = dense.make([2], 999)
        self.assertEqual(plans[0]['shape'], (57, 61))
        self.assertEqual(actual.rgb_gt.numel(), 57*61*3)
        torch.testing.assert_close(actual.rgb_gt, (ds.rgb[2]/255.).reshape(1, 1, -1, 3), atol=0, rtol=0)
        torch.testing.assert_close(actual.rays_dir, ds.rd.reshape(1, 1, -1, 3), atol=0, rtol=0)

    def test_historical_grid_and_dense_formula_and_gradients(self):
        # Same packed RGB layout and one fused call over independent view images.
        with patch.dict('sys.modules', fused_ssim=SimpleNamespace(fused_ssim=cpu_ssim)):
            for views, fn in ((1, grid_objective), (4, grid_objective), (1, dense_objective)):
                config = MethodConfig(views=views, l1_weight=.8)
                h, w = 13, 17
                pred = torch.rand(views, 1, h*w, 3, dtype=torch.float64, requires_grad=True)
                target = torch.rand_like(pred)
                density = torch.rand(5, 1, dtype=torch.float64, requires_grad=True)
                scale = torch.rand(5, 3, dtype=torch.float64, requires_grad=True)
                plans = [dict(shape=(h, w)) for _ in range(views)]
                actual, _ = fn(pred, target, plans, density, scale, config)
                xp = pred.reshape(views, h, w, 3).permute(0, 3, 1, 2).contiguous()
                yp = target.reshape(views, h, w, 3).permute(0, 3, 1, 2).contiguous()
                reg = .01*density.abs().mean()+.01*scale.abs().mean()
                expected = .8*(pred-target).abs().mean()+.2*(1-cpu_ssim(xp, yp))+reg
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                inputs = (pred, density, scale)
                a = torch.autograd.grad(actual, inputs, retain_graph=True)
                b = torch.autograd.grad(expected, inputs)
                for x, y in zip(a, b):
                    torch.testing.assert_close(x, y, atol=0, rtol=0)

    def test_checkpoint_rejects_missing_or_wrong_shaped_adam_moments(self):
        positions = torch.nn.Parameter(torch.zeros(2, 3))
        frozen = torch.nn.Parameter(torch.zeros(2, 1), requires_grad=False)
        optimizer = torch.optim.Adam([dict(params=[positions], name='positions'),
                                      dict(params=[frozen], name='a_raw')], lr=.001)
        positions.square().sum().backward()
        optimizer.step()
        state = dict(global_step=2000, training_seconds=12., positions=positions.detach(),
                     a_raw=frozen.detach(), optimizer=optimizer.state_dict(),
                     config=dict(optimizer=dict(params=dict(positions={}, a_raw={}))))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'checkpoint.pt'
            torch.save(state, path)
            receipt = P.verify_checkpoint(SimpleNamespace(torch=torch), path, 2000, 12., 2)
            self.assertEqual(receipt['optimizer_initialized_groups'], ['positions'])
            self.assertTrue(receipt['optimizer_moment_shapes_verified'])
            for corruption in ('missing_second_moment', 'wrong_moment_shape', 'wrong_dtype',
                               'missing_group', 'unknown_group', 'wrong_step'):
                bad = copy.deepcopy(state)
                slot = bad['optimizer']['state'][0]
                if corruption == 'missing_second_moment':
                    del slot['exp_avg_sq']
                elif corruption == 'wrong_moment_shape':
                    slot['exp_avg'] = torch.zeros(3, 3)
                elif corruption == 'wrong_dtype':
                    slot['exp_avg_sq'] = slot['exp_avg_sq'].double()
                elif corruption == 'missing_group':
                    bad['optimizer']['param_groups'].pop()
                elif corruption == 'unknown_group':
                    bad['optimizer']['param_groups'][0]['name'] = 'absent'
                elif corruption == 'wrong_step':
                    slot['step'] = torch.tensor(2001.)
                torch.save(bad, path)
                with self.subTest(corruption=corruption), self.assertRaises(RuntimeError):
                    P.verify_checkpoint(SimpleNamespace(torch=torch), path, 2000, 12., 2)

    def test_30000_update_schedule_persists_seven_optimizer_checkpoints_without_eval(self):
        # The full runner executes on a fake CPU backend; checkpoint I/O is real.
        from sparse_grt import training as T
        from sparse_grt import standalone_backend as B
        config = MethodConfig(views=1, block=4, cap=2, l1_weight=.8, boundary='all')
        model = SimpleNamespace(num_gaussians=2, positions=torch.zeros(2, 3),
                                optimizer=SimpleNamespace(state={}), train=lambda: None)
        def state():
            return dict(positions=model.positions, optimizer=dict(state={0: dict(
                step=torch.tensor(float(model.positions[0, 0])), exp_avg=torch.ones(2, 3),
                exp_avg_sq=torch.ones(2, 3))}, param_groups=[dict(params=[0], name='positions')]))
        model.get_model_parameters = state
        runtime = B.load()
        runtime.S.KEYS = ['positions']
        runtime.B.CachedDataset = lambda ds: ds
        runtime.M.clear = lambda runner: None
        strategy = SimpleNamespace(paired_step=0)
        runner = SimpleNamespace(native=lambda: None)
        fake_inputs = SimpleNamespace(resident=False, resident_bytes=0, geometry={})
        batch = SimpleNamespace(**{key: torch.ones(1, 1, 1, 3)
                                   for key in ('rays_ori', 'rays_dir', 'rgb_gt')})
        fake_inputs.make = lambda views, seed: (batch, None, [])
        def fake_update(model, strategy, data, batch, plans, step, method, **kwargs):
            model.positions.fill_(step+1)
            return dict(N=2, delta_N=0, L1=.1, coordinate_SSIM=.2)
        ticks = iter(range(1000000))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'config.json').write_text(json.dumps(vars(config)))
            args = SimpleNamespace(output=root/'run', scene=root/'scene', config=root/'config.json',
                                   manifest=None, protocol='cvpr2027', structural_loss='coordinate',
                                   keep_checkpoints='7000,15000,30000', budget='fixed', budget_views='all',
                                   growth='log_horizon', growth_end=5000, cache_mib=0,
                                   downsample=2, mode='train')
            with patch.object(T.backend, 'load', return_value=runtime), \
                 patch.object(T.backend, 'dataset', return_value=[0, 1]), \
                 patch.object(T.backend, 'initialize', return_value=(model, strategy, runner, None)), \
                 patch.object(T.backend, 'evaluate', side_effect=AssertionError('Evaluation during training')), \
                 patch.object(T, 'SelectedInputs', return_value=fake_inputs), \
                 patch.object(T, 'update', side_effect=fake_update), \
                 patch.object(T, 'loaded_runtime', return_value={}), \
                 patch.object(T.time, 'perf_counter', side_effect=lambda: next(ticks)), \
                 patch.object(torch.cuda, 'synchronize'), \
                 patch.object(torch.cuda, 'current_device', return_value=0), \
                 patch.object(torch.cuda, 'get_device_name', return_value='NVIDIA GeForce RTX 4090'), \
                 patch.object(torch.cuda, 'get_rng_state_all', return_value=[]), redirect_stdout(io.StringIO()):
                T.main(args)
            output = args.output
            self.assertEqual(sorted(int(p.stem.split('_')[1]) for p in output.glob('ckpt_*.pt')),
                             list(P.CHECKPOINTS))
            for step in P.CHECKPOINTS:
                checkpoint = torch.load(output/f'ckpt_{step}.pt', weights_only=False)
                self.assertEqual(checkpoint['global_step'], step)
                self.assertEqual(checkpoint['training_seconds'], step)
                self.assertEqual(checkpoint['evaluation_seconds'], 0)
                self.assertEqual(checkpoint['positions'][0, 0].item(), step)
                self.assertTrue(checkpoint['optimizer']['state'])
                receipt = json.loads((output/f'checkpoint_verified_{step}.json').read_text())
                self.assertTrue(receipt['load_verified'])
            self.assertEqual(list(output.glob('evaluation_*.json')), [])
            with (output/'step_times.csv').open() as stream:
                logs = list(csv.DictReader(stream))
            self.assertEqual(len(logs), 300)
            self.assertEqual([int(x['step']) for x in logs], list(range(100, 30001, 100)))
            self.assertTrue(all(float(x['mean_step_seconds_100']) == 1. for x in logs))
            complete = json.loads((output/'training_complete.json').read_text())
            self.assertEqual(complete['training_seconds'], 30000)
            self.assertTrue(complete['evaluation_pending'])
            self.assertEqual(complete['evaluation_seconds'], 0)


if __name__ == '__main__':
    unittest.main()
