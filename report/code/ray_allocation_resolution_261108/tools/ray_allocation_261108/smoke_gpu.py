#!/usr/bin/env python3
"""Disposable real-GPU validation; never a campaign training/evaluation result.

Keep the supplied production 30k configuration unchanged. This diagnostic calls
the same backend, sampler, update and checkpoint functions for either one update
or 610 updates (crossing the first native MCMC birth at completed step 601).
No production guard is changed, no campaign receipt/gate is written, and no
checkpoint is deleted. Run on an explicitly assigned, idle RTX 4090 only.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
import unittest

try:
    from .common import now, read, sha, write
    from .worker import execute, gpu_inventory, machine_lock
except ImportError:
    from common import now, read, sha, write
    from worker import execute, gpu_inventory, machine_lock

CODE = Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path, help='Campaign root; shares its GPU lock')
    p.add_argument('--scene', required=True, type=Path)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--downsample', required=True, type=int)
    p.add_argument('--arm', required=True, choices=('uniform', 'schedule', 'residual', 'combined'))
    p.add_argument('--steps', type=int, choices=(1, 610), default=610)
    p.add_argument('--cache-mib', type=int, default=1200)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--_phase', choices=('kernels', 'training'), help=argparse.SUPPRESS)
    return p


def validate_paths(a):
    for name in ('root', 'scene', 'config', 'output'):
        setattr(a, name, getattr(a, name).expanduser().resolve())
    if not a.root.is_dir() or not a.scene.is_dir() or not a.config.is_file():
        raise ValueError('Existing campaign root, scene and production config are required')
    if a.output == a.root / 'runs' or (a.root / 'runs') in a.output.parents:
        raise ValueError('Smoke output must be separate from campaign runs/')
    if a.output == a.scene or a.scene in a.output.parents:
        raise ValueError('Do not write diagnostic outputs inside input data')
    if a.downsample <= 0 or a.cache_mib < 0:
        raise ValueError('Invalid downsample/cache budget')
    if not a._phase and a.output.exists():
        raise FileExistsError(a.output)
    return a


def checkpoint_steps(limit):
    if limit not in (1, 610):
        raise ValueError('Only the bounded 1- or 610-update smoke is supported')
    return {1} if limit == 1 else {1, 601, 610}


def kernels(out):
    """Existing opt-in numerical GPU tests, in a separate fresh process."""
    import torch
    os.environ['RUN_COORDINATE_POINTS_CUDA'] = '1'
    path = CODE / 'tests/test_coordinate_ssim_points.py'
    spec = importlib.util.spec_from_file_location('points_smoke_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(module.PointSSIMCUDA))
    torch.cuda.synchronize()
    record = dict(passed=result.wasSuccessful() and result.testsRun == 3 and not result.skipped,
                  tests_run=result.testsRun, skipped=result.skipped,
                  failures=[dict(test=str(t), trace=s) for t, s in result.failures],
                  errors=[dict(test=str(t), trace=s) for t, s in result.errors],
                  wall_seconds=time.perf_counter()-started,
                  peak_torch_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                  peak_torch_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                  coverage=['irregular points/duplicates/boundaries: sigma 1.5,6,12 vs float64 reference',
                            'regular-grid CUDA loss and RGB-gradient parity: sigma 6,12',
                            '8192 repeated coordinates: multiplicity and nonquadratic grouping'])
    write(out/'kernel_results.json', record)
    if not record['passed']:
        raise RuntimeError('Coordinate SSIM CUDA numerical validation failed')


def phase_check(T, model, runner, inputs, method, rows, P, C):
    """Actual K4 GRT forward/backward at each allocation phase, without updates."""
    import torch
    from sparse_grt.engine import PARAMETER_NAMES
    from sparse_grt.losses.coordinate_ssim_points import validate_coordinates
    checks = []
    before = T.S.digest({k: getattr(model, k) for k in T.S.KEYS})
    for step in (1, 15001):
        row = rows[step-1]
        inputs.set_step(step)
        model.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); tick = time.perf_counter()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            batch, _, plans = inputs.make(row['views'], row['ray_seed'])
            if len(plans) != 4 or len({p['view_id'] for p in plans}) != 4:
                raise RuntimeError('Actual sampler did not produce four distinct views')
            for p in plans:
                h, w = p['image_shape']
                validate_coordinates(p['y'].reshape(-1), p['x'].reshape(-1), h, w)
                if p['end']-p['start'] != p['allocation_budget']['target_count']:
                    raise RuntimeError('Actual sampled count does not match its phase quota')
                if inputs.allocation_arm in ('residual', 'combined'):
                    ids = p['score_blocks'].reshape(-1)
                    if not bool((ids[1:] >= ids[:-1]).all()):
                        raise RuntimeError('Merged rays are not sorted by score block')
            runner.native(); T.M.clear(runner)
            pred = model(batch, train=True, frame_id=row['step'])['pred_rgb']
            total, parts = P.point_objective(pred, batch.rgb_gt, plans, model.get_density(),
                                            model.get_scale(), method, C.loss)
            if not bool(torch.isfinite(total)):
                raise FloatingPointError('Nonfinite real-renderer diagnostic loss')
            total.backward()
            gradients = {}
            for name in PARAMETER_NAMES:
                grad = getattr(model, name).grad
                if grad is not None:
                    if not bool(torch.isfinite(grad).all()):
                        raise FloatingPointError('Nonfinite actual GRT gradient: '+name)
                    gradients[name] = float(grad.abs().max())
            if not gradients or not any(value > 0 for value in gradients.values()):
                raise RuntimeError('Actual GRT backward produced no nonzero model gradient')
            torch.cuda.synchronize()
            checks.append(dict(step=step, rays=sum(p['end']-p['start'] for p in plans),
                               base_rays=sum(p['base_count'] for p in plans),
                               bonus_rays=sum(p['bonus_count'] for p in plans),
                               views=row['views'], loss=float(total), gradient_max_abs=gradients,
                               forward_backward_seconds=time.perf_counter()-tick))
            del batch, plans, pred, total, parts
        model.optimizer.zero_grad(set_to_none=True)
    if T.S.digest({k: getattr(model, k) for k in T.S.KEYS}) != before:
        raise RuntimeError('Diagnostic forward/backward changed model parameters')
    if model.optimizer.state:
        raise RuntimeError('Diagnostic phases initialized Adam without an optimizer update')
    if any(inputs.last_view_updates.values()):
        raise RuntimeError('Diagnostic phases updated residual maps')
    return checks


def training(a, out):
    import torch
    from sparse_grt import allocation_protocol as P, standalone_backend as backend
    from sparse_grt.allocation_inputs import AllocationInputs
    from sparse_grt.config import MethodConfig
    from sparse_grt.contracts import make_manifest, validate_manifest
    from sparse_grt.engine import update
    from sparse_grt.growth import NativeGrowthAudit
    from sparse_grt.losses import coordinate_ssim_all as C
    from sparse_grt.provenance import loaded_runtime
    from sparse_grt.training import save

    method = MethodConfig.read(a.config)
    protocol_args = SimpleNamespace(scene=a.scene, downsample=a.downsample,
        eval_downsample=a.downsample, growth='native_5percent', growth_end=None,
        budget='fixed', structural_loss='coordinate', allocation_arm=a.arm)
    P.validate(method, protocol_args)  # Config still has 30k; no guard monkeypatch.
    model = None; completed = 0; train_seconds = 0.; saved_seconds = 0.
    stage = 'initialization'; attempted = None
    try:
        T = backend.load(); S = T.S
        P.validate_gpu(torch)
        cfg = backend.configure(T, a.scene, a.downsample, method)
        P.configure_native_add(cfg)
        data = T.B.CachedDataset(backend.dataset(T, cfg))
        model, strategy, runner, _ = backend.initialize(T, cfg, method, data)
        audit = NativeGrowthAudit(model.num_gaussians, method.cap)
        initial = dict(N=model.num_gaussians, optimizer_states=len(model.optimizer.state))
        model.train()
        inputs = AllocationInputs(data, S.Batch, method, allocation_arm=a.arm,
                                  resident_budget_bytes=a.cache_mib*2**20, weight_fn=C.weights)
        manifest = make_manifest(method, len(data))
        rows = validate_manifest(manifest, method, len(data)); mh = S.digest(rows)
        write(out/'method.json', vars(method)); write(out/'runtime_manifest.json', loaded_runtime())
        write(out/'resolved_config.json', S.OmegaConf.to_container(cfg, resolve=True))
        write(out/'ray_geometry.json', inputs.geometry)
        write(out/'training_manifest.json', manifest)
        stage = 'sampler_phase_forward_backward'
        write(out/'phase_checks.json', phase_check(T, model, runner, inputs, method, rows, P, C))
        torch.cuda.reset_peak_memory_stats()
        keep = checkpoint_steps(a.steps); growth = 0; cumulative_rays = 0; proofs = []
        with (out/'smoke_steps.jsonl').open('x') as log, P.StepLog(out) as timer:
            for row in rows[:a.steps]:
                t = row['step']; attempted = t+1; stage = 'training_update'
                torch.cuda.synchronize(); tick = time.perf_counter()
                inputs.set_step(t+1)
                batch, _, plans = inputs.make(row['views'], row['ray_seed'])
                runner.native(); T.M.clear(runner)
                values = update(model, strategy, data, batch, plans, t, method, loss_backend=C.loss,
                    feature_step_condition=T.B.check_step_condition, objective_backend=P.point_objective,
                    observation_callback=inputs.update_residual)
                torch.cuda.synchronize(); elapsed = time.perf_counter()-tick
                train_seconds += elapsed
                if not all(math.isfinite(v) for v in values.values()) or values['N'] != audit.count_after(t):
                    raise RuntimeError('Nonfinite update or native MCMC count mismatch: '+str(values))
                rays = sum(p['end']-p['start'] for p in plans); cumulative_rays += rays
                growth += int(values['delta_N'] > 0); completed = t+1
                log.write(json.dumps(dict(step=completed, seconds=elapsed, rays=rays, **values))+'\n')
                timer.append(completed, elapsed, values['N'])
                del batch, plans
                if completed in keep:
                    stage = 'checkpoint_save_verify'; tick = time.perf_counter()
                    state = inputs.state_dict()
                    save(T, model, strategy, out, completed, (train_seconds, 0., saved_seconds), mh,
                         keep, input_state=state)
                    proof = P.verify_checkpoint(T, out/f'ckpt_{completed}.pt', completed,
                                                train_seconds, model.num_gaussians)
                    saved = torch.load(out/f'ckpt_{completed}.pt', map_location='cpu', weights_only=False)
                    if S.digest(saved.get('allocation_input_state')) != S.digest(state):
                        raise RuntimeError('Checkpoint did not retain exact residual-map sampler state')
                    del saved, state
                    proofs.append(proof); saved_seconds += time.perf_counter()-tick
                    write(out/f'checkpoint_verified_{completed}.json', proof)
                if completed == 1 or completed % 50 == 0 or completed == a.steps:
                    log.flush()
                    progress = dict(step=completed, total_steps=a.steps, G=model.num_gaussians,
                                    training_seconds=train_seconds, last_step_seconds=elapsed)
                    write(out/'progress.json', progress); print('SMOKE_PROGRESS', json.dumps(progress), flush=True)
        if a.steps == 610 and initial['N'] < method.cap and growth != 1:
            raise RuntimeError('Expected exactly one native 5% MCMC birth at completed step 601')
        write(out/'smoke_training_result.json', dict(passed=True, benchmark_result=False,
            arm=a.arm, planned_config_steps=method.steps, diagnostic_updates=completed, initial=initial,
            final_G=model.num_gaussians, growth_events=growth, cumulative_rays=cumulative_rays,
            training_seconds=train_seconds, checkpoint_seconds=saved_seconds,
            mean_update_seconds=train_seconds/completed, checkpoint_proofs=proofs,
            peak_torch_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            peak_torch_reserved_bytes=int(torch.cuda.max_memory_reserved()),
            full_training_2k_gate_satisfied=False))
    except BaseException as exc:
        record = dict(passed=False, benchmark_result=False, stage=stage, completed_step=completed,
                      attempted_step=attempted, G=model.num_gaussians if model is not None else None,
                      error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
                      training_seconds=train_seconds)
        for key in ('memory_allocated', 'memory_reserved', 'max_memory_allocated', 'max_memory_reserved'):
            try:
                record[key+'_bytes'] = int(getattr(torch.cuda, key)())
            except Exception:
                pass
        write(out/'failure.json', record)
        raise


def main(argv=None):
    a = validate_paths(parser().parse_args(argv))
    sys.path.insert(0, str(CODE/'src'))
    if a._phase:
        out = a.output/a._phase; out.mkdir(exist_ok=False)
        if a._phase == 'kernels':
            kernels(out)
        else:
            training(a, out)
        return 0
    if os.environ.get('GRT_GRAD_ACCUM_DOUBLE') != '1':
        raise RuntimeError('Set GRT_GRAD_ACCUM_DOUBLE=1, as in the frozen campaign')
    with machine_lock(a.root):
        hardware = gpu_inventory()
        a.output.mkdir(parents=True, exist_ok=False)
        write(a.output/'request.json', dict(timestamp_utc=now(), benchmark_result=False,
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
            hardware=hardware, config_sha256=sha(a.config), script_sha256=sha(__file__),
            frozen_commit=os.environ.get('SPARSE_GRT_FROZEN_COMMIT'),
            purpose='Numerical, actual-renderer and checkpoint smoke only; does not open the full 2k gate'))
        env = dict(os.environ, PYTHONPATH=str(CODE/'src'), PYTHONUNBUFFERED='1')
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--root', str(a.root),
                   '--scene', str(a.scene), '--config', str(a.config), '--downsample', str(a.downsample),
                   '--arm', a.arm, '--steps', str(a.steps), '--cache-mib', str(a.cache_mib),
                   '--output', str(a.output)]
        results = {}
        for phase in ('kernels', 'training'):
            error = None
            try:
                execute(command+['--_phase', phase], a.output/f'{phase}.log',
                        a.output/f'{phase}_vram.json', CODE, env, phase, hardware)
            except BaseException as exc:
                error = exc
            memory_path = a.output/f'{phase}_vram.json'
            memory = read(memory_path) if memory_path.is_file() else {}
            results[phase] = dict(exit_code=memory.get('process_exit_code'), memory=memory)
            if error is not None:
                results[phase]['error'] = str(error)
            write(a.output/'smoke_summary.json', dict(passed=False, benchmark_result=False,
                                                    phases=results, full_training_2k_gate_satisfied=False))
            if error is not None:
                raise RuntimeError(f'{phase} smoke failed; see {a.output / (phase+".log")}') from error
        write(a.output/'smoke_summary.json', dict(passed=True, benchmark_result=False,
            phases=results, training=read(a.output/'training/smoke_training_result.json'),
            full_training_2k_gate_satisfied=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
