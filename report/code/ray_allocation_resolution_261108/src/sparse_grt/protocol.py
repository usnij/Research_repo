"""Execution and recording contract for the 2026-11-04 CVPR experiment order.

This module does not change sampling, the renderer, MCMC, or the loss formula.
Evaluation is deliberately a separate phase, run by eval_full.py after training.
"""
import csv
import math
from pathlib import Path


CHECKPOINTS = (2000, 5000, 7000, 10000, 15000, 20000, 30000)


def validate(method, args):
    if method.grid_window != 11 or method.grid_sigma != 1.5:
        raise ValueError('The frozen CVPR protocol keeps the established 11x11 grid SSIM')
    if method.steps != 30000:
        raise ValueError('CVPR protocol requires 30000 planned updates')
    if args.growth != 'log_horizon' or args.growth_end != 5000:
        raise ValueError('CVPR protocol requires the established 600..5000 log growth')
    expected = dict(l1_weight=.8, structure_weight=.2, opacity_weight=.01, scale_weight=.01)
    for key, value in expected.items():
        if getattr(method, key) != value:
            raise ValueError(f'CVPR protocol requires {key}={value}')
    if args.structural_loss == 'coordinate' and (method.sigma != 6 or method.boundary != 'all'):
        raise ValueError('Fixed-scale SSIM requires sigma=6 and all observed centres')
    if args.structural_loss == 'dense' and (method.views != 1 or method.block != 1):
        raise ValueError('Dense SSIM requires K=1 and block=1 (all rays)')


def validate_gpu(torch):
    name = torch.cuda.get_device_name(torch.cuda.current_device())
    if 'RTX 4090' not in name:
        raise RuntimeError(f'CVPR training requires RTX 4090, found {name!r}')
    return name


class StepLog:
    """Write synchronized update-only time; do not include this logger's I/O."""
    def __init__(self, output):
        self.path = Path(output) / 'step_times.csv'
        self.file = self.path.open('w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(('step', 'mean_step_seconds_100', 'G'))
        self.seconds = []

    def append(self, step, elapsed, count):
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Invalid measured update duration')
        self.seconds.append(elapsed)
        if len(self.seconds) == 100:
            self.writer.writerow((step, sum(self.seconds) / 100, count))
            self.file.flush()
            self.seconds.clear()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def verify_checkpoint(runtime, path, step, expected_seconds, count):
    """Read the saved file back; the 2k gate must reflect a real, usable file."""
    torch = runtime.torch
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('global_step') != step or state.get('training_seconds') != expected_seconds:
        raise RuntimeError('Saved checkpoint has incorrect step/time metadata')
    if len(state['positions']) != count:
        raise RuntimeError('Saved checkpoint has incorrect Gaussian count')
    optimizer = state.get('optimizer')
    if not isinstance(optimizer, dict) or not optimizer.get('state') or not optimizer.get('param_groups'):
        raise RuntimeError('Saved checkpoint is missing initialized optimizer state')
    # The established Adam groups each contain one named Gaussian tensor. Do not
    # infer trainability from saved tensors: cpu() intentionally detaches them.
    cfg = state.get('config', {})
    model_cfg = cfg.get('model', {})
    medium_v2 = cfg.get('render', {}).get('splat', {}).get('medium_v2', False)
    flags = dict(positions='optimize_position', density='optimize_density',
                 features_albedo='optimize_features_albedo',
                 features_specular='optimize_features_specular', rotation='optimize_rotation',
                 scale='optimize_scale')
    names, parameter_ids, checked = set(), set(), []
    for group in optimizer['param_groups']:
        name, ids = group.get('name'), group.get('params', [])
        parameter = state.get(name)
        if (name in names or len(ids) != 1 or ids[0] in parameter_ids
                or not torch.is_tensor(parameter) or parameter.ndim == 0
                or parameter.shape[0] != count):
            raise RuntimeError(f'Invalid named Adam parameter group: {name!r}')
        names.add(name)
        parameter_ids.add(ids[0])
        slot = optimizer['state'].get(ids[0])
        frozen = ((name == 'a_raw' and not medium_v2)
                  or (name == 'density' and medium_v2)
                  or (name in flags and not model_cfg.get(flags[name], True)))
        if not slot and frozen:
            continue
        if not isinstance(slot, dict):
            raise RuntimeError(f'Missing Adam state for trainable parameter {name}')
        for key in ('exp_avg', 'exp_avg_sq'):
            moment = slot.get(key)
            if (not torch.is_tensor(moment) or moment.shape != parameter.shape
                    or moment.dtype != parameter.dtype):
                raise RuntimeError(f'Invalid Adam {key} shape/dtype for {name}')
        adam_step = slot.get('step')
        if torch.is_tensor(adam_step):
            if adam_step.numel() != 1:
                raise RuntimeError(f'Invalid Adam step shape for {name}')
            adam_step = adam_step.item()
        if not isinstance(adam_step, (int, float)) or not 0 < adam_step <= step:
            raise RuntimeError(f'Invalid Adam step for {name}')
        checked.append(name)
    if not checked or set(optimizer['state']) - parameter_ids:
        raise RuntimeError('Adam state is empty or contains unmapped parameters')
    expected_groups = set(cfg.get('optimizer', {}).get('params', {}))
    if expected_groups and names != expected_groups:
        raise RuntimeError('Saved Adam groups differ from checkpoint optimizer config')
    return dict(step=step, path=str(Path(path).resolve()), bytes=Path(path).stat().st_size,
                G=count, training_seconds=expected_seconds, optimizer_present=True,
                optimizer_moment_shapes_verified=True, optimizer_initialized_groups=checked,
                load_verified=True)
