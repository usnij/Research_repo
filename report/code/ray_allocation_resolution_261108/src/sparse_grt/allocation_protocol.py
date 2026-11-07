"""Frozen November 8 ray-allocation and high-resolution comparisons."""
from pathlib import Path
from .protocol import CHECKPOINTS, StepLog, validate_gpu, verify_checkpoint
from .native5_protocol import configure_native_add

CAPS = dict(counter=1229851, garden=2044672, bonsai=1209886, bicycle=3021939)
STANDARD_DS = dict(counter=2, garden=4, bicycle=4)


def validate(method, args):
    scene = Path(args.scene).name
    if scene not in CAPS or method.cap != CAPS[scene]:
        raise ValueError('Use the registered scene cap, independent of image resolution')
    if method.steps != 30000 or method.initialization_seed != 777:
        raise ValueError('Require 30,000 updates and seed 777')
    if args.growth != 'native_5percent' or args.growth_end is not None or args.budget != 'fixed':
        raise ValueError('Require native 5% growth with no forced cap-arrival step')
    for key, value in dict(l1_weight=.8, structure_weight=.2, opacity_weight=.01, scale_weight=.01).items():
        if getattr(method, key) != value:
            raise ValueError(f'Unexpected {key}: expected {value}')
    if method.boundary != 'all' or method.sampler != 'point':
        raise ValueError('Use all observed centres and point sampling')
    if method.grid_window != 11 or method.grid_sigma != 1.5:
        raise ValueError('Dense SSIM stays window 11, sigma 1.5 training pixels')
    arm = args.allocation_arm
    if arm != 'none':
        if (scene not in STANDARD_DS or args.downsample != STANDARD_DS[scene]
                or args.eval_downsample != args.downsample or method.views != 4
                or method.block != 6 or method.sigma != 6 or args.structural_loss != 'coordinate'):
            raise ValueError('E1 requires standard resolution, MV4, reference block6, sigma6')
    else:
        target_ds = 2 if scene == 'garden' else 1
        if scene == 'bicycle' or args.eval_downsample != target_ds:
            raise ValueError('E2 target is DS1 Bonsai/Counter or DS2 Garden')
        if args.structural_loss == 'dense':
            allowed = (2, 8) if scene == 'garden' else (1, 4, 8)
            if args.downsample not in allowed or method.views != 1 or method.block != 1:
                raise ValueError('Unregistered E2 dense resolution/view count')
        elif args.structural_loss == 'coordinate':
            pairs = ((8, 6.),) if scene == 'garden' else ((8, 6.), (16, 12.))
            if args.downsample != target_ds or method.views != 4 or (method.block, method.sigma) not in pairs:
                raise ValueError('Unregistered E2 sparse resolution/bandwidth')
        else:
            raise ValueError('Only dense or coordinate SSIM is registered')


def point_objective(pred, gt, plans, density, scale, config, backend=None):
    """Preserve the established formula and per-view reduction on ragged samples."""
    import torch
    from .losses.coordinate_ssim_points import loss
    rgb = pred.reshape(-1, 3)
    target = gt.reshape(-1, 3)
    terms = []
    for p in plans:
        if p.get('point_mode'):
            h, w = p['image_shape']
            terms.append(loss(rgb[p['start']:p['end']], target[p['start']:p['end']],
                              p['y'].reshape(-1), p['x'].reshape(-1), config.sigma, h, w))
        else:
            terms.append(backend(rgb[p['start']:p['end']].reshape(*p['shape'], 3),
                                 p['gt'][0], p['weights'], p['k']))
    structure = torch.stack(terms).mean()
    l1 = (pred-gt).abs().mean()
    regularization = config.opacity_weight*density.abs().mean()+config.scale_weight*scale.abs().mean()
    return config.l1_weight*l1+config.structure_weight*structure+regularization, dict(
        L1=l1, coordinate_SSIM=structure, regularization=regularization)
