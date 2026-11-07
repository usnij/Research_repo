"""The separately authorized 2026-11-05 single-view/native-growth experiment.

Recording/evaluation use the established seven-checkpoint protocol. Scientific
differences are explicit here and do not relax the frozen cvpr2027 protocol.
"""
from .protocol import CHECKPOINTS, StepLog, validate_gpu, verify_checkpoint


def validate(method, args):
    if method.steps != 30000 or method.initialization_seed != 777 or method.views != 1:
        raise ValueError('Native5 comparison requires 30000 updates, seed777 and K=1')
    if args.growth != 'native_5percent' or args.growth_end is not None:
        raise ValueError('Native5 comparison forbids a log-horizon or growth-end override')
    if args.budget != 'fixed':
        raise ValueError('Reuse the previously registered dense MCMC cap explicitly')
    for key, value in dict(l1_weight=.8, structure_weight=.2,
                           opacity_weight=.01, scale_weight=.01).items():
        if getattr(method, key) != value:
            raise ValueError(f'Native5 comparison requires {key}={value}')
    if method.boundary != 'all' or method.grid_sigma != 1.5:
        raise ValueError('Keep all coordinate centres and grid sigma1.5')
    if args.structural_loss == 'dense':
        if method.block != 1 or method.grid_window != 11:
            raise ValueError('Dense arm requires all rays and the established 11x11 SSIM')
    elif args.structural_loss == 'grid':
        if method.block != 3 or method.grid_window not in (4, 11):
            raise ValueError('Grid arms require sparse1/9 and window4 or11')
    elif args.structural_loss == 'coordinate':
        if method.block != 3 or method.sigma not in (1., 1.5, 2., 3., 4.):
            raise ValueError('Coordinate arms require sparse1/9 and sigma1,1.5,2,3,4')
    else:
        raise ValueError('Unknown structural loss')


def configure_native_add(cfg):
    """Restore the original native add schedule, not the template's old 5k stop.

    The untouched MCMC method checks start<step<end and step%frequency==0,
    hence additions are eligible at 600..24900. Parent selection, relocation,
    perturbation, Adam behaviour, and the 1.05 multiplier are not replaced.
    """
    cfg.strategy.add.start_iteration = 500
    cfg.strategy.add.end_iteration = 25000
    cfg.strategy.add.frequency = 100
    cfg.strategy.add.placement = 'opacity'
