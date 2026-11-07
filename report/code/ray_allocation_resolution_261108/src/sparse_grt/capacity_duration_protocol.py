"""Explicit capacity/iteration follow-up to N5; old N5 is not modified.

30k changes only the registered scene cap and keeps N5's native schedules.
60k doubles time-dependent LR horizons and MCMC end bounds. The user
explicitly preserves start=500, the native 1.05 birth multiplier and frequencies.
"""
import bisect
from dataclasses import dataclass

from .native5_protocol import configure_native_add
from .protocol import CHECKPOINTS, StepLog, validate_gpu, verify_checkpoint


HIGH_CAPS = {'bicycle': 6026079, 'garden': 5709543}
BASE_CAPS = {'bicycle': 3021939, 'garden': 2044672}


def schedule_scale(method):
    if method.steps not in (30000, 60000):
        raise ValueError('Capacity/duration experiment requires 30000 or 60000 steps')
    return method.steps // 30000


def checkpoint_steps(method):
    return CHECKPOINTS + ((45000, 60000) if schedule_scale(method) == 2 else ())


def evaluation_steps(method):
    return (30000, 45000, 60000) if schedule_scale(method) == 2 else CHECKPOINTS


def validate(method, args):
    scale = schedule_scale(method)
    scene = args.scene.name.lower()
    caps = HIGH_CAPS if scale == 1 else BASE_CAPS
    if scene not in caps or method.cap != caps[scene]:
        raise ValueError(f'Incorrect registered {scene} cap for {method.steps} steps')
    if method.initialization_seed != 777 or method.views != 1:
        raise ValueError('Capacity/duration experiment requires seed777 and K=1')
    if args.growth != 'native_5percent' or args.growth_end is not None or args.budget != 'fixed':
        raise ValueError('Use the explicit registered cap and native5%, with no target-step override')
    for key, value in dict(l1_weight=.8, structure_weight=.2,
                           opacity_weight=.01, scale_weight=.01).items():
        if getattr(method, key) != value:
            raise ValueError(f'Capacity/duration experiment requires {key}={value}')
    if method.boundary != 'all' or method.grid_sigma != 1.5 or method.grid_window != 11:
        raise ValueError('Keep N5 all-centre coordinate boundary and default dense SSIM')
    if args.structural_loss == 'dense' and scale == 1:
        if method.block != 1:
            raise ValueError('Dense arm requires all rays')
    elif args.structural_loss == 'coordinate':
        allowed = (2., 4.) if scale == 1 else (4.,)
        if method.block != 3 or method.sigma not in allowed:
            raise ValueError('Coordinate arm requires 1/9 rays and registered sigma')
    else:
        raise ValueError('Unregistered structural loss for capacity/duration experiment')


def configure(cfg, method):
    """Modify timing fields only, before model/optimizer construction."""
    scale = schedule_scale(method)
    configure_native_add(cfg)
    if cfg.scheduler.positions.max_steps != 30000:
        raise ValueError('Unexpected source position-LR horizon')
    if (cfg.strategy.relocate.start_iteration, cfg.strategy.relocate.end_iteration,
            cfg.strategy.relocate.frequency) != (500, 25000, 100):
        raise ValueError('Unexpected source relocation schedule')
    if (cfg.strategy.perturb.start_iteration, cfg.strategy.perturb.end_iteration,
            cfg.strategy.perturb.frequency, cfg.strategy.perturb.noise_lr) != (0, 27500, 1, 500000.):
        raise ValueError('Unexpected source noise schedule')
    for name, scheduler in cfg.scheduler.items():
        if scheduler.type == 'skip':
            continue
        if scheduler.type != 'exp' or 'max_steps' not in scheduler:
            raise ValueError(f'Unaudited LR schedule: {name}')
        scheduler.max_steps *= scale
    for name in ('add', 'relocate', 'perturb'):
        operation = cfg.strategy[name]
        operation.end_iteration *= scale
    return dict(schedule_scale=scale, learning_rate_horizons={
        name: int(scheduler.max_steps) for name, scheduler in cfg.scheduler.items()
        if scheduler.type != 'skip'}, mcmc_schedules={
        name: dict(start_exclusive=int(cfg.strategy[name].start_iteration),
                   end_exclusive=int(cfg.strategy[name].end_iteration),
                   frequency=int(cfg.strategy[name].frequency))
        for name in ('add', 'relocate', 'perturb')},
        perturb_noise_lr=float(cfg.strategy.perturb.noise_lr),
        native_growth_multiplier=1.05, native_add_method_preserved=True,
        sh_increase_frequency=int(cfg.model.progressive_training.increase_frequency),
        sh_schedule_changed=False, from_scratch=True,
        n5_30k_direct_schedule_match=(scale == 1))


@dataclass(frozen=True)
class CountAudit:
    """Pure count predictor; does not replace native MCMC methods."""
    initial: int
    target: int
    start: int
    end: int
    frequency: int = 100

    def __post_init__(self):
        if (self.initial <= 0 or self.target < self.initial
                or (self.start, self.end, self.frequency) not in
                ((500, 25000, 100), (500, 50000, 100))):
            raise ValueError('Invalid capacity/duration count audit')

    @property
    def events(self):
        first = (self.start // self.frequency + 1) * self.frequency
        return tuple(range(first, self.end, self.frequency))

    def count_after(self, step):
        count = self.initial
        for _ in self.events[:bisect.bisect_right(self.events, step)]:
            count = min(self.target, int(1.05 * count))
        return count

    def record(self):
        count, rows = self.initial, []
        for step in self.events:
            after = min(self.target, int(1.05 * count))
            rows.append(dict(step=step, completed_steps=step+1, before=count,
                             after=after, added=after-count, relative_add=(after-count)/count))
            count = after
        first = (0 if self.initial == self.target else
                 next((r['completed_steps'] for r in rows if r['after'] == self.target), None))
        return dict(policy='native_5percent', audit_only=True, **vars(self), events=rows,
                    first_cap_completed_step=first,
                    step_indexing='zero-based backend step; completed_steps=step+1')


def count_audit(initial, method, cfg):
    return CountAudit(initial, method.cap, int(cfg.strategy.add.start_iteration),
                      int(cfg.strategy.add.end_iteration), int(cfg.strategy.add.frequency))
