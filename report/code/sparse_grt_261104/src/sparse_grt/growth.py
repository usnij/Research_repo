"""Separate input target capacity from the time needed to reach it.

An explicit common horizon defines a log-linear target trajectory. This is a
schedule choice, not a theorem about quality or an adaptive capacity signal.
The existing opacity parent selection and mass/scale correction are retained.
"""
from dataclasses import dataclass
import bisect
import math


@dataclass(frozen=True)
class GrowthPlan:
    initial: int
    target: int
    first_event: int
    last_event: int
    frequency: int = 100

    def __post_init__(self):
        if any(type(x) is not int for x in (self.initial, self.target, self.first_event,
                                            self.last_event, self.frequency)):
            raise ValueError("Integer counts and steps are required")
        if self.initial <= 0 or self.target < self.initial:
            raise ValueError("Growth never deletes initial points; target must be >= initial")
        if self.frequency <= 0 or self.first_event < 0 or self.last_event < self.first_event:
            raise ValueError("Invalid growth interval")
        if self.first_event % self.frequency or self.last_event % self.frequency:
            raise ValueError("Events must align with the backend frequency")

    @property
    def events(self):
        return tuple(range(self.first_event, self.last_event+1, self.frequency))

    def count_after(self, step):
        j = bisect.bisect_right(self.events, step)
        if j == 0:
            return self.initial
        if j == len(self.events):
            return self.target
        value = self.initial*math.exp(math.log(self.target/self.initial)*j/len(self.events))
        return min(self.target, max(self.initial, math.ceil(value)))

    def record(self):
        n = self.initial
        rows = []
        for t in self.events:
            target = self.count_after(t)
            rows.append(dict(step=t, completed_steps=t+1, before=n, after=target, added=target-n,
                             relative_add=(target-n)/n))
            n = target
        return dict(**vars(self), events=rows, step_indexing='zero-based backend step; completed_steps=step+1',
                    unrounded_relative_add=math.expm1(math.log(self.target/self.initial)/len(rows)))


def add_to_target(strategy, target):
    """Same established MCMC birth operation, with an explicit target count."""
    import torch
    with torch.no_grad():
        current = strategy.model.num_gaussians
        if not current <= target <= strategy.conf.strategy.add.max_n_gaussians:
            raise ValueError("Birth target must preserve count and respect the cap")
        count = target-current
        if not count:
            return
        density = strategy.model.get_density()
        alive = torch.where(density > strategy.conf.strategy.opacity_threshold)[0]
        if not len(alive):
            raise RuntimeError("No eligible MCMC parents")
        parents, new_density, new_scale = strategy.sample_new_gaussians(count, alive)
        def parameters(name, param):
            if name == 'density':
                param[parents] = new_density
            elif name == 'scale':
                param[parents] = new_scale
            return torch.nn.Parameter(torch.cat([param, param[parents]]), requires_grad=param.requires_grad)
        def optimizer(key, value):
            return torch.cat([value, value.new_zeros((len(parents), *value.shape[1:]))])
        strategy._update_param_with_optimizer(parameters, optimizer)


def install(strategy, plan):
    """Call before wrapping MCMC in its per-step reproducible RNG context."""
    import types
    config = strategy.conf.strategy.add
    config.start_iteration = plan.first_event-1
    config.end_iteration = plan.last_event+1
    config.frequency = plan.frequency
    config.max_n_gaussians = plan.target
    def add(self):
        step = self.paired_step
        if step not in plan.events:
            raise RuntimeError(f"Unexpected growth call at {step}")
        add_to_target(self, plan.count_after(step))
    strategy.add_new_gaussians = types.MethodType(add, strategy)
