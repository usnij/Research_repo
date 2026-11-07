"""Fixed-budget ray allocation for the four E1 arms.

The uniform arm delegates to the existing adapter without changing its draws,
layout, camera transforms, or structural objective. Other arms expose point-set
plans; they MUST be consumed by the arbitrary-coordinate SSIM objective rather
than reshaped into a synthetic image. All additional random draws use private
generators, independent of the view schedule and MCMC generator.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .inputs import SelectedInputs
from .sampling import StratifiedGrid
from .losses import coordinate_ssim_all


ARMS = ("uniform", "schedule", "residual", "combined")
SCORE_BLOCK = 8
PHASE_STEPS = 15000


def grid_ray_count(height, width, block):
    """Number of real strata, including partial image-edge strata."""
    if any(type(v) is not int or v <= 0 for v in (height, width, block)):
        raise ValueError("Image and block dimensions must be positive integers")
    return math.ceil(height / block) * math.ceil(width / block)


def allocation_budget(height, width, arm, step):
    """Exact per-view quota; ``step`` is the update about to run, starting at 1.

    The scheduled second-half quota is 2*N6-N8, not the slightly larger N5.
    Equality over 30k updates requires matching image geometry/view occurrence
    counts between the halves; the campaign must verify its saved view ledger.
    """
    if arm not in ARMS or type(step) is not int or not 1 <= step <= 30000:
        raise ValueError("Unknown allocation arm or update outside 1..30000")
    n6, n8 = (grid_ray_count(height, width, b) for b in (6, 8))
    late = step > PHASE_STEPS
    target = (2*n6-n8 if late else n8) if arm in ("schedule", "combined") else n6
    block = {"uniform": 6, "schedule": 5 if late else 8,
             "residual": 8, "combined": 7 if late else 11}[arm]
    base = grid_ray_count(height, width, block)
    thinned = max(0, base-target) if arm == "schedule" else 0
    bonus = target-base if arm in ("residual", "combined") else 0
    if target <= 0 or bonus < 0 or (arm == "schedule" and target > base):
        raise ValueError("Image geometry cannot support the exact allocation quota")
    return dict(base_block=block, target_count=target, base_grid_count=base,
                base_count=base-thinned, bonus_count=bonus, thinned_count=thinned,
                uniform_count=n6, early_count=n8, late_count=2*n6-n8,
                phase="late" if late else "early")


def initial_score_map(image, block=SCORE_BLOCK):
    """GT luminance sample standard deviation, including partial blocks.

    ``image`` is uint8 HWC or 1xHWC. Returned values are a flat, row-major map.
    This runs once per training image; neither padded zeros nor RGB-channel
    variance are counted as image observations. ``correction=1`` matches the
    August EVER initializer's torch.std default; one-pixel blocks get zero.
    """
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != torch.uint8:
        raise TypeError("Score initialization needs a uint8 HWC RGB image")
    h, w = image.shape[:2]
    rows, cols = math.ceil(h/block), math.ceil(w/block)
    rgb = image.to(torch.float64)/255.0
    luminance = rgb @ rgb.new_tensor([.2126, .7152, .0722])
    padded = F.pad(luminance, (0, cols*block-w, 0, rows*block-h))
    tiles = padded.reshape(rows, block, cols, block)
    hh = (h-torch.arange(rows, device=image.device)*block).clamp(max=block)
    ww = (w-torch.arange(cols, device=image.device)*block).clamp(max=block)
    area = (hh[:, None]*ww[None, :]).to(luminance.dtype)
    mean = tiles.sum((1, 3))/area
    variance = (tiles.square().sum((1, 3))/area-mean.square()).clamp_min(0)
    # sqrt of subtractive roundoff must not give texture to constant blocks.
    variance = torch.where(variance <= 8*torch.finfo(variance.dtype).eps,
                           torch.zeros_like(variance), variance)
    variance = torch.where(area > 1, variance*area/(area-1).clamp_min(1),
                           torch.zeros_like(variance))
    return variance.sqrt().reshape(-1).float()


def _generator(device, seed, stream=0):
    return torch.Generator(device=device).manual_seed((int(seed)+stream) % (2**63-1))


@dataclass
class AllocationSample:
    y: torch.Tensor
    x: torch.Tensor
    score_blocks: torch.Tensor
    is_base: torch.Tensor
    budget: dict


def sample_coordinates(height, width, arm, step, seed, scores, *, grid_cache=None):
    """Pure tensor sampler usable on CPU or CUDA, preserving duplicate draws."""
    budget = allocation_budget(height, width, arm, step)
    if scores.ndim != 1 or scores.numel() != grid_ray_count(height, width, SCORE_BLOCK):
        raise ValueError("The score map must have one value per 8x8 image block")
    device = scores.device
    cache = {} if grid_cache is None else grid_cache
    def grid(block):
        key = (height, width, block)
        if key not in cache:
            cache[key] = StratifiedGrid(height, width, block, device)
        return cache[key]

    base = grid(budget['base_block']).sample(_generator(device, seed))
    y, x = base.y, base.x
    if budget['thinned_count']:
        # Independent subset selection; sorting restores the native stratum
        # order and never advances the camera/MCMC RNG.
        keep = torch.randperm(y.numel(), device=device,
                              generator=_generator(device, seed, 982451653))[:budget['target_count']]
        keep = keep.sort().values
        y, x = y[keep], x[keep]
    is_base = torch.ones(y.numel(), dtype=torch.bool, device=device)
    if budget['bonus_count']:
        bonus_grid = grid(SCORE_BLOCK)
        # Exact zero-score fallback stays on-device; no host synchronization.
        probabilities = scores.clamp_min(0)
        probabilities = probabilities + (probabilities.sum() == 0).to(scores.dtype)
        rng = _generator(device, seed, 961748941)
        chosen = torch.multinomial(probabilities, budget['bonus_count'],
                                   replacement=True, generator=rng)
        by = bonus_grid.block_y[chosen]
        bx = bonus_grid.block_x[chosen]
        by = by + (torch.rand(chosen.shape, device=device, generator=rng)*
                   bonus_grid.valid_h[chosen]).floor().long()
        bx = bx + (torch.rand(chosen.shape, device=device, generator=rng)*
                   bonus_grid.valid_w[chosen]).floor().long()
        y, x = torch.cat((y, by)), torch.cat((x, bx))
        is_base = torch.cat((is_base, torch.zeros(chosen.numel(), dtype=torch.bool, device=device)))
    ids = (y//SCORE_BLOCK)*math.ceil(width/SCORE_BLOCK)+x//SCORE_BLOCK
    if arm in ("residual", "combined"):
        order = torch.argsort(ids, stable=True)
        y, x, ids, is_base = y[order], x[order], ids[order], is_base[order]
    return AllocationSample(y, x, ids, is_base, budget)


class AllocationInputs(SelectedInputs):
    """K=4 image adapter with persistent, observed-only residual score maps."""
    def __init__(self, dataset, batch_factory, config, *, allocation_arm="uniform",
                 device="cuda", resident_budget_bytes=0,
                 weight_fn=coordinate_ssim_all.weights):
        if allocation_arm not in ARMS:
            raise ValueError(f"allocation_arm must be one of {ARMS}")
        if config.views != 4 or config.block != 6 or config.boundary != "all" or config.sampler != "point":
            raise ValueError("E1 allocation requires K=4, baseline block=6, all centres, point sampler")
        super().__init__(dataset, batch_factory, config, device=device,
                         resident_budget_bytes=resident_budget_bytes, weight_fn=weight_fn)
        self.allocation_arm, self.step = allocation_arm, 1
        self.allocation_grids = {}
        self.score_maps, self.observation_counts, self.update_counts, self.last_steps = {}, {}, {}, {}
        self.last_view_updates = {vid: 0 for vid in self.items}
        for vid, item in self.items.items():
            score = initial_score_map(item['data'][0]).to(self.device)
            self.score_maps[vid] = score
            self.observation_counts[vid] = torch.zeros_like(score, dtype=torch.long)
            self.update_counts[vid] = torch.zeros_like(score, dtype=torch.long)
            self.last_steps[vid] = torch.zeros_like(score, dtype=torch.long)
        # Fail before training if a view's edge geometry cannot support a phase.
        for step in (1, PHASE_STEPS+1):
            counts = []
            for geom in self.geometry.values():
                counts.append(allocation_budget(geom['height'], geom['width'], allocation_arm, step)['target_count'])
            if len(set(counts)) != 1:
                raise ValueError("This renderer adapter needs equal per-view quotas in each phase")

    def set_step(self, update_1based):
        if type(update_1based) is not int or not 1 <= update_1based <= 30000:
            raise ValueError("set_step expects the next update number in 1..30000")
        self.step = update_1based

    def _metadata(self, plan, vid, budget, is_base):
        geom = self.geometry[vid]
        y, x = plan['y'], plan['x']
        ids = (y//SCORE_BLOCK)*math.ceil(geom['width']/SCORE_BLOCK)+x//SCORE_BLOCK
        # Unique counting sorts variable-length tensors and can synchronize
        # CUDA; measure it at log intervals rather than every training step.
        unique = y.numel() if not budget['bonus_count'] else None
        if unique is None and (self.step == 1 or self.step % 100 == 0):
            unique = torch.unique(y*geom['width']+x).numel()
        plan.update(view_id=vid, allocation_step=self.step, allocation_arm=self.allocation_arm,
                    image_shape=(geom['height'], geom['width']), score_blocks=ids,
                    score_grid_shape=(math.ceil(geom['height']/8), math.ceil(geom['width']/8)),
                    is_base=is_base, ray_count=y.numel(), unique_ray_count=unique,
                    base_count=budget['base_count'], bonus_count=budget['bonus_count'],
                    thinned_count=budget['thinned_count'], allocation_budget=budget)

    def make(self, views, seed, auxiliary="point_ssim"):
        if auxiliary != "point_ssim" or len(views) != self.config.views:
            raise ValueError("Only the configured K=4 observed-ray batch is supported")
        if len(set(map(int, views))) != len(views) or any(int(v) not in self.items for v in views):
            raise ValueError("Batch views must be distinct valid training indices")
        if self.allocation_arm == "uniform":
            batch, unused, plans = super().make(views, seed, auxiliary)
            for plan, vid in zip(plans, map(int, views)):
                geom = self.geometry[vid]
                budget = allocation_budget(geom['height'], geom['width'], "uniform", self.step)
                self._metadata(plan, vid, budget, torch.ones_like(plan['y'], dtype=torch.bool))
                plan['point_mode'] = False
            return batch, unused, plans

        origins, directions, targets, plans = [], [], [], []
        offset = 0
        for j, vid in enumerate(map(int, views)):
            _, ro, rd, _ = self.cameras[self.intrinsics[vid]]
            h, w = rd.shape[1:3]
            sample = sample_coordinates(h, w, self.allocation_arm, self.step, seed+104729*j,
                                        self.score_maps[vid], grid_cache=self.allocation_grids)
            y, x = sample.y, sample.x
            transform = self.poses[vid][0]
            origins.append((ro[:, y, x] @ transform[:3, :3].T+transform[:3, 3])
                           .reshape(1, 1, -1, 3).contiguous())
            directions.append((rd[:, y, x] @ transform[:3, :3].T)
                              .reshape(1, 1, -1, 3).contiguous())
            image = (self.images[vid] if self.resident else
                     self.items[vid]['data'][0].to(self.device, non_blocking=True))
            target = (image[:, y, x]/255.0).reshape(1, 1, -1, 3)
            targets.append(target)
            n = y.numel()
            coords = torch.stack((y, x), dim=-1).contiguous()
            plan = dict(gt=target.reshape(1, n, 1, 3), shape=(n, 1), weights=None, k=None,
                        start=offset, end=offset+n, y=y, x=x, point_coords=coords,
                        coordinates=coords, point_mode=True)
            self._metadata(plan, vid, sample.budget, sample.is_base)
            plans.append(plan)
            offset += n
        if len({target.shape for target in targets}) != 1:
            raise ValueError("Unequal per-view quotas need a ragged renderer adapter")
        batch = self.batch_factory(rays_ori=torch.cat(origins).contiguous(),
                                   rays_dir=torch.cat(directions).contiguous(),
                                   rgb_gt=torch.cat(targets).contiguous(),
                                   T_to_world=self.identity, intrinsics=None)
        return batch, None, plans

    @torch.no_grad()
    def update_residual(self, pred, gt, plans):
        """Update once per observed block, after this batch's forward pass.

        Repeated pixels remain repeated observations. The per-block mean is
        computed first, followed by exactly one EMA, independent of its count.
        No unobserved block is decayed or reset.
        """
        if pred.shape != gt.shape or pred.shape[-1] != 3:
            raise ValueError("Residual RGB tensors must have identical shapes")
        map_device = next(iter(self.score_maps.values())).device
        if pred.device != map_device or gt.device != map_device:
            raise ValueError("Residuals must be on the adapter device")
        residual = (pred.detach()-gt.detach()).abs().reshape(-1, 3).mean(-1)
        if not plans or plans[0]['start'] != 0 or plans[-1]['end'] != residual.numel():
            raise ValueError("Plans must cover the full residual batch")
        # Validate the whole batch before mutating any map. A stale or repeated
        # later view must not leave an earlier view partially updated.
        offset, seen = 0, set()
        for plan in plans:
            vid = plan['view_id']
            if vid not in self.items or vid in seen:
                raise ValueError("Residual plans must reference distinct training views")
            if plan['start'] != offset or plan['end'] <= offset:
                raise ValueError("Residual plans must form a contiguous partition")
            if plan['allocation_step'] != self.step or self.last_view_updates[vid] >= self.step:
                raise ValueError("Residual maps may be updated once per view per current update")
            if plan['score_blocks'].numel() != plan['end']-plan['start']:
                raise ValueError("Score-block indices and observed residuals disagree")
            seen.add(vid)
            offset = plan['end']
        for plan in plans:
            vid = plan['view_id']
            score = self.score_maps[vid]
            ids = plan['score_blocks']
            values = residual[plan['start']:plan['end']].to(score.dtype)
            sums = torch.zeros_like(score).scatter_add_(0, ids, values)
            counts = torch.zeros_like(score, dtype=torch.long).scatter_add_(0, ids, torch.ones_like(ids))
            observed = counts > 0
            mean = sums/counts.clamp_min(1)
            score.copy_(torch.where(observed, .9*score+.1*mean, score))
            self.observation_counts[vid].add_(counts)
            self.update_counts[vid].add_(observed.long())
            self.last_steps[vid].copy_(torch.where(observed, self.step, self.last_steps[vid]))
            self.last_view_updates[vid] = self.step

    def state_dict(self):
        return dict(version=1, arm=self.allocation_arm, step=self.step, score_block=SCORE_BLOCK,
                    ema_alpha=.1, luminance_std_correction=1,
                    geometry={vid: (g['height'], g['width']) for vid, g in self.geometry.items()},
                    last_view_updates=dict(self.last_view_updates),
                    score_maps={v: t.detach().cpu().clone() for v, t in self.score_maps.items()},
                    observation_counts={v: t.detach().cpu().clone() for v, t in self.observation_counts.items()},
                    update_counts={v: t.detach().cpu().clone() for v, t in self.update_counts.items()},
                    last_steps={v: t.detach().cpu().clone() for v, t in self.last_steps.items()})

    def load_state_dict(self, state):
        geometry = {v: (g['height'], g['width']) for v, g in self.geometry.items()}
        if (state.get('version') != 1 or state.get('arm') != self.allocation_arm or
                state.get('score_block') != SCORE_BLOCK or state.get('ema_alpha') != .1 or
                state.get('luminance_std_correction') != 1 or
                state.get('geometry') != geometry):
            raise ValueError("Allocation state has incompatible arm, score geometry, or EMA")
        for key in ('score_maps', 'observation_counts', 'update_counts', 'last_steps'):
            if set(state[key]) != set(self.items):
                raise ValueError("Allocation state does not contain every training view")
            for vid, value in state[key].items():
                target = getattr(self, key)[vid]
                if value.shape != target.shape or value.dtype != target.dtype:
                    raise ValueError(f"Incompatible {key} tensor in view {vid}")
                if not torch.isfinite(value).all() or (value < 0).any():
                    raise ValueError(f"Invalid {key} values in view {vid}")
        updates = state['last_view_updates']
        if set(updates) != set(self.items) or any(type(v) is not int or not 0 <= v <= state['step'] for v in updates.values()):
            raise ValueError("Invalid last-view-update state")
        self.set_step(state['step'])
        for key in ('score_maps', 'observation_counts', 'update_counts', 'last_steps'):
            for vid, value in state[key].items():
                getattr(self, key)[vid].copy_(value.to(self.device))
        self.last_view_updates = dict(updates)
