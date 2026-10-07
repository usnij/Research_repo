"""Train a 3DGRT scene from scratch with 1/64 stratified primary rays.

This is the first arm of the joint-training plan: the scene is optimised on the
supervisor's sparse objective and the denoiser is **not** in the gradient path.
It is only evaluated, so a denoiser trained on a different (converged) scene
cannot distort the geometry while its inputs are out of distribution. If the
scene survives this, letting the denoiser loss reach the Gaussians is the second
arm.

The whole substitution happens in the batch. `Trainer.train_epoch` renders
`model(gpu_batch)` and then calls `get_losses(gpu_batch, outputs)`, so replacing
the batch with its 1/64 stratified subset -- compact rays, and the ground truth
gathered at exactly those pixels -- leaves densification, the learning-rate
schedule, progressive spherical harmonics and every other mechanism untouched.

It also produces the supervisor's loss for free. L1 lands on the sampled pixels
because those are the only pixels present, and the structural term is computed
on the gathered 1/64 grid, which is the jitter-mask downsample.

Schedule: dense until `--sparse-from`, sparse after. Densification is most
sensitive to gradient noise, and worklog section 53 measured 0.19 dB of drift
from sparse supervision even on a converged scene, so the early phase is left
dense on purpose.
"""

from __future__ import annotations

import argparse
import math
import zlib
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from cached_dataset import CachedDataset
from stratified_mc import sample_permuted_pixels, sample_stratified_pixels
from threedgrut.datasets.protocols import Batch
from threedgrut.model.losses import ssim

if not OmegaConf.has_resolver("int_list"):
    OmegaConf.register_new_resolver("int_list", lambda l: [int(x) for x in l])


def _pose_key(T):
    """pose 로 view 를 식별한다. 프로세스 간 안정적이어야 하므로 crc32 를 쓴다."""
    return zlib.crc32(np.ascontiguousarray(
        T.detach().float().cpu().numpy()).tobytes()) & 0x7FFFFFFF


class StepPermSampler(torch.utils.data.Sampler):
    """학습 view 순서를 global step 의 순수 함수로 만든다.

    분기 실험에서 두 갈래가 같은 view 를 같은 순서로 보게 하려면 dataloader 의
    난수 상태를 저장·복원하는 대신 순서 자체를 step 으로 결정하는 편이 안전하다.
    epoch e 의 순열은 rng(seed, e) 로 정해지고, step s 는 그 순열의 s % n 번째다.

    시작 step 은 이터레이터를 만들 때 한 번만 읽는다.  워커가 prefetch 로 앞서
    당겨가도 순서가 흔들리지 않게 하기 위함이다 (기본 경로는 num_workers=0).
    """

    def __init__(self, allowed, seed, start_step_fn):
        self.allowed = list(allowed)
        self.seed = int(seed)
        self.start_step_fn = start_step_fn

    def __len__(self):
        return len(self.allowed)

    def __iter__(self):
        n = len(self.allowed)
        s = int(self.start_step_fn())
        while True:
            e, off = divmod(s, n)
            # hash() 는 문자열이 섞이면 PYTHONHASHSEED 로 프로세스마다 달라진다.
            # 재개 실행이 같은 순열을 얻어야 하므로 SeedSequence 를 쓴다.
            perm = np.random.default_rng([self.seed, 1, e]).permutation(n)
            for k in range(off, n):
                yield self.allowed[int(perm[k])]
            s = (e + 1) * n


class SparseBatchDataset:
    """Wraps a training dataset so every batch becomes its stratified subset.

    `step_holder` is a one-element list the trainer writes the global step into,
    which is how the warm-up boundary is applied without touching the loop.
    """

    half_budget = False

    def __init__(self, dataset, block, sparse_from, step_holder, seed=1234,
                 permuted=False, sparse_until=-1, mv_k=1, mv_from=-1):
        self.dataset = dataset
        self.block = block
        self.sparse_from = sparse_from
        # -1 keeps sparse on to the end; a positive value returns full batches
        # once the step reaches it, so the ray budget can be cut at densification
        # end and the two training phases separated.
        self.sparse_until = sparse_until
        self.step_holder = step_holder
        self.generator = torch.Generator(device="cuda").manual_seed(seed)
        self.announced = False
        self.last_full = None
        self.last_samples = None
        # temporal stratification state: which visit each image is on, and the
        # per-block permutation parameters, keyed by grid shape
        self.permuted = permuted
        self.visits = {}
        self.perm_state = {}
        # 다중 view: mv_from 스텝부터 한 배치에 view 를 mv_k 개 담는다. 예산을
        # 맞추려고 블록을 sqrt(k) 배로 키우므로 총 Ray 수는 단일 view 와 같다.
        # 배치 차원이 곧 view 라서 추적기 호출은 한 번이다.
        self.mv_k = mv_k
        self.mv_from = mv_from
        self.mv_rng = np.random.default_rng(seed + 7)
        self.mv_announced = False
        # 격차 감시가 쓰는 보류 집합.  비어 있으면 아무 영향이 없다.
        self.holdout = set()
        # --- 분기 실험용 결정론 표집 -------------------------------------
        # allowed 가 비어 있지 않으면 학습 표집은 전부 이 목록에서만 이뤄진다.
        # 일반 view 는 샘플러가, 추가 multi-view 는 아래 draw_multi 가 쓴다.
        # 기존 holdout 의 "다음 인덱스로 밀어내기" 는 일부 view 를 두 번
        # 뽑으므로 쓰지 않는다.
        self.allowed = None
        # det_seed 가 있으면 Ray 표집 난수를 (seed, step, view, 용도, 호출번호)
        # 로 파생시킨다.  전용 generator 만 재시드하므로 MCMC 의 CUDA 난수는
        # 건드리지 않는다.
        self.det_seed = None
        self._call_key = None
        self._call_n = 0

    def _reseed(self, purpose, view_key):
        """전용 generator 만 재시드한다. torch.manual_seed 는 절대 쓰지 않는다."""
        if self.det_seed is None:
            return
        step = int(self.step_holder[0])
        key = (step, view_key, purpose)
        if key != self._call_key:
            self._call_key, self._call_n = key, 0
        ss = np.random.SeedSequence(
            [self.det_seed, 3, step, int(view_key) & 0x7FFFFFFF,
             int(purpose[0]), int(purpose[1]), self._call_n])
        self._call_n += 1
        self.generator.manual_seed(int(ss.generate_state(1, dtype=np.uint32)[0]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # allowed 를 쓰면 샘플러가 이미 허용 목록에서만 뽑으므로 대체가 필요 없다.
        if self.allowed is not None:
            return self.dataset[index]
        # 보류된 view 는 학습에 쓰지 않는다.  다음 인덱스로 밀어 대체한다.
        if self.holdout and index in self.holdout:
            n = len(self.dataset)
            for k in range(1, n):
                j = (index + k) % n
                if j not in self.holdout:
                    index = j
                    break
        return self.dataset[index]

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def draw(self, gpu_batch, advance=True, block_w=None, block_h=None):
        """One stratified subset of an already-materialised full batch.

        block_w 를 주면 셀이 self.block x block_w 인 직사각형이 된다. split-half
        의 두 벌 A, B 는 각각 폭을 두 배로 잡아 표집률을 절반으로 낮춘다. 그러면
        A 와 B 를 합친 Ray 수가 arm 1 의 1/block^2 과 같아져 세 arm 이 같은 Ray
        예산에서 비교된다. 폭만 두 배로 잡는 이유는 반환 격자가 직사각형이어야
        SSIM 이 성립하기 때문이다.
        """
        height, width = gpu_batch.rays_dir.shape[1:3]
        bw = self.block if block_w is None else block_w
        bh = self.block if block_h is None else block_h
        grid_h, grid_w = math.ceil(height / bh), math.ceil(width / bw)
        if self.det_seed is not None:
            # pose 가 dataloader 순서와 무관하게 view 를 식별한다.
            self._reseed((bh, bw), _pose_key(gpu_batch.T_to_world))
        if self.permuted:
            # the pose identifies the image without relying on dataloader ordering
            key = hash(tuple(gpu_batch.T_to_world.flatten().tolist()))
            visit = self.visits.get(key, 0)
            if advance:
                self.visits[key] = visit + 1
            samples = sample_permuted_pixels(
                height, width, bh, bw, visit, self.perm_state,
                generator=self.generator, device="cuda",
            )
        else:
            samples = sample_stratified_pixels(
                height, width, bh, bw,
                generator=self.generator, device="cuda",
            )
        self.last_samples = np.stack(
            [samples.y.detach().cpu().numpy().ravel(),
             samples.x.detach().cpu().numpy().ravel()])
        return Batch(
            rays_ori=gpu_batch.rays_ori[:, samples.y, samples.x]
            .reshape(1, grid_h, grid_w, 3).contiguous(),
            rays_dir=gpu_batch.rays_dir[:, samples.y, samples.x]
            .reshape(1, grid_h, grid_w, 3).contiguous(),
            T_to_world=gpu_batch.T_to_world,
            rgb_gt=gpu_batch.rgb_gt[:, samples.y, samples.x]
            .reshape(1, grid_h, grid_w, 3).contiguous(),
            intrinsics=gpu_batch.intrinsics,
        )

    def _multiview(self, gpu_batch):
        """이 배치의 view 에 다른 view 들을 더해 배치 차원에 쌓는다."""
        k = self.mv_k
        blk = max(1, int(round(self.block * math.sqrt(k))))
        n = len(self.dataset)
        # 추가 view 도 반드시 허용 목록에서만 뽑는다. 예전에는 원본 dataset 을
        # 직접 찍어 holdout 을 우회했고, 그러면 검증 view 가 학습에 샌다.
        pool = self.allowed if self.allowed is not None else (
            [i for i in range(n) if i not in self.holdout] if self.holdout
            else list(range(n)))
        if self.det_seed is not None:
            rng = np.random.default_rng([self.det_seed, 2, int(self.step_holder[0])])
        else:
            rng = self.mv_rng
        extra = rng.choice(pool, size=min(k - 1, len(pool)), replace=False)
        parts = [self.draw(gpu_batch, block_w=blk, block_h=blk)]
        for i in extra:
            gb = self.dataset.get_gpu_batch_with_intrinsics(
                torch.utils.data.default_collate([self.dataset[int(i)]]))
            parts.append(self.draw(gb, block_w=blk, block_h=blk))
        if not self.mv_announced:
            h, w = parts[0].rays_ori.shape[1:3]
            print(f"[multiview] step {self.step_holder[0]}: view {k}개 x "
                  f"{h}x{w} = {k * h * w:,} rays (block {blk})", flush=True)
            self.mv_announced = True
        # 추적기는 배치 원소별 T_to_world 를 쓰지 않고 [0] 을 전부에 적용한다
        # (검증: 배치[1] 이 배치[0] 의 카메라로 렌더링됨).  그래서 Ray 를 미리
        # 월드 공간으로 옮기고 변환은 단위행렬로 넘긴다.  view 별 격자 모양은
        # 유지하므로 SSIM 은 각 view 안에서 성립한다.
        wo, wd = [], []
        for part in parts:
            T = part.T_to_world[0]
            R, t = T[:3, :3], T[:3, 3]
            wo.append(part.rays_ori @ R.T + t)
            wd.append(part.rays_dir @ R.T)
        eye = torch.eye(4, dtype=parts[0].T_to_world.dtype,
                        device=parts[0].T_to_world.device)[None].repeat(k, 1, 1)
        return Batch(
            rays_ori=torch.cat(wo, 0).contiguous(),
            rays_dir=torch.cat(wd, 0).contiguous(),
            rgb_gt=torch.cat([p.rgb_gt for p in parts], 0),
            T_to_world=eye,
            intrinsics=parts[0].intrinsics,
        )

    def get_gpu_batch_with_intrinsics(self, batch):
        gpu_batch = self.dataset.get_gpu_batch_with_intrinsics(batch)
        if self.step_holder[0] < self.sparse_from or (
                0 <= self.sparse_until <= self.step_holder[0]):
            self.last_full = None
            return gpu_batch
        if self.mv_k > 1 and 0 <= self.mv_from <= self.step_holder[0]:
            self.last_full = None
            return self._multiview(gpu_batch)

        height, width = gpu_batch.rays_dir.shape[1:3]
        bw = self.block * 2 if self.half_budget else self.block
        grid_h, grid_w = math.ceil(height / self.block), math.ceil(width / bw)
        if not self.announced:
            extra = (f"  x2 (A+B 합쳐 {2 * grid_h * grid_w:,})"
                     if self.half_budget else "")
            print(f"[sparse] step {self.step_holder[0]}: "
                  f"{height}x{width} -> {grid_h}x{grid_w} "
                  f"({grid_h * grid_w:,} of {height * width:,} rays){extra}", flush=True)
            self.announced = True
        # kept so the densification correction can draw a second, independent
        # subset of the *same* view; it is the batch that was built anyway
        self.last_full = gpu_batch
        if self.half_budget:
            # split-half: 이 벌이 A 이고, densification 쪽에서 같은 폭으로 B 를
            # 한 벌 더 뽑는다. 합이 arm 1 의 Ray 수와 같아진다.
            return self.draw(gpu_batch, block_w=self.block * 2)
        return self.draw(gpu_batch)


def install_debiased_densification(trainer, conf, args):
    """Feed the densification buffer a variance-corrected statistic.

    The buffer averages per-step gradient *norms*. Under 1/64 supervision that
    per-step gradient is unbiased but noisy, and a norm is convex, so the average
    inflates by an amount set by the sampling variance -- measured at 1.63x at
    ds=1 and 2.61x at ds=2, which is why a fixed rescale had to be retuned per
    configuration and still overshot. Drawing a second independent subset of the
    same view makes the variance observable:

        E[||g_A - g_B||^2] = 2 tr(S),   g = (g_A + g_B)/2,   Var(g) = S/2
        ||dense gradient||^2 ~= ||g||^2 - ||g_A - g_B||^2 / 4

    The correction is applied inside the per-step norm, because the target is a
    mean of norms; accumulating the squared terms first produces a root-mean-square
    instead and measured 8-14 points worse on ranking overlap.

    The second pass costs one extra forward/backward on 1/64 rays, and only until
    densification ends, so `--debias-every` can thin it further.
    """
    strategy = trainer.strategy
    model = trainer.model
    dataset = trainer.train_dataset
    end = int(conf.strategy.densify.end_iteration)
    l1_weight = float(conf.loss.lambda_l1)
    ssim_weight = float(conf.loss.lambda_ssim)
    original = strategy.post_backward
    stats = {"applied": 0, "negative": 0.0}

    def post_backward(step, scene_extent, train_dataset, batch=None, writer=None):
        full = getattr(dataset, "last_full", None)
        if (full is None or step >= end or step % args.debias_every != 0
                or model.positions.grad is None):
            return original(step, scene_extent, train_dataset, batch, writer)

        g_a = model.positions.grad.detach().clone()
        model.zero_grad(set_to_none=True)
        second = dataset.draw(full, advance=False,
                              block_w=dataset.block * 2 if dataset.half_budget else None)
        out = model(second, train=True)
        loss = l1_weight * torch.abs(out["pred_rgb"] - second.rgb_gt).mean()
        if ssim_weight > 0:
            loss = loss + ssim_weight * (1.0 - ssim(
                out["pred_rgb"].permute(0, 3, 1, 2),
                second.rgb_gt.permute(0, 3, 1, 2)))
        loss.backward()
        g_b = model.positions.grad.detach().clone()
        del out, loss, second

        if getattr(dataset, "half_budget", False):
            # A 와 B 가 각각 절반 표집률이므로, 둘을 평균하면 arm 1 과 같은
            # Ray 수로 만든 gradient 가 된다. A 것만 쓰면 optimizer 가 절반만
            # 보게 되어 densification 과 무관한 이유로 품질이 떨어진다.
            for parameter, grad in saved:
                if grad is None:
                    continue
                parameter.grad = (grad if parameter.grad is None
                                  else (grad + parameter.grad) / 2)
        else:
            for parameter, grad in saved:
                parameter.grad = grad

        with torch.no_grad():
            sensor = batch.T_to_world[0, :3, 3]
            distance = (model.positions.detach() - sensor).norm(dim=1, keepdim=True)
            mean = (g_a + g_b) / 2 * distance
            diff = (g_a - g_b) * distance
            squared = mean.pow(2).sum(-1, keepdim=True) - diff.pow(2).sum(-1, keepdim=True) / 4
            touched = ((g_a != 0) | (g_b != 0)).max(dim=1)[0]
            stats["applied"] += 1
            stats["negative"] += float((squared[touched] < 0).float().mean())
            if args.shrink_only:
                # 축소만: 교차 내적을 쓰지 않는다. 양쪽 다 맞은 step 에서는 보통의
                # norm 을 쌓고, 한쪽만 맞은 step 에는 0 을 쌓되 분모는 센다.
                # probe 에서 이 형태가 6/6 조건에서 무보정보다 순위가 높았고,
                # 교차 내적판보다도 4/6 에서 높았다. 배율도 0.95~1.32 로 1 근처다.
                both = ((g_a != 0).max(dim=1)[0]) & ((g_b != 0).max(dim=1)[0])
                corrected = ((g_a + g_b) / 2 * distance).norm(dim=-1, keepdim=True) / 2
                corrected = torch.where(both.unsqueeze(-1), corrected,
                                        torch.zeros_like(corrected))
            else:
                corrected = squared.clamp_min(0).sqrt() / 2
            strategy.densify_grad_norm_accum[touched] += corrected[touched]
            strategy.densify_grad_norm_denom[touched] += 1
        del g_a, g_b, distance

        # everything post_backward does *besides* the buffer update still has to
        # run, so it is called with the buffer write suppressed
        keep = strategy.update_gradient_buffer
        strategy.update_gradient_buffer = lambda *a, **k: None
        try:
            return original(step, scene_extent, train_dataset, batch, writer)
        finally:
            strategy.update_gradient_buffer = keep

    strategy.post_backward = post_backward
    trainer._debias_stats = stats
    print(f"[debias] split-half correction on, every {args.debias_every} step(s), "
          f"until step {end}", flush=True)


def install_opacity_boundary_barrier(trainer, weight, guard, temperature, start,
                                     selected_mask=None):
    """Keep opacity away from the renderer's absorbing boundary.

    3DGRT accepts a hit only when ``alpha * gaussian_response > 1/255``.
    Once alpha crosses that boundary the primitive normally loses every
    gradient.  This smooth exterior penalty is the first, deliberately small
    test of treating that implementation threshold as a constrained
    optimisation boundary rather than pruning particles after they cross it.

    The mean is taken over particles inside the guard band only.  Dividing by
    all ~1.5M particles would make the force depend on model size and nearly
    erase it precisely when few particles are approaching the boundary.
    """
    original = trainer.get_losses

    def get_losses(gpu_batch, outputs):
        losses = original(gpu_batch, outputs)
        if trainer.global_step < start or weight <= 0:
            return losses

        alpha = trainer.model.get_density().reshape(-1)
        endangered = alpha < guard
        if selected_mask is not None:
            endangered = endangered & selected_mask
        if bool(endangered.any()):
            # softplus has a finite, non-zero slope on both sides of the guard
            # and avoids the singular numerical behaviour of -log(alpha-a_min).
            barrier = torch.nn.functional.softplus(
                (guard - alpha[endangered]) / temperature
            ).mean() * temperature
        else:
            barrier = alpha.new_zeros(())
        weighted = weight * barrier
        losses["total_loss"] = losses["total_loss"] + weighted
        losses["opacity_boundary_barrier"] = weighted
        return losses

    trainer.get_losses = get_losses
    print(f"[opacity-barrier] start={start}, weight={weight:g}, "
          f"guard={guard:g}, temperature={temperature:g}, "
          f"selected={int(selected_mask.sum()) if selected_mask is not None else 'all'}",
          flush=True)


# counter ds=2, 1/16 표집에서 probe 로 실측한 k(h). h 구간별 중앙값이다.
# 두 벌을 각각 목표 표집률로 뽑아 쟀으므로(--full-rate-pairs) 여기 값이 곧 통계량
# 보정에 쓸 k 이고 alpha 보정이 필요 없다. 절반 표집률로 재던 이전 곡선은 값이
# 약 2 배였고 alpha=0.5 가 필요했다 — h 가 큰 구간에서 두 측정의 비가 2.00 으로
# 수렴해 분산 반비례 가정이 확인됐다. 씬 배율 1.692 에서 역산한 k=1.863 과
# 이 곡선의 중앙값 1.91 이 1.5% 이내로 일치한다.
_K_OF_H_H = (1.5, 2.5, 3.5, 5.0, 7.5, 11.0, 18.0, 28.0, 47.0, 80.0, 150.0)
_K_OF_H_K = (3.317, 3.289, 2.747, 2.187, 1.653, 1.237, 0.916, 0.6675, 0.4735, 0.3113, 0.143)


_K_CACHE = {}

# 실측 부풀림 phi(h). block 4 는 counter·room·kitchen·bonsai 네 씬의 중앙값,
# block 8 은 counter 단일이다 (dense 체크포인트 300 view 누적).
#
# 지금까지 쓰던 sqrt(1+k) 는 k 를 부풀림으로 환산하는 유도인데 실측과 -70% ~
# +36% 어긋난다. k 는 제곱 노름 · 한 step · 참값 분모 기준이고 실제 부풀림은
# 노름 · n_i step 평균 · dense 통계량 분모 기준이라 세 군데가 다르다.
# 그래서 유도를 버리고 실측 곡선을 직접 쓴다.
#
# 곡선은 h 에 대해 U 자다. h<2 에서 2.6~3.9 로 치솟고 h=11~17 에서 최소를
# 찍은 뒤 완만히 오른다. sqrt(1+k) 는 단조 감소라 모양 자체가 다르다.
# 씬 간에는 10~12% 편차로 전이되지만 표집률 간에는 전이되지 않는다
# (같은 h=11 에서 1/16 은 1.36, 1/64 는 1.86).
_PHI_KNOTS = {
    4: ((1.385, 2.42, 3.431, 4.842, 7.222, 10.9, 17.0, 27.077, 44.611, 74.978, 162.408),
        (2.568, 1.8156, 1.637, 1.5202, 1.4261, 1.3557, 1.3315, 1.3458, 1.3646, 1.3555, 1.2748)),
    8: ((1.237, 2.349, 3.392, 4.765, 7.157, 10.875, 16.88, 26.487, 43.393, 73.863, 142.034),
        (3.9377, 2.2625, 2.0478, 1.9527, 1.8878, 1.8562, 1.8234, 1.7687, 1.665, 1.5527, 1.3249)),
}
_PHI_CACHE = {}


def _phi_of_h(h, block):
    """실측 부풀림을 log-log 선형 보간한다. 측정하지 않은 block 은 가장 가까운 것을 쓴다."""
    key = (block, h.device)
    if key not in _PHI_CACHE:
        b = block if block in _PHI_KNOTS else min(_PHI_KNOTS, key=lambda x: abs(x - block))
        hs, ps = _PHI_KNOTS[b]
        _PHI_CACHE[key] = (torch.tensor(hs, device=h.device).log(),
                           torch.tensor(ps, device=h.device).log())
    lh, lp = _PHI_CACHE[key]
    x = h.clamp_min(1e-6).log().clamp(float(lh[0]), float(lh[-1]))
    i = torch.searchsorted(lh, x).clamp(1, len(lh) - 1)
    x0, x1 = lh[i - 1], lh[i]
    y0, y1 = lp[i - 1], lp[i]
    return torch.exp(y0 + (y1 - y0) * (x - x0) / (x1 - x0).clamp_min(1e-9))
import os
_MEMDBG = os.environ.get('RADC_MEMDBG') == '1'
_SM_DBG = os.environ.get('SHRINKMEAN_DBG') == '1'
_MEMDBG_START = int(os.environ.get('RADC_MEMDBG_START', '7000'))
_MEMDBG_DUMP = int(os.environ.get('RADC_MEMDBG_DUMP', '9000'))


def _k_of_h(h, knots=None):
    """실측 k(h) 를 로그-로그 선형 보간. 구간 밖은 양 끝 값으로 고정한다.

    knots 를 주면 그 곡선을 쓴다 (씬마다 따로 측정한 경우). 안 주면 counter 에서
    측정한 기본 곡선을 쓴다.
    """
    hs, ks = knots if knots is not None else (_K_OF_H_H, _K_OF_H_K)
    # 매 step 새로 만들면 작은 CUDA 할당이 3 만 번 이상 쌓여 큰 구간을 조각낸다.
    # 곡선이 바뀔 때만 다시 만들고 그 외에는 재사용한다.
    key = (id(hs), h.device, h.dtype)
    cached = _K_CACHE.get(key)
    if cached is None or cached[2] is not hs:
        xs = torch.tensor(hs, device=h.device, dtype=h.dtype).log()
        ys = torch.tensor(ks, device=h.device, dtype=h.dtype).log()
        _K_CACHE.clear()
        _K_CACHE[key] = (xs, ys, hs)
    else:
        xs, ys, _ = cached
    lh = torch.log(h.clamp_min(1e-6))
    idx = torch.bucketize(lh, xs).clamp(1, len(hs) - 1)
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    t = ((lh - x0) / (x1 - x0)).clamp(0.0, 1.0)
    return torch.exp(y0 + t * (y1 - y0))


class OnlineKCurve:
    """학습 중에 k(h) 곡선을 스스로 재서 갱신한다.

    k(h) 는 두 곳에서 움직인다. Gaussian 마다 다르고(사분위폭이 중앙값의 1.94 배),
    학습 단계에 따라서도 변한다(step 7,000 의 k 가 수렴 후보다 30% 낮다). 그래서
    미리 한 번 재둔 곡선으로는 학습 전 구간을 덮지 못한다. 실제로 수렴 체크포인트
    곡선을 쓰면 초반에 과도하게 축소해 Gaussian 이 dense 의 91% 에서 멈춘다.

    곡선은 Gaussian 100 만 개를 모아 만드는 집단 통계이므로 매 step 잴 필요가 없다.
    every step 마다 한 번만 split-half 를 돌려 표본을 쌓고, densify 직전에 구간별
    중앙값으로 곡선을 갱신한다. every=20 이면 추가 비용이 5% 다.

    층화 표집에서는 블록마다 화소가 하나뿐이라 블록 안 편차를 한 벌로는 알 수 없다.
    커널에서 Ray 별 기여의 제곱합을 받아 쓰는 방법을 시험했으나 실측에서 split-half
    추정과 상위 1% 중첩이 1.7% 로 무관했다. 두 벌을 뽑는 것이 표집 구조상 필연이다.
    """

    EDGES = (1, 2, 3, 4, 6, 9, 14, 22, 35, 60, 100, 10 ** 9)

    def __init__(self, n, device, momentum=0.5, decay=0.5):
        self.n = n
        self.k_sum = torch.zeros(n, device=device)
        self.v_sum = torch.zeros(n, device=device)
        self.h_sum = torch.zeros(n, device=device)
        self.den = torch.zeros(n, device=device)
        # 전역 크기용 h 는 probe 와 정의를 맞춘다. 곡선의 h 축은 양쪽 다 맞은
        # step 으로 세는데, 그렇게 조건화하면 적중이 많은 step 만 골라져 h 가
        # 부풀려진다 (counter step 3000 에서 32.2 대 probe 8.5). 전역 부풀림 식은
        # probe 의 h 로 적합했으므로 같은 정의로 따로 쌓는다.
        self.hp_sum = torch.zeros(n, device=device)
        self.hp_den = torch.zeros(n, device=device)
        self.momentum = momentum
        self.decay = decay
        self.knots = None            # (h 배열, k 배열)
        self.k_med = None            # 인구 중앙 k (전역 크기용)
        self.k_pool = None           # 선별 없는 합-먼저 k
        self.h_med = None            # 인구 중앙 h
        self.gate_knots = None       # 게이트 h_bar 축으로 묶은 (h 배열, k 배열)

    def grow(self, n, device):
        """Gaussian 이 늘면 누적 버퍼도 늘린다.

        densify 마다 정확히 n 으로 다시 잡으면 크기가 매번 달라 할당기가 옛 구간을
        재사용하지 못한다. 1.3 배 여유를 두고 잡아 재할당 횟수를 줄인다.
        """
        if self.den.numel() >= n:
            self.n = n
            return
        cap = int(n * 1.3) + 1024
        for name in ("k_sum", "v_sum", "h_sum", "den", "hp_sum", "hp_den"):
            old = getattr(self, name)
            new = torch.zeros(cap, device=device)
            new[: old.numel()] = old
            setattr(self, name, new)
            del old
        self.n = n

    @torch.no_grad()
    def remap_topology(self, survivors, n_new=0, reset_old=None):
        """Keep survivor histories in model order; newborn histories start at zero.

        reset_old marks changed surviving parents (EVER clone). The fitted global
        curve is retained; only per-Gaussian evidence is remapped.
        """
        if survivors.dtype != torch.bool or survivors.ndim != 1:
            raise ValueError("online-k topology requires a 1-D survivor mask")
        if survivors.numel() != self.n or n_new < 0:
            raise ValueError("online-k topology/model size mismatch")
        names = ("k_sum", "v_sum", "h_sum", "den", "hp_sum", "hp_den")
        kept = {}
        for name in names:
            values = getattr(self, name)[:self.n].clone()
            if reset_old is not None:
                values[reset_old] = 0
            kept[name] = values[survivors]
        count = kept["den"].numel()
        self.grow(count + n_new, survivors.device)
        for name in names:
            buf = getattr(self, name)
            buf.zero_()  # Also clear spare capacity so stale rows cannot reappear.
            buf[:count].copy_(kept[name])

    @torch.no_grad()
    def observe(self, g_a, g_b, distance, hits):
        """한 step 의 두 표본으로 ||mu||^2 과 sigma^2 을 쌓는다."""
        both = ((g_a != 0).max(dim=1)[0]) & ((g_b != 0).max(dim=1)[0])
        if not bool(both.any()):
            return
        mean = (g_a + g_b) / 2 * distance
        diff = (g_a - g_b) * distance
        n = both.numel()
        self.k_sum[:n][both] += (mean.pow(2).sum(-1) - diff.pow(2).sum(-1) / 4)[both]
        self.v_sum[:n][both] += (diff.pow(2).sum(-1) / 2)[both]
        if hits is not None:
            self.h_sum[:n][both] += hits[both]
        self.den[:n][both] += 1

    @torch.no_grad()
    def observe_hits(self, hits):
        """매 step 의 적중 수만 따로 쌓는다.

        split-half 관측은 비용 때문에 every step 마다 한 번인데, 그 빈도로 h 를
        세면 드물게 맞는 Gaussian 이 창 안에서 한 번도 안 잡혀 중앙값이 위로
        쏠린다 (counter step 6000 에서 13.1 대 probe 8.5). 적중 수는 매 step
        이미 있으므로 전부 쌓는다.
        """
        if hits is None:
            return
        n = min(hits.numel(), self.hp_sum.numel())
        seen = hits[:n] > 0
        self.hp_sum[:n][seen] += hits[:n][seen]
        self.hp_den[:n][seen] += 1

    @torch.no_grad()
    def refresh(self, min_count=200):
        """모은 표본을 h 구간별로 묶어 곡선을 갱신하고 누적을 비운다."""
        n = self.n or self.den.numel()
        ok = self.den[:n] > 0.5
        if int(ok.sum()) < 1000:
            return
        den = self.den[:n][ok]
        mu2 = self.k_sum[:n][ok] / den
        var = self.v_sum[:n][ok] / den
        hbar = self.h_sum[:n][ok] / den
        good = mu2 > 0
        if int(good.sum()) < 1000:
            self._clear(); return
        kk, hh = var[good] / mu2[good], hbar[good]
        # 전역 크기 보정용 인구 중앙값. 씬·block·downsample·학습 단계 33 조건에서
        # 부풀림 = 1.1552 * (1 + k_med/h_med)^0.7999 가 20% 안에 맞았다.
        # 구간 단위로는 R2 0.26 이라 개체별로 쓰면 안 되고 전역 크기로만 쓴다.
        self.k_med = float(torch.quantile(kk, 0.5))
        # 선별 없는 합-먼저 추정. 개체마다 var/mu2 를 구해 중앙값을 내면, 표본이
        # 적을 때 mu2 가 흔들려 음수가 된 개체가 good 에서 빠진다. 빠지는 쪽은
        # 신호가 약한 (k 가 큰) 집단이라 남은 중앙값이 아래로 쏠린다. 합을 먼저
        # 내고 나누면 선별이 없어 표본이 적어도 편향되지 않는다.
        _num = float(self.v_sum[:n][ok].sum())
        _den = float(self.k_sum[:n][ok].sum())
        self.k_pool = (_num / _den) if _den > 0 else None
        # 감쇠 누적이라 드물게 맞은 Gaussian 은 hp_den 이 0.5 아래로 내려간다.
        # 0.5 로 자르면 그들이 통째로 빠져 중앙값이 위로 쏠린다.
        _hp = self.hp_den[:n] > 1e-6
        self.h_med = (float(torch.quantile(
            (self.hp_sum[:n][_hp] / self.hp_den[:n][_hp]), 0.5))
            if bool(_hp.any()) else float(torch.quantile(hh, 0.5)))
        # 게이트가 쓰는 h 축으로도 같은 k 를 묶는다. 위의 hh 는 양쪽 반쪽이 모두
        # 맞은 step 으로만 세서 h 가 부풀려져 있고 (counter step 3000 에서 32.2 대
        # probe 8.5), 게이트의 h_bar 는 관측된 step 당 평균 적중이다. 축이 다르면
        # 곡선에서 읽은 값을 theta 로 쓸 수 없다. hp 는 probe 와 정의를 맞춘 h 라
        # 게이트 축과 같으므로, 이쪽으로 묶은 곡선을 theta 용으로 따로 둔다.
        _hpv = torch.zeros_like(hh)
        _hpd = self.hp_den[:n][ok][good]
        _hps = self.hp_sum[:n][ok][good]
        _m = _hpd > 1e-6
        _hpv[_m] = _hps[_m] / _hpd[_m]
        gh, gk = [], []
        for lo, hi in zip(self.EDGES[:-1], self.EDGES[1:]):
            m = _m & (_hpv >= lo) & (_hpv < hi)
            if int(m.sum()) < min_count:
                continue
            gh.append(float(torch.quantile(_hpv[m], 0.5)))
            gk.append(float(torch.quantile(kk[m], 0.5)))
        if len(gh) >= 2:
            self.gate_knots = (gh, gk)
        hs, ks = [], []
        for lo, hi in zip(self.EDGES[:-1], self.EDGES[1:]):
            m = (hh >= lo) & (hh < hi)
            if int(m.sum()) < min_count:
                continue
            hs.append(float(torch.quantile(hh[m], 0.5)))
            ks.append(float(torch.quantile(kk[m], 0.5)))
        if len(hs) >= 3:
            if self.knots is None:
                self.knots = (hs, ks)
            else:
                # 구간 구성이 바뀔 수 있으므로 새 곡선을 그대로 받되 급변을 눌러 준다
                oh, ok_ = self.knots
                if len(oh) == len(hs):
                    b = self.momentum
                    ks = [b * a + (1 - b) * n for a, n in zip(ok_, ks)]
                self.knots = (hs, ks)
        self._clear()

    def _clear(self):
        """누적을 비우지 않고 감쇠시킨다.

        곡선은 Gaussian 수십만 개를 모아 만드는 집단 통계라 표본을 자주 모을
        필요가 없다. 갱신마다 비우면 간격을 늘렸을 때 표본이 모자라는데, 감쇠
        누적을 쓰면 최근 여러 주기의 표본이 함께 반영되어 간격 100 에서도
        안정적이다. decay=0.5 면 유효 기억이 약 두 주기다.
        """
        for t in (self.k_sum, self.v_sum, self.h_sum, self.den,
                  self.hp_sum, self.hp_den):
            t.mul_(self.decay)


# 33 조건(씬 7 / block 2,4,6,8 / downsample 1,2,4 / step 7000,30000)에서 적합한
# 전역 부풀림. h 를 (1+k) 안에 k/h 로 넣는 형태만 적합 범위 밖으로 외삽해도 무너지지
# 않았다. h 를 별도 거듭제곱으로 붙이면 미사용 조건 예측 오차가 52% 로 커진다.
_PHI_C, _PHI_A = 1.1552, 0.7999


def _phi_global(k_med, h_med):
    """인구 중앙 k 와 h 로 그 시점 전체 부풀림을 예측한다."""
    return _PHI_C * (1.0 + k_med / max(h_med, 1e-6)) ** _PHI_A


_GATE_AUTO_STEP = 1500


def install_hit_shrinkage(trainer, conf, hit_holder, alpha=1.0, knots=None,
                          online=None, dataset=None, every=20, grad_shrink=False,
                          step_shrink=False, shrink_mean=False, ndim=3,
                          phi_div=False, blk=4, auto_scale=False, pooled_k=False,
                          gate_min_hits=0.0, step_min_hits=0.0, target_g=0, quantile=0.0,
                          tau_phi=False, waste_target=0.0, gate_z=0.0, gate_auto=0.0,
                          eb_shrink=False):
    """Ray 적중 횟수로 densification 통계량을 축소한다.

    젠센 인플레이션의 크기는 Gaussian 마다 다르고, 그 차이는 그 Gaussian 이 몇
    개의 Ray 에 맞았는지(h)로 정해진다. 적게 맞을수록 분산이 크고 통계량이 크게
    부푼다. arm 2 의 씬 상수 하나로는 이 차이를 담지 못해 순서를 바꾸지 못한다.

    축소 인자는 split-half 판이 암묵적으로 쓰던 것과 같은 값을 해석적으로 쓴다.
    Ray h 개를 무작위로 반씩 나눌 때 양쪽에 최소 하나씩 들어갈 확률이

        P = 1 - 2^(1-h)

    이고, 앞선 실행에서 이 인자가 통계량에 곱해지고 있었다. 여기서는 h 를 직접
    세므로 표집을 두 벌로 나눌 필요가 없다. 표집과 손실이 arm 1 과 완전히 같아져
    비교에서 통계량만 남는다.
    """
    strategy = trainer.strategy
    # tau_phi 가 매 주기 임계값을 덮어쓰므로 원본을 보관한다
    _TAU0 = (float(strategy.clone_grad_threshold), float(strategy.split_grad_threshold))
    # h >= step_min_hits 인 관측만 따로 쌓는 버퍼 (누적, 관측 수)
    hi_holder = [None, None]
    # 온라인 theta: [현재 theta, 시작 G]. target_g 가 있을 때만 쓴다.
    theta_state = [gate_min_hits if gate_min_hits > 0 else 1.0, None]
    # 직전 densify 가 만든 자식의 표식. prune 때 같이 잘라야 인덱스가 맞는다.
    fresh_holder = [None]
    # 유의성 게이트용 반쪽 누적. [합A, 수A, 합B, 수B] 로, 한 densify 주기 안의
    # 관측을 step 홀짝으로 갈라 같은 양을 두 번 독립으로 추정한다.
    sh_holder = [None, None, None, None]
    original = strategy.update_gradient_buffer
    model = trainer.model

    if online is not None:
        # GS clone/split/prune notify with the exact pre-change survivor mask.
        strategy._online_k_topology = online.remap_topology

    if shrink_mean:
        # 평균으로의 축소 (경험적 베이즈).
        #
        #   T_i = S_bar + w_i (S_i - S_bar),   w_i = tau^2 / (tau^2 + v_i)
        #
        # 지금까지 쓰던 1/sqrt(1+k) 는 "부풀림을 되돌린다" 는 유도인데, 실측과
        # 대조하면 씬에 따라 -70% ~ +36% 어긋난다. k 는 제곱 노름 · 한 step ·
        # 참값 분모 기준이고 실제 부풀림은 노름 · n_i step 평균 · dense 통계량
        # 분모 기준이라 세 군데가 다르다. 실측 phi(h) 는 h<2 에서 2.4~3.9,
        # h>=2 에서 1.25~2.26 으로 사실상 두 집단이고 h 의존이 약하다.
        #
        # 그래서 부풀림을 되돌리는 대신 신뢰도로 가중한다. ||g|| ~ sqrt(M)(1+u/2)
        # 전개에서 Var(||g||) ~= M k / d 이므로
        #
        #   v_i = Var(||g||) / n_i ~= k_i S_i^2 / (d n_i)
        #
        # 이고 tau^2 = max(Var_i(S_i) - mean_i(v_i), 0) 로 적률 추정한다.
        # v_i 에 관측값 S_i^2 를 대입하는 것이 요점이다. 참값이 작은데 운으로 큰
        # 값을 뽑은 것은 S_i 가 커서 v_i 도 커지므로 평균으로 끌려가고, 참값이
        # 큰 것은 k_i 가 작고 n_i 가 커서 거의 손대지 않는다.
        # prune 은 100 step 마다 Gaussian 을 제거하며 strategy 의 누적 버퍼들을
        # 함께 잘라낸다 (prune_densification_buffers). 적중 누적도 같이 잘라야
        # densify 시점에 크기가 맞는다.
        orig_prune_bufs = strategy.prune_densification_buffers

        def prune_bufs_with_hits(valid_mask):
            out = orig_prune_bufs(valid_mask)
            if hit_holder[1] is not None and hit_holder[1].shape[0] == valid_mask.shape[0]:
                hit_holder[1] = hit_holder[1][valid_mask]
            return out

        strategy.prune_densification_buffers = prune_bufs_with_hits

        dfreq_dbg = int(conf.strategy.densify.frequency)
        orig_densify = strategy.densify_gaussians

        @torch.no_grad()
        def densify_with_shrinkage(scene_extent):
            accum = strategy.densify_grad_norm_accum
            denom = strategy.densify_grad_norm_denom.float()
            hits = hit_holder[1]
            ok = (denom.squeeze(-1) > 0) if denom.dim() > 1 else (denom > 0)
            if int(ok.sum()) >= 100 and hits is not None and hits.shape[0] == accum.shape[0]:
                n = denom.clamp_min(1.0)
                S = accum / n
                hbar = (hits / n).clamp_min(1.0)
                live_k = online.knots if (online is not None and online.knots is not None) else knots
                k = _k_of_h(hbar.squeeze(-1), live_k).unsqueeze(-1)
                v = k * S.pow(2) / (ndim * n)
                sel = ok.unsqueeze(-1) if S.dim() > 1 else ok
                # RADC-Gate: 구간 누적 적중 수가 임계 미만이면 densification 후보에서
                # 뺀다. 근거는 "적중이 적으면 noise 가 크다" 가 아니다 — 그건 축소가
                # 이미 처리한다. 문제는 split-half 로 추정하는 rho 자체가 표본 수가
                # 적으면 정의되지 않는다는 것이다. 표본오차가 1/sqrt(k) 라 k 가 1~2 면
                # rho 가 우연히 크게 나와 축소되지 않고 통과한다. 그래서 tau2 와 Sbar
                # 를 추정하는 개체군에서도 먼저 빼고, 통계량도 0 으로 만든다.
                gate_mask = None
                if gate_min_hits > 0:
                    gate_mask = hbar <= float(gate_min_hits)
                    sel = sel & (~gate_mask)
                Sv = S[sel]
                tau2 = (Sv.var() - v[sel].mean()).clamp_min(1e-20)
                if phi_div:
                    # 실측 부풀림으로 직접 나눈다. 곱셈 축소를 유지하므로 개수
                    # 통제라는 1차 주장이 살아 있고, 배율만 유도에서 실측으로
                    # 바뀐다. 누적 평균 h 로 한 번 적용해야 한다 — step 별로
                    # 곱하면 mean(g/phi(h)) 가 되는데 필요한 것은
                    # mean(g)/phi(h_bar) 다.
                    T = S / _phi_of_h(hbar.squeeze(-1), blk).unsqueeze(-1)
                    w = None
                else:
                    w = tau2 / (tau2 + v)
                    Sbar = Sv.mean()
                    T = Sbar + w * (S - Sbar)
                if _SM_DBG and w is not None:
                    wv = w[sel]
                    # 통제 실패의 기전을 가리려면 Sbar 가 임계값의 어느 쪽에 있는지,
                    # 그리고 축소가 Gaussian 을 구분하고 있는지를 함께 봐야 한다.
                    # w 가 전부 같은 값이면 T 는 Sbar 한 점으로 모이고, 선택은
                    # 개체별 성질이 아니라 Sbar 대 임계값의 전역 비교가 된다.
                    _th = float(conf.strategy.densify.clone_grad_threshold)
                    _sb = float(Sbar)
                    _nS = int((S > _th).sum())
                    _nT = int((T > _th).sum())
                    print(f"[sm] step {trainer.global_step:>6} G={int(strategy.model.positions.shape[0]):>9,} "
                          f"n={int(ok.sum()):>8} "
                          f"Sbar={_sb:.3e} Sbar/th={_sb/_th:.3f} "
                          f"선택 S>{_nS:,} → T>{_nT:,} "
                          f"tau2={float(tau2):.3e} v_mean={float(v[sel].mean()):.3e} "
                          f"v/tau2={float(v[sel].mean()/tau2):.3e} "
                          f"w: p10={float(wv.quantile(0.1)):.4f} "
                          f"p50={float(wv.median()):.4f} p90={float(wv.quantile(0.9)):.4f} "
                          f"min={float(wv.min()):.4f}", flush=True)
                if gate_mask is not None:
                    n_gated = int(gate_mask.sum())
                    n_ok = int(ok.sum())
                    T = T.masked_fill(gate_mask, 0.0)
                    if _SM_DBG or (trainer.global_step % (dfreq_dbg * 5) == 0):
                        print(f"[gate] step {trainer.global_step:>6} "
                              f"관측 {n_ok:,} 중 차단 {n_gated:,} "
                              f"({100.0*n_gated/max(1,n_ok):.1f}%) "
                              f"임계 {gate_min_hits:g}", flush=True)
                # gs.py 는 accum/denom 을 다시 나누므로 denom 을 곱해 되돌린다
                strategy.densify_grad_norm_accum = (T * n).contiguous()
            out = orig_densify(scene_extent)
            hit_holder[1] = torch.zeros_like(strategy.densify_grad_norm_accum)
            return out

        strategy.densify_gaussians = densify_with_shrinkage
    rho_holder = [None]

    if step_shrink:
        # Adam 은 gradient 크기에 불변이다. g -> c*g 이면 m -> c*m, v -> c^2*v 라
        #     m_hat / sqrt(v_hat) = c*m_hat / (c*sqrt(v_hat))
        # 로 c 가 상쇄된다. 따라서 gradient 에 rho 를 곱해도 갱신량이 바뀌지 않는다
        # (counter 실측 +0.031 dB, 실행 편차의 2.4 배에 그침).
        # 신뢰도를 반영하려면 Adam 정규화가 끝난 뒤의 갱신량에 곱해야 한다.
        #     theta_new = theta_old + rho * (theta_adam - theta_old)
        # 이는 Gaussian 마다 학습률을 rho 배 하는 것과 같다.
        _opt = model.optimizer
        _orig_step = _opt.step

        def _step_with_rho(*a, **kw):
            r = rho_holder[0]
            if r is None or r.shape[0] != model.positions.shape[0]:
                rho_holder[0] = None
                return _orig_step(*a, **kw)
            before = model.positions.data.clone()
            out = _orig_step(*a, **kw)
            with torch.no_grad():
                model.positions.data.copy_(before + r * (model.positions.data - before))
            rho_holder[0] = None
            return out

        _opt.step = _step_with_rho
    l1w = float(conf.loss.lambda_l1)
    ssimw = float(conf.loss.lambda_ssim)

    def calibrate(sensor_position):
        """split-half 표본을 하나 모은다. every step 마다 한 번만 돈다."""
        full = getattr(dataset, "last_full", None)
        if full is None or model.positions.grad is None:
            return
        g_a = model.positions.grad.detach().clone()
        h_a = hit_holder[0].detach().clone() if hit_holder[0] is not None else None
        second = dataset.draw(full, advance=False)
        try:
            out = model(second, train=True)
            loss = l1w * torch.abs(out["pred_rgb"] - second.rgb_gt).mean()
            if ssimw > 0:
                loss = loss + ssimw * (1.0 - ssim(out["pred_rgb"].permute(0, 3, 1, 2),
                                                  second.rgb_gt.permute(0, 3, 1, 2)))
            # .grad 버퍼를 건드리지 않으므로 전체 파라미터를 복제했다 되돌릴 필요가 없다
            g_b = torch.autograd.grad(loss, model.positions, retain_graph=False)[0].detach()
            del out, loss, second
            with torch.no_grad():
                distance = (model.positions.detach() - sensor_position).norm(dim=1, keepdim=True)
                online.grow(int(model.num_gaussians), g_a.device)
                online.observe(g_a, g_b, distance, h_a)
        finally:
            # The second forward overwrites the visibility hook; ADC uses g_A.
            hit_holder[0] = h_a
        del g_a, g_b
        torch.cuda.empty_cache()

    # densify 와 계측이 같은 step 에 겹치면 메모리 봉우리가 쌓인다. 본 학습 forward
    # 와 backward 위에 계측용 렌더 한 벌이 더 올라가고, 그 상태에서 densify 가
    # 파라미터마다 torch.cat 으로 순간 두 배를 잡는다. kitchen 4 회가 전부 300 과
    # 100 의 공배수 step(10,500 · 10,800)에서 OOM 으로 죽었다. 겹치는 step 은 건너뛴다.
    dfreq = int(conf.strategy.densify.frequency)
    dstart = int(conf.strategy.densify.start_iteration)
    dend = int(conf.strategy.densify.end_iteration)

    def _densify_soon(step):
        return dstart <= step <= dend and (step % dfreq) in (0, 1)

    @torch.no_grad()
    def update_gradient_buffer(sensor_position, batch=None):
        # 원본 gs.py 의 update_gradient_buffer 는 @torch.no_grad() 가 붙어 있다.
        # 이걸 빠뜨리면 positions[mask] 에서 autograd 그래프가 생기고, 그 그래프가
        # densify_grad_norm_accum 에 누적되어 densify 주기(300 step) 내내 살아남는다.
        # 매 step 의 중간 텐서가 통째로 붙들려 kitchen 이 126 만 개에서 OOM 났다.
        # 같은 코드의 무보정 실행은 230 만 개까지 완주한다.
        if (online is not None and dataset is not None
                and trainer.global_step % every == 0
                and not _densify_soon(trainer.global_step)):
            # 계측은 gradient 계산이 필요하므로 no_grad 를 잠시 푼다
            with torch.enable_grad():
                calibrate(sensor_position)
        h = hit_holder[0]
        if online is not None and h is not None:
            online.grow(int(strategy.model.positions.shape[0]), h.device)
            online.observe_hits(h)
        if h is None:
            return original(sensor_position, batch)
        params_grad = strategy.model.positions.grad
        if params_grad is None:
            return original(sensor_position, batch)
        mask = (params_grad != 0).max(dim=1)[0]
        distance = (strategy.model.positions[mask] - sensor_position).norm(dim=1, keepdim=True)
        grad_norm = torch.norm(params_grad[mask] * distance, dim=-1, keepdim=True) / 2
        hm = h[mask].clamp_min(1.0)
        # 축소 인자 = 1 / sqrt(1 + k(h)).  k(h) 는 split-half 로 실측한 곡선을
        # 로그-로그 공간에서 선형 보간해 쓴다. k ∝ 1/h 를 가정했다가 반증됐다 —
        # 실측은 k×h 가 h 에 따라 4.97 에서 44.7 로 9 배 커진다. 한 Gaussian 을
        # 맞추는 Ray 들이 독립 표본이 아니기 때문으로 본다.
        # alpha 는 k(h) 의 눈금이다. k 는 1/(2*block^2) 짜리 두 반쪽에서 쟀는데
        # 실제 통계량은 1/block^2 한 벌로 계산되므로, Ray 가 두 배인 만큼 분산이
        # 절반이다. 이론상 alpha = 0.5 이지만 중앙값의 비와 비의 중앙값이 다르고
        # 제곱근 젠센도 섞여 정확히 떨어지지 않아 실측으로 정한다.
        live = online.knots if (online is not None and online.knots is not None) else knots
        # RADC-Gate 는 densify 주기 동안 누적된 적중 수를 쓴다. shrink_mean 여부와
        # 무관하게 필요하므로 여기서 항상 쌓는다 (텐서 하나, 비용 무시 가능).
        if (gate_min_hits > 0 or step_min_hits > 0 or target_g > 0
                or quantile > 0 or waste_target > 0 or shrink_mean
                or gate_z > 0 or gate_auto > 0):
            if (hit_holder[1] is None
                    or hit_holder[1].shape[0] != strategy.densify_grad_norm_accum.shape[0]):
                hit_holder[1] = torch.zeros_like(strategy.densify_grad_norm_accum)
            hit_holder[1][mask] += hm.unsqueeze(-1)
        if shrink_mean:
            # 평균 축소판에서는 여기서 곱하지 않는다. 축소 인자가 h 의 비선형
            # 함수이므로 step 별로 곱하고 평균하면 mean(w(h)*g) 가 되는데,
            # 필요한 것은 w(h_bar)*mean(g) 다. densify 시점에 누적 평균으로 한 번
            # 적용한다. 여기서는 원시 노름과 적중 수만 쌓는다.
            shrink = torch.ones_like(grad_norm)
        elif auto_scale:
            # 모양과 크기를 분리한다. 개체별 항은 h 에 따른 상대 순위만 담당하도록
            # 인구 중앙 h 에서 1 이 되게 정규화하고, 크기는 실측 전역 부풀림이 정한다.
            #     shrink_i = sqrt((1+k(h_med)) / (1+k(h_i))) / phi_global
            # alpha 를 손으로 넣지 않는다. 인구 중앙값이 아직 없으면 (첫 densify
            # 이전) 기존 식으로 돌린다.
            km = (getattr(online, "k_pool", None) if pooled_k
                  else getattr(online, "k_med", None)) if online is not None else None
            hmd = getattr(online, "h_med", None) if online is not None else None
            if km is None or hmd is None:
                shrink = 1.0 / torch.sqrt(1.0 + alpha * _k_of_h(hm, live))
            else:
                ref = 1.0 + float(_k_of_h(torch.tensor([hmd], device=hm.device), live)[0])
                shape = torch.sqrt(ref / (1.0 + _k_of_h(hm, live)))
                shrink = shape / _phi_global(km, hmd)
                if trainer.global_step % 3000 == 0:
                    print(f"[auto-scale] step {trainer.global_step} "
                          f"k={km:.3f}({'pool' if pooled_k else 'med'}) "
                          f"k_med={getattr(online,'k_med',float('nan')):.3f} "
                          f"k_pool={getattr(online,'k_pool',None) if getattr(online,'k_pool',None) is None else round(getattr(online,'k_pool'),3)} "
                          f"h_med={hmd:.2f} "
                          f"phi={_phi_global(km, hmd):.3f} "
                          f"shrink p50={shrink.median():.4f}", flush=True)
            shrink = shrink.unsqueeze(-1)
        else:
            shrink = 1.0 / torch.sqrt(1.0 + alpha * _k_of_h(hm, live))
            shrink = shrink.unsqueeze(-1)
        if (grad_shrink or step_shrink) and trainer.global_step % 2000 == 0:
            _r = (shrink * shrink).flatten()
            print(f"[rho] step {trainer.global_step} n={_r.numel()} "
                  f"mean={_r.mean():.4f} p10={_r.quantile(0.1):.4f} "
                  f"p50={_r.median():.4f} p90={_r.quantile(0.9):.4f}", flush=True)
        if step_shrink:
            # 맞은 Gaussian 만 rho 를 쓰고 나머지는 1 로 둔다 (갱신량 그대로).
            rho = torch.ones((strategy.model.positions.shape[0], 1),
                             device=params_grad.device, dtype=params_grad.dtype)
            rho[mask] = (shrink * shrink).to(rho.dtype)
            rho_holder[0] = rho
        if grad_shrink:
            # 같은 신뢰도를 optimizer 에도 적용한다. 관측 gradient 를 g = mu + eps,
            # Var(eps) = sigma^2 라 하면 mu 에 대한 평균제곱오차 최소 선형 추정은
            #     mu_hat = ||mu||^2 / (||mu||^2 + sigma^2) * g = rho * g,
            # rho = 1/(1+k) 이다 (Wiener 축소). densification 통계량에 쓰는 인자는
            # sqrt(rho) 인데, 그쪽은 노름 ||mu|| 의 추정이고 이쪽은 벡터 mu 의
            # 추정이라 지수가 하나 다르다. 이중 적용이 아니다.
            # post_backward 는 optimizer.step() 보다 먼저 돈다 (trainer.py:813 대 845).
            params_grad[mask] = params_grad[mask] * (shrink * shrink)
        strategy.densify_grad_norm_accum[mask] += grad_norm * shrink
        strategy.densify_grad_norm_denom[mask] += 1
        if step_min_hits > 0 and h is not None and h.numel() == mask.numel():
            # 개체 게이트를 통과한 Gaussian 의 통계량을 다시 만들 때 쓸 누적이다.
            # 보정 후 잔여 부풀림이 1 을 넘는 h < step_min_hits 관측을 빼고 쌓는다.
            # 기존 누적을 건드리지 않으므로 개체 게이트의 판정 근거(h_bar)와
            # 배제의 확실성이 유지된다.
            hi = mask & (h >= step_min_hits)
            if hi_holder[0] is None or hi_holder[0].shape != strategy.densify_grad_norm_accum.shape:
                hi_holder[0] = torch.zeros_like(strategy.densify_grad_norm_accum)
                hi_holder[1] = torch.zeros_like(strategy.densify_grad_norm_denom)
            if bool(hi.any()):
                _sel = hi[mask]
                hi_holder[0][hi] += (grad_norm * shrink)[_sel]
                hi_holder[1][hi] += 1
        if _MEMDBG and trainer.global_step == _MEMDBG_START:
            torch.cuda.memory._record_memory_history(max_entries=200000, stacks="python")
            print(f"[mem] step {trainer.global_step} 할당 이력 기록 시작", flush=True)
        if _MEMDBG and trainer.global_step == _MEMDBG_DUMP:
            import collections
            snap = torch.cuda.memory_snapshot()
            # 살아 있는 블록을 할당 지점(스택 최상단 프레임)별로 합산한다
            by_site = collections.Counter()
            for sg in snap:
                for b in sg["blocks"]:
                    if b["state"] != "active_allocated":
                        continue
                    fr = b.get("frames") or []
                    # 우리 코드 프레임을 우선 고르고, 없으면 최상단을 쓴다
                    site = "unknown"
                    for f in fr:
                        nm = f.get("filename", "")
                        if any(t in nm for t in ("3dgrut/", "mc_sparse_training", "threedgrt")):
                            site = f"{nm.split('/')[-1]}:{f.get('line',0)} {f.get('name','')}"
                            break
                    else:
                        if fr:
                            f = fr[0]
                            site = f"{f.get('filename','?').split('/')[-1]}:{f.get('line',0)} {f.get('name','')}"
                    by_site[site] += b["size"]
            print(f"[mem] step {trainer.global_step} 활성 블록 할당 지점 상위 12", flush=True)
            for site, sz in by_site.most_common(12):
                print(f"        {sz/2**20:>9,.1f} MiB   {site}", flush=True)
            torch.cuda.memory._record_memory_history(enabled=None)
        if _MEMDBG and trainer.global_step % 1000 == 0:
            import collections
            snap = torch.cuda.memory_snapshot()
            seg = collections.Counter()
            act = 0
            for sg in snap:
                seg[(sg["segment_type"], round(sg["total_size"] / 2**20))] += 1
                act += sum(b["size"] for b in sg["blocks"] if b["state"] == "active_allocated")
            top = ", ".join(f"{t}/{mb}MiB×{c}" for (t, mb), c in seg.most_common(6))
            print(f"[mem] step {trainer.global_step} G {int(strategy.model.num_gaussians):,} "
                  f"할당 {torch.cuda.memory_allocated()/2**20:,.0f} "
                  f"예약 {torch.cuda.memory_reserved()/2**20:,.0f} "
                  f"구간 {len(snap)}개 활성 {act/2**20:,.0f}MiB | {top}", flush=True)
        # forward 마다 새로 만든 텐서를 다음 step 까지 붙들면, expandable_segments 가
        # 잡아 둔 구간이 그 텐서 하나 때문에 반환되지 않고 계속 쌓인다. 같은 step 안에서
        # 다 썼으므로 여기서 놓아 준다. kitchen 이 이것 때문에 129 만 개에서 OOM 났고,
        # 같은 코드의 plain 은 230 만 개까지 갔다.
        hit_holder[0] = None

    strategy.update_gradient_buffer = update_gradient_buffer

    if online is not None:
        # densify 직전에 그 구간에 모은 표본으로 곡선을 갱신한다
        original_densify = strategy.densify_gaussians

        def densify_gaussians(scene_extent):
            online.refresh()
            if online.knots is not None:
                hs, ks = online.knots
                print(f"[k(h)] step {trainer.global_step}: {len(hs)} 점  "
                      + " ".join(f"h{h:.0f}={k:.2f}" for h, k in list(zip(hs, ks))[:6]),
                      flush=True)
            return original_densify(scene_extent)

        strategy.densify_gaussians = densify_gaussians

    if step_min_hits > 0:
        print(f"[step-gate] 한 step 에서 Ray 를 {step_min_hits:g} 개 미만 받은 관측은 "
              "densification 통계량에 누적하지 않는다", flush=True)
    if (gate_min_hits > 0 or step_min_hits > 0 or target_g > 0
            or quantile > 0 or tau_phi or waste_target > 0 or gate_z > 0 or gate_auto > 0
            or eb_shrink) and not shrink_mean:
        # 표준 RADC 경로(곱셈 축소)에서도 게이트를 건다. densify 직전에 통계량을
        # 0 으로 만들어 clone/split 후보에서 뺀다. prune 은 별도 경로이므로
        # 제거가 아니라 성장 배제다.
        _orig_prune_bufs_g = strategy.prune_densification_buffers

        def _prune_bufs_gate(valid_mask):
            out = _orig_prune_bufs_g(valid_mask)
            if hit_holder[1] is not None and hit_holder[1].shape[0] == valid_mask.shape[0]:
                hit_holder[1] = hit_holder[1][valid_mask]
            for _i in (0, 1):
                if hi_holder[_i] is not None and hi_holder[_i].shape[0] == valid_mask.shape[0]:
                    hi_holder[_i] = hi_holder[_i][valid_mask]
            if (fresh_holder[0] is not None
                    and fresh_holder[0].shape[0] == valid_mask.shape[0]):
                fresh_holder[0] = fresh_holder[0][valid_mask]
            for _j in range(4):
                if (sh_holder[_j] is not None
                        and sh_holder[_j].shape[0] == valid_mask.shape[0]):
                    sh_holder[_j] = sh_holder[_j][valid_mask]
            return out

        strategy.prune_densification_buffers = _prune_bufs_gate
        _orig_densify_g = strategy.densify_gaussians

        @torch.no_grad()
        def _densify_with_gate(scene_extent):
            # quantile / target_g 분기가 이 이름에 대입하므로 파이썬이 지역 변수로
            # 취급한다. 바깥 인자를 그대로 읽고 쓰려면 nonlocal 이 필요하다.
            nonlocal gate_min_hits
            hits = hit_holder[1]
            acc = strategy.densify_grad_norm_accum
            if hits is not None and hits.shape[0] == acc.shape[0]:
                # 기준은 h = 관측된 step 당 평균 적중 수다. k(h) 곡선의 정의역이
                # h >= 1.5 이고 그 밖은 끝값으로 고정되므로, 보정량을 모르는 영역을
                # h 로 지정한다. 누적 적중으로 자르면 관측 step 수가 섞여 기준이
                # 조건마다 달라진다.
                dnm = strategy.densify_grad_norm_denom.float().clamp_min(1.0)
                hb = hits / dnm
                if waste_target > 0 and fresh_holder[0] is not None:
                    # 직전 주기에 만들어진 자식 중 이번 구간에 한 번도 관측되지
                    # 않은 비율. 반사실이 필요 없다 — 만들어진 개체가 실제로
                    # 어떻게 됐는지를 본다.
                    _fr = fresh_holder[0]
                    if _fr.shape[0] == acc.shape[0] and bool(_fr.any()):
                        # "한 번도 관측 안 됨" 은 너무 느슨하다 — densify 주기가
                        # 300 step 이라 실측 낭비율이 0.1% 로 나온다. 대신 신뢰도
                        # 바닥을 쓴다: h_bar <= 1 은 관측된 모든 step 에서 ray 를
                        # 정확히 하나만 받은 층이고, split-half 로 잰 rho 가
                        # 0.0001 이다. 그 상태로 태어난 자식은 통계량이 무의미하다.
                        _seen = (strategy.densify_grad_norm_denom.reshape(-1) > 0)
                        _bad = (~_seen) | (hb.reshape(-1) <= 1.0)
                        _nf = int(_fr.sum())
                        _w = float(_bad[_fr].float().mean())
                        _r = _w / waste_target
                        _st = min(1.25, max(0.80, _r ** 0.5))
                        theta_state[0] = float(min(20.0, max(0.5, theta_state[0] * _st)))
                        gate_min_hits = theta_state[0]
                        if trainer.global_step % (int(conf.strategy.densify.frequency) * 5) == 0:
                            print(f"[waste] step {int(trainer.global_step):>6} "
                                  f"자식 {_nf:,} 중 낭비(h̄≤1) {_w*100:.1f}% "
                                  f"(목표 {waste_target*100:.0f}%) → theta {gate_min_hits:.2f}",
                                  flush=True)
                if tau_phi and online is not None and online.k_med is not None:
                    # tau 를 실측 부풀림으로 올린다. 자유 매개변수가 없다.
                    # 전역 배율이므로 어느 Gaussian 이 뽑히는지의 순서는 바뀌지 않고
                    # 몇 개가 임계를 넘느냐만 바뀐다.
                    _phi = _phi_global(online.k_med, online.h_med or 1.0)
                    strategy.clone_grad_threshold = _TAU0[0] * _phi
                    strategy.split_grad_threshold = _TAU0[1] * _phi
                    if trainer.global_step % (int(conf.strategy.densify.frequency) * 5) == 0:
                        print(f"[tau-phi] step {int(trainer.global_step):>6} "
                              f"k_med={online.k_med:.3f} h_med={online.h_med:.2f} "
                              f"phi={_phi:.3f} → tau {strategy.clone_grad_threshold:.3e}",
                              flush=True)
                if quantile > 0:
                    # 관측된 개체의 h_bar 분포에서 q 분위수를 임계로 쓴다.
                    # 미관측(h_bar=0)은 어차피 accum 이 0 이라 분포에서 뺀다.
                    _c = (strategy.densify_grad_norm_denom.reshape(-1) > 0)
                    if int(_c.sum()) > 100:
                        gate_min_hits = float(hb.reshape(-1)[_c].quantile(quantile))
                        theta_state[0] = gate_min_hits
                        if trainer.global_step % (int(conf.strategy.densify.frequency) * 5) == 0:
                            print(f"[quantile-theta] step {int(trainer.global_step):>6} "
                                  f"q={quantile:.2f} → theta {gate_min_hits:.3f} "
                                  f"(후보 {int(_c.sum()):,})", flush=True)
                if target_g > 0:
                    # 개수 목표 제어. densify 구간(500~end) 동안 G 를 지수 궤적으로
                    # target_g 까지 끌고 간다. 현재 G 가 궤적보다 많으면 theta 를 올려
                    # 성장을 조인다. 품질 신호를 쓰지 않으므로 반사실도 test 누출도 없다.
                    _s0 = int(conf.strategy.densify.start_iteration)
                    _s1 = int(conf.strategy.densify.end_iteration)
                    _now = int(trainer.global_step)
                    if theta_state[1] is None:
                        theta_state[1] = float(acc.shape[0])
                    _f = min(1.0, max(0.0, (_now - _s0) / max(1, _s1 - _s0)))
                    _goal = theta_state[1] * (target_g / max(theta_state[1], 1.0)) ** _f
                    _r = float(acc.shape[0]) / max(_goal, 1.0)
                    # 제곱근 이득. 한 주기에 theta 가 과하게 튀지 않게 [0.7, 1.4] 로 자른다.
                    _step = min(1.4, max(0.7, _r ** 0.5))
                    theta_state[0] = float(min(20.0, max(0.5, theta_state[0] * _step)))
                    gate_min_hits = theta_state[0]
                    if trainer.global_step % (int(conf.strategy.densify.frequency) * 5) == 0:
                        print(f"[online-theta] step {_now:>6} G={int(acc.shape[0]):,} "
                              f"목표 {_goal:,.0f} 비 {_r:.3f} → theta {gate_min_hits:.2f}",
                              flush=True)
                # gate_min_hits 가 0 이면 개체 배제는 하지 않는다. 이때 이 래퍼는
                # 통계량을 h >= step_min_hits 인 관측만으로 다시 만드는 일만 한다.
                gm = (hb <= float(gate_min_hits)) if gate_min_hits > 0 \
                    else torch.zeros_like(hb, dtype=torch.bool)
                if (gate_auto > 0 and theta_state[1] != "auto_done"
                        and trainer.global_step >= _GATE_AUTO_STEP):
                    # 첫 주기에서 한 번만 정하고 그대로 둔다. 매 주기 다시 정하면
                    # 차단 강도가 학습 내내 일정해져 초반 성장을 막는다 (garden
                    # 실측: 상시 35% 차단이 고정 theta 와 같은 Gaussian 수에서
                    # PSNR 을 0.192 dB 떨어뜨렸다). 고정 theta 의 차단율이 1% 에서
                    # 47% 로 오르는 것은 h_bar 분포가 내려가기 때문이고, theta 를
                    # 붙잡아 두면 그 스케줄이 저절로 생긴다.
                    # 측정 시점을 step 1500 으로 고정한다. h_bar 의 중앙값은 학습
                    # 초반에 급락하므로 (counter 67.7@600 → 36.2@1500) 어느 주기에서
                    # 재느냐가 theta 를 두 배 바꾼다. 계수 0.1361 은 1500 step 값으로
                    # 맞춘 것이라 같은 시점에서 재야 한다.
                    _c = (strategy.densify_grad_norm_denom.reshape(-1) > 0)
                    if int(_c.sum()) > 1000:
                        _p50 = float(hb.reshape(-1)[_c].median())
                        gate_min_hits = float(min(20.0, max(1.0, gate_auto * _p50)))
                        theta_state[0] = gate_min_hits
                        theta_state[1] = "auto_done"
                        gm = hb <= float(gate_min_hits)
                        print(f"[gate-auto] step {trainer.global_step} "
                              f"후보 {int(_c.sum()):,} h_bar p50={_p50:.2f} "
                              f"x C={gate_auto:g} → theta {gate_min_hits:.3f} (이후 고정)",
                              flush=True)
                if gate_z > 0 and online is not None and getattr(online, 'gate_knots', None) is not None:
                    # theta 를 실측 k(h) 곡선에서 정한다.
                    #
                    # 게이트가 빼야 하는 것은 "통계량이 제 noise 에 묻힌 개체" 다.
                    # 그 정도를 재는 양은 k = sigma^2/||mu||^2 이고, OnlineKCurve 가
                    # 한 step 안에서 두 벌을 뽑아(split-half) h 구간별로 이미 재고
                    # 있다. k 는 h 에 대해 감소하므로
                    #     theta = k(h) = K* 가 되는 h
                    # 로 잡으면 경계가 하나로 정해진다. K* 는 무차원 신뢰도 목표라
                    # 씬·block·해상도가 바뀌어도 같은 값을 쓴다. 곡선이 씬과 학습
                    # 단계를 따라 움직이므로 theta 가 그만큼 따라 움직인다.
                    #
                    # 주기 평균의 표준오차로 잡으려던 앞선 설계는 쓰지 못한다.
                    # 주기 안에서 100 번 평균하면 표준오차가 sqrt(n) 로 줄어
                    # S/SE 가 어느 h 에서나 4 를 넘었다 (counter step 1500 실측
                    # h=2.2 에서 4.09). 부풀림은 평균해도 줄지 않으므로 표준오차는
                    # 부풀림의 대리 변수가 못 된다. k 는 step 안에서 재므로 준다.
                    _kh, _kk = online.gate_knots
                    _new_theta = None
                    if len(_kh) >= 2:
                        if _kk[0] <= gate_z:
                            _new_theta = 1.0          # 최저 h 에서 이미 목표 이하
                        else:
                            for _i in range(1, len(_kh)):
                                if _kk[_i] <= gate_z:
                                    _x0, _x1 = math.log(_kh[_i-1]), math.log(_kh[_i])
                                    _y0, _y1 = _kk[_i-1], _kk[_i]
                                    _t = (_y0 - gate_z) / max(_y0 - _y1, 1e-9)
                                    _new_theta = math.exp(_x0 + _t * (_x1 - _x0))
                                    break
                            if _new_theta is None:
                                _new_theta = _kh[-1]  # 전 구간이 목표 위
                    if _new_theta is not None:
                        _prev = theta_state[0]
                        _tgt = float(min(50.0, max(1.0, _new_theta)))
                        # 주기 간 요동을 줄이려고 로그 공간에서 절반씩 옮긴다
                        gate_min_hits = math.exp(0.5 * math.log(max(_prev, 1e-3))
                                                 + 0.5 * math.log(_tgt))
                        theta_state[0] = gate_min_hits
                        gm = hb <= float(gate_min_hits)
                    if trainer.global_step % (int(conf.strategy.densify.frequency) * 5) == 0:
                        print(f"[online-theta] step {trainer.global_step:>6} "
                              f"목표 k={gate_z:g} | 매듭 {len(_kh)} | "
                              f"교차 h={_new_theta if _new_theta else float('nan'):.2f} "
                              f"→ theta {gate_min_hits:.2f} | k(h): "
                              + " ".join(f"{h:.1f}:{k:.2f}"
                                         for h, k in zip(_kh[:7], _kk[:7])), flush=True)
                n_g = int(gm.sum())
                new_acc = acc.masked_fill(gm, 0.0) if (gate_min_hits > 0 or gate_z > 0 or gate_auto > 0) else acc
                if (step_min_hits > 0 and hi_holder[0] is not None
                        and hi_holder[0].shape == acc.shape):
                    # 통계량을 h>=step_min_hits 인 관측만으로 다시 만든다.
                    # 개체 게이트가 켜져 있으면 그것을 통과한 것에만 적용한다.
                    # 이때 분모(관측 수)는 줄이지 않는다.
                    #
                    #   S = ( h>=m 인 관측의 합 ) / ( 전체 관측 수 )
                    #
                    # 분모까지 줄이면 평균이 거의 안 내려가고 표본만 줄어 분산이
                    # 커진다. 분자에서만 빼면 평균과 분산이 함께 내려간다.
                    # 보정할 수 없는 관측은 통계량에 아무것도 넣지 않되 관측
                    # 횟수로는 센다는 뜻이다. 자주 보이지만 매번 Ray 를 하나만
                    # 받는 Gaussian 은 통계량이 0 에 가까워진다.
                    #
                    # gs.py 가 accum/denom 을 다시 나누므로 accum 자리에
                    # h>=m 누적을 그대로 넣으면 위 식이 된다.
                    new_acc = torch.where(~gm, hi_holder[0], new_acc)
                    if trainer.global_step % (int(conf.strategy.densify.frequency) * 5) == 0:
                        # h<m 관측을 뺀 통계량이 실제로 얼마나 내려가는지. 선택 수가
                        # 줄지 않으면 이 방식은 count 를 통제하지 못한다.
                        _c = (strategy.densify_grad_norm_denom.reshape(-1) > 0)
                        _d = dnm.reshape(-1)[_c]
                        _th = float(conf.strategy.densify.clone_grad_threshold)
                        _Sa = acc.reshape(-1)[_c] / _d
                        _Sh = hi_holder[0].reshape(-1)[_c] / _d
                        _dh = hi_holder[1].float().reshape(-1)[_c]
                        print(f"[stepgate] step {trainer.global_step:>6} "
                              f"m={step_min_hits:g} 후보 {int(_c.sum()):,} | "
                              f"S 전체 p50={_Sa.median():.2e} 선택 {int((_Sa>_th).sum()):,} | "
                              f"S(h>=m) p50={_Sh.median():.2e} 선택 "
                              f"{int((_Sh>_th).sum()):,} | "
                              f"남은 관측 비율 p50={(_dh/_d).median():.3f} "
                              f"전부배제 {int((_dh==0).sum()):,}", flush=True)
                if eb_shrink:
                    # densification 은 개체마다 정밀도가 다른 추정치 10^6 개로
                    # 동시에 내리는 결정이다. 고정 임계값은 모든 추정치의 정밀도가
                    # 같을 때만 옳다. dense 학습에서는 시야 안 개체가 대체로
                    # 충분한 Ray 를 받아 근사적으로 성립하지만, 1/16 sparse 에서는
                    # h 가 1 에서 100 이상까지 흩어져 정밀도가 100 배 벌어진다.
                    #
                    # 여기서는 정규-정규 경험적 베이즈로 개체별 정밀도를 반영한다.
                    #     S_i ~ N(mu_i, sigma_i^2),   mu_i ~ N(m, T^2)
                    #     사후 평균 = m + T^2/(T^2+sigma_i^2) * (S_i - m)
                    # sigma_i^2 는 실측 k(h) 곡선에서 얻는다. 주기 안에서 n_i 번
                    # 평균했으므로 평균의 분산은 k(h_i)*S_i^2/n_i 다. 부풀림(bias)
                    # 은 평균해도 줄지 않지만 평균의 분산은 n 으로 줄기 때문이다.
                    # m 과 T^2 는 표본에서 추정하므로 맞출 상수가 없다.
                    _dv = dnm.reshape(-1).float()
                    _cand = _dv > 0
                    if int(_cand.sum()) > 1000:
                        _live = (online.knots if (online is not None
                                 and online.knots is not None) else knots)
                        _S = new_acc.reshape(-1)[_cand] / _dv[_cand]
                        _k = _k_of_h(hb.reshape(-1)[_cand].clamp_min(1.0), _live)
                        _sig2 = (_k * _S * _S / _dv[_cand]).clamp_min(1e-30)
                        _m = _S.mean()
                        _tot = _S.var(unbiased=False)
                        _T2 = (_tot - _sig2.mean()).clamp_min(1e-30)
                        _B = _T2 / (_T2 + _sig2)
                        _Snew = (_m + _B * (_S - _m)).clamp_min(0.0)
                        _flat = new_acc.reshape(-1).clone()
                        _flat[_cand] = _Snew * _dv[_cand]
                        new_acc = _flat.reshape(new_acc.shape)
                        if trainer.global_step % (int(conf.strategy.densify.frequency) * 5) == 0:
                            _th = float(conf.strategy.densify.clone_grad_threshold)
                            print(f"[eb] step {trainer.global_step:>6} 후보 {int(_cand.sum()):,} "
                                  f"| B p10={_B.quantile(0.1):.3f} p50={_B.median():.3f} "
                                  f"p90={_B.quantile(0.9):.3f} | T2={float(_T2):.3e} "
                                  f"sig2 p50={_sig2.median():.3e} | 선택 "
                                  f"{int((_S > _th).sum()):,} -> {int((_Snew > _th).sum()):,}",
                                  flush=True)
                strategy.densify_grad_norm_accum = new_acc.contiguous()
                if trainer.global_step % (int(conf.strategy.densify.frequency) * 5) == 0:
                    # h_bar 는 0 (그 주기에 한 번도 관측 안 됨) 아니면 1 이상이다.
                    # 각 step 의 h 가 clamp_min(1) 이므로 hits >= denom 이기 때문이다.
                    # 미관측 개체는 accum 이 이미 0 이라 차단해도 동작이 없다.
                    # 따라서 후보(denom>0) 를 분모로 다시 세야 실제 차단 강도가 나온다.
                    _cand = (strategy.densify_grad_norm_denom.reshape(-1) > 0)
                    _gmf = gm.reshape(-1)
                    _nc = int(_cand.sum()); _ng_eff = int((_gmf & _cand).sum())
                    _hv = hb.reshape(-1)[_cand]
                    _q = [float(_hv.quantile(x)) for x in (0.1, 0.25, 0.5, 0.75, 0.9)] if _nc else [0]*5
                    _dv = strategy.densify_grad_norm_denom.float().reshape(-1)[_cand]
                    _dq = [float(_dv.quantile(x)) for x in (0.1, 0.5, 0.9)] if _nc else [0]*3
                    print(f"[gate] step {trainer.global_step:>6} "
                          f"G={int(acc.shape[0]):,} 후보 {_nc:,} "
                          f"미관측 {int(acc.shape[0])-_nc:,} | "
                          f"차단(전체기준) {n_g:,} ({100.0*n_g/max(1,int(acc.shape[0])):.1f}%) "
                          f"차단(후보기준) {_ng_eff:,} ({100.0*_ng_eff/max(1,_nc):.1f}%) | "
                          f"h_bar p10={_q[0]:.2f} p25={_q[1]:.2f} p50={_q[2]:.2f} "
                          f"p75={_q[3]:.2f} p90={_q[4]:.2f} | "
                          f"denom p10={_dq[0]:.0f} p50={_dq[1]:.0f} p90={_dq[2]:.0f} | "
                          f"임계 {gate_min_hits:g}", flush=True)
                    if _nc > 1000:
                        # h_bar 와 관측 수가 얼마나 함께 움직이는지. 상관이 강하면
                        # h 게이트가 관측이 적은 개체도 대부분 함께 걸러낸다.
                        _lh = torch.log(_hv.clamp_min(1e-6)); _ld = torch.log(_dv.clamp_min(1.0))
                        _r = float(((_lh-_lh.mean())*(_ld-_ld.mean())).mean()
                                   / (_lh.std().clamp_min(1e-9)*_ld.std().clamp_min(1e-9)))
                        _pass = _hv > float(gate_min_hits)          # 게이트를 통과한 것
                        _dmed = float(_dv.median())
                        _lown = _pass & (_dv < 0.25*_dmed)          # 통과했지만 관측이 매우 적음
                        # 이들의 통계량이 임계값을 넘는 비율 (실제로 densify 될 후보)
                        _acc = acc.reshape(-1)[_cand]
                        _S = _acc / _dv.clamp_min(1.0)
                        _th = float(conf.strategy.densify.clone_grad_threshold)
                        _sel = _S > _th
                        print(f"[gate2] step {trainer.global_step:>6} "
                              f"corr(log h_bar, log denom) = {_r:+.3f} | "
                              f"통과 {int(_pass.sum()):,} 중 관측 하위(denom<{0.25*_dmed:.0f}) "
                              f"{int(_lown.sum()):,} ({100.0*int(_lown.sum())/max(1,int(_pass.sum())):.1f}%) | "
                              f"선택 {int(_sel.sum()):,} 중 그 집단 "
                              f"{int((_sel & _lown).sum()):,} "
                              f"({100.0*int((_sel & _lown).sum())/max(1,int(_sel.sum())):.1f}%)", flush=True)
            _n_before = int(strategy.model.positions.shape[0])
            out = _orig_densify_g(scene_extent)
            _n_after = int(strategy.model.positions.shape[0])
            if waste_target > 0:
                _f = torch.zeros(_n_after, dtype=torch.bool,
                                 device=strategy.model.positions.device)
                if _n_after > _n_before:
                    _f[_n_before:] = True
                fresh_holder[0] = _f
            hit_holder[1] = torch.zeros_like(strategy.densify_grad_norm_accum)
            if gate_z > 0:
                for _j in range(4):
                    sh_holder[_j] = torch.zeros_like(
                        strategy.densify_grad_norm_accum)
            if step_min_hits > 0:
                hi_holder[0] = torch.zeros_like(strategy.densify_grad_norm_accum)
                hi_holder[1] = torch.zeros_like(strategy.densify_grad_norm_denom)
            return out

        strategy.densify_gaussians = _densify_with_gate
        if gate_min_hits <= 0:
            print("[gate] 개체 배제는 꺼져 있다. 통계량을 "
                  f"h >= {step_min_hits:g} 인 관측만으로 다시 만든다 "
                  "(분모는 전체 관측 수 그대로)", flush=True)
        else:
            print(f"[gate] RADC-Gate 켜짐: 관측 step 당 평균 적중 h <= {gate_min_hits:g} 인 "
                  "Gaussian 을 clone/split 후보에서 제외한다 "
                  "(k(h) 곡선 정의역 h>=1.5 밖은 보정량 미상)", flush=True)

    src = "씬 자체 측정" if knots is not None else "counter 기본 곡선"
    print(f"[hit-shrink] Ray 적중 횟수로 통계량 축소, 인자 1/sqrt(1+{alpha:g}*k(h)), "
          f"k(h) = {src} ({len((knots or _K_OF_H_H)[0] if knots else _K_OF_H_H)} 점)", flush=True)


def install_percentile_threshold(trainer, percentile):
    """Select a fixed top fraction instead of thresholding an absolute value.

    Correction restores the *ordering* of the statistic but not its scale, which
    still tracks resolution about as strongly as before. An absolute threshold is
    therefore the wrong interface; densification only ever needed to pick the
    largest, and a percentile expresses that decision without depending on scale.
    """
    strategy = trainer.strategy
    original = strategy.densify_gaussians

    def densify_gaussians(scene_extent):
        with torch.no_grad():
            statistic = (strategy.densify_grad_norm_accum
                         / strategy.densify_grad_norm_denom.clamp_min(1))
            statistic[statistic.isnan()] = 0.0
            positive = statistic[statistic > 0]
            if positive.numel() > 0:
                k = max(1, int(percentile * positive.numel()))
                value = float(torch.topk(positive.flatten(), k).values[-1])
                strategy.clone_grad_threshold = value
                strategy.split_grad_threshold = value
        return original(scene_extent)

    strategy.densify_gaussians = densify_gaussians
    print(f"[percentile] densify selects the top {percentile*100:.2f}% "
          f"of the statistic", flush=True)


def install_mcmc_observe(trainer, every=1):
    """MCMC 전략에 개체별 관측 횟수를 넘긴다.

    relocate 는 opacity 만 보고 사망을 판정한다. multi-view 를 켜면 Ray 예산이
    같아도 view 당 격자가 성겨져 (block 4 -> 8) 작은 개체가 Ray 를 못 받고,
    image gradient 없이 정규화만 받아 opacity 가 깎인다. 죽는 개체가 저관측
    쪽에 몰려 있는지를 재기 위해 forward 출력의 hit 수를 누적한다.
    """
    # 전략이 학습 도중 교체될 수 있으므로 (--radc-to-mcmc) 객체를 잡아두지 않고
    # 매번 trainer.strategy 를 다시 본다. 교체 전 GS 구간에서는 observe 가 없어
    # 조용히 넘어가고, 교체 시점부터 누적이 시작된다.
    trainer._observe_every = every
    if hasattr(trainer.strategy, "observe"):
        trainer.strategy._audit_every = every
    original_forward = trainer.model.forward

    def forward(gpu_batch, *a, **k):
        out = original_forward(gpu_batch, *a, **k)
        vis = out.get("mog_visibility") if isinstance(out, dict) else None
        st = trainer.strategy
        if vis is not None and vis.dtype == torch.float32 and hasattr(st, "observe"):
            # float32 버퍼에 int32 로 쓰인 적중 횟수다 (referenceOptix.cu 참고)
            st.observe(vis.view(torch.int32).reshape(-1).float())
        return out

    trainer.model.forward = forward
    print("[observe] 개체별 관측 횟수 누적 활성화", flush=True)


def install_death_audit(trainer, every=500):
    """회복 불가 상태에 들어간 Gaussian 의 수를 주기적으로 기록한다.

    3DGRT 는 `gres * alpha > 1/255` 인 교차만 수락하고 이 조건이 backward 의
    gradient 누적도 막는다. gres <= 1 이므로 alpha 가 1/255 미만인 개체는 어떤
    Ray 에도 수락되지 않고, 모든 파라미터의 gradient 가 0 이 되어 되살아날 수
    없다 (2026-08-20 리포트 6절). 렌더링 기여도 정확히 0 이므로 제거해도 결과가
    바뀌지 않는다.

    출생(densify)과 사망(이 상태로의 진입)을 함께 세면 개체군의 수용력을 볼 수
    있다. 사망률이 개수에 따라 증가하면 평형 개수가 정의된다.
    """
    st = {"prev": None}
    orig = trainer.model.forward

    def forward(gpu_batch, *a, **k):
        out = orig(gpu_batch, *a, **k)
        if trainer.global_step % every == 0:
            with torch.no_grad():
                o = trainer.model.get_density().reshape(-1)
                dead = o < (1.0 / 255.0)
                n, nd = int(o.numel()), int(dead.sum())
                new = ""
                if st["prev"] is not None and st["prev"].numel() == n:
                    new = f" 신규사망 {int((dead & ~st['prev']).sum()):,}"
                st["prev"] = dead.clone()
                print(f"[death] step {trainer.global_step:>6} G={n:,} "
                      f"사망 {nd:,} ({100.0 * nd / max(n, 1):.2f}%){new} "
                      f"| alpha p10={float(o.kthvalue(max(1, n // 10)).values):.5f} "
                      f"p50={float(o.median()):.4f}", flush=True)
        return out

    trainer.model.forward = forward
    print(f"[death] 회복 불가 개체 계측 활성화 ({every} step 마다)", flush=True)


def install_radc_to_mcmc(trainer, switch_step, cap=-1, keep_reg=True,
                         end_iteration=-1, min_obs_ratio=0.0, prune_stop=0):
    """RADC 로 배치를 만들고 switch_step 부터 MCMC 로 이어받는다.

    RADC(휴리스틱 ADC + hit gate)는 500~15,000 에서 개수를 늘리고, MCMC 는
    개수를 고정한 채 죽은 개체를 살아 있는 곳으로 되돌린다. 두 구조의 담당
    구간이 겹치지 않으므로 이어 붙일 수 있다.

    이어 붙일 때 손대야 하는 것이 네 가지다.

    1. conf.strategy 서브트리. GSStrategy 는 densify/prune/reset_density 를,
       MCMCStrategy 는 relocate/add/perturb/binom_n_max/opacity_threshold 를
       읽는다. 교체 시점에 뒤쪽 키를 주입한다.

    2. opacity·scale 정규화. MCMC 의 회수는 lambda_opacity 가 opacity 를 계속
       0 쪽으로 미는 데 의존한다. base_gs.yaml 은 use_opacity=false 라 켜주지
       않으면 죽는 개체가 생기지 않아 relocate 가 빈 채로 돈다.

    3. 사망 문턱. RADC 의 prune.density_threshold 와 MCMC 의 opacity_threshold
       가 둘 다 0.005 다. RADC 가 방금 그 아래를 전부 잘라냈으므로 교체 직후의
       dead 집합은 반드시 비어 있다. 정규화를 켜서 다시 채워질 때까지 기다리는
       구조가 된다.

    4. perturb 의 크기. noise 는 positions 의 현재 lr 에 비례하는데 lr 은 지수
       감쇠라 15,000 시점에 초기값의 1/10 이다. 15,000~30,000 에 주입되는 총
       noise 는 0~30,000 전체의 9% 에 그친다.
    """
    from omegaconf import OmegaConf
    from threedgrut.strategy.mcmc import MCMCStrategy

    conf = trainer.conf
    OmegaConf.set_struct(conf, False)
    state = {"done": False}
    old_post_backward = trainer.strategy.post_backward

    def swap():
        n = int(trainer.model.num_gaussians)
        cap_v = int(cap) if cap > 0 else n            # 기본은 현재 개수 고정
        end_v = int(end_iteration) if end_iteration > 0 else 25000

        conf.strategy.method = "MCMCStrategy"
        conf.strategy.binom_n_max = 51
        conf.strategy.opacity_threshold = 0.005
        conf.strategy.relocate = OmegaConf.create(
            {"start_iteration": switch_step, "end_iteration": end_v, "frequency": 100})
        conf.strategy.add = OmegaConf.create(
            {"placement": "opacity", "start_iteration": switch_step,
             "end_iteration": end_v, "frequency": 100, "max_n_gaussians": cap_v})
        conf.strategy.perturb = OmegaConf.create(
            {"start_iteration": switch_step, "end_iteration": 27500,
             "frequency": 1, "noise_lr": 500000.0})
        conf.strategy.min_obs_ratio = float(min_obs_ratio)
        # GS 전용 스케줄을 확실히 끈다. 교체 후에는 호출되지 않지만, 설정이
        # 남아 있으면 나중에 읽는 쪽이 생겼을 때 조용히 되살아난다.
        for k in ("densify", "prune", "reset_density"):
            if k in conf.strategy:
                conf.strategy[k].end_iteration = switch_step - 1

        if keep_reg:
            conf.loss.use_opacity = True
            conf.loss.lambda_opacity = 0.01
            conf.loss.use_scale = True
            conf.loss.lambda_scale = 0.01

        st = MCMCStrategy(conf, trainer.model)
        st._audit_every = getattr(trainer, "_observe_every", 0)
        trainer.strategy = st

        lr = 0.0
        for g in trainer.model.optimizer.param_groups:
            if g["name"] == "positions":
                lr = g["lr"]
        # positions 의 lr 은 scene_extent 가 곱해져 있어 (model.py 의 setup_optimizer)
        # 설정값과 직접 비교할 수 없다. 지수 스케줄의 감쇠 비율만 계산한다.
        try:
            _sc = trainer.conf.scheduler.positions
            _r = (float(_sc.lr_final) / float(_sc.lr_init)) ** (
                switch_step / float(_sc.max_steps))
        except Exception:
            _r = float("nan")
        # 이 시점은 optimizer.step() 직전이다. relocate 는 그 다음에 돌기 때문에
        # 여기서 센 수와 첫 relocate 가 옮기는 수는 같지 않다. 문턱 근처의 분포도
        # 같이 찍어 얼마나 몰려 있는지 본다.
        dens = trainer.model.get_density().reshape(-1)
        n_dead = int((dens <= 0.005).sum())
        n_near = int(((dens > 0.005) & (dens <= 0.0055)).sum())
        print(f"[radc->mcmc] step {switch_step}: G={n:,} cap={cap_v:,} "
              f"end={end_v} | optimizer.step 직전 죽은 개체 {n_dead:,} "
              f"({100.0 * n_dead / max(n, 1):.3f}%), 문턱 바로 위(0.005~0.0055) "
              f"{n_near:,} ({100.0 * n_near / max(n, 1):.3f}%) "
              f"| positions lr={lr:.3e} (스케줄 감쇠 {_r:.3f}배) | 정규화 "
              f"{'on' if keep_reg else 'off'}", flush=True)

    def post_backward(step, *a, **k):
        if not state["done"] and step >= switch_step:
            state["done"] = True
            swap()
            return False        # 교체한 step 에서는 GS 의 densify 를 건너뛴다
        return old_post_backward(step, *a, **k)

    if prune_stop > 0 and "prune" in conf.strategy:
        # RADC 의 prune 과 MCMC 의 relocate 는 문턱이 0.005 로 같다. 판정은 같고
        # 처분만 다르다 -- RADC 는 지우고 MCMC 는 살아 있는 곳으로 되돌린다.
        # 교체 전 prune_stop step 동안 prune 을 멈추면 그 사이에 문턱 아래로
        # 내려간 개체가 지워지지 않고 남아, 교체 직후 MCMC 의 dead 집합이 된다.
        # split/clone 은 그대로 두므로 개수 증가 경로는 건드리지 않는다.
        conf.strategy.prune.end_iteration = switch_step - prune_stop
        print(f"[radc->mcmc] prune 을 {switch_step - prune_stop} 에서 멈춘다 "
              f"(교체 {prune_stop} step 전). densify 는 {switch_step} 까지 유지",
              flush=True)

    trainer.strategy.post_backward = post_backward
    print(f"[radc->mcmc] {switch_step} step 에서 MCMC 로 교체 예약", flush=True)


def install_mcmc_audit(trainer):
    """Report model state after every MCMC strategy step, to locate a crash.

    MCMC + sparse rays fails in `trace_bwd` with an illegal memory access around
    step 4,000 in five of six runs. The relocation kernel is not the cause --
    probed directly over ratios 1-51 and opacities 0.01-0.99 it produces no
    non-finite or negative scale. So the next question is what the model looks
    like when the tracer is handed it, which nothing currently records.
    """
    strategy = trainer.strategy
    model = trainer.model
    original = strategy.post_optimizer_step

    def audit(step, *args, **kwargs):
        result = original(step, *args, **kwargs)
        if step % 500 == 0:
            # Deliberately light: no torch.unique, which sorts 1.2M rows and adds
            # a synchronisation point of its own -- that would confound any
            # measurement of a race.
            with torch.no_grad():
                pos = model.positions
                scale = model.get_scale()
                nonfinite = int((~torch.isfinite(pos)).sum()) + int((~torch.isfinite(scale)).sum())
                print(f"[audit] step {step:6d}  N {pos.shape[0]:>9,}  "
                      f"scale_max {scale.max():.3e}  "
                      f"|pos|max {pos.abs().max():.3e}  "
                      f"비유한 {nonfinite}", flush=True)
                # sparse 가 opacity 분포를 누르는지 잰다. noise 가중치
                # op_sigmoid(1-o) 는 k=100 이라 o 가 0.05 에서 0.005 로 흔들리면
                # 가중치가 45 배로 뛴다. 기댓값이 같아도 분산이 커지면 이 볼록한
                # 함수를 지나며 평균 noise 가 올라간다. 노름에서 일어나는 Jensen
                # 부풀림과 같은 구조이고, 축만 opacity 다.
                o = model.get_density().reshape(-1)
                w = 1.0 / (1.0 + torch.exp(-100.0 * ((1.0 - o) - 0.995)))
                q = [float(o.quantile(x)) for x in (0.1, 0.25, 0.5, 0.75, 0.9)]
                c = trainer.model.positions.detach()
                ctr = c.median(0).values
                r = (c - ctr).norm(dim=1)
                rmed = max(float(r.median()), 1e-9)
                print(f"[opacity] step {step:6d}  "
                      f"o p10={q[0]:.4f} p25={q[1]:.4f} p50={q[2]:.4f} "
                      f"p75={q[3]:.4f} p90={q[4]:.4f} | "
                      f"o<0.02 {float((o < 0.02).float().mean()) * 100:.1f}% | "
                      f"noise가중 평균={float(w.mean()):.4f} "
                      f">0.1인 비율={float((w > 0.1).float().mean()) * 100:.1f}% | "
                      f"r/rmed>10 {int((r > 10 * rmed).sum()):,} "
                      f">100 {int((r > 100 * rmed).sum()):,}", flush=True)
        return result

    strategy.post_optimizer_step = audit
    print("[audit] MCMC 상태 감사 활성화", flush=True)


def install_fast_perturb(trainer):
    """Rewrite MCMC's position noise so it stops building covariance matrices.

    `MCMCStrategy.perturb_gaussians` runs every step and needs one quantity, the
    covariance acting on a noise vector. `model.get_covariance()` produces it the
    expensive way: allocate an (N,3,3) zero tensor for the diagonal scale matrix,
    convert quaternions to (N,3,3) rotations, then three batched 3x3 matmuls for
    `R S S^T R^T`, and finally a fourth to hit the vector.

    The same value factors, since `S` is diagonal:

        R S S^T R^T v  =  R (s^2 * (R^T v))

    which is two rotations and one elementwise multiply, with no (N,3,3)
    intermediates. Identical arithmetic, so nothing about the algorithm changes.

    It matters here and not upstream because the cost is per-Gaussian, not
    per-ray. Measured at 34 ms per step on 1.27M Gaussians, that is 5% of a
    737 ms dense step but 27% of a 126 ms step at 1/16 sampling -- the same code
    carries five times the relative weight once rays are the thing being cut.
    """
    from threedgrut.utils.misc import quaternion_to_so3

    strategy = trainer.strategy
    model = trainer.model
    if not hasattr(strategy, "perturb_gaussians"):
        print("[fast-perturb] strategy has no perturb_gaussians; skipped", flush=True)
        return

    @torch.no_grad()
    def perturb_gaussians():
        positions = model.get_positions()
        densities = model.get_density()
        scales = model.get_scale()
        rotations = quaternion_to_so3(model.get_rotation())

        current_lr = 0.0
        for group in model.optimizer.param_groups:
            if group["name"] == "positions":
                current_lr = group["lr"]

        def op_sigmoid(x, k: int = 100, x0: float = 0.995):
            return 1 / (1 + torch.exp(-k * (x - x0)))

        noise = (torch.randn_like(positions) * op_sigmoid(1 - densities)
                 * strategy.conf.strategy.perturb.noise_lr * current_lr)
        # R^T v, then scale^2 elementwise, then R w
        noise = torch.einsum("nji,nj->ni", rotations, noise)
        noise = noise * scales.pow(2)
        noise = torch.einsum("nij,nj->ni", rotations, noise)
        model.positions.add_(noise)

    strategy.perturb_gaussians = perturb_gaussians
    print("[fast-perturb] MCMC position noise rewritten without covariance "
          "matrices", flush=True)


def install_gap_monitor(trainer, args, step_holder):
    """덴서피케이션 주기마다 학습-검증 격차를 기록한다.

    개수를 늘리면 학습 view 적합도는 언제나 오르지만 test 는 씬마다 갈린다.
    3 씬 측정에서 두 값의 격차가 벌어지는 순서와 test 이득이 줄어드는 순서가
    일치했다 (room -0.052 / counter +0.196 / kitchen +0.993 대 +0.425 / +0.153 /
    -0.497).  즉 격차가 '지금 개수를 더 늘려도 되는가' 를 알려준다.

    3DGS 의 종료 시점 15000 은 dense 성장률에 맞춘 상수다.  sparse 에서는
    인플레이션 때문에 같은 스텝에 1.7 배 빨리 자라므로 그 상수가 무효가 된다.
    스텝이 아니라 데이터로 멈춰야 한다는 것이 이 장치의 근거다.

    이 판은 기록만 한다.  실제 정지 규칙은 세 씬의 궤적을 보고 정한다.
    """
    ds = trainer.train_dataset
    base = getattr(ds, "dataset", ds)
    n = len(base)
    step = max(2, n // max(1, args.gap_val_views))
    val_idx = list(range(0, n, step))[: args.gap_val_views]
    val_set = set(val_idx)
    train_idx = [i for i in range(n) if i not in val_set]
    ds.holdout = val_set          # SparseBatchDataset 이 학습에서 제외한다
    rng = np.random.default_rng(1234)
    probe_train = list(rng.choice(train_idx, size=min(len(val_idx), len(train_idx)),
                                  replace=False))
    hist = []

    @torch.no_grad()
    def psnr_over(idxs):
        tot = 0.0
        for i in idxs:
            gb = base.get_gpu_batch_with_intrinsics(
                torch.utils.data.default_collate([base[int(i)]]))
            b = ds.draw(gb, advance=False) if args.gap_sparse else gb
            out = trainer.model(b, train=False)
            mse = ((out["pred_rgb"] - b.rgb_gt) ** 2).mean()
            tot += float(-10.0 * torch.log10(mse.clamp_min(1e-12)))
            del out, b, gb
        return tot / max(len(idxs), 1)

    original = trainer.strategy.densify_gaussians

    def densify_with_gap(scene_extent):
        # 덴서피케이션 뒤에 재면 가속 구조가 아직 갱신되기 전이라 렌더가 깨진다.
        # 직전 주기의 결과를 재는 것이 의미상으로도 맞으므로 앞에서 잰다.
        st = trainer.global_step
        tr, va = psnr_over(probe_train), psnr_over(val_idx)
        g = trainer.model.num_gaussians
        out = original(scene_extent)
        hist.append({"step": st, "gaussians": int(g), "train": tr, "val": va,
                     "gap": tr - va})
        print(f"[gap] step {st:>6} G={int(g):>9,} train={tr:.3f} "
              f"val={va:.3f} gap={tr - va:+.3f}", flush=True)
        return out

    trainer.strategy.densify_gaussians = densify_with_gap
    trainer._gap_hist = hist
    print(f"[gap] 검증 {len(val_idx)}장 보류, 학습 {len(train_idx)}장, "
          f"프로브 {len(probe_train)}장, "
          f"{'sparse' if args.gap_sparse else 'dense'} Ray 로 측정", flush=True)
    return hist


def install_lr_scale(trainer, scale, beta_scale):
    """Grendel 의 배치 학습률 규칙을 적용한다.

    "On Scaling Up 3D Gaussian Splatting Training" (arXiv 2406.18533) 은
    배치 학습에서 lr' = lr * sqrt(batch) 와 beta' = beta^sqrt(batch) 를 쓴다.
    근거는 배치 갱신이 개별 갱신의 합과 같아지도록 Adam 의 2 차 모멘트 정규화를
    되돌리는 것이다.

    다만 그쪽 배치는 이미지를 B 장 '더' 보는 것이고 우리 다중뷰는 같은 Ray 를
    K 개로 '나누는' 것이라 배수가 그대로 맞을 이유는 없다.  그래서 배수를 인자로
    받아 훑는다.

    scheduler_step 이 매 스텝 param_group['lr'] 을 덮어쓰므로 그 뒤에 곱해야 한다.
    """
    model = trainer.model
    original = model.scheduler_step
    # density 는 schedulers 에 있지만 타입이 skip 이라 값을 돌려주지 않는다.
    # 등록 여부로 판단하면 곱이 누적되어 0.05 * scale^n 으로 폭주한다.
    # 스케줄러 값을 직접 물어보고, None 이면 기준값을 쓴다 (멱등).
    base = {g["name"]: g["lr"] for g in model.optimizer.param_groups}

    def scheduler_step(step):
        original(step)
        for g in model.optimizer.param_groups:
            name = g["name"]
            sch = model.schedulers.get(name)
            v = sch(step) if sch is not None else None
            g["lr"] = (v if v is not None else base[name]) * scale

    model.scheduler_step = scheduler_step
    if beta_scale and beta_scale != 1.0:
        for g in model.optimizer.param_groups:
            b1, b2 = g.get("betas", (0.9, 0.999))
            g["betas"] = (b1 ** beta_scale, b2 ** beta_scale)
        print(f"[lr-scale] beta 를 ^{beta_scale:g} 로 조정 "
              f"(0.9 -> {0.9 ** beta_scale:.4f}, 0.999 -> {0.999 ** beta_scale:.5f})",
              flush=True)
    print(f"[lr-scale] 모든 파라미터 학습률에 {scale:g} 배", flush=True)


def install_selective_adam(trainer):
    """Let SelectiveAdam run by supplying the visibility mask it expects.

    The optimiser is the largest fixed cost in a sparse step -- 35.4 ms of the
    84.1 ms floor at 3.06M Gaussians -- and it updates every primitive whether or
    not that primitive was touched. Under 1/64 supervision only 17-19% receive a
    gradient, so most of that work moves nothing but Adam's own decay terms.

    `SelectiveAdam` already exists here and `model.py` will construct it, but the
    trainer reads the mask from `outputs["mog_visibility"]` and nothing in the
    repository ever writes that key, so the path is dead. The mask cannot be
    produced during forward either, since it depends on gradients. Returning a
    dict that derives it on access works because the trainer reads it after
    `loss.backward()`.
    """
    model = trainer.model
    original = model.forward

    class LazyVisibility(dict):
        def __missing__(self, key):
            if key != "mog_visibility":
                raise KeyError(key)
            grad = model.positions.grad
            if grad is None:
                return torch.ones_like(model.density, dtype=torch.bool)
            return (grad != 0).any(dim=1, keepdim=True)

    def forward(gpu_batch, *call_args, **call_kwargs):
        return LazyVisibility(original(gpu_batch, *call_args, **call_kwargs))

    model.forward = forward
    print("[selective-adam] 그래디언트를 받은 Gaussian만 갱신", flush=True)


def install_scale_decay(trainer, conf, gamma, every):
    """Shrink Gaussians multiplicatively so rays intersect fewer of them.

    Section 55.10 found the hit threshold's 0.593 dB is not capacity, not an
    energy bias, and not the high band -- it is the *low* band, because deleting
    weakly-responding Gaussians at render time cuts hits/ray from 28.3 to 14.1
    and alpha compositing cannot build a smooth gradient out of six terms. GRay
    reaches the same hits/ray by keeping primitives small instead, so nothing it
    skips was going to contribute and no compositing term is truncated.

    An L1 penalty on scale was tried first (`loss.use_scale`, lambda 0.3) and
    collapsed the scene: median scale reached 0.00000 and densification answered
    with 2.03M Gaussians. L1's gradient does not vanish as the scale does, so it
    pushes to zero regardless of what reconstruction wants. A multiplicative decay
    shrinks proportionally, so it weakens exactly as the primitive gets small and
    settles wherever the data pushes back -- the same form `decay_density` already
    uses, applied in log space because scale is stored logarithmically.
    """
    strategy = trainer.strategy
    model = trainer.model
    shift = math.log(gamma)
    original = strategy.post_optimizer_step
    applications = max(1, int(conf.n_iterations) // every)

    def post_optimizer_step(step, scene_extent, train_dataset, batch=None, writer=None):
        result = original(step, scene_extent, train_dataset, batch, writer)
        if step % every == 0:
            with torch.no_grad():
                model.scale.data += shift
        return result

    strategy.post_optimizer_step = post_optimizer_step
    print(f"[scale-decay] x{gamma} every {every} steps "
          f"(nominal cumulative {gamma ** applications:.3f} over training)", flush=True)


def install_count_controller(trainer, conf, dense_run, gain, scale=1.0):
    """Pick the densification threshold from the Gaussian-count deficit.

    Every constant tried so far missed, in both directions: a rescale measured on
    the converged scene (1.63x, 2.62x) overshot to 1.7-2.0M or blew the cap, and a
    fixed 2.80 percent percentile undershot to 0.22M. They share a cause -- the
    statistic's distribution is not stationary over training, so any constant
    calibrated at one point is wrong everywhere else.

    Counting sidesteps the calibration entirely. The dense run's Gaussian count
    per step is a *measurement*, not a tuned value, and at each densification the
    number of Gaussians still needed is known. Selecting exactly that many by
    taking the corresponding order statistic makes the threshold a consequence
    rather than an input, and the loop closes each event, so pruning and the
    scene's own growth are absorbed without being modelled.

    The scale of the statistic then stops mattering, which is what the split-half
    correction could not deliver on its own -- it restored the ordering but left
    the scale tracking resolution as strongly as before.
    """
    from tensorboard.backend.event_processing import event_accumulator
    import glob as _glob

    events = sorted(_glob.glob(str(Path(dense_run).resolve() / "events*")))
    reader = event_accumulator.EventAccumulator(events[-1], size_guidance={"scalars": 0})
    reader.Reload()
    tag = "train/num_GS" if "train/num_GS" in reader.Tags()["scalars"] else "num_particles/train"
    trajectory = [(x.step, float(x.value)) for x in reader.Scalars(tag)]
    steps = torch.tensor([x[0] for x in trajectory], dtype=torch.float64)
    counts = torch.tensor([x[1] for x in trajectory], dtype=torch.float64)

    frequency = int(conf.strategy.densify.frequency)
    strategy = trainer.strategy
    original = strategy.densify_gaussians
    log = []

    def target_at(step):
        index = int(torch.searchsorted(steps, torch.tensor(float(step))).clamp(0, len(steps) - 1))
        return float(counts[index]) * scale

    def densify_gaussians(scene_extent):
        step = trainer.global_step
        current = int(trainer.model.num_gaussians)
        # the count this densification should reach by the next event
        target = target_at(step + frequency)
        deficit = target - current
        with torch.no_grad():
            statistic = (strategy.densify_grad_norm_accum
                         / strategy.densify_grad_norm_denom.clamp_min(1))
            statistic[statistic.isnan()] = 0.0
            positive = statistic[statistic > 0].flatten()
            if positive.numel() == 0:
                return original(scene_extent)
            k = int(max(1, min(positive.numel(), round(gain * deficit))))
            value = float(torch.topk(positive, k).values[-1])
            strategy.clone_grad_threshold = value
            strategy.split_grad_threshold = value
        log.append((step, current, int(target), k, value))
        if len(log) % 10 == 1:
            print(f"[count] step {step}: {current:,} -> 목표 {int(target):,} "
                  f"(부족 {int(deficit):+,}), 선택 {k:,}, 임계값 {value:.3e}", flush=True)
        return original(scene_extent)

    strategy.densify_gaussians = densify_gaussians
    trainer._count_log = log
    print(f"[count] dense 궤적 추종 x{scale}: {dense_run} ({len(trajectory):,}점, "
          f"최종 목표 {int(counts[-1] * scale):,})", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="apps/colmap_3dgrt.yaml")
    parser.add_argument("--path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--downsample-factor", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--multiview-k", type=int, default=1,
                        help="한 스텝에 담을 view 수 (예산 고정, 블록을 sqrt(k) 배로 키움)")
    parser.add_argument("--multiview-from", type=int, default=-1,
                        help="이 스텝부터 다중 view 를 켠다 (-1 이면 끄기)")
    parser.add_argument("--sparse-until", type=int, default=-1,
                        help="이 스텝부터 다시 dense Ray 로 학습한다 (-1 이면 끝까지 sparse)")
    parser.add_argument("--sparse-from", type=int, default=3000,
                        help="global step at which supervision switches to 1/64")
    parser.add_argument("--debias", action="store_true",
                        help="replace the densification statistic with the "
                             "split-half debiased one")
    parser.add_argument("--online-k", action="store_true",
                        help="k(h) 곡선을 학습 중에 스스로 재서 갱신한다. 사전 측정 불필요")
    parser.add_argument("--online-k-every", type=int, default=97,
                        help="split-half 표본을 몇 step 마다 모을지. 감쇠 누적이라 100 이어도 "
                             "최근 여러 주기 표본이 반영된다 (비용 약 2%%)")
    parser.add_argument("--quiet", action="store_true",
                        help="학습 시간 측정용. 진행 표시줄과 통계 출력을 끈다. logger.log_progress 가 "
                             "매 step loss 텐서를 문자열로 만들며 GPU 동기화를 강제하므로 "
                             "(threedgrut/utils/logger.py 의 _concat_additional_progress_info), "
                             "켜 둔 채로 잰 시간은 sparse 의 가속을 과소평가한다")
    parser.add_argument("--phi-div", action="store_true",
                        help="유도한 sqrt(1+k) 대신 실측 부풀림 phi(h) 로 나눈다. "
                             "phi 는 counter/room/kitchen/bonsai 의 dense 체크포인트에서 "
                             "300 view 누적으로 쟀고 씬 간 편차가 10~12%% 다. "
                             "--shrink-mean 과 같은 자리에 적용되며 함께 주면 이쪽이 우선한다")
    parser.add_argument("--shrink-mean", action="store_true",
                        help="부풀림을 1/sqrt(1+k) 로 되돌리는 대신 신뢰도로 평균에 축소한다. "
                             "실측 phi(h) 가 h<2 에서 2.4~3.9, h>=2 에서 1.25~2.26 으로 "
                             "사실상 두 집단이고 h 의존이 약해, 부풀림 유도가 -70%%~+36%% "
                             "어긋난다. 관측이 없는 것을 신뢰하지 않는 쪽으로 근거를 옮긴다")
    parser.add_argument("--grad-shrink", action="store_true",
                        help="같은 신뢰도 rho=1/(1+k(h)) 로 position gradient 도 축소한다. "
                             "densification 통계량에는 sqrt(rho), gradient 에는 rho 를 쓴다")
    parser.add_argument("--step-shrink", action="store_true",
                        help="Adam 갱신량에 rho=1/(1+k(h)) 를 곱한다. gradient 에 곱하는 "
                             "--grad-shrink 는 Adam 의 크기 불변성 때문에 효과가 없다")
    parser.add_argument("--k-curve-json", type=str, default="",
                        help="probe 가 저장한 k(h) 곡선 json. 안 주면 counter 기본 곡선을 쓴다")
    parser.add_argument("--shrink-alpha", type=float, default=1.0,
                        help="축소 인자 1/sqrt(1+alpha*k(h)) 의 alpha. 이론값 0.5")
    parser.add_argument("--pooled-k", action="store_true",
                        help="전역 크기용 k 를 개체별 비의 중앙값 대신 합-먼저"
                             "(sum(var)/sum(mu2))로 구한다. mu2>0 선별이 없어져"
                             " 표본이 적을 때의 하향 편향이 사라진다")
    parser.add_argument("--auto-scale", action="store_true",
                        help="alpha 대신 실측 전역 부풀림으로 크기를 정한다. 개체별 "
                             "항은 인구 중앙 h 에서 1 이 되게 정규화해 순위만 담당하고, "
                             "크기는 1.1552*(1+k_med/h_med)^0.7999 이 정한다. "
                             "--online-k 가 함께 필요하다")
    parser.add_argument("--step-min-hits", type=float, default=0.0,
                        help="step 단위 게이트: 그 step 에서 받은 Ray 가 이 값 미만인 "
                             "관측은 densification 통계량에 누적하지 않는다. 개체를 "
                             "통째로 배제하는 --radc-gate 와 달리 나쁜 관측만 버린다. "
                             "2 로 두면 phi(h) 와 sqrt(1+k(h)) 의 교차점(h*=1.95) 위의 "
                             "관측만 쓰는 것이 된다. 0 이면 끈다")
    parser.add_argument("--gate-waste", type=float, default=0.0,
                        help="theta 를 자식 Gaussian 의 낭비율로 제어한다. densify 가 만든 "
                             "자식 중 다음 주기에 한 번도 관측되지 않은 비율이 이 값을 넘으면 "
                             "theta 를 올린다. 로스와 달리 학습 진행과 섞이지 않고, 만들어진 "
                             "개체가 실제로 어떻게 됐는지를 직접 본다. 미교차 개체는 품질에 "
                             "기여하지 않으면서 G 만 늘리므로 (0901 리포트, 아홉 번 확인) "
                             "낭비율을 낮추는 것이 곧 G 최소화다")
    parser.add_argument("--eb-shrink", action="store_true",
                        help="densify 통계량에 정규-정규 경험적 베이즈 축소를 적용한다. "
                             "개체별 분산은 실측 k(h) 곡선에서 얻고 사전분산은 표본에서 "
                             "추정하므로 맞출 상수가 없다.")
    parser.add_argument("--tau-phi", action="store_true",
                        help="densify 임계값을 실측 전역 부풀림으로 보정한다. "
                             "tau=0.0002 는 dense 에서 맞춘 값인데 sparse 는 통계량을 "
                             "phi 배 부풀리므로 tau_eff = tau x phi 를 쓴다. phi 는 "
                             "--online-k 가 매 주기 갱신하는 k_med/h_med 로 계산되며 "
                             "고를 값이 없다. 전역 배율이라 순위는 바뀌지 않는다")
    parser.add_argument("--gate-auto", type=float, default=0.0,
                        help="theta 자동 선택. 첫 densify 주기에서 후보의 h_bar "
                             "중앙값을 step 1500 에서 재고 theta <- C * p50 으로 정한 뒤 "
                             "때까지 고정한다. 8개 씬 실측에서 최선 theta 가 "
                             "theta/p50 = 0.109~0.162 에 모였고 중앙값이 0.1361 이다. "
                             "매 주기 다시 정하지 않는 것이 핵심이다. 차단 강도는 "
                             "h_bar 분포가 학습 중 내려가면서 저절로 올라간다.")
    parser.add_argument("--gate-online", dest="gate_z", type=float, default=0.0,
                        help="online theta. 목표 신뢰도 K*. 매 densify 주기에 "
                             "실측 k(h) 곡선에서 k(h)=K* 가 되는 h 를 theta 로 쓴다. "
                             "게이트 구조(h_bar <= theta 배제)는 그대로이고, 바뀌는 "
                             "것은 theta 를 손으로 정하느냐 곡선에서 읽느냐다. K* 는 "
                             "무차원 신뢰도 목표라 씬·block 과 무관하다.")
    parser.add_argument("--gate-quantile", type=float, default=0.0,
                        help="theta 를 절대값이 아니라 h_bar 의 분위수로 정한다. "
                             "h_bar 분포는 ray 수에 비례해 통째로 이동하므로 (block 3/4/6 "
                             "에서 p50 이 3.9/2.2/1.0), 고정 theta 는 블록마다 다르게 문다. "
                             "분위수로 두면 어느 블록에서도 같은 비율을 막는다. 예산 불필요")
    parser.add_argument("--gate-target-g", type=int, default=0,
                        help="densify 종료 시점의 Gaussian 목표 수. 주면 --radc-gate 가 "
                             "고정값이 아니라 이 목표를 맞추도록 매 주기 조정된다. "
                             "theta 를 자유 변수에서 빼야 블록 비교가 오염되지 않고, "
                             "MCMC 의 max_n_gaussians 와 같은 조건에서 비교할 수 있다")
    parser.add_argument("--radc-gate", type=float, default=0.0,
                        help="RADC-Gate: 관측된 step 당 평균 Ray 적중 수 h 가 이 값 "
                             "이하인 Gaussian 을 densification 후보에서 제외한다. "
                             "0 이면 끈다. 근거는 k(h) 곡선의 정의역이 h>=1.5 라서 "
                             "그 아래는 보정량을 모른 채 끝값이 적용된다는 것이다. "
                             "1.0 이면 step 당 Ray 를 하나만 받는 것들을 뺀다")
    parser.add_argument("--hit-shrink", action="store_true",
                        help="Ray 적중 횟수로 통계량을 축소한다. 표집을 나누지 않는다")
    parser.add_argument("--shrink-only", action="store_true",
                        help="split-half 의 교차 내적을 빼고 축소만 적용한다")
    parser.add_argument("--debias-double-budget", action="store_true",
                        help="split-half 의 두 벌을 각각 목표 표집률로 뽑는다 "
                             "(arm 1 의 두 배 Ray 를 쓰게 되므로 기본은 꺼둔다)")
    parser.add_argument("--debias-every", type=int, default=1,
                        help="apply the correction every N steps; the statistic "
                             "is accumulated over 300 steps so it need not be "
                             "every one")
    parser.add_argument("--densify-percentile", type=float, default=0.0,
                        help="if set, select this top fraction of the statistic "
                             "instead of comparing against an absolute threshold")
    parser.add_argument("--count-target", default=None,
                        help="run directory of a dense training whose Gaussian "
                             "count trajectory should be tracked; the threshold "
                             "then follows from the deficit instead of a constant")
    parser.add_argument("--count-gain", type=float, default=1.0)
    parser.add_argument("--count-scale", type=float, default=1.0,
                        help="multiply the tracked trajectory; smaller primitives "
                             "need more of them to keep a surface covered")
    parser.add_argument("--scale-decay", type=float, default=0.0,
                        help="multiply every Gaussian scale by this factor every "
                             "--scale-decay-every steps, shrinking primitives so "
                             "rays meet fewer of them")
    parser.add_argument("--scale-decay-every", type=int, default=50)
    parser.add_argument("--permuted-sampling", action="store_true",
                        help="cycle each block through its positions instead of "
                             "drawing independently, so no pixel is left unused")
    parser.add_argument("--radc-to-mcmc", type=int, default=-1,
                        help="이 step 부터 densification 전략을 RADC(휴리스틱+게이트)"
                             "에서 MCMC 로 교체한다. -1 이면 끄기.")
    parser.add_argument("--radc-mcmc-cap", type=int, default=-1,
                        help="교체 후 MCMC 의 max_n_gaussians. -1 이면 교체 시점의 "
                             "개수로 고정해 성장을 막는다.")
    parser.add_argument("--radc-mcmc-end", type=int, default=-1,
                        help="교체 후 relocate/add 의 end_iteration. -1 이면 25000.")
    parser.add_argument("--radc-mcmc-prune-stop", type=int, default=0,
                        help="교체 이 step 전부터 RADC 의 prune 을 멈춘다. 문턱 아래로 "
                             "내려간 개체가 지워지지 않고 남아 MCMC 의 dead 집합이 "
                             "된다. densify(split/clone) 는 교체 시점까지 유지된다.")
    parser.add_argument("--radc-mcmc-no-reg", action="store_true",
                        help="교체 시 opacity/scale 정규화를 켜지 않는다. 켜지 않으면 "
                             "죽는 개체가 생기지 않아 relocate 가 빈 채로 돈다.")
    parser.add_argument("--train-views", type=int, default=0,
                        help="학습 view 를 이 개수로 균등 간격 부분표집한다. 0 이면 전부 쓴다. "
                             "장면 내용을 고정한 채 관측 제약만 줄여 Gaussian 수가 무엇에 "
                             "의해 정해지는지 분리하기 위한 것이다.")
    parser.add_argument("--death-audit", type=int, default=0,
                        help="이 step 주기로 alpha < 1/255 인 개체 수를 기록한다. 0 이면 끈다.")
    parser.add_argument("--mcmc-observe", action="store_true",
                        help="MCMC 의 relocate 시점에 죽는 개체의 관측 횟수 분포를 "
                             "전체 분포와 함께 기록한다.")
    parser.add_argument("--mcmc-audit", action="store_true",
                        help="MCMC only: dump model state every 100 steps to "
                             "locate the tracer crash")
    parser.add_argument("--fast-perturb", action="store_true",
                        help="MCMC only: compute the position noise as "
                             "R (s^2 * (R^T v)) instead of building covariance "
                             "matrices. Same arithmetic, fewer allocations.")
    parser.add_argument("--lr-scale", type=float, default=1.0,
                        help="모든 학습률에 곱할 배수 (Grendel 규칙은 sqrt(batch))")
    parser.add_argument("--beta-scale", type=float, default=1.0,
                        help="Adam beta 를 이 지수로 올린다 (Grendel 규칙은 sqrt(batch))")
    parser.add_argument("--gap-monitor", action="store_true",
                        help="덴서피케이션 주기마다 학습-검증 격차를 기록한다")
    parser.add_argument("--gap-val-views", type=int, default=8,
                        help="학습에서 빼서 검증에 쓸 view 수")
    parser.add_argument("--gap-sparse", action="store_true", default=True,
                        help="격차를 sparse Ray 로 잰다 (학습과 같은 예산)")
    parser.add_argument("--selective-adam", action="store_true",
                        help="update only the Gaussians that received a gradient")
    parser.add_argument("--opacity-barrier-weight", type=float, default=0.0,
                        help="weight of the smooth opacity boundary barrier")
    parser.add_argument("--opacity-barrier-guard", type=float, default=0.008,
                        help="start penalising opacity below this guard value")
    parser.add_argument("--opacity-barrier-temperature", type=float, default=0.001,
                        help="softplus temperature of the opacity barrier")
    parser.add_argument("--opacity-barrier-start", type=int, default=15000,
                        help="first iteration at which the barrier is active")
    parser.add_argument("--opacity-barrier-mask", default="",
                        help="optional future-dead utility .pt; protect selected slots only")
    parser.add_argument("--opacity-barrier-min-z", type=float, default=2.0,
                        help="minimum z_increase when --opacity-barrier-mask is used")
    parser.add_argument("--no-cache", action="store_true")
    # --- 분기 실험 (유지 대 증가) 용 --------------------------------------
    parser.add_argument("--holdout-views", type=int, default=0,
                        help="학습에서 처음부터 제외할 view 수. 균등 간격으로 고른다. "
                             "일반 view 와 추가 multi-view 모두 허용 목록에서만 뽑는다")
    parser.add_argument("--deterministic-views", type=int, default=0,
                        help="0 이 아니면 그 값을 seed 로 view 순서와 Ray 표집을 "
                             "global step 의 함수로 고정한다. 두 분기가 같은 입력을 "
                             "같은 순서로 보게 한다")
    parser.add_argument("--save-full-state", type=str, default="",
                        help="CPU/CUDA 난수 상태와 교체 상태를 이 경로에 저장한다")
    parser.add_argument("--resume-full", type=str, default="",
                        help="--save-full-state 파일에서 난수·교체 상태를 복원한다. "
                             "모델·optimizer·global_step 은 체크포인트가 담당")
    parser.add_argument("--prune-to", type=int, default=0,
                        help="재개 직후 개체 수를 이 값으로 줄이고 cap 을 재고정한다. "
                             "순위 하위부터 제거. --prune-rank 로 기준 선택")
    parser.add_argument("--prune-rank", default="hitalpha", choices=["hitalpha", "alpha"],
                        help="삭제 순위 기준. hitalpha 는 (학습 view 적중수 x alpha) 로, "
                             "실제 영상 기여도가 아니라 그 대리 지표다")
    parser.add_argument("--prune-rank-views", type=int, default=40,
                        help="적중수를 누적할 학습 view 수 (holdout 은 제외됨)")
    parser.add_argument("--grow-adds", type=int, default=0,
                        help="재개 직후 optimizer step 없이 MCMC 의 add 를 N 회 연속 "
                             "적용하고 cap 을 그 결과 개수로 다시 고정한다. "
                             "1.05^N 배 (N=6 이면 약 +34%%)")
    parser.add_argument("--resume-config-from", type=str, default="",
                        help="이 체크포인트에 저장된 설정을 통째로 기준으로 복원한다. "
                             "교체 후(MCMC) 체크포인트를 재개할 때 필수. "
                             "--radc-to-mcmc 는 같이 주지 않는다 (교체 중복 방지)")
    parser.add_argument("--dump-state-at", type=str, default="",
                        help="STEP:경로. 그 step 의 첫 입력을 처리하기 직전에 "
                             "모든 학습 파라미터·optimizer·전략 버퍼·난수 상태를 저장한다 "
                             "(연속 실행과 재개 실행의 복원 일치 확인용)")
    parser.add_argument("--trace-inputs", type=str, default="",
                        help="매 step 의 view ID·Ray 인덱스 해시·학습률을 이 파일에 기록 "
                             "(관문 검사용)")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    if args.quiet:
        # 진행 표시줄은 매 step loss 텐서를 포맷하며 GPU 동기화를 강제한다.
        # 학습 시간을 재는 실행에서는 이걸 끄지 않으면 sparse 의 가속이
        # 과소평가된다. 표시만 없애고 학습 자체는 건드리지 않는다.
        from threedgrut.utils import logger as _grt_logger
        _grt_logger.logger.log_progress = lambda *a, **kw: None
        print("[quiet] 진행 표시줄과 통계 출력을 끈다 (시간 측정용)", flush=True)

    from hydra import compose, initialize_config_dir
    from threedgrut.trainer import Trainer3DGRUT

    config_dir = str(Path("configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        overrides = [
            f"path={args.path}",
            f"out_dir={args.out_dir}",
            f"experiment_name={args.experiment_name}",
            f"dataset.downsample_factor={args.downsample_factor}",
            # The 3DGRT path reads render.splat unconditionally through the
            # Medium V2 code even though its own config does not define it.
            "+render.splat.medium_v2=false",
        ] + ([
            "strategy.print_stats=false",
            "enable_writer=false",
        ] if args.quiet else []) + [
            "+render.splat.medium=false",
            "+render.splat.ever_medium=false",
        ] + (["optimizer.type=selective_adam"] if args.selective_adam else []) + list(args.overrides)
        conf = compose(config_name=args.config_name, overrides=overrides)

    # 교체(RADC -> MCMC) 후 체크포인트는 MCMCStrategy 가 저장한 것이라
    # GS 의 densify 버퍼 키가 없다. 교체 전 설정으로 재개하면 trainer 가 GS
    # 복원 경로를 타다 KeyError 로 죽는다. 그래서 체크포인트에 함께 저장된
    # 교체 후 설정을 통째로 기준으로 삼고, 실행 관리 항목만 덮어쓴다.
    if args.resume_config_from:
        _ck = torch.load(args.resume_config_from, map_location="cpu", weights_only=False)
        _saved = _ck["config"]
        OmegaConf.set_struct(_saved, False)
        _keep = {"path": conf.path, "out_dir": conf.out_dir,
                 "experiment_name": conf.experiment_name,
                 "n_iterations": conf.n_iterations,
                 "resume": args.resume_config_from}
        for _k, _v in _keep.items():
            OmegaConf.update(_saved, _k, _v, merge=False)
        for _k in ("checkpoint", "val_frequency", "test_last", "num_workers",
                   "enable_writer", "compute_extra_metrics"):
            if _k in conf:
                OmegaConf.update(_saved, _k, conf[_k], merge=False)
        conf = _saved
        del _ck
        print(f"[resume-cfg] 체크포인트의 교체 후 설정으로 복원: "
              f"strategy={conf.strategy.method} "
              f"cap={getattr(getattr(conf.strategy,'add',None),'max_n_gaussians',None)}",
              flush=True)

    trainer = Trainer3DGRUT(conf)

    # base trainer 의 재개 경로(conf.resume)는 다른 초기화 분기와 달리
    # build_acc() 를 호출하지 않는다. bvh_update_frequency=1 이라 step 끝에서
    # 다시 세워지므로 둘째 step 부터는 회복되지만, 재개 후 첫 forward 는
    # 가속 구조 없이 렌더되어 loss 가 크게 어긋난다 (실측 0.139 대 0.458).
    if getattr(conf, "resume", ""):
        trainer.model.build_acc(rebuild=True)
        print("[resume] 재개 직후 가속 구조 재구축", flush=True)

    if args.train_views > 0:
        _n = int(trainer.train_dataset.poses.shape[0])
        if args.train_views < _n:
            _sel = np.linspace(0, _n - 1, args.train_views).round().astype(int)
            _sel = np.unique(_sel)
            _ds = trainer.train_dataset
            _ds.poses = _ds.poses[_sel]
            _ds.image_paths = _ds.image_paths[_sel]
            _ds.n_frames = _ds.poses.shape[0]
            for _attr in ("images", "intrinsics", "camera_ids", "cam_centers", "K"):
                _v = getattr(_ds, _attr, None)
                if _v is not None and hasattr(_v, "__len__") and len(_v) == _n:
                    setattr(_ds, _attr, _v[_sel] if not isinstance(_v, list)
                            else [_v[i] for i in _sel])
            print(f"[views] 학습 view {_n} -> {_ds.n_frames} 로 균등 부분표집", flush=True)
        else:
            print(f"[views] 요청 {args.train_views} >= 전체 {_n}, 전부 사용", flush=True)

    step_holder = [0]
    train_dataset = trainer.train_dataset
    if not args.no_cache:
        train_dataset = CachedDataset(train_dataset)
    trainer.train_dataset = SparseBatchDataset(
        train_dataset, args.block_size, args.sparse_from, step_holder,
        permuted=args.permuted_sampling, sparse_until=args.sparse_until,
        mv_k=args.multiview_k, mv_from=args.multiview_from,
    )
    if args.permuted_sampling:
        print(f"[permuted] 블록마다 {args.block_size**2}개 위치를 순열로 소진", flush=True)
    # The loop takes raw batches from the dataloader but converts them through
    # `self.train_dataset`, so only the dataset needs wrapping. The dataloader
    # keeps pointing at the uncached original, which is fine: it only yields
    # indices and the collated dict, and the cache sits behind __getitem__.
    # --- 분기 실험: 허용 학습 view 목록 -----------------------------------
    _sbd = trainer.train_dataset
    _allowed = None
    if args.holdout_views > 0:
        _n_all = len(train_dataset)
        _hold = set(int(v) for v in np.linspace(0, _n_all - 1, args.holdout_views).round())
        _allowed = [i for i in range(_n_all) if i not in _hold]
        _sbd.allowed = _allowed
        _sbd.holdout = _hold
        print(f"[holdout] 학습 view {_n_all} 중 {len(_hold)}장 제외 -> 허용 {len(_allowed)}장. "
              f"제외 ID {sorted(_hold)}", flush=True)
    if args.deterministic_views:
        _sbd.det_seed = int(args.deterministic_views)
        print(f"[det] view 순서와 Ray 표집을 seed {args.deterministic_views} 로 "
              f"global step 의 함수로 고정", flush=True)

    if not args.no_cache:
        # DataLoader 는 iterator 를 만들 때 기본 CPU RNG 를 소비할 수 있다.
        # 연속 실행과 재개 실행은 생성 시점이 달라 그 소비가 어긋난다.
        _kw = dict(batch_size=1, num_workers=0,
                   collate_fn=torch.utils.data.default_collate,
                   generator=torch.Generator().manual_seed(
                       int(args.deterministic_views) or 1234))
        if args.deterministic_views:
            _kw["sampler"] = StepPermSampler(
                _allowed if _allowed is not None else range(len(train_dataset)),
                args.deterministic_views, lambda: trainer.global_step)
        elif _allowed is not None:
            _kw["sampler"] = torch.utils.data.SubsetRandomSampler(_allowed)
        else:
            _kw["shuffle"] = True
        trainer.train_dataloader = torch.utils.data.DataLoader(train_dataset, **_kw)

    # The wrapper needs the step the trainer is on; the trainer keeps it on
    # itself, so mirroring it into the holder each render is enough.
    # step_holder 는 반드시 Ray 표집 '전' 에 갱신해야 한다. forward 안에서
    # 갱신하면 get_gpu_batch_with_intrinsics -> draw 가 먼저 돌아 한 step
    # 뒤처진 값을 쓰고, 재개 첫 step 에서는 초기값 0 을 참조한다.
    _orig_getb = trainer.train_dataset.get_gpu_batch_with_intrinsics

    # "600:경로a,601:경로b" 형태로 여러 지점을 덤프한다
    _dump_map = {}
    for _it in (args.dump_state_at.split(",") if args.dump_state_at else []):
        _k, _v = _it.split(":", 1)
        _dump_map[int(_k)] = _v

    def _dump_full_state(path):
        st = trainer.strategy
        d = {"global_step": trainer.global_step,
             "num_gaussians": int(trainer.model.num_gaussians),
             "params": {n: p.detach().cpu().clone()
                        for n, p in trainer.model.named_parameters()},
             "opt": trainer.model.optimizer.state_dict(),
             "cpu_rng": torch.get_rng_state(),
             "cuda_rng": (torch.cuda.get_rng_state()
                          if torch.cuda.is_available() else None),
             "np_rng": np.random.get_state()}
        for b in ("densify_grad_norm_accum", "densify_grad_norm_denom",
                  "tau_accum", "hit_count_tau"):
            v = getattr(st, b, None)
            d[b] = v.detach().cpu().clone() if torch.is_tensor(v) else None
        # 비파라미터 모델 상태
        d["n_active_features"] = getattr(trainer.model, "n_active_features", None)
        # 교체 상태: 전략 종류·cap·정규화·relocate/perturb 스케줄
        _c = trainer.conf
        d["switch"] = {
            "strategy": type(trainer.strategy).__name__,
            "method": getattr(getattr(_c, "strategy", None), "method", None),
            "cap": getattr(getattr(getattr(_c, "strategy", None), "add", None),
                           "max_n_gaussians", None),
            "add_start": getattr(getattr(getattr(_c, "strategy", None), "add", None),
                                 "start_iteration", None),
            "reloc": {k: getattr(getattr(getattr(_c, "strategy", None), "relocate", None), k, None)
                      for k in ("start_iteration", "end_iteration", "frequency")},
            "perturb": {k: getattr(getattr(getattr(_c, "strategy", None), "perturb", None), k, None)
                        for k in ("start_iteration", "end_iteration", "noise_lr")},
            "loss": {k: getattr(getattr(_c, "loss", None), k, None)
                     for k in ("use_opacity", "lambda_opacity", "use_scale", "lambda_scale")},
            "densify_end": getattr(getattr(getattr(_c, "strategy", None), "densify", None),
                                   "end_iteration", None),
        }
        # RADC 의 online k(h) 상태
        ok = getattr(trainer, "_online_k", None)
        if ok is not None:
            d["online_k"] = {k: (v.detach().cpu().clone() if torch.is_tensor(v)
                                 else (tuple(x.detach().cpu().clone() for x in v)
                                       if isinstance(v, tuple) and v and torch.is_tensor(v[0])
                                       else v))
                             for k, v in vars(ok).items()
                             if not k.startswith("__") and not callable(v)}
        torch.save(d, path)
        print(f"[dump] step {trainer.global_step} 전체 상태 -> {path}", flush=True)

    def _getb(batch, *a, **kw):
        step_holder[0] = trainer.global_step
        if trainer.global_step in _dump_map:
            _dump_full_state(_dump_map.pop(trainer.global_step))
        return _orig_getb(batch, *a, **kw)
    trainer.train_dataset.get_gpu_batch_with_intrinsics = _getb

    original_forward = trainer.model.forward
    # [0] 직전 forward 의 Gaussian 별 Ray 적중 횟수
    # [1] densify 주기 동안의 적중 누적 (--shrink-mean 에서만 씀)
    hit_holder = [None, None]

    def forward(gpu_batch, *call_args, **call_kwargs):
        step_holder[0] = trainer.global_step
        out = original_forward(gpu_batch, *call_args, **call_kwargs)
        # mog_visibility 는 float32 버퍼에 int32 로 쓰인 Ray 적중 횟수다.
        # .view(torch.int32) 로 재해석해야 개수가 나온다 (referenceOptix.cu 주석 참고).
        vis = out.get("mog_visibility") if isinstance(out, dict) else None
        hit_holder[0] = (vis.view(torch.int32).reshape(-1).float()
                         if vis is not None and vis.dtype == torch.float32 else None)
        return out

    trainer.model.forward = forward

    if args.opacity_barrier_weight > 0:
        selected_mask = None
        if args.opacity_barrier_mask:
            utility = torch.load(args.opacity_barrier_mask, map_location="cpu",
                                 weights_only=False)
            chosen = utility["z_increase"] > args.opacity_barrier_min_z
            selected_slots = utility["slots"][chosen].cuda()
            selected_mask = torch.zeros(int(trainer.model.num_gaussians), dtype=torch.bool,
                                        device="cuda")
            selected_mask[selected_slots] = True
        install_opacity_boundary_barrier(
            trainer,
            args.opacity_barrier_weight,
            args.opacity_barrier_guard,
            args.opacity_barrier_temperature,
            args.opacity_barrier_start,
            selected_mask,
        )

    if args.selective_adam:
        install_selective_adam(trainer)
    if args.gap_monitor:
        install_gap_monitor(trainer, args, step_holder)
    if args.lr_scale != 1.0 or args.beta_scale != 1.0:
        install_lr_scale(trainer, args.lr_scale, args.beta_scale)
    if args.fast_perturb:
        install_fast_perturb(trainer)
    if args.death_audit > 0:
        install_death_audit(trainer, args.death_audit)
    if args.mcmc_observe:
        install_mcmc_observe(trainer)
    if args.radc_to_mcmc > 0:
        install_radc_to_mcmc(trainer, args.radc_to_mcmc,
                             cap=args.radc_mcmc_cap,
                             keep_reg=not args.radc_mcmc_no_reg,
                             end_iteration=args.radc_mcmc_end,
                             prune_stop=args.radc_mcmc_prune_stop)
    if args.mcmc_audit:
        install_mcmc_audit(trainer)
    if args.hit_shrink:
        knots = None
        if args.k_curve_json:
            import json as _json
            c = _json.loads(Path(args.k_curve_json).read_text())["k_curve"]
            knots = ([e["h"] for e in c], [e["k"] for e in c])
            print(f"[hit-shrink] k(h) 곡선을 {args.k_curve_json} 에서 읽음: "
                  + ", ".join(f"h={e['h']:.3g}→k={e['k']:.3g}" for e in c[:4]) + " ...", flush=True)
        online = OnlineKCurve(int(trainer.model.num_gaussians), "cuda") if args.online_k else None
        trainer._online_k = online   # 상태 덤프에서 참조
        if args.shrink_mean or args.phi_div:
            hit_holder[1] = torch.zeros_like(trainer.strategy.densify_grad_norm_accum)
        install_hit_shrinkage(trainer, conf, hit_holder, args.shrink_alpha, knots,
                              online, trainer.train_dataset, args.online_k_every,
                              grad_shrink=args.grad_shrink, step_shrink=args.step_shrink,
                              shrink_mean=args.shrink_mean or args.phi_div,
                              phi_div=args.phi_div, blk=args.block_size,
                              auto_scale=args.auto_scale, pooled_k=args.pooled_k,
                              gate_min_hits=args.radc_gate,
                              step_min_hits=args.step_min_hits,
                              target_g=args.gate_target_g,
                              quantile=args.gate_quantile,
                              tau_phi=args.tau_phi,
                              waste_target=args.gate_waste,
                              gate_z=args.gate_z,
                              gate_auto=args.gate_auto,
                              eb_shrink=args.eb_shrink)
        if args.auto_scale:
            print("[auto-scale] 전역 크기를 실측 부풀림 "
                  f"{_PHI_C:g}*(1+k_med/h_med)^{_PHI_A:g} 으로 정한다. "
                  "shrink-alpha 를 쓰지 않는다", flush=True)
        if args.phi_div:
            print("[phi-div] 실측 부풀림 phi(h) 로 나눈다 "
                  f"(block {args.block_size}, 누적 평균 h 기준)", flush=True)
        elif args.shrink_mean:
            print("[shrink-mean] 부풀림 보정 대신 평균으로 축소한다. "
                  "T = S_bar + w (S - S_bar), w = tau^2/(tau^2 + k S^2/(3 n))", flush=True)
        if args.step_shrink:
            print("[step-shrink] Adam 갱신량에 rho=1/(1+k(h)) 를 곱한다", flush=True)
        if args.grad_shrink:
            print("[grad-shrink] position gradient 에 rho=1/(1+k(h)) 를 곱한다", flush=True)
        if online is not None:
            print(f"[k(h)] 학습 중 자체 측정, {args.online_k_every} step 마다 표본 수집",
                  flush=True)
    if args.debias and not args.debias_double_budget:
        # A, B 각각 절반 표집률로 뽑아 합이 arm 1 의 Ray 수와 같게 한다.
        # --debias-double-budget 을 주면 예전처럼 두 벌 모두 목표 표집률로 뽑는다.
        trainer.train_dataset.half_budget = True
        print("[debias] A, B 를 각각 절반 표집률로 뽑아 arm 1 과 Ray 예산을 맞춘다",
              flush=True)
    if args.debias:
        install_debiased_densification(trainer, conf, args)
    if args.densify_percentile > 0:
        install_percentile_threshold(trainer, args.densify_percentile)
    if args.count_target:
        install_count_controller(trainer, conf, args.count_target, args.count_gain,
                                 args.count_scale)
    if args.scale_decay > 0:
        install_scale_decay(trainer, conf, args.scale_decay, args.scale_decay_every)

    print(f"sparse from step {args.sparse_from}, block {args.block_size} "
          f"(1/{args.block_size ** 2} of rays)", flush=True)
    # --- 분기 실험: 난수·교체 상태 복원 -----------------------------------
    # 모델·optimizer·global_step 은 체크포인트가 담당하므로, 여기서는 데이터
    # 로더와 전략이 모두 세워진 뒤 "다음 학습 입력을 만들기 직전" 에만 복원한다.
    if args.resume_full:
        _st = torch.load(args.resume_full, map_location="cpu", weights_only=False)
        torch.set_rng_state(_st["cpu_rng"])
        if _st.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(_st["cuda_rng"])
        np.random.set_state(_st["np_rng"])
        print(f"[resume] step {_st['global_step']} 의 난수 상태 복원 "
              f"(현재 trainer.global_step={trainer.global_step})", flush=True)
        if _st["global_step"] != trainer.global_step:
            raise SystemExit(
                f"[resume] 중단: 상태 파일 step {_st['global_step']} 과 "
                f"체크포인트 step {trainer.global_step} 이 다르다")

    # --- 분기 실험: 재개 직후 축소 -------------------------------------
    # 순위는 학습 view 에서만 계산한다 (holdout 미사용). 적중수 x alpha 는
    # 실제 기여도가 아니라 대리 지표다.
    if args.prune_to > 0:
        _n0 = int(trainer.model.num_gaussians)
        if args.prune_rank == "alpha":
            _score = trainer.model.get_density().detach().reshape(-1).float()
            _how = "alpha 단독"
        else:
            _pool = (_sbd.allowed if _sbd.allowed is not None
                     else list(range(len(train_dataset))))
            _rng2 = np.random.default_rng([int(args.deterministic_views) or 1234, 9])
            _vs = _rng2.choice(_pool, size=min(args.prune_rank_views, len(_pool)),
                               replace=False)
            # _orig_getb 는 SparseBatchDataset 의 재정의판이라 이미 1/16 로
            # 표집된 배치를 준다. 여기에 draw 를 또 걸면 1/256 이 되어
            # 적중수가 지나치게 희소해진다. 그대로 렌더한다.
            _sbd.step_holder[0] = int(trainer.global_step)
            _hits = torch.zeros(_n0, device="cuda")
            with torch.no_grad():
                for _v in _vs:
                    _b = _orig_getb(torch.utils.data.default_collate(
                        [train_dataset[int(_v)]]))
                    _o = trainer.model(_b, train=True)
                    _vis = _o.get("mog_visibility")
                    if _vis is None or _vis.dtype != torch.float32:
                        raise SystemExit("[prune] mog_visibility 를 얻지 못했다. "
                                         "적중수 기반 순위를 만들 수 없다")
                    _hits += _vis.view(torch.int32).reshape(-1).float().clamp_min(0)
                    del _o, _b
            torch.cuda.empty_cache()
            _pos = float((_hits > 0).float().mean())
            print(f"[prune] 적중수 계측: view {len(_vs)}장, "
                  f"적중>0 개체 비율 {_pos:.4f}", flush=True)
            if _pos < 0.05:
                raise SystemExit(f"[prune] 적중>0 비율이 {_pos:.4f} 로 너무 낮다. "
                                 "계측이 실패했을 가능성이 크다")
            _score = _hits * trainer.model.get_density().detach().reshape(-1).float()
            _how = f"적중수 x alpha (학습 view {len(_vs)}장)"
        _order = torch.argsort(_score, descending=True)
        _keep = torch.zeros(_n0, dtype=torch.bool, device=_score.device)
        _keep[_order[:args.prune_to]] = True
        _thr = float(_score[_order[args.prune_to - 1]])
        _ties = int((_score == _thr).sum())
        print(f"[prune] 순위 경계값 {_thr:.6e}, 동점 {_ties:,}개", flush=True)
        trainer.strategy._update_param_with_optimizer(
            lambda name, param: torch.nn.Parameter(param[_keep],
                                                   requires_grad=param.requires_grad),
            lambda key, v: v[_keep])
        if hasattr(trainer.strategy, "prune_densification_buffers"):
            try: trainer.strategy.prune_densification_buffers(_keep)
            except Exception: pass
        _n1 = int(trainer.model.num_gaussians)
        conf.strategy.add.max_n_gaussians = _n1
        trainer.model.build_acc(rebuild=True)
        print(f"[prune] {_how}: {_n0:,} -> {_n1:,} ({_n1/_n0:.3f}배). "
              f"BVH 재구축. cap {_n1:,} 재고정", flush=True)

    # --- 분기 실험: 재개 직후 증식 -------------------------------------
    # 통상적인 100 step 간격 성장과 다른 일회성 개입이다. 측정 대상은
    # "순수 개수 효과" 가 아니라 "이 증식 연산을 추가한 효과" 다.
    if args.grow_adds > 0:
        _st = trainer.strategy
        _before = int(trainer.model.num_gaussians)
        _oldcap = conf.strategy.add.max_n_gaussians
        conf.strategy.add.max_n_gaussians = int(_before * (1.05 ** args.grow_adds)) + 16
        for _ in range(args.grow_adds):
            _st.add_new_gaussians()
        _after = int(trainer.model.num_gaussians)
        conf.strategy.add.max_n_gaussians = _after
        # add_new_gaussians() 는 파라미터·optimizer 만 갱신하고 가속 구조는
        # 건드리지 않는다. 다시 세우지 않으면 증식 후 첫 forward 가 증식 전
        # BVH 로 렌더된다 (신규 개체 누락 + 부모 scale 변경 미반영).
        trainer.model.build_acc(rebuild=True)
        print(f"[grow] add {args.grow_adds}회: {_before:,} -> {_after:,} "
              f"({_after/_before:.3f}배). BVH 재구축. cap {_after:,} 재고정 "
              f"(이전 cap {_oldcap:,})", flush=True)

    # --- 분기 실험: 입력 추적 --------------------------------------------
    if args.trace_inputs:
        _trace = open(args.trace_inputs, "w")
        # 직전 step 의 광도 loss.  forward/backward 에서 수치가 갈리는지 본다.
        _loss_holder = [float("nan")]
        _orig_getloss = trainer.get_losses

        def _getloss(*a, **kw):
            r = _orig_getloss(*a, **kw)
            try:
                v = r[0] if isinstance(r, (tuple, list)) else r
                _loss_holder[0] = float(v["total_loss"] if isinstance(v, dict) else v)
            except Exception:
                pass
            return r
        trainer.get_losses = _getloss
        # In full-ray mode SparseBatchDataset.draw() is intentionally bypassed,
        # so tracing only draw() leaves an empty file. Trace the already wrapped
        # batch fetch instead; this records the view sequence without changing
        # either the rendered rays or the RNG state.
        if args.block_size == 1:
            _orig_getb_trace = trainer.train_dataset.get_gpu_batch_with_intrinsics

            def _getb_traced(batch, *a, **kw):
                out = _orig_getb_trace(batch, *a, **kw)
                vk = _pose_key(out.T_to_world)
                lr = trainer.model.optimizer.param_groups[0]["lr"]
                _trace.write(f"{trainer.global_step}\t{vk}\t-1\t{lr:.6e}\t"
                             f"{int(trainer.model.num_gaussians)}\t"
                             f"{_loss_holder[0]:.10e}\n")
                if trainer.global_step % 200 == 0:
                    _trace.flush()
                return out

            trainer.train_dataset.get_gpu_batch_with_intrinsics = _getb_traced
        _orig_draw = trainer.train_dataset.draw

        def _draw_traced(gpu_batch, *a, **kw):
            out = _orig_draw(gpu_batch, *a, **kw)
            vk = _pose_key(gpu_batch.T_to_world)
            # 합이 아니라 실제 표본 좌표의 해시. 서로 다른 표본이 같은 합을
            # 가질 수 있으므로 합으로는 표집 일치를 확인할 수 없다.
            _s = trainer.train_dataset.last_samples
            rh = (zlib.crc32(np.ascontiguousarray(_s).tobytes()) & 0x7FFFFFFF
                  if _s is not None else -1)
            lr = trainer.model.optimizer.param_groups[0]["lr"]
            _trace.write(f"{trainer.global_step}\t{vk}\t{rh}\t{lr:.6e}\t"
                         f"{int(trainer.model.num_gaussians)}\t"
                         f"{_loss_holder[0]:.10e}\n")
            if trainer.global_step % 200 == 0:
                _trace.flush()
            return out
        if args.block_size != 1:
            trainer.train_dataset.draw = _draw_traced

    # --- 분기 실험: 난수·교체 상태 저장 -----------------------------------
    if args.save_full_state:
        _orig_save = trainer.save_checkpoint

        def _save_with_state(*a, **kw):
            r = _orig_save(*a, **kw)
            torch.save({"global_step": trainer.global_step,
                        "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": (torch.cuda.get_rng_state()
                                     if torch.cuda.is_available() else None),
                        "np_rng": np.random.get_state(),
                        "strategy": type(trainer.strategy).__name__,
                        "cap": getattr(getattr(getattr(trainer.conf, "strategy", None),
                                               "add", None), "max_n_gaussians", None),
                        # 실제 적용 중인 online k(h). 정적 곡선으로 평가한 결과를
                        # online 경로의 검증으로 쓰면 안 되므로 함께 남긴다.
                        "online_knots": (lambda o: (list(o.knots[0]), list(o.knots[1]))
                                         if o is not None and o.knots is not None
                                         else None)(getattr(trainer, "_online_k", None)),
                        "online_k_med": getattr(getattr(trainer, "_online_k", None),
                                                "k_med", None),
                        "online_h_med": getattr(getattr(trainer, "_online_k", None),
                                                "h_med", None)},
                       f"{args.save_full_state}.{trainer.global_step}")
            print(f"[state] step {trainer.global_step} 상태 저장 "
                  f"-> {args.save_full_state}.{trainer.global_step}", flush=True)
            return r
        trainer.save_checkpoint = _save_with_state

    trainer.run_training()

    # --quiet 는 print_stats 를 끄는데, Gaussian 개수를 읽던 곳이 거기다
    # ("Density-pruned X / Y" 로그). 개수는 교수님 지시의 부 지표이므로
    # 학습이 끝난 뒤 모델에서 직접 찍는다. 표집 방식과 무관한 값이다.
    print(f"[final] gaussians={int(trainer.model.num_gaussians)}", flush=True)


if __name__ == "__main__":
    main()
