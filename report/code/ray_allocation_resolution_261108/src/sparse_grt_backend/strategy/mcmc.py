# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MCMC implementation was adpoted from gSplat library (https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/strategy/mcmc.py),
# which is based on the original implementation https://github.com/ubc-vision/3dgs-mcmc that uderlines the work
#
# 3D Gaussian Splatting as Markov Chain Monte Carlo by
# Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi and Kwang Moo Yie
#
# If you use this code in your research, please cite the above works.

import math
from typing import Optional, Tuple

import torch

from sparse_grt_backend.model.model import MixtureOfGaussians
from sparse_grt_backend.strategy.base import BaseStrategy
from sparse_grt_backend.utils.logger import logger
from sparse_grt_backend.utils.misc import _multinomial_sample, check_step_condition

_mcmc_plugin = None


def load_mcmc_plugin():
    global _mcmc_plugin
    if _mcmc_plugin is None:
        from sparse_grt.native_build import build_mcmc
        _mcmc_plugin = build_mcmc()


class MCMCStrategy(BaseStrategy):
    """Densification and prunning strategy that follows the paper:

    `3D Gaussian Splatting as Markov Chain Monte Carlo <https://arxiv.org/abs/2404.09591>`

    MCMC Strategy interprets the training process of placing and optimizing Gaussians
    as a sampling process.

    Specifically, it periodically:
    - Moves "dead" Gaussians (low opacity) to the location of "live" Gaussians (high opacity).
    - Adds covariance dependent noise to the positions of the Gaussians.
    - Introduces new Gaussians sampled based on the opacity distribution.

    """

    def __init__(self, config, model: MixtureOfGaussians) -> None:
        self._obs = None          # 개체별 누적 관측 횟수 (계측용)
        self._audit_every = 0     # 0 이면 계측하지 않는다
        super().__init__(config=config, model=model)

        load_mcmc_plugin()
        # Precompute the look up table for binomial coefficients (Eq 9 in the MCMC paper)
        self.binoms = torch.tensor(
            [
                [math.comb(n, k) if k <= n else 0 for k in range(config.strategy.binom_n_max)]
                for n in range(config.strategy.binom_n_max)
            ],
            dtype=torch.float32,
            device=self.model.device,
        )

    def post_optimizer_step(self, step: int, scene_extent: float, train_dataset, batch=None, writer=None) -> bool:
        # Relocate dead gaussians to the alive areas
        if check_step_condition(
            step,
            self.conf.strategy.relocate.start_iteration,
            self.conf.strategy.relocate.end_iteration,
            self.conf.strategy.relocate.frequency,
        ):
            self.relocate_gaussians()

        # Add new Gaussians if the maximum number has not been reached
        if check_step_condition(
            step,
            self.conf.strategy.add.start_iteration,
            self.conf.strategy.add.end_iteration,
            self.conf.strategy.add.frequency,
        ):
            self.add_new_gaussians()

        # Perturb the positions of the Gaussians
        if check_step_condition(
            step,
            self.conf.strategy.perturb.start_iteration,
            self.conf.strategy.perturb.end_iteration,
            self.conf.strategy.perturb.frequency,
        ):
            self.perturb_gaussians()

        return True

    @torch.no_grad()
    def observe(self, visibility) -> None:
        """개체별 관측 횟수를 누적한다.

        relocate 는 opacity 만 보고 죽은 개체를 정한다. multi-view 를 켜면
        Ray 예산이 같아도 view 당 격자가 성겨지므로 (block 4 -> 8, 1/16 -> 1/64)
        화면에서 작은 면적을 차지하는 개체는 Ray 를 아예 못 받는다. 그런 개체는
        image gradient 를 못 받는데 opacity 정규화는 Ray 와 무관하게 매 step
        들어오므로 opacity 가 깎여 죽는다. 쓸모없어서가 아니라 관측이 안 돼서
        죽는 것이라면, 사망 판정에 관측량이 들어가야 한다. 여기서는 그 여부를
        재기 위해 누적만 한다.
        """
        if visibility is None:
            return
        v = visibility.reshape(-1).float()
        if self._obs is None or self._obs.numel() != v.numel():
            prev = self._obs
            self._obs = torch.zeros_like(v)
            if prev is not None and prev.numel() < v.numel():
                self._obs[: prev.numel()] = prev
        self._obs += (v > 0).float()

    @torch.no_grad()
    def relocate_gaussians(self) -> None:
        # Get the per Gaussian densities and scales (after sigmoid)
        densities = self.model.get_density()
        # Find the dead indices
        dead_idxs = torch.where(densities <= self.conf.strategy.opacity_threshold)[0]
        alive_idxs = torch.where(densities > self.conf.strategy.opacity_threshold)[0]
        # 죽는 개체가 저관측 쪽에 몰려 있는지 본다. 전체 분포와 죽는 개체의
        # 분포를 같은 시점에 찍어 비교한다.
        if self._audit_every and self._obs is not None \
                and self._obs.numel() == densities.reshape(-1).numel() \
                and len(dead_idxs) > 0:
            o = self._obs
            q = lambda t: [float(t.quantile(x)) for x in (0.1, 0.25, 0.5, 0.75, 0.9)]
            qa, qd = q(o), q(o[dead_idxs])
            logger.info(
                f"[obs] 전체 {o.numel():,} 관측 p10={qa[0]:.0f} p25={qa[1]:.0f} "
                f"p50={qa[2]:.0f} p75={qa[3]:.0f} p90={qa[4]:.0f} | "
                f"죽는 개체 {len(dead_idxs):,} p10={qd[0]:.0f} p25={qd[1]:.0f} "
                f"p50={qd[2]:.0f} p75={qd[3]:.0f} p90={qd[4]:.0f} | "
                f"관측 0회 전체 {float((o == 0).float().mean()) * 100:.1f}% "
                f"죽는쪽 {float((o[dead_idxs] == 0).float().mean()) * 100:.1f}%"
            )
        n_dead_gaussians = len(dead_idxs)

        if n_dead_gaussians:
            sampled_idxs, new_densities, new_scales = self.sample_new_gaussians(n_dead_gaussians, alive_idxs)

            def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
                if name == "density":
                    param[sampled_idxs] = new_densities
                elif name == "scale":
                    param[sampled_idxs] = new_scales
                param[dead_idxs] = param[sampled_idxs]
                return torch.nn.Parameter(param, requires_grad=param.requires_grad)

            def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
                v[sampled_idxs] = 0
                return v

            self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)

        if self.conf.strategy.print_stats:
            logger.info(f"Relocated {n_dead_gaussians} ({n_dead_gaussians / len(densities) * 100:.2f}%) gaussians")

    @torch.no_grad()
    def add_new_gaussians(self) -> None:
        # Get the current number of gaussians
        current_num_gaussians = self.model.num_gaussians
        target_num_gaussians = min(self.conf.strategy.add.max_n_gaussians, int(1.05 * current_num_gaussians))
        num_gaussians_to_add = max(0, target_num_gaussians - current_num_gaussians)

        if num_gaussians_to_add:
            # relocate 는 alive_idxs 를 넘기는데 이쪽만 전체를 넘겨 dead 도 부모가
            # 되었다. dead 의 opacity 는 0 에 가까워 relocation 커널의 분모를 0 으로
            # 만든다. 두 경로의 후보 집합을 맞춘다.
            densities = self.model.get_density()
            alive_idxs = torch.where(densities > self.conf.strategy.opacity_threshold)[0]
            sampled_idxs, new_densities, new_scales = self.sample_new_gaussians(
                num_gaussians_to_add, alive_idxs
            )

            def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
                if name == "density":
                    param[sampled_idxs] = new_densities
                elif name == "scale":
                    param[sampled_idxs] = new_scales
                param_new = torch.cat([param, param[sampled_idxs]])
                return torch.nn.Parameter(param_new, requires_grad=param.requires_grad)

            def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
                v_new = torch.zeros((len(sampled_idxs), *v.shape[1:]), device=v.device)
                return torch.cat([v, v_new])

            # 배치 계측. 새 Gaussian 이 어디에 놓이는지는 부모를 뽑는 다항분포가
            # 정한다. 그 분포가 얼마나 퍼져 있는지(유효표본수)와 후보 pool 의
            # opacity 분포, 그리고 부모가 그 시점에 몇 번 관측됐는지를 남긴다.
            # multi-view 가 배치 구간에서 정보를 늘려주는지 보려면 이 셋이 필요하다.
            if self._audit_every:
                _p = densities[alive_idxs].flatten()
                _p = _p / _p.sum().clamp_min(1e-30)
                _ess = float(1.0 / (_p * _p).sum().clamp_min(1e-30))
                _uniq = int(torch.unique(sampled_idxs).numel())
                _q = lambda t: [float(t.quantile(x)) for x in (0.1, 0.5, 0.9)]
                _qo = _q(densities[alive_idxs].flatten())
                _msg = (f"[place] G={current_num_gaussians:,} 추가 {num_gaussians_to_add:,} "
                        f"| 부모 후보 {alive_idxs.numel():,} 유효표본수 {_ess:,.0f} "
                        f"({100.0 * _ess / max(alive_idxs.numel(), 1):.1f}%) "
                        f"고유 부모 {_uniq:,} | opacity p10={_qo[0]:.4f} "
                        f"p50={_qo[1]:.4f} p90={_qo[2]:.4f}")
                if self._obs is not None and self._obs.numel() == densities.reshape(-1).numel():
                    _op = self._obs[sampled_idxs]
                    _qp = _q(_op)
                    _msg += (f" | 부모 관측 p10={_qp[0]:.0f} p50={_qp[1]:.0f} "
                             f"p90={_qp[2]:.0f}, 0회 {100.0 * float((_op == 0).float().mean()):.1f}%")
                logger.info(_msg)

            self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)

        if self.conf.strategy.print_stats:
            logger.info(
                f"Added {num_gaussians_to_add} ({num_gaussians_to_add / current_num_gaussians * 100:.2f}%) gaussians"
            )

    @torch.no_grad()
    def perturb_gaussians(self) -> None:
        covariance = self.model.get_covariance()
        positions = self.model.get_positions()
        densities = self.model.get_density()

        current_lr = 0.0
        for param_gorup in self.model.optimizer.param_groups:
            if param_gorup["name"] == "positions":
                current_lr = param_gorup["lr"]

        def op_sigmoid(x: torch.Tensor, k: int = 100, x0: float = 0.995) -> torch.Tensor:
            return 1 / (1 + torch.exp(-k * (x - x0)))

        # Current positional learning rate multiplied by the config paramater scale
        noise = (
            torch.randn_like(positions) * (op_sigmoid(1 - densities)) * self.conf.strategy.perturb.noise_lr * current_lr
        )
        noise = torch.bmm(covariance, noise.unsqueeze(-1)).squeeze(-1)

        self.model.positions.add_(noise)

    def sample_new_gaussians(
        self, num_gaussians: int, valid_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        densities = self.model.get_density()
        scales = self.model.get_scale()

        if valid_indices is None:
            valid_indices = torch.arange(0, int(densities.shape[0]), device=densities.device, dtype=torch.int32)

        probabilities = densities[valid_indices].flatten()  # ensure its shape is [N,]
        # 배치 신호의 기여를 재는 진단 경로. uniform 으로 두면 살아 있는 개체가
        # 모두 같은 확률로 부모가 되어, 위치 선택에서 장면 정보가 사라진다.
        if getattr(self.conf.strategy.add, "placement", "opacity") == "uniform":
            probabilities = torch.ones_like(probabilities)

        # Sample the locations to which the dead Gaussians will be moved proportional to the opacity of the alive Gaussians
        sampled_idxs = _multinomial_sample(probabilities, num_gaussians, replacement=True)
        sampled_idxs = valid_indices[sampled_idxs]

        ratios = (
            (torch.bincount(sampled_idxs)[sampled_idxs] + 1).clamp_(min=1, max=self.conf.strategy.binom_n_max).int()
        )

        new_densities, new_scales = _mcmc_plugin.compute_relocation_tensor(
            densities[sampled_idxs].contiguous(),
            scales[sampled_idxs].contiguous(),
            ratios.contiguous(),
            self.binoms,
            self.conf.strategy.binom_n_max,
        )

        new_densities = self.model.density_activation_inv(
            torch.clamp(
                new_densities, max=1.0 - torch.finfo(torch.float32).eps, min=self.conf.strategy.opacity_threshold
            )
        )

        # denom_sum 이 0 으로 반올림되면 coeff 가 inf/nan 이 되어 scale 이 오염되고,
        # 그 경계상자로 만든 가속 구조를 순회하면 잘못된 메모리를 읽는다. 식이 무너진
        # 경우의 올바른 대체는 부모를 그대로 복사하는 것이다.
        parent_scales = scales[sampled_idxs]
        broken = ~torch.isfinite(new_scales) | (new_scales <= 0)
        if broken.any():
            new_scales = torch.where(broken, parent_scales, new_scales)
        new_scales = new_scales.clamp(min=1e-8, max=float(scales.max()))

        new_scales = self.model.scale_activation_inv(new_scales)

        return sampled_idxs, new_densities, new_scales
