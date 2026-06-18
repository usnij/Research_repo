# ever/splinetracers/fast_ellipsoid_splinetracer.py

---

## 핵심 설명

### SplineTracer 클래스 역할

`torch.autograd.Function`을 상속한 **PyTorch 커스텀 autograd 클래스**.
`.apply()`로 호출하면 PyTorch가 자동으로 forward/backward를 분기하고, backward 시 `ctx`에 저장된 값들을 전달한다.

### forward() 핵심 흐름

```python
# 1. Gaussian → AABB 생성
ctx.prims = sp.Primitives(ctx.device)
ctx.prims.add_primitives(mean, scale, quat, half_attribs, density, color)
#   → C++ fesPyPrimitives::add_primitives()  [py_binding.cpp]
#        → create_aabbs(model)               [create_aabbs.cu]
#             → kern_create_aabbs<<<>>>()    [GPU 커널: Gaussian마다 AABB 계산]
#        → D_AABBS = model.aabbs             [전역 캐시 업데이트]

# 2. AABB → BVH 빌드
ctx.gas = sp.GAS(otx, ctx.device, ctx.prims, True, False, True)
#   → C++ fesPyGas 생성자 → GAS::build()    [GAS.cpp]
#        → optixAccelBuild()                [BVH 트리 생성]
#        → gas_handle 획득                  [BVH 루트 핸들]

# 3. OptiX 파이프라인 셋업
ctx.forward = sp.Forward(otx, ctx.device, ctx.prims, True)
#   → C++ fesPyForward 생성자 → Forward::Forward()  [Forward.cpp]
#        → PTX 로딩 (shaders.slang 컴파일 결과)
#        → 프로그램 그룹 등록 (raygen / anyhit+intersection / miss)
#        → SBT 설정
#        → params에 Gaussian 파라미터 포인터 연결

# 4. GPU 렌더링 실행
out = ctx.forward.trace_rays(ctx.gas, rayo, rayd, tmin, tmax, ctx.max_iters, max_prim_size)
#   → C++ fesPyForward::trace_rays()        [py_binding.cpp]
#        → Forward::trace_rays()            [Forward.cpp]
#             → cudaMemcpy(params → GPU)
#             → optixLaunch(num_rays, 1, 1) → GPU 셰이더 실행
```

| Step | Python 호출 | C++ 진입점 | 상세 파일 |
|------|------------|-----------|---------|
| 1 | `sp.Primitives(...).add_primitives(...)` | `fesPyPrimitives::add_primitives()` | [py_binding_cpp.md](py_binding_cpp.md) → [create_aabbs_cu.md](create_aabbs_cu.md) |
| 2 | `sp.GAS(...)` | `GAS::build()` | [GAS_cpp.md](GAS_cpp.md) |
| 3 | `sp.Forward(...)` | `Forward::Forward()` | [Forward_cpp.md](Forward_cpp.md) |
| 4 | `.trace_rays(...)` | `fesPyForward::trace_rays()` → `Forward::trace_rays()` | [py_binding_cpp.md](py_binding_cpp.md) → [Forward_cpp.md](Forward_cpp.md) |

### backward() 저장하는 것

```python
ctx.save_for_backward(
    mean, scale, quat, density, color,
    rayo, rayd,
    tri_collection,   # ← forward에서 각 ray가 처리한 이벤트 순서 (backward 재생용)
    wcts,
    out['initial_drgb'],
    initial_inds,
    half_attribs
)
```

### backward() 핵심 흐름

```python
kernels.backwards_kernel(
    last_state=ctx.saved.states,     # 마지막 SplineState
    last_dirac=ctx.saved.diracs,     # 마지막 ControlPoint.dirac
    iters=ctx.saved.iters,           # ray별 처리 이벤트 수
    tri_collection=tri_collection,   # forward에서 기록한 이벤트 순서
    ...
    dL_doutputs=grad_output,         # PyTorch에서 전달된 gradient
).launchRaw(blockSize=(16, 1, 1), gridSize=(num_rays // 16 + 1, 1, 1))
```

Slang `backwards_kernel`이 `tri_collection`을 역순으로 순회하며 `bwd_diff(update)` 자동미분으로 각 Gaussian 파라미터의 gradient를 계산한다.

### .apply() vs .forward() 직접 호출의 차이

| | `.apply()` | `.forward()` 직접 |
|---|---|---|
| gradient 추적 | O (PyTorch autograd 그래프 연결) | X |
| backward 자동 실행 | O | X |
| ctx 전달 | O | X |

---

## 전체 코드

```python
import time
from pathlib import Path
from typing import *

import slangtorch
import torch
from torch.autograd import Function

import sys
sys.path.append(str(Path(__file__).parent))

from build.splinetracer.extension import fast_ellipsoid_splinetracer_cpp_extension as sp
kernels = slangtorch.loadModule(
    str(Path(__file__).parent / "fast_ellipsoid_splinetracer/slang/backwards_kernel.slang"),
    includePaths=[str(Path(__file__).parent / 'slang')]
)

otx = sp.OptixContext(torch.device("cuda:0"))


class SplineTracer(Function):
    @staticmethod
    def forward(
        ctx: Any,
        mean: torch.Tensor,
        scale: torch.Tensor,
        quat: torch.Tensor,
        density: torch.Tensor,
        color: torch.Tensor,
        rayo: torch.Tensor,
        rayd: torch.Tensor,
        tmin: float,
        tmax: float,
        max_prim_size: float,
        mean2D: torch.Tensor,
        wcts: torch.Tensor,
        max_iters: int,
        return_extras: bool = False,
    ):
        ctx.device = rayo.device
        ctx.prims = sp.Primitives(ctx.device)
        mean = mean.contiguous()
        scale = scale.contiguous()
        density = density.contiguous()
        quat = quat.contiguous()
        color = color.contiguous()
        half_attribs = torch.cat([mean, scale, quat], dim=1).half().contiguous()
        ctx.prims.add_primitives(mean, scale, quat, half_attribs, density, color)

        ctx.gas = sp.GAS(otx, ctx.device, ctx.prims, True, False, True)
        ctx.forward = sp.Forward(otx, ctx.device, ctx.prims, True)
        ctx.max_iters = max_iters

        out = ctx.forward.trace_rays(ctx.gas, rayo, rayd, tmin, tmax, ctx.max_iters, max_prim_size)
        ctx.saved = out["saved"]
        ctx.max_prim_size = max_prim_size
        ctx.tmin = tmin
        ctx.tmax = tmax
        tri_collection = out["tri_collection"]

        states = ctx.saved.states.reshape(rayo.shape[0], -1)
        distortion_loss = (states[:, 0] - states[:, 1])
        color_and_loss = torch.cat([out["color"], distortion_loss.reshape(-1, 1)], dim=1)

        initial_inds = out['initial_touch_inds'][:out['initial_touch_count'][0]]
        ctx.save_for_backward(
            mean, scale, quat, density, color, rayo, rayd,
            tri_collection, wcts, out['initial_drgb'], initial_inds, half_attribs
        )

        if return_extras:
            return color_and_loss, dict(
                tri_collection=tri_collection,
                iters=ctx.saved.iters,
                opacity=out["color"][:, 3],
                touch_count=ctx.saved.touch_count,
                distortion_loss=distortion_loss,
                saved=ctx.saved,
            )
        else:
            return color_and_loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, return_extras=False):
        (mean, scale, quat, density, features,
         rayo, rayd, tri_collection, wcts,
         initial_drgb, initial_inds, half_attribs) = ctx.saved_tensors
        device = ctx.device

        num_prims = mean.shape[0]
        num_rays = rayo.shape[0]

        dL_dmeans     = torch.zeros((num_prims, 3),  dtype=torch.float32, device=device)
        dL_dscales    = torch.zeros((num_prims, 3),  dtype=torch.float32, device=device)
        dL_dquats     = torch.zeros((num_prims, 4),  dtype=torch.float32, device=device)
        dL_ddensities = torch.zeros((num_prims),     dtype=torch.float32, device=device)
        dL_dfeatures  = torch.zeros_like(features)
        dL_drayo      = torch.zeros((num_rays, 3),   dtype=torch.float32, device=device)
        dL_drayd      = torch.zeros((num_rays, 3),   dtype=torch.float32, device=device)
        dL_dmeans2D   = torch.zeros((num_prims, 2),  dtype=torch.float32, device=device)
        touch_count   = torch.zeros((num_prims),     dtype=torch.int32,   device=device)
        dL_dinital_drgb = torch.zeros((num_rays, 4), dtype=torch.float32, device=device)

        block_size = 16
        if ctx.saved.iters.sum() > 0:
            dual_model = (
                mean, scale, quat, density, features,
                dL_dmeans, dL_dscales, dL_dquats, dL_ddensities,
                dL_dfeatures, dL_drayo, dL_drayd, dL_dmeans2D,
            )
            kernels.backwards_kernel(
                last_state=ctx.saved.states,
                last_dirac=ctx.saved.diracs,
                iters=ctx.saved.iters,
                tri_collection=tri_collection,
                ray_origins=rayo,
                ray_directions=rayd,
                model=dual_model,
                initial_drgb=initial_drgb,
                dL_dinital_drgb=dL_dinital_drgb,
                touch_count=touch_count,
                dL_doutputs=grad_output.contiguous(),
                wcts=wcts if wcts is not None else torch.ones((1,4,4), device=device, dtype=torch.float32),
                tmin=ctx.tmin, tmax=ctx.tmax,
                max_prim_size=ctx.max_prim_size,
                max_iters=ctx.max_iters,
            ).launchRaw(
                blockSize=(block_size, 1, 1),
                gridSize=(num_rays // block_size + 1, 1, 1),
            )

        v = 1e+3
        mean_v = 1e+3
        return (
            dL_dmeans.clip(min=-mean_v, max=mean_v),
            dL_dscales.clip(min=-v, max=v),
            dL_dquats.clip(min=-v, max=v),
            dL_ddensities.clip(min=-50, max=50).reshape(density.shape),
            dL_dfeatures.clip(min=-v, max=v),
            dL_drayo.clip(min=-v, max=v),
            dL_drayd.clip(min=-v, max=v),
            None, None, None, None, None, None, None,
        )


def trace_rays(
    mean, scale, quat, density, features,
    rayo, rayd,
    tmin=0.0, tmax=1000,
    max_prim_size=3,
    dL_dmeans2D=None, wcts=None,
    max_iters=500, return_extras=False,
):
    out = SplineTracer.apply(
        mean, scale, quat, density, features,
        rayo, rayd, tmin, tmax, max_prim_size,
        dL_dmeans2D, wcts, max_iters, return_extras,
    )
    return out

trace_rays.uses_density = True
```
