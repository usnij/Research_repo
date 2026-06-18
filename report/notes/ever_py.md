# gaussian_renderer/ever.py

---

## 핵심 설명

### splinerender() 역할

렌더링 파이프라인의 **진입점**. 카메라 정보와 GaussianModel을 받아서 GPU 렌더링에 필요한 세 가지 입력을 준비하고 `trace_rays()`를 호출한다.

### 1. camera → ray 변환

```python
rays_o, rays_d = camera2rays(view, random=random)
# 또는 어안/왜곡 카메라의 경우
rays_o, rays_d = camera2rays_full(view, random=False)
```

- `world_view_transform` (카메라 외부 파라미터) → 역행렬 → world space 변환 행렬
- 픽셀 좌표 → FoV, 초점거리 적용 → 각 픽셀의 world space ray 방향
- 출력 shape: `(H*W, 3)` 각각

### 2. SH → 색상 계산

```python
net_color = eval_sh2(pc.get_xyz, shs, cam_pos, pc.active_sh_degree)
net_color = torch.nn.functional.softplus(net_color, beta=10)
features = RGB2SH(net_color).reshape(-1, 1, 3)
```

- 각 Gaussian의 구면 조화 함수 계수에서 현재 시점 방향의 색상 계산
- `softplus`로 음수 색상 방지
- 다시 SH 공간으로 변환해서 셰이더에 전달

### 3. Gaussian 파라미터 준비

```python
scales, density = pc.get_scale_and_density_for_rendering(per_point_2d_filter_scale, scaling_modifier)
tmin = pc.tmin if tmin is None else tmin
```

### 4. trace_rays 호출

`trace_rays`는 [fast_ellipsoid_splinetracer.py](fast_ellipsoid_splinetracer_py.md)에 정의된 래퍼 함수. 내부에서 `SplineTracer.apply()`를 호출한다.

```python
out, extras = trace_rays(          # → SplineTracer.apply() 호출
    pc.get_xyz,    # means: Gaussian 중심 위치
    scales,        # 크기
    pc.get_rotation,  # 쿼터니언
    density,       # 밀도
    features,      # 색상 (SH 계수)
    rays_o, rays_d,
    tmin, tmax,
    100,           # max_prim_size (하드코딩)
    means2D, full_wct.reshape(1, 4, 4),
    max_iters=MAX_ITERS,  # 400
    return_extras=True,
)
# → SplineTracer.forward()
#     → create_aabbs()      [create_aabbs_cu.md]
#     → GAS::build()        [GAS_cpp.md]
#     → Forward::Forward()  [Forward_cpp.md]
#     → Forward::trace_rays() → optixLaunch → __raygen__rg_float()
```

### 5. 반환값 구성

```python
return {
    "render": rendered_image,          # (3, H, W) 최종 렌더 이미지
    "visibility_filter": num_pixels >= 4,
    "radii": radii,
    "iters": extras["iters"],          # ray별 처리한 이벤트 수
    "opacity": out[:, 3],
    "distortion_loss": out[:, 4],
}
```

---

## 전체 코드

```python
# coding=utf-8
import torch
import math
from scene.gaussian_model import GaussianModel
from utils import camera_utils_zipnerf
from ever.splinetracers.fast_ellipsoid_splinetracer import trace_rays

MAX_ITERS = 400

from ever.eval_sh import eval_sh as eval_sh2
from utils.sh_utils import eval_sh, RGB2SH, SH2RGB
from kornia import create_meshgrid
import numpy as np
from scene.dataset_readers import ProjectionType

def get_ray_directions(H, W, focal, center=None, random=True):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    if random:
        grid = grid + torch.rand_like(grid)
    else:
        grid = grid + 0.5
    i, j = grid.unbind(-1)
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack(
        [(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1
    )
    return directions

def get_rays(directions, c2w):
    rays_d = directions @ c2w[:3, :3].T
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    return rays_o, rays_d

def camera2rays_full(view, **kwargs):
    w = view.image_width
    h = view.image_height
    device = torch.device('cuda')
    x, y = torch.meshgrid(torch.arange(w, device=device), torch.arange(h, device=device), indexing='xy')
    fx = 0.5 * w / np.tan(0.5 * view.FoVx)
    fy = 0.5 * h / np.tan(0.5 * view.FoVy)
    pixtocams = torch.eye(3, device=device)
    pixtocams[0, 0] = 1/fx
    pixtocams[1, 1] = 1/fy
    pixtocams[0, 2] = -w/2/fx
    pixtocams[1, 2] = -h/2/fy
    T = torch.linalg.inv(view.world_view_transform.T).to(device)
    origins, _, directions, _, _ = camera_utils_zipnerf.pixels_to_rays(
        x.reshape(-1), y.reshape(-1),
        pixtocams.reshape(1, 3, 3),
        T[:3].reshape(1, 3, 4),
        camtype=view.model,
        distortion_params=view.distortion_params,
        xnp=torch
    )
    return origins.float().cuda().contiguous(), directions.float().cuda().contiguous()

def camera2rays(view, **kwargs):
    w = view.image_width
    h = view.image_height
    fx = 0.5 * w / math.tan(0.5 * view.FoVx)
    fy = 0.5 * h / math.tan(0.5 * view.FoVy)
    directions = get_ray_directions(h, w, [fx, fy], **kwargs).cuda()
    directions = (directions / torch.norm(directions, dim=-1, keepdim=True))
    T = torch.linalg.inv(view.world_view_transform.T.cuda())
    rays_o, rays_d = get_rays(directions, T)
    return rays_o.contiguous(), rays_d

def splinerender(
    view,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    random=False,
    tmin=None,
    tmax=1e7,
):
    device = pc.get_xyz.device
    if view.model == ProjectionType.PERSPECTIVE:
        rays_o, rays_d = camera2rays(view, random=random)
    else:
        rays_o, rays_d = camera2rays_full(view, random=False)

    means2D = torch.zeros_like(pc.get_xyz[..., :2])
    means2D.requires_grad = True

    w = view.image_width
    h = view.image_height
    fx = 0.5 * w / np.tan(0.5 * view.FoVx)
    fy = 0.5 * h / np.tan(0.5 * view.FoVy)
    K = torch.tensor([[fx, 0, w/2, 0],[0, fy, h/2, 0],[0, 0, 1, 0]], device="cuda").float()
    wct = view.world_view_transform.cuda().float()
    full_wct = torch.eye(4, device="cuda")
    full_wct[:, :3] = wct @ K.T

    shs = pc.get_features
    if pipe.enable_GLO:
        glo_vector = view.glo_vector if view.glo_vector is not None else torch.zeros((1, 64), device='cuda')
        shs = pc.glo_network(glo_vector.reshape(1, -1), shs.reshape(shs.shape[0], -1)).reshape(shs.shape)

    cam_pos = view.camera_center.to("cuda")
    net_color = eval_sh2(pc.get_xyz, shs, cam_pos, pc.active_sh_degree)
    net_color = torch.nn.functional.softplus(net_color, beta=10)
    features = RGB2SH(net_color).reshape(-1, 1, 3)

    per_point_2d_filter_scale = torch.zeros(pc._xyz.shape[0], device=pc._xyz.device)
    if trace_rays.uses_density:
        scales, density = pc.get_scale_and_density_for_rendering(per_point_2d_filter_scale, scaling_modifier)
    else:
        scales, density = pc.get_scale_and_opacity_for_rendering(per_point_2d_filter_scale, scaling_modifier)

    tmin = pc.tmin if tmin is None else tmin
    out, extras = trace_rays(
        pc.get_xyz, scales, pc.get_rotation, density, features,
        rays_o, rays_d, tmin, tmax, 100,
        means2D, full_wct.reshape(1, 4, 4),
        max_iters=MAX_ITERS, return_extras=True,
    )

    torch.cuda.synchronize()
    rendered_image = out[:, :3].T.reshape(3, view.image_height, view.image_width)
    num_pixels = (extras['touch_count'] // 2)
    side_length = (num_pixels).float().sqrt()
    radii = side_length / 2 * np.sqrt(2) * 2.5 * 5

    return {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter": num_pixels >= 4,
        "touch_count": extras['touch_count'],
        "radii": radii,
        "iters": extras["iters"].reshape(view.image_height, view.image_width),
        "opacity": out[:, 3].reshape(-1, 1),
        "distortion_loss": out[:, 4].reshape(-1, 1),
    }
```
