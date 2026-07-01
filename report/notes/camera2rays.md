# camera2rays 코드 분석

## 1. 위치 및 호출 맥락

**정의:** `gaussian_renderer/ever.py:149`

`splinerender()`에서 카메라 모델에 따라 분기 호출된다:

```python
if view.model == ProjectionType.PERSPECTIVE:
    rays_o, rays_d = camera2rays(view, random=random)       # 핀홀
else:
    rays_o, rays_d = camera2rays_full(view, random=False)   # fisheye / 왜곡
```

---

## 2. camera2rays (핀홀 카메라)

### 전체 코드

```python
def get_ray_directions(H, W, focal, center=None, random=True):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    if random:
        grid = grid + torch.rand_like(grid)   # 서브픽셀 jitter (학습 시)
    else:
        grid = grid + 0.5                     # 픽셀 중심 (추론 시)
    i, j = grid.unbind(-1)
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack(
        [(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1
    )
    return directions   # shape: (H, W, 3), 카메라 공간 방향벡터

def get_rays(directions, c2w):
    rays_d = directions @ c2w[:3, :3].T   # 카메라 공간 → 월드 공간 (회전만)
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # 카메라 위치 = ray 원점
    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    return rays_o, rays_d

def camera2rays(view, **kwargs):
    w = view.image_width
    h = view.image_height
    fx = 0.5 * w / math.tan(0.5 * view.FoVx)   # 초점거리 계산
    fy = 0.5 * h / math.tan(0.5 * view.FoVy)
    directions = get_ray_directions(h, w, [fx, fy], **kwargs).cuda()
    directions = (directions / torch.norm(directions, dim=-1, keepdim=True))  # 정규화
    T = torch.linalg.inv(view.world_view_transform.T.cuda())  # c2w 행렬
    rays_o, rays_d = get_rays(directions, T)
    return rays_o.contiguous(), rays_d
```

### 단계별 동작

```
1. 픽셀 격자 생성   create_meshgrid(H, W)       → (H, W, 2) 픽셀 좌표
2. 방향벡터 계산    (i-cx)/fx, (j-cy)/fy, 1     → (H, W, 3) 카메라 공간
3. 정규화           directions / ||directions||  → 단위 방향벡터
4. 행렬 역산        inv(world_view_transform)    → c2w (카메라→월드 변환)
5. 좌표 변환        directions @ c2w[:3,:3].T    → (H×W, 3) 월드 공간 방향
6. 원점 설정        c2w[:3, 3]                   → 카메라 위치 = 모든 ray의 origin
```

### 출력

| 변수 | shape | 의미 |
|------|-------|------|
| `rays_o` | (H×W, 3) | 모든 ray의 원점 (카메라 위치로 동일) |
| `rays_d` | (H×W, 3) | 픽셀마다 다른 ray 방향 (정규화된 단위벡터) |

---

## 3. camera2rays_full (fisheye / 왜곡 카메라)

```python
def camera2rays_full(view, **kwargs):
    x, y = torch.meshgrid(torch.arange(w, device=device),
                          torch.arange(h, device=device), indexing='xy')
    pixtocams = torch.eye(3, device=device)
    pixtocams[0, 0] = 1/fx;  pixtocams[0, 2] = -w/2/fx
    pixtocams[1, 1] = 1/fy;  pixtocams[1, 2] = -h/2/fy
    T = torch.linalg.inv(view.world_view_transform.T).to(device)
    origins, _, directions, _, _ = camera_utils_zipnerf.pixels_to_rays(
        x.reshape(-1), y.reshape(-1),
        pixtocams.reshape(1, 3, 3),
        T[:3].reshape(1, 3, 4),
        camtype=view.model,
        distortion_params=view.distortion_params,   # 왜곡 파라미터 적용
        xnp=torch
    )
```

핀홀과 달리 `pixels_to_rays()`가 distortion_params를 이용해 왜곡된 픽셀 좌표를 역보정한 후 ray를 생성한다.

---

## 4. 왜 16.39ms (Forward의 39.7%)나 걸리는가

### 실제 실행 흐름과 CUDA launch 위치

```
[ get_ray_directions ]  ← CPU에서 실행 (CUDA launch 없음)
  create_meshgrid(H, W)          CPU 텐서 생성
  grid + torch.rand_like(grid)   CPU 연산
  torch.ones_like(i)             CPU 연산
  torch.stack([...])             CPU 연산
  return directions              CPU 텐서 (H, W, 3)

[ camera2rays ]
  .cuda()                        ← PCIe 전송: 800×800×3×4byte = 7.68MB
  torch.norm(directions, ...)    ← CUDA launch 1
  directions / norm              ← CUDA launch 2
  view.world_view_transform.cuda() ← PCIe 전송: 4×4 matrix (tiny)
  torch.linalg.inv(...)          ← CUDA launch 3
  directions @ c2w[:3,:3].T      ← CUDA launch 4
  rays_o.contiguous()            ← CUDA launch 5
```

CUDA launch: **총 5회** / PCIe 전송: **7.68MB (CPU→GPU)**

---

### 병렬임에도 느린 이유

640,000개 ray 계산 자체는 GPU에서 병렬로 처리되므로 빠르다.  
실제 비용은 아래 세 가지에서 발생한다:

**① PCIe 전송**
`get_ray_directions`가 CPU에서 실행되어 결과(7.68MB)를 GPU로 전송한다.
```
7.68MB / PCIe 3.0 대역폭 ~16GB/s ≈ 0.48ms (매 iter)
```

**② GPU 메모리 할당**
매 iteration마다 H×W×3 텐서를 새로 할당하고 해제한다.
GPU 메모리 할당은 공짜가 아니며, 반복 시 overhead가 쌓인다.

**③ 캐싱 없음**
`create_meshgrid`(해상도가 같으면 결과 동일)와  
`torch.linalg.inv`(같은 카메라면 결과 동일)를  
매 30,000 iteration마다 처음부터 재계산한다.

---

### 3DGS와의 비교

| | EVER | 3DGS |
|--|------|------|
| 렌더링 방식 | ray tracing | rasterization |
| 픽셀별 ray 생성 | 필요 (H×W, 매 iter) | 불필요 |
| CPU→GPU 전송 | 7.68MB/iter | 없음 |

3DGS는 Gaussian을 2D로 투영하므로 픽셀별 ray 생성 자체가 없다.  
EVER는 ray 기반 구조상 이 비용이 구조적으로 발생한다.

---

### 최적화 가능한 지점

```
- get_ray_directions를 GPU에서 직접 실행 → PCIe 전송 제거
- meshgrid / inv 결과를 캐싱 → 재계산 제거
- 학습 중 전체 픽셀 대신 일부만 샘플링 → 전송량 감소
```
