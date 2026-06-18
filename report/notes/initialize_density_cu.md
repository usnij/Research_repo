# initialize_density.cu — ray origin 내부 Gaussian 초기화

---

## 핵심 설명

### 역할

`Forward::trace_rays()`가 `optixLaunch` 직전에 호출한다. **ray origin이 Gaussian 내부에 있는 경우** `initial_drgb`를 미리 채워주는 전처리 단계다.

`rg_float()`의 첫 줄 `state.drgb = initial_drgb[idx.x]`가 이 값을 읽는다. ray가 씬 외부에서 시작하는 일반적인 경우엔 0이지만, 카메라가 Gaussian 안에 있을 때는 그 Gaussian의 밀도가 이미 running sum에 포함되어 있어야 한다.

### 두 단계 처리

```
1. kern_prefilter(): 후보 Gaussian 추리기
   모든 Gaussian의 AABB와 ray origin 사이의 최소 거리를 계산
   거리 ≤ tmin 이면 "후보"로 touch_indices[]에 기록
   → AABB로만 추리기 때문에 O(num_prims) 병렬

2. kern_initialize_density(): 정밀 내부 판정 + drgb 누적
   touch_indices의 후보 Gaussian들에 대해서만 실행
   ray origin을 Gaussian 로컬 좌표로 변환: Trayo = Rᵀ(rayo - center) / scale
   |Trayo|² ≤ 1 이면 내부 → initial_drgb에 atomicAdd
```

### initial_drgb 포맷

```
initial_drgb[ray_idx * 4 + 0] += density          (σ 합계)
initial_drgb[ray_idx * 4 + 1] += density * color.x (σ·r)
initial_drgb[ray_idx * 4 + 2] += density * color.y (σ·g)
initial_drgb[ray_idx * 4 + 3] += density * color.z (σ·b)
```

SplineState의 `drgb` float4와 동일한 포맷. ray origin이 여러 Gaussian 내부에 동시에 있을 수 있으므로 atomicAdd로 누적한다.

### 색상 처리

SH 차수 0만 사용 (상수항):
```cpp
color = features[prim_ind * 3 + k] * SH_C0 + 0.5   (k = 0,1,2)
SH_C0 = 0.28209...  (l=0 SH 계수)
```

---

## 전체 코드

```cuda
// Copyright 2024 Google LLC (Apache 2.0 License)

#define TRI_PER_G 4
#define PT_PER_G 4
#define SQR(x) (x)*(x)

#include "Forward.h"
#include "structs.h"
#include "glm/glm.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

__device__ static const float SH_C0 = 0.28209479177387814f;

// ─── ray origin → Gaussian 로컬 좌표 변환 ─────────────────────────────────────
__device__ glm::vec3 get_Trayo(
    const glm::vec3 center, const glm::vec4 quat,
    const glm::vec3 size,   const glm::vec3 rayo)
{
    const float r = quat.x, x = quat.y, y = quat.z, z = quat.w;
    const glm::mat3 Rt = {
        1.0 - 2.0*(y*y + z*z),  2.0*(x*y - r*z),        2.0*(x*z + r*y),
        2.0*(x*y + r*z),         1.0 - 2.0*(x*x + z*z),  2.0*(y*z - r*x),
        2.0*(x*z - r*y),         2.0*(y*z + r*x),         1.0 - 2.0*(x*x + y*y)
    };
    return (Rt * (rayo - center)) / size;
}

// ─── 1단계: AABB 기반 후보 추리기 ─────────────────────────────────────────────
// 스레드당 Gaussian 1개. ray origin과 AABB 사이 최소 거리가 tmin 이하면 후보로 기록.
__global__ void kern_prefilter(
    const OptixAabb *aabbs, const size_t num_prims, const float tmin,
    const glm::vec3 *rayos, int *touch_indices, int *touch_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 0 || i >= num_prims) return;

    OptixAabb aabb = aabbs[i];
    const glm::vec3 rayo = rayos[0];

    // Jim Arvo, Graphics Gems: point-to-AABB 최소 거리²
    float dmin = 0;
    if (rayo.x < aabb.minX)      dmin += SQR(rayo.x - aabb.minX);
    else if (rayo.x > aabb.maxX) dmin += SQR(rayo.x - aabb.maxX);
    if (rayo.y < aabb.minY)      dmin += SQR(rayo.y - aabb.minY);
    else if (rayo.y > aabb.maxY) dmin += SQR(rayo.y - aabb.maxY);
    if (rayo.z < aabb.minZ)      dmin += SQR(rayo.z - aabb.minZ);
    else if (rayo.z > aabb.maxZ) dmin += SQR(rayo.z - aabb.maxZ);

    if (dmin <= tmin * tmin) {
        int pos = atomicAdd(touch_count, 1);
        touch_indices[pos] = i;
    }
}

// ─── 2단계: 정밀 내부 판정 + initial_drgb 누적 ──────────────────────────────
// 스레드 (j=ray_idx, i=touch_candidate_idx)의 2D 그리드
__global__ void kern_initialize_density(
    const glm::vec3 *means, const glm::vec3 *scales,
    const glm::vec4 *quats, const float *densities, const float *features,
    const size_t num_prims, const size_t num_rays, const float tmin,
    const glm::vec3 *rayos, const glm::vec3 *rayds,
    float *initial_drgb, int *touch_indices, int *touch_count)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // ray 인덱스
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // 후보 Gaussian 인덱스
    if (i >= *touch_count) return;
    if (j >= num_rays) return;

    glm::vec3 rayo = rayos[j] + tmin * glm::normalize(rayds[j]);
    const int prim_ind = touch_indices[i];

    const glm::vec3 Trayo = get_Trayo(
        means[prim_ind], glm::normalize(quats[prim_ind]),
        scales[prim_ind], rayo);

    const float dist = Trayo.x*Trayo.x + Trayo.y*Trayo.y + Trayo.z*Trayo.z;
    if (dist <= 1) {   // ray origin이 이 Gaussian 내부
        const float density = densities[prim_ind];
        const glm::vec3 color = {
            features[prim_ind * 3 + 0] * SH_C0 + 0.5f,
            features[prim_ind * 3 + 1] * SH_C0 + 0.5f,
            features[prim_ind * 3 + 2] * SH_C0 + 0.5f,
        };
        // SplineState.drgb 포맷과 동일: (σ, σ·r, σ·g, σ·b)
        atomicAdd(initial_drgb + 4*j + 0, density);
        atomicAdd(initial_drgb + 4*j + 1, density * color.x);
        atomicAdd(initial_drgb + 4*j + 2, density * color.y);
        atomicAdd(initial_drgb + 4*j + 3, density * color.z);
    }
}

// ─── initialize_density(): 두 단계 커널을 순서대로 호출 ──────────────────────
void initialize_density(Params *params, OptixAabb *aabbs,
                        int *d_touch_count, int *d_touch_inds)
{
    const size_t block_size       = 1024;
    const size_t ray_block_size   = 64;
    const size_t second_block_size = 16;

    int num_prims = params->means.size;
    int num_rays  = params->initial_drgb.size;

    bool initialize_tensors = (d_touch_count == NULL);
    if (initialize_tensors) {
        cudaMalloc((void**)&d_touch_inds,  num_prims * sizeof(int));
        cudaMalloc((void**)&d_touch_count, sizeof(int));
    }
    cudaMemset(d_touch_count, 0, sizeof(int));

    // 1단계: 후보 추리기
    kern_prefilter<<<
        (num_prims + block_size - 1) / block_size,
        block_size>>>(
            aabbs, num_prims, params->tmin,
            (glm::vec3 *)(params->ray_origins.data),
            d_touch_inds, d_touch_count);

    int touch_count;
    cudaMemcpy(&touch_count, d_touch_count, sizeof(int), cudaMemcpyDeviceToHost);

    // 2단계: 내부 판정 + drgb 누적 (후보가 있을 때만)
    if (touch_count > 0) {
        dim3 init_grid_dim(
            (num_rays  + ray_block_size    - 1) / ray_block_size,
            (touch_count + second_block_size - 1) / second_block_size,
            1);
        dim3 init_block_dim(ray_block_size, second_block_size, 1);

        kern_initialize_density<<<init_grid_dim, init_block_dim>>>(
            (glm::vec3 *)(params->means.data),
            (glm::vec3 *)(params->scales.data),
            (glm::vec4 *)(params->quats.data),
            (float *)(params->densities.data),
            (float *)(params->features.data),
            num_prims, num_rays, params->tmin,
            (glm::vec3 *)(params->ray_origins.data),
            (glm::vec3 *)(params->ray_directions.data),
            (float *)(params->initial_drgb.data),
            d_touch_inds, d_touch_count);

        CUDA_SYNC_CHECK();
    }

    if (initialize_tensors) {
        cudaFree(d_touch_inds);
        cudaFree(d_touch_count);
    }
}
```
