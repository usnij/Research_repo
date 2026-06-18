# create_aabbs.cu — AABB GPU 커널

---

## 핵심 설명

### 역할

각 Gaussian(ellipsoid)을 감싸는 **AABB(Axis-Aligned Bounding Box)** 를 GPU에서 병렬 계산한다. BVH 빌드(`GAS::build()`)에 필요한 입력이다.

### 왜 AABB가 필요한가

OptiX의 `optixAccelBuild()`는 AABB를 기반으로만 BVH를 빌드할 수 있다. Ellipsoid는 커스텀 프리미티브로 직접 지원되지 않으므로, 각 Gaussian을 감싸는 최소 직육면체(AABB)를 먼저 계산하고 이걸로 BVH를 구성한다.

BVH는 AABB로 "후보"를 추려주고, 실제 ellipsoid 교차는 `__intersection__ellipsoid`에서 정밀 계산한다.

### AABB 계산 수식

```
ellipsoid = R·S·(x - center) ≤ 1    (R: 회전행렬, S: 스케일)

AABB의 반지름 = 변환 행렬 M = S·Rᵀ 의 각 행의 L2 norm

minX = center.x - ‖M[0,:]‖₂
maxX = center.x + ‖M[0,:]‖₂
(Y, Z 동일)
```

쿼터니언 → 회전행렬 변환 → `M = S·Rᵀ` → 각 축 방향 최대 반지름 계산의 순서.

### create_aabbs() — GPU 메모리 관리

```
prims.prev_alloc_size < prims.num_prims 이면 cudaMalloc (재할당)
                          그렇지 않으면  기존 메모리 재사용 (malloc 없음)
kern_create_aabbs<<<blocks, 1024>>>(...) 실행
cudaDeviceSynchronize()
```

`D_AABBS` / `NUM_AABBS` 전역 캐시 덕분에 Gaussian 수가 줄어들어도 메모리를 해제하지 않고 재사용한다. 매 iteration마다 malloc을 피하는 최적화.

---

## 전체 코드

```cuda
// Copyright 2024 Google LLC (Apache 2.0 License)

#define TRI_PER_G 8
#define PT_PER_G 6

#include "create_aabbs.h"
#include "structs.h"
#include "glm/glm.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

// ─── AABB 계산 GPU 커널 ────────────────────────────────────────────────────────
// 스레드당 Gaussian 1개 처리
__global__ void
kern_create_aabbs(const glm::vec3 *means, const glm::vec3 *scales,
                  const glm::vec4 *quats, const float *densities,
                  const size_t num_prims, OptixAabb *aabbs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 0 || i >= num_prims) return;

    // ─── 쿼터니언 → 회전행렬 ─────────────────────────────────────────────────
    const glm::vec4 quat   = glm::normalize(quats[i]);
    const glm::vec3 center = means[i];
    const glm::vec3 size   = scales[i];

    const float r = quat.x;
    const float x = quat.y;
    const float y = quat.z;
    const float z = quat.w;

    // Rᵀ (column-major 순서)
    const glm::mat3 Rt = {
        1.0 - 2.0*(y*y + z*z),  2.0*(x*y - r*z),        2.0*(x*z + r*y),
        2.0*(x*y + r*z),         1.0 - 2.0*(x*x + z*z),  2.0*(y*z - r*x),
        2.0*(x*z - r*y),         2.0*(y*z + r*x),         1.0 - 2.0*(x*x + y*y)
    };
    const glm::mat3 R = glm::transpose(Rt);

    // ─── M = S·Rᵀ ──────────────────────────────────────────────────────────
    float s = 1.0;
    glm::mat3 S = glm::mat3(1.0);
    S[0][0] = s * size.x;
    S[1][1] = s * size.y;
    S[2][2] = s * size.z;

    glm::mat4 M = glm::mat4(S * Rt);
    M[0][3] = center.x;
    M[1][3] = center.y;
    M[2][3] = center.z;

    // ─── AABB = center ± ‖M[row,:]‖₂ ──────────────────────────────────────
    // 각 축 방향 최대 반지름 = 해당 행의 L2 norm
    aabbs[i] = {
        .minX = (float)(center.x - sqrt(M[0][0]*M[0][0] + M[0][1]*M[0][1] + M[0][2]*M[0][2])),
        .minY = (float)(center.y - sqrt(M[1][0]*M[1][0] + M[1][1]*M[1][1] + M[1][2]*M[1][2])),
        .minZ = (float)(center.z - sqrt(M[2][0]*M[2][0] + M[2][1]*M[2][1] + M[2][2]*M[2][2])),
        .maxX = (float)(center.x + sqrt(M[0][0]*M[0][0] + M[0][1]*M[0][1] + M[0][2]*M[0][2])),
        .maxY = (float)(center.y + sqrt(M[1][0]*M[1][0] + M[1][1]*M[1][1] + M[1][2]*M[1][2])),
        .maxZ = (float)(center.z + sqrt(M[2][0]*M[2][0] + M[2][1]*M[2][1] + M[2][2]*M[2][2])),
    };
}

// ─── create_aabbs(): 메모리 관리 + 커널 launch ──────────────────────────────
void create_aabbs(Primitives &prims)
{
    const size_t block_size = 1024;

    // 이전보다 Gaussian이 늘었을 때만 재할당
    if (prims.prev_alloc_size < prims.num_prims) {
        if (prims.prev_alloc_size > 0) {
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(prims.aabbs)));
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&prims.aabbs),
                              prims.num_prims * sizeof(OptixAabb)));
    }

    kern_create_aabbs<<<
        (prims.num_prims + block_size - 1) / block_size,
        block_size>>>(
            (glm::vec3 *)prims.means,
            (glm::vec3 *)prims.scales,
            (glm::vec4 *)prims.quats,
            prims.densities,
            prims.num_prims,
            prims.aabbs);

    CUDA_SYNC_CHECK();   // GPU 동기화
}
```
