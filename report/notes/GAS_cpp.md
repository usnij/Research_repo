# GAS.cpp / GAS.h — BVH 빌드

---

## 핵심 설명

### 역할

`ctx.prims.aabbs[]`(Gaussian마다 하나의 AABB)를 입력으로 받아 **BVH(Bounding Volume Hierarchy)** 를 GPU에서 빌드한다. 빌드 결과로 `gas_handle`을 획득하고, 이것이 `optixTrace`에서 씬 탐색의 루트가 된다.

### BVH란

씬의 공간을 계층적으로 나눈 가속 자료구조. ray tracing 시 O(N) 순차 탐색 대신 O(log N)으로 후보 Gaussian을 추려준다. EVER에서는 AABB를 기반으로 BVH를 구성하고, BVH가 추려준 후보에 대해서만 `__intersection__ellipsoid`가 정밀 교차를 계산한다.

### GAS::build() 흐름

```
1. optixAccelComputeMemoryUsage()
   → 이번 빌드에 필요한 temp/output 버퍼 크기 계산

2. 메모리 할당 (전역 캐시 재사용)
   D_GAS_OUTPUT_BUFFER  — BVH 출력 버퍼 (크기 늘 때만 재할당)
   D_TEMP_BUFFER_GAS    — 빌드용 임시 버퍼 (크기 늘 때만 재할당)

3. optixAccelBuild()
   입력: AABB 배열 (model.aabbs, num_prims개)
   출력: gas_handle (BVH traversable handle)

4. Compaction (선택적)
   optixAccelBuild 후 실제 사용 크기(compactedSize)를 측정
   compactedSize < outputSize 이면 D_COMPACT_GAS_BUFFER에 복사
   → 메모리 절약
```

### 전역 캐시 변수

| 변수 | 역할 |
|------|------|
| `D_GAS_OUTPUT_BUFFER` | BVH 결과 저장 버퍼 (GPU) |
| `OUTPUT_BUFFER_SIZE` | 현재 할당 크기 |
| `D_TEMP_BUFFER_GAS` | 빌드 중 임시 버퍼 (GPU) |
| `TEMP_BUFFER_SIZE` | 현재 할당 크기 |
| `D_COMPACT_GAS_BUFFER` | 압축된 BVH 버퍼 (GPU) |
| `COMPACT_GAS_BUFFER_SIZE` | 현재 할당 크기 |

모두 전역 변수로 iteration 간 재사용. 크기가 증가할 때만 `cudaFree` + `cudaMalloc`.

### `gas_handle`

`optixAccelBuild()`가 채워주는 `OptixTraversableHandle`. Python 레이어에서 `ctx.gas.gas_handle`로 접근하며, `Forward::trace_rays()`에 넘겨 `params.handle`로 셰이더에 전달된다. 셰이더에서 `optixTrace(traversable, ...)` 호출 시 이 handle을 씬 루트로 사용한다.

---

## 전체 코드

### GAS.h — 클래스 선언

```cpp
#pragma once
#include <cuda.h>
#include <optix.h>
#include "structs.h"

class GAS {
public:
    OptixTraversableHandle gas_handle = 0;
    OptixTraversableHandle compactedAccelHandle = 0;

    GAS() noexcept;
    GAS(const OptixDeviceContext &context, const uint8_t device,
        const bool enable_backwards, const bool fast_build)
        : device(device), context(context),
          enable_backwards(enable_backwards), fast_build(fast_build) {}

    // 생성자에서 바로 build() 호출
    GAS(const OptixDeviceContext &context, const uint8_t device,
        const Primitives &model,
        const bool enable_backwards=false,
        const bool fast_build=false)
        : GAS(context, device, enable_backwards, fast_build) {
        build(model);
    }

    ~GAS() noexcept(false);
    GAS(const GAS &) = delete;
    GAS &operator=(const GAS &) = delete;
    GAS(GAS &&other) noexcept;
    GAS &operator=(GAS &&other);

    bool defined() const { return gas_handle != 0; }

private:
    void build(const Primitives &model);   // BVH 빌드 핵심 함수
    void release();
    bool enable_backwards, fast_build;
    OptixDeviceContext context = nullptr;
    int8_t device = -1;
};
```

### GAS.cpp — BVH 빌드 구현

```cpp
// Copyright 2024 Google LLC (Apache 2.0 License)

#include <optix_stubs.h>
#include "GAS.h"

// ─── 전역 캐시 버퍼 ───────────────────────────────────────────────────────────
CUdeviceptr D_GAS_OUTPUT_BUFFER    = 0;   size_t OUTPUT_BUFFER_SIZE       = 0;
CUdeviceptr D_TEMP_BUFFER_GAS      = 0;   size_t TEMP_BUFFER_SIZE         = 0;
CUdeviceptr D_COMPACT_GAS_BUFFER   = 0;   size_t COMPACT_GAS_BUFFER_SIZE  = 0;

// ─── GAS::build(): BVH 빌드 ───────────────────────────────────────────────────
void GAS::build(const Primitives &model)
{
    release();
    CUDA_CHECK(cudaSetDevice(device));

    // 1. BVH 빌드 옵션
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
                             | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    // 2. AABB 입력 설정
    uint32_t aabb_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    CUdeviceptr d_aabbs = (CUdeviceptr)model.aabbs;

    OptixBuildInput aabb_input = {};
    aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers   = &d_aabbs;
    aabb_input.customPrimitiveArray.numPrimitives = model.num_prims;
    aabb_input.customPrimitiveArray.flags         = aabb_input_flags;
    aabb_input.customPrimitiveArray.numSbtRecords = 1;

    // 3. 필요 메모리 크기 계산
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context, &accel_options, &aabb_input, 1, &gas_buffer_sizes));

    // 4. 출력/임시 버퍼 재할당 (크기 증가 시에만)
    if (OUTPUT_BUFFER_SIZE <= gas_buffer_sizes.outputSizeInBytes) {
        if (D_GAS_OUTPUT_BUFFER != 0)
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(D_GAS_OUTPUT_BUFFER)));
        OUTPUT_BUFFER_SIZE = gas_buffer_sizes.outputSizeInBytes;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&D_GAS_OUTPUT_BUFFER),
                              OUTPUT_BUFFER_SIZE));
    }
    if (TEMP_BUFFER_SIZE <= gas_buffer_sizes.tempSizeInBytes) {
        if (D_TEMP_BUFFER_GAS != 0)
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(D_TEMP_BUFFER_GAS)));
        TEMP_BUFFER_SIZE = gas_buffer_sizes.tempSizeInBytes;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&D_TEMP_BUFFER_GAS),
                              TEMP_BUFFER_SIZE));
    }

    // 5. Compaction 크기 측정용 버퍼
    size_t *d_compactedSize;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_compactedSize), sizeof(size_t)));
    OptixAccelEmitDesc property = {};
    property.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    property.result = (CUdeviceptr)d_compactedSize;

    // 6. BVH 빌드 실행 → gas_handle 획득
    OPTIX_CHECK(optixAccelBuild(
        context,
        0,                              // CUDA stream
        &accel_options,
        &aabb_input,
        1,                              // num build inputs
        D_TEMP_BUFFER_GAS,  gas_buffer_sizes.tempSizeInBytes,
        D_GAS_OUTPUT_BUFFER, gas_buffer_sizes.outputSizeInBytes,
        &gas_handle,
        &property,
        1                               // num emitted properties
    ));

    // 7. Compaction: 실제 사용 크기가 더 작으면 압축 복사
    size_t compactedSize;
    cudaMemcpy(&compactedSize, d_compactedSize, sizeof(size_t), cudaMemcpyDeviceToHost);

    if (compactedSize < gas_buffer_sizes.outputSizeInBytes) {
        if (COMPACT_GAS_BUFFER_SIZE <= compactedSize) {
            if (D_COMPACT_GAS_BUFFER != 0)
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(D_COMPACT_GAS_BUFFER)));
            COMPACT_GAS_BUFFER_SIZE = compactedSize;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&D_COMPACT_GAS_BUFFER),
                                  COMPACT_GAS_BUFFER_SIZE));
            OPTIX_CHECK(optixAccelCompact(context, 0, gas_handle,
                                          D_COMPACT_GAS_BUFFER,
                                          COMPACT_GAS_BUFFER_SIZE, &gas_handle));
        }
    }
}

// ─── 기타 ────────────────────────────────────────────────────────────────────
GAS::GAS() noexcept : device(-1), context(nullptr), gas_handle(0) {}

GAS::GAS(GAS &&other) noexcept
    : device(std::exchange(other.device, -1)),
      context(std::exchange(other.context, nullptr)),
      gas_handle(std::exchange(other.gas_handle, 0)) {}

void GAS::release() { gas_handle = 0; }

GAS::~GAS() noexcept(false) {
    if (this->device != -1) release();
    std::exchange(this->device, -1);
}
```
