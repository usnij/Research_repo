# Forward.cpp / Forward.h

---

## 핵심 설명

### 역할

C++ OptiX 파이프라인 설정 + 실행을 담당하는 클래스. Python에서 `sp.Forward(otx, ctx.device, ctx.prims, True)`로 생성하면 셰이더를 로딩·컴파일하고, `trace_rays()`로 optixLaunch를 실행한다.

### 연결된 Slang 셰이더 파일

`enable_backward=True`일 때 로드하는 PTX는 아래 Slang 파일들의 컴파일 결과다.

| 셰이더 함수 | Slang 소스 | notes |
|------------|-----------|-------|
| `__raygen__rg_float` | shaders.slang | [03_rg_float.md](03_rg_float.md) |
| `__intersection__ellipsoid` | shaders.slang | [04_intersection_anyhit.md](04_intersection_anyhit.md) |
| `__anyhit__ah` | shaders.slang | [04_intersection_anyhit.md](04_intersection_anyhit.md) |
| `update()` / `SplineState` | spline-machine.slang | [02_update.md](02_update.md) · [01_SplineState.md](01_SplineState.md) |

### Forward::Forward() 생성자 — 5단계 파이프라인 구축

```
1. PTX 로딩
   enable_backward=True  → shaders.slang 컴파일된 ptx_code_file (전체 셰이더)
   enable_backward=False → fast_ptx_code_file (backward 생략, 더 빠름)
   optixModuleCreateFromPTX() → module 객체 생성

2. 프로그램 그룹 등록
   raygen   : __raygen__rg_float    → 픽셀당 루프 전체
   miss     : __miss__ms            → ray가 씬에 hit 없을 때
   hitgroup : __anyhit__ah          → 후보 hit 정렬
              __intersection__ellipsoid → 정밀 ellipsoid 교차 계산

3. 파이프라인 링크
   optixPipelineCreate([raygen, miss, hitgroup]) → pipeline 객체
   optixPipelineSetStackSize() → GPU 스택 크기 설정

4. SBT(Shader Binding Table) 설정
   각 셰이더 그룹마다 cudaMalloc + optixSbtRecordPackHeader + cudaMemcpy
   sbt.raygenRecord, missRecordBase, hitgroupRecordBase 설정

5. Gaussian 파라미터 포인터 연결 (ctx.prims에서)
   params.means.data     = model.means    (float3*)
   params.scales.data    = model.scales   (float3*)
   params.quats.data     = model.quats    (float4*)
   params.densities.data = model.densities (float*)
   params.features.data  = model.features
   → 이 포인터들이 GPU 셰이더에서 전역 배열로 접근됨
```

**ctx.prims가 생성자에 필요한 이유**: 포인터를 `params`에 연결해두어야 이후 `trace_rays()`가 셰이더로 Gaussian 데이터를 전달할 수 있다.

### Forward::trace_rays() — optixLaunch 실행

```
1. params에 렌더링 대상 버퍼 채우기
   - fimage (출력 색상), last_state, last_dirac, tri_collection, iters
   - ray_origins, ray_directions
   - tmin, tmax, max_prim_size, max_iters

2. initialize_density()  →  [initialize_density_cu.md](initialize_density_cu.md)
   - ray origin이 Gaussian 내부에 있는 경우 initial_drgb 설정
   - optixLaunch 전에 호출해서 rg_float()의 state.drgb 초기값 준비

3. cudaMemcpy(d_param ← params)
   → params 구조체를 GPU 메모리로 복사 (셰이더의 SLANG_globalParams에 바인딩)

4. optixLaunch(pipeline, stream, d_param, sbt, num_rays, 1, 1)
   → num_rays개 스레드 launch
   → 각 스레드가 __raygen__rg_float() 실행

5. cudaStreamSynchronize()
   → GPU 작업 완료 대기
```

**ctx.gas가 trace_rays()에 필요한 이유**: `handle = ctx.gas.handle` = BVH traversable handle. optixLaunch 시점에 `params.handle`로 전달해야 BVH를 탐색할 수 있다.

### Params 구조체 역할

셰이더(`rg_float`, `ellipsoid`, `ah`)가 읽는 **전역 파라미터 블록**. `SLANG_globalParams` 이름으로 PTX에 노출된다.

| 필드 | 타입 | 역할 |
|------|------|------|
| `means/scales/quats` | StructuredBuffer | Gaussian 위치·크기·회전 |
| `densities/features` | StructuredBuffer | 밀도·SH 색상 계수 |
| `ray_origins/directions` | StructuredBuffer | 렌더링할 ray 배열 |
| `tri_collection` | StructuredBuffer | forward 이벤트 기록 (backward용) |
| `last_state/last_dirac` | StructuredBuffer | 마지막 SplineState 저장 |
| `handle` | OptixTraversableHandle | BVH 루트 핸들 (GAS) |
| `tmin/tmax/max_iters` | float/size_t | 렌더링 파라미터 |

---

## 전체 코드

### Forward.h — Params 구조체 및 클래스 선언

```cpp
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include "structs.h"

extern unsigned char ptx_code_file[];
extern unsigned char ptx_code_file2[];
extern unsigned char fast_ptx_code_file[];

struct RayGenData {};
struct MissData { float3 bg_color; };
typedef SbtRecord<RayGenData>  RayGenSbtRecord;
typedef SbtRecord<MissData>    MissSbtRecord;

struct HitGroupData {};
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

struct Params
{
    StructuredBuffer<uchar4>      image;
    StructuredBuffer<float4>      fimage;
    StructuredBuffer<uint>        iters;
    StructuredBuffer<uint>        last_face;
    StructuredBuffer<uint>        touch_count;
    StructuredBuffer<float4>      last_dirac;
    StructuredBuffer<SplineState> last_state;
    StructuredBuffer<int>         tri_collection;
    StructuredBuffer<float3>      ray_origins;
    StructuredBuffer<float3>      ray_directions;
    Cam                           camera;

    StructuredBuffer<__half>      half_attribs;

    StructuredBuffer<float3>      means;
    StructuredBuffer<float3>      scales;
    StructuredBuffer<float4>      quats;
    StructuredBuffer<float>       densities;
    StructuredBuffer<float>       features;

    size_t sh_degree;
    size_t max_iters;
    float  tmin;
    float  tmax;
    StructuredBuffer<float4>      initial_drgb;
    float  max_prim_size;
    OptixTraversableHandle        handle;
};

class Forward {
public:
    Forward() = default;
    Forward(const OptixDeviceContext &context, int8_t device,
            const Primitives &model, const bool enable_backward);
    ~Forward() noexcept(false);

    void trace_rays(const OptixTraversableHandle &handle,
                    const size_t num_rays,
                    float3 *ray_origins,
                    float3 *ray_directions,
                    void *image_out,
                    uint sh_degree,
                    float tmin,
                    float tmax,
                    float4 *initial_drgb,
                    Cam *camera=NULL,
                    const size_t max_iters=10000,
                    const float max_prim_size=3,
                    uint *iters=NULL,
                    uint *last_face=NULL,
                    uint *touch_count=NULL,
                    float4 *last_dirac=NULL,
                    SplineState *last_state=NULL,
                    int *tri_collection=NULL,
                    int *d_touch_count=NULL,
                    int *d_touch_inds=NULL);
    void reset_features(const Primitives &model);

    bool enable_backward = false;
    size_t num_prims = 0;

private:
    Params params;
    OptixDeviceContext context = nullptr;
    int8_t device = -1;
    const Primitives *model;
    OptixModule module = nullptr;
    OptixShaderBindingTable sbt = {};
    OptixPipeline pipeline = nullptr;
    CUdeviceptr d_param = 0;
    CUstream stream = nullptr;
    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;
    float eps = 1e-6;

    static std::string load_ptx_data2() { return std::string((char *)ptx_code_file2); }
    static std::string load_ptx_data()  { return std::string((char *)ptx_code_file); }
    static std::string load_fast_ptx_data() { return std::string((char *)fast_ptx_code_file); }
};
```

### Forward.cpp — 생성자 및 trace_rays() 구현

```cpp
// Copyright 2024 Google LLC (Apache 2.0 License)

#include <optix_stack_size.h>
#include <optix_stubs.h>
#include "cuda_util.h"
#include "Forward.h"
#include "CUDABuffer.h"
#include "initialize_density.h"

// ─── trace_rays: optixLaunch 실행 ─────────────────────────────────────────────

void Forward::trace_rays(
    const OptixTraversableHandle &handle,
    const size_t num_rays, float3 *ray_origins,
    float3 *ray_directions, void *image_out, uint sh_deg,
    float tmin, float tmax, float4 *initial_drgb,
    Cam *camera,
    const size_t max_iters,
    const float max_prim_size,
    uint *iters, uint *last_face,
    uint *touch_count,
    float4 *last_dirac, SplineState *last_state,
    int *tri_collection, int *d_touch_count, int *d_touch_inds)
{
    CUDA_CHECK(cudaSetDevice(device));
    {
        params.fimage.data         = (float4 *)image_out;
        params.last_state.data     = last_state;
        params.last_state.size     = num_rays;
        params.last_dirac.data     = last_dirac;
        params.last_dirac.size     = num_rays;
        params.tri_collection.data = tri_collection;
        params.tri_collection.size = num_rays * max_iters;
        params.iters.data          = iters;
        params.iters.size          = num_rays;
        params.last_face.data      = last_face;
        params.last_face.size      = num_rays;
        params.touch_count.data    = touch_count;
        params.sh_degree           = sh_deg;
        params.max_prim_size       = max_prim_size;
        params.max_iters           = max_iters;
        params.ray_origins.data    = ray_origins;
        params.ray_origins.size    = num_rays;
        params.ray_directions.data = ray_directions;
        params.ray_directions.size = num_rays;
        if (camera != NULL)
            params.camera = *camera;
        params.tmin = tmin;
        params.tmax = tmax;

        CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(initial_drgb), 0,
                              num_rays * sizeof(float4)));
        params.initial_drgb.data = initial_drgb;
        params.initial_drgb.size = num_rays;

        initialize_density(&params, model->aabbs, d_touch_count, d_touch_inds);

        params.handle = handle;
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_param), &params,
                              sizeof(params), cudaMemcpyHostToDevice));

        if (camera != NULL) {
            OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt,
                                    camera->width, camera->height, 1));
        } else {
            OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt,
                                    num_rays, 1, 1));
        }
        CUDA_SYNC_CHECK();
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

// ─── Forward::Forward(): OptiX 파이프라인 구축 ────────────────────────────────

Forward::Forward(const OptixDeviceContext &context, int8_t device,
                 const Primitives &model, const bool enable_backward)
    : enable_backward(enable_backward), context(context), device(device), model(&model)
{
    CUDA_CHECK(cudaSetDevice(device));
    char log[2048];
    size_t sizeof_log = sizeof(log);

    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur         = false;
    pipeline_compile_options.traversableGraphFlags  = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues       = 32;   // payload 슬롯 수 (16 hit × 2 = 32)
    pipeline_compile_options.numAttributeValues     = 1;
    pipeline_compile_options.exceptionFlags         = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "SLANG_globalParams";
    pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    // ─── 1. PTX 로딩 ─────────────────────────────────────────────────────────
    OptixModule module = nullptr;
    {
        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        std::string input = enable_backward ? Forward::load_ptx_data()
                                           : Forward::load_fast_ptx_data();
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
            context, &module_compile_options, &pipeline_compile_options,
            input.c_str(), input.size(), log, &sizeof_log, &module));

        std::string input2 = Forward::load_ptx_data2();
    }

    // ─── 2. 프로그램 그룹 등록 ────────────────────────────────────────────────
    {
        OptixProgramGroupOptions program_group_options = {};

        // raygen: __raygen__rg_float
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module           = module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg_float";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &raygen_prog_group_desc,
                                                1, &program_group_options,
                                                log, &sizeof_log, &raygen_prog_group));

        // miss: __miss__ms
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module           = module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &miss_prog_group_desc,
                                                1, &program_group_options,
                                                log, &sizeof_log, &miss_prog_group));

        // hitgroup: __anyhit__ah + __intersection__ellipsoid
        OptixProgramGroupDesc hitgroup_prog_group_desc = {};
        hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleAH          = module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
        hitgroup_prog_group_desc.hitgroup.moduleIS          = module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__ellipsoid";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &hitgroup_prog_group_desc,
                                                1, &program_group_options,
                                                log, &sizeof_log, &hitgroup_prog_group));
    }

    // ─── 3. 파이프라인 링크 ───────────────────────────────────────────────────
    {
        const uint32_t max_trace_depth = 1;
        OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };
        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = max_trace_depth;
        pipeline_link_options.debugLevel    = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        OPTIX_CHECK_LOG(optixPipelineCreate(
            context, &pipeline_compile_options, &pipeline_link_options,
            program_groups, sizeof(program_groups)/sizeof(program_groups[0]),
            log, &sizeof_log, &pipeline));

        OptixStackSizes stack_sizes = {};
        for (auto &prog_group : program_groups)
            OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        OPTIX_CHECK(optixUtilComputeStackSizes(
            &stack_sizes, max_trace_depth, 0, 0,
            &direct_callable_stack_size_from_traversal,
            &direct_callable_stack_size_from_state, &continuation_stack_size));
        OPTIX_CHECK(optixPipelineSetStackSize(
            pipeline,
            direct_callable_stack_size_from_traversal,
            direct_callable_stack_size_from_state,
            continuation_stack_size,
            1));
    }

    // ─── 4. SBT(Shader Binding Table) 설정 ───────────────────────────────────
    {
        // raygen record
        CUdeviceptr raygen_record;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raygen_record), sizeof(RayGenSbtRecord)));
        RayGenSbtRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(raygen_record), &rg_sbt,
                              sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice));

        // miss record
        CUdeviceptr miss_record;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&miss_record), sizeof(MissSbtRecord)));
        MissSbtRecord ms_sbt;
        ms_sbt.data = { 0.3f, 0.1f, 0.2f };
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(miss_record), &ms_sbt,
                              sizeof(MissSbtRecord), cudaMemcpyHostToDevice));

        // hitgroup record
        CUdeviceptr hitgroup_record;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&hitgroup_record), sizeof(HitGroupSbtRecord)));
        HitGroupSbtRecord hg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(hitgroup_record), &hg_sbt,
                              sizeof(HitGroupSbtRecord), cudaMemcpyHostToDevice));

        sbt.raygenRecord              = raygen_record;
        sbt.missRecordBase            = miss_record;
        sbt.missRecordStrideInBytes   = sizeof(MissSbtRecord);
        sbt.missRecordCount           = 1;
        sbt.hitgroupRecordBase        = hitgroup_record;
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        sbt.hitgroupRecordCount       = 1;
    }

    // ─── 5. Gaussian 파라미터 포인터 연결 ─────────────────────────────────────
    {
        params.half_attribs.data  = model.half_attribs;
        params.half_attribs.size  = model.num_prims;

        params.means.data         = (float3 *)model.means;
        params.means.size         = model.num_prims;
        params.scales.data        = (float3 *)model.scales;
        params.scales.size        = model.num_prims;
        params.quats.data         = (float4 *)model.quats;
        params.quats.size         = model.num_prims;

        params.densities.data     = (float *)model.densities;
        params.densities.size     = model.num_prims;
        params.features.data      = model.features;
        params.features.size      = model.num_prims * model.feature_size;

        num_prims = model.num_prims;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_param), sizeof(Params)));
    }
}

// ─── reset_features / Destructor ──────────────────────────────────────────────

void Forward::reset_features(const Primitives &model) {
    params.features.data = model.features;
    params.features.size = model.num_prims * model.feature_size;
}

Forward::~Forward() noexcept(false) {
    if (d_param != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(d_param, 0))));
    if (sbt.raygenRecord != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.raygenRecord, 0))));
    if (sbt.missRecordBase != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.missRecordBase, 0))));
    if (sbt.hitgroupRecordBase != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.hitgroupRecordBase, 0))));
    if (stream != nullptr)
        CUDA_CHECK(cudaStreamDestroy(std::exchange(stream, nullptr)));
    if (pipeline != nullptr)
        OPTIX_CHECK(optixPipelineDestroy(std::exchange(pipeline, nullptr)));
    if (raygen_prog_group != nullptr)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(raygen_prog_group, nullptr)));
    if (miss_prog_group != nullptr)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(miss_prog_group, nullptr)));
    if (hitgroup_prog_group != nullptr)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(hitgroup_prog_group, nullptr)));
    if (module != nullptr)
        OPTIX_CHECK(optixModuleDestroy(std::exchange(module, nullptr)));
}
```
