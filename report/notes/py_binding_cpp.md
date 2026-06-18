# py_binding.cpp — pybind11 C++ 바인딩

---

## 핵심 설명

### 역할

Python(`sp.Primitives`, `sp.GAS`, `sp.Forward`)에서 호출하는 C++ 클래스들이 여기서 정의된다. pybind11을 통해 Python ↔ C++ 브릿지를 담당한다.

### 구조체 맵핑표

| Python 이름 | C++ 구조체 | 내부 핵심 멤버 |
|------------|-----------|--------------|
| `sp.OptixContext(device)` | `fesOptixContext` | `OptixDeviceContext context` — OptiX 전역 컨텍스트 |
| `sp.Primitives(device)` | `fesPyPrimitives` | `Primitives model` — Gaussian 파라미터 포인터 + AABB 배열 |
| `sp.GAS(otx, device, prims, ...)` | `fesPyGas` | `GAS gas` — BVH 트리 + `gas_handle` |
| `sp.Forward(otx, device, prims, enable_bwd)` | `fesPyForward` | `Forward forward` — OptiX 파이프라인 |

### fesPyPrimitives::add_primitives() 흐름

```
Python: ctx.prims.add_primitives(mean, scale, quat, half_attribs, density, color)
│
├─ torch.Tensor → raw GPU 포인터 변환
│  model.means    = (float3*)means.data_ptr()
│  model.scales   = (float3*)scales.data_ptr()
│  model.quats    = (float4*)quats.data_ptr()
│  model.densities = (float*)densities.data_ptr()
│  model.features  = (float*)colors.data_ptr()
│
├─ GPU 메모리 캐시 연결
│  model.prev_alloc_size = NUM_AABBS   ← 이전 할당 크기
│  model.aabbs = D_AABBS               ← 전역 캐시 포인터
│
├─ create_aabbs(model)                  ← AABB GPU 커널 호출
│       → 새 Gaussian 수 > 이전 할당 크기면 cudaMalloc, 아니면 재사용
│       → kern_create_aabbs<<<...>>>() 실행
│
└─ D_AABBS = model.aabbs               ← 캐시 포인터 업데이트
   NUM_AABBS = max(num_prims, NUM_AABBS)
```

**`D_AABBS` / `NUM_AABBS`**: 전역 변수로 AABB GPU 메모리를 캐싱한다. Gaussian 수가 이전보다 줄어도 메모리를 해제하지 않고 재사용해서 반복적인 malloc을 피한다.

### fesPyForward::trace_rays() — pybind11 레이어

Python에서 `ctx.forward.trace_rays(ctx.gas, rayo, rayd, ...)`를 호출하면:
1. torch.Tensor → raw 포인터 변환
2. `fesSavedForBackward` 버퍼 할당 (states, diracs, iters, faces, touch_count)
3. `Forward::trace_rays(gas.gas.gas_handle, ...)` 호출
4. 결과를 `py::dict`로 패키징해서 Python으로 반환

### fesSavedForBackward — backward 버퍼

`Forward::trace_rays()` 실행 중 GPU가 채우는 backward용 버퍼들.

| 버퍼 | 크기 | 내용 |
|------|------|------|
| `states` | (num_rays, 16) float32 | 마지막 SplineState (float 16개) |
| `diracs` | (num_rays, 4) float32 | 마지막 ControlPoint.dirac |
| `iters` | (num_rays,) int32 | ray별 처리 이벤트 수 |
| `faces` | (num_rays,) int32 | 마지막 tri_id |
| `touch_count` | (num_prims,) int32 | Gaussian별 hit 횟수 |

---

## 전체 코드

```cpp
// Copyright 2024 Google LLC (Apache 2.0 License)

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <optix.h>
#include <optix_stubs.h>
#include "Forward.h"
#include "GAS.h"
#include "create_aabbs.h"

namespace py = pybind11;
using namespace pybind11::literals;

// ─── 입력 검증 매크로 ─────────────────────────────────────────────────────────
#define CHECK_CUDA(x)        TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_DEVICE(x)      TORCH_CHECK(x.device() == this->device, #x " must be on the same device")
#define CHECK_CONTIGUOUS(x)  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x)       TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must have float32 type")
#define CHECK_INPUT(x)       CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT_DIM3(x)  CHECK_INPUT(x); CHECK_DEVICE(x); CHECK_FLOAT(x); \
                             TORCH_CHECK(x.size(-1) == 3, #x " must have last dimension with size 3")
#define CHECK_FLOAT_DIM4(x)  CHECK_INPUT(x); CHECK_DEVICE(x); CHECK_FLOAT(x); \
                             TORCH_CHECK(x.size(-1) == 4, #x " must have last dimension with size 4")

// ─── AABB GPU 메모리 전역 캐시 ─────────────────────────────────────────────────
OptixAabb *D_AABBS = 0;
size_t NUM_AABBS = 0;

// ─── fesOptixContext: OptiX 초기화 ────────────────────────────────────────────
struct fesOptixContext {
public:
    OptixDeviceContext context = nullptr;
    uint device;
    fesOptixContext(const torch::Device &device) : device(device.index()) {
        CUDA_CHECK(cudaSetDevice(device.index()));
        CUDA_CHECK(cudaFree(0));               // CUDA 초기화
        OPTIX_CHECK(optixInit());              // OptiX API 초기화
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;
        CUcontext cuCtx = 0;
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
    }
    ~fesOptixContext() { OPTIX_CHECK(optixDeviceContextDestroy(context)); }
private:
    static void context_log_cb(unsigned int level, const char *tag, const char *message, void *) {}
};

// ─── fesPyPrimitives: Gaussian 파라미터 + AABB ────────────────────────────────
struct fesPyPrimitives {
public:
    Primitives model;
    torch::Device device;
    fesPyPrimitives(const torch::Device &device) : device(device) {}

    void add_primitives(const torch::Tensor &means, const torch::Tensor &scales,
                        const torch::Tensor &quats, const torch::Tensor half_attribs,
                        const torch::Tensor &densities, const torch::Tensor &colors) {
        const int64_t numPrimitives = means.size(0);
        CHECK_FLOAT_DIM3(means);
        CHECK_FLOAT_DIM3(scales);
        CHECK_FLOAT_DIM4(quats);
        CHECK_FLOAT_DIM3(colors);
        TORCH_CHECK(colors.size(2) == 3, "Features must have 3 channels. (N, d, 3)")

        model.feature_size  = colors.size(1);
        model.half_attribs  = reinterpret_cast<half *>(half_attribs.data_ptr<torch::Half>());
        model.means         = reinterpret_cast<float3 *>(means.data_ptr());
        model.scales        = reinterpret_cast<float3 *>(scales.data_ptr());
        model.quats         = reinterpret_cast<float4 *>(quats.data_ptr());
        model.densities     = reinterpret_cast<float *>(densities.data_ptr());
        model.features      = reinterpret_cast<float *>(colors.data_ptr());
        model.num_prims     = numPrimitives;

        // AABB 캐시 연결 후 GPU 커널 호출
        model.prev_alloc_size = NUM_AABBS;
        model.aabbs           = D_AABBS;
        create_aabbs(model);       // ← create_aabbs.cu
        D_AABBS   = model.aabbs;
        NUM_AABBS = std::max(model.num_prims, NUM_AABBS);
    }

    void set_features(const torch::Tensor &colors) {
        CHECK_FLOAT_DIM3(colors);
        model.features = reinterpret_cast<float *>(colors.data_ptr());
    }
};

// ─── fesPyGas: BVH 트리 ───────────────────────────────────────────────────────
struct fesPyGas {
public:
    GAS gas;
    fesPyGas(const fesOptixContext &context, const torch::Device &device,
             const fesPyPrimitives &model, const bool enable_anyhit,
             const bool fast_build, const bool enable_rebuild)
        : gas(context.context, device.index(), model.model, enable_anyhit, fast_build) {}
};

// ─── fesSavedForBackward: backward 버퍼 ───────────────────────────────────────
struct fesSavedForBackward {
public:
    torch::Tensor states, diracs, faces, touch_count, iters;
    size_t num_prims, num_rays;
    size_t num_float_per_state;
    torch::Device device;

    fesSavedForBackward(torch::Device device)
        : num_prims(0), num_rays(0),
          num_float_per_state(sizeof(SplineState) / sizeof(float)),
          device(device) {}

    fesSavedForBackward(size_t num_rays, size_t num_prims, torch::Device device)
        : num_prims(num_prims),
          num_float_per_state(sizeof(SplineState) / sizeof(float)),
          device(device) { allocate(num_rays); }

    // raw 포인터 접근자
    uint        *iters_data_ptr()       { return reinterpret_cast<uint *>(iters.data_ptr()); }
    uint        *touch_count_data_ptr() { return reinterpret_cast<uint *>(touch_count.data_ptr()); }
    uint        *faces_data_ptr()       { return reinterpret_cast<uint *>(faces.data_ptr()); }
    float4      *diracs_data_ptr()      { return reinterpret_cast<float4 *>(diracs.data_ptr()); }
    SplineState *states_data_ptr()      { return reinterpret_cast<SplineState *>(states.data_ptr()); }

    // Python 반환용
    torch::Tensor get_states()      { return states; }
    torch::Tensor get_diracs()      { return diracs; }
    torch::Tensor get_faces()       { return faces; }
    torch::Tensor get_iters()       { return iters; }
    torch::Tensor get_touch_count() { return touch_count; }

    void allocate(size_t num_rays) {
        states      = torch::zeros({(long)num_rays, (long)num_float_per_state},
                                   torch::device(device).dtype(torch::kFloat32));
        diracs      = torch::zeros({(long)num_rays, 4},
                                   torch::device(device).dtype(torch::kFloat32));
        faces       = torch::zeros({(long)num_rays},
                                   torch::device(device).dtype(torch::kInt32));
        touch_count = torch::zeros({(long)num_prims},
                                   torch::device(device).dtype(torch::kInt32));
        iters       = torch::zeros({(long)num_rays},
                                   torch::device(device).dtype(torch::kInt32));
        this->num_rays = num_rays;
    }
};

// ─── fesPyForward: OptiX 파이프라인 래퍼 ─────────────────────────────────────
struct fesPyForward {
public:
    Forward forward;
    torch::Device device;
    size_t num_prims;
    uint sh_degree;

    fesPyForward(const fesOptixContext &context, const torch::Device &device,
                 const fesPyPrimitives &model, const bool enable_backward)
        : device(device),
          forward(context.context, device.index(), model.model, enable_backward),
          num_prims(model.model.num_prims),
          sh_degree(sqrt(model.model.feature_size) - 1) {}

    void update_model(const fesPyPrimitives &model) {
        forward.reset_features(model.model);
    }

    py::dict trace_rays(const fesPyGas &gas,
                        const torch::Tensor &ray_origins,
                        const torch::Tensor &ray_directions,
                        float tmin, float tmax,
                        const size_t max_iters,
                        const float max_prim_size) {
        torch::AutoGradMode enable_grad(false);
        CHECK_FLOAT_DIM3(ray_origins);
        CHECK_FLOAT_DIM3(ray_directions);
        const size_t num_rays = ray_origins.numel() / 3;

        // 출력 버퍼 할당
        torch::Tensor color = torch::zeros({(long)num_rays, 4},
            torch::device(device).dtype(torch::kFloat32));
        torch::Tensor tri_collection = torch::zeros({(long)(num_rays * max_iters)},
            torch::device(device).dtype(torch::kInt32));
        torch::Tensor initial_drgb = torch::zeros({(long)num_rays, 4},
            torch::device(device).dtype(torch::kFloat32));
        torch::Tensor initial_touch_count = torch::zeros({1},
            torch::device(device).dtype(torch::kInt32));
        torch::Tensor initial_touch_inds = torch::zeros({(long)num_prims},
            torch::device(device).dtype(torch::kInt32));

        fesSavedForBackward saved_for_backward(num_rays, num_prims, device);

        // C++ Forward::trace_rays() 호출
        forward.trace_rays(
            gas.gas.gas_handle, num_rays,
            reinterpret_cast<float3 *>(ray_origins.data_ptr()),
            reinterpret_cast<float3 *>(ray_directions.data_ptr()),
            reinterpret_cast<void *>(color.data_ptr()),
            sh_degree, tmin, tmax,
            reinterpret_cast<float4 *>(initial_drgb.data_ptr()),
            NULL,
            max_iters, max_prim_size,
            saved_for_backward.iters_data_ptr(),
            saved_for_backward.faces_data_ptr(),
            saved_for_backward.touch_count_data_ptr(),
            saved_for_backward.diracs_data_ptr(),
            saved_for_backward.states_data_ptr(),
            reinterpret_cast<int *>(tri_collection.data_ptr()),
            reinterpret_cast<int *>(initial_touch_count.data_ptr()),
            reinterpret_cast<int *>(initial_touch_inds.data_ptr()));

        return py::dict(
            "color"_a              = color,
            "saved"_a              = saved_for_backward,
            "tri_collection"_a     = tri_collection,
            "initial_drgb"_a       = initial_drgb,
            "initial_touch_inds"_a = initial_touch_inds,
            "initial_touch_count"_a = initial_touch_count);
    }
};

// ─── pybind11 모듈 등록 ───────────────────────────────────────────────────────
PYBIND11_MODULE(fast_ellipsoid_splinetracer_cpp_extension, m) {
    py::class_<fesOptixContext>(m, "OptixContext")
        .def(py::init<const torch::Device &>());

    py::class_<fesSavedForBackward>(m, "SavedForBackward")
        .def_property_readonly("states",      &fesSavedForBackward::get_states)
        .def_property_readonly("diracs",      &fesSavedForBackward::get_diracs)
        .def_property_readonly("touch_count", &fesSavedForBackward::get_touch_count)
        .def_property_readonly("iters",       &fesSavedForBackward::get_iters)
        .def_property_readonly("faces",       &fesSavedForBackward::get_faces);

    py::class_<fesPyPrimitives>(m, "Primitives")
        .def(py::init<const torch::Device &>())
        .def("add_primitives", &fesPyPrimitives::add_primitives)
        .def("set_features",   &fesPyPrimitives::set_features);

    py::class_<fesPyGas>(m, "GAS")
        .def(py::init<const fesOptixContext &, const torch::Device &,
                      const fesPyPrimitives &, const bool, const bool, const bool>());

    py::class_<fesPyForward>(m, "Forward")
        .def(py::init<const fesOptixContext &, const torch::Device &,
                      const fesPyPrimitives &, const bool>())
        .def("trace_rays",    &fesPyForward::trace_rays)
        .def("update_model",  &fesPyForward::update_model);
}
```
