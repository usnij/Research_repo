# __raygen__rg_float() — shaders.slang

---

## 핵심 설명

### 역할

`optixLaunch(num_rays, 1, 1)` 호출 시 **픽셀당 스레드 하나**가 이 함수를 실행한다. ray 하나의 전체 렌더링 루프를 담당.

### 전역 배열 출처 (Params struct → Forward.cpp)

셰이더 안의 전역 배열들은 모두 `SLANG_globalParams`를 통해 전달된다. C++ `Params` 구조체에 연결된 포인터들이다. → [Forward_cpp.md](Forward_cpp.md)

| 셰이더 전역 | Params 필드 | C++ 설정 위치 |
|------------|------------|-------------|
| `ray_origins[]` | `params.ray_origins.data` | `Forward::trace_rays()` |
| `ray_directions[]` | `params.ray_directions.data` | `Forward::trace_rays()` |
| `tri_collection[]` | `params.tri_collection.data` | `Forward::trace_rays()` |
| `fimage[]` | `params.fimage.data` | `Forward::trace_rays()` |
| `last_state[]` | `params.last_state.data` | `Forward::trace_rays()` |
| `traversable` | `params.handle` = `ctx.gas.gas_handle` | [GAS_cpp.md](GAS_cpp.md) |
| `initial_drgb[]` | `params.initial_drgb.data` | [initialize_density_cu.md](initialize_density_cu.md) |
| `means[]`, `scales[]` 등 | `params.means.data`, ... | `Forward::Forward()` 생성자 |

### initial_drgb의 의미

`state.drgb = initial_drgb[idx.x]` — ray origin이 Gaussian 내부에 있을 때 그 Gaussian의 밀도가 이미 running sum에 포함되어야 한다. `optixLaunch` 직전에 `initialize_density()`([initialize_density_cu.md](initialize_density_cu.md))가 이 값을 채운다.

### 연결된 파일

| 호출 | 파일 | 설명 |
|------|------|------|
| `SplineState state` | [01_SplineState.md](01_SplineState.md) | 자료구조 선언 (spline-machine.slang) |
| `update(state, ctrl_pt)` | [02_update.md](02_update.md) | 볼륨 렌더링 색상 누적 (spline-machine.slang) |
| `get_ctrl_pt(tri, t)` / `optixTrace` | [04_intersection_anyhit.md](04_intersection_anyhit.md) | ellipsoid 교차 + payload 정렬 (shaders.slang) |
| `initial_drgb[idx.x]` | [initialize_density_cu.md](initialize_density_cu.md) | ray origin 내부 Gaussian 초기화 (initialize_density.cu) |
| `traversable` (BVH handle) | [GAS_cpp.md](GAS_cpp.md) | BVH 빌드 결과 (GAS.cpp) |
| 전역 버퍼 전체 | [Forward_cpp.md](Forward_cpp.md) | Params struct 통해 전달 (Forward.cpp) |

### 스트리밍 구조

한 번의 `optixTrace`로 전체 씬을 처리하지 않고, **16개씩 배치로 스트리밍**한다.

```
1회차: optixTrace(start_t=0.0  ~ tmax) → 최대 16개 hit 수집 → 16개 update()
2회차: optixTrace(start_t=t_16 ~ tmax) → 최대 16개 hit 수집 → 16개 update()
3회차: ...
```

`start_t`가 매 루프마다 `abs(state.t)`로 갱신됨 = 이전 루프 마지막 이벤트 위치부터 재탐색.

### 스트리밍이 전구간 처리와 수학적으로 등가인 이유

`state`가 루프 간 **carry**되기 때문. `drgb`, `logT`, `C`, `t` 모두 루프를 넘어 유지되므로, 16개 단위로 나눠 처리하든 한 번에 처리하든 누적 결과가 동일하다.

### tri_collection 기록 (backward용)

```slang
tri_collection[idx.x + iter * dim.x] = tri;
```

ray 인덱스 × 이터레이션 인덱스의 2D 배열. forward에서 처리한 이벤트 순서를 저장해두고, backward에서 역순으로 재생한다.

### 종료 조건

| 조건 | 의미 |
|------|------|
| `state.logT >= LOG_CUTOFF (5.54)` | transmittance < 0.004, 충분히 불투명해서 더 볼 필요 없음 |
| `iter >= max_iters (400)` | 최대 반복 도달 (최대 6400개 이벤트/픽셀) |
| `ctrl_pt.t > 1e9` | payload 빈 슬롯 = 씬 안에 더 이상 hit 없음 |

### payload 구조

```
payload[0]  = t값 (가장 작은 t)    payload[1]  = tri_id
payload[2]  = t값                  payload[3]  = tri_id
...
payload[30] = t값 (가장 큰 t)      payload[31] = tri_id
```

짝수 슬롯 = t값, 홀수 슬롯 = tri_id. 초기값 `1e10f`는 "빈 슬롯" 의미.

---

## 전체 코드

```slang
#define RT_EPS 0
#define tri_per_g 2
#define LOG_CUTOFF 5.54
#define BUFFER_SIZE 16

[shader("raygeneration")]
void rg_float()
{
    let FAST_MODE = false;

    const uint3 idx = DispatchRaysIndex();
    const uint3 dim = DispatchRaysDimensions();
    float3 direction = l2_normalize(ray_directions[idx.x]);
    float3 origin = ray_origins[idx.x] + tmin * ray_directions[idx.x];

    SplineState state = make_empty_state();
    state.t = 0;
    state.drgb = initial_drgb[idx.x];   // ray origin 내부 Gaussian 처리값

    let start_id = idx.x * max_iters;

    ControlPoint ctrl_pt = {};
    uint last_tri = -1;
    float prev_t = state.t;
    uint next_tri = -1;
    float next_t = 1e20;

    uint tri;

    int iter = 0;
    while (state.logT < LOG_CUTOFF && iter < max_iters)
    {
        let start_t = abs(state.t);                  // 이전 루프 마지막 t부터 재시작

        uint payload[2*BUFFER_SIZE];
        for (int i=0; i<BUFFER_SIZE; i++) {
            payload[2*i] = asuint(1e10f);            // 짝수 슬롯 초기화 (빈 슬롯 = 1e10)
        }

        optixTraceP32(
                traversable,
                origin,
                direction,
                start_t,
                tmax,
                payload);

        bool end = false;
        for (int i=0; i<BUFFER_SIZE; i++) {
            ctrl_pt.t = asfloat(payload[2*i]);
            tri = payload[2*i+1];
            if (ctrl_pt.t > 1e9) {
                end = true;
                break;
            }
            ctrl_pt = get_ctrl_pt(tri, ctrl_pt.t);
            state = update(state, ctrl_pt, tmin, tmax, max_prim_size);
            touch_count[tri / tri_per_g]++;
            tri_collection[idx.x + iter * dim.x] = tri;   // backward용 이벤트 기록
            iter++;
            if (!(state.logT < LOG_CUTOFF && iter < max_iters)) break;
        }
        if (end) break;
    }

    let output = extract_color(state, tmin);
    fimage[idx.x] = {output.C.x, output.C.y, output.C.z, output.depth};

    // backward용 저장
    let dual_state = to_dual(state, ctrl_pt);
    last_state[idx.x] = dual_state;
    last_dirac[idx.x] = ctrl_pt.dirac;
    last_face[idx.x] = last_tri;
    iters[idx.x] = iter;
}
```
