# SplineState / ControlPoint — spline-machine.slang

---

## 핵심 설명

### 연결된 파일

| 방향 | 파일 | 설명 |
|------|------|------|
| 선언된 곳 | spline-machine.slang (이 파일) | — |
| 사용하는 곳 (forward) | [03_rg_float.md](03_rg_float.md) | rg_float()에서 make_empty_state()로 생성 |
| 갱신 함수 | [02_update.md](02_update.md) | update(state, ctrl_pt) → 새 SplineState 반환 |
| 역산 함수 | [05_backward.md](05_backward.md) | inverse_update_dual()로 역방향 복원 |
| ControlPoint 생성 | [04_intersection_anyhit.md](04_intersection_anyhit.md) | get_ctrl_pt(tri, t)에서 생성 |

### SplineState — ray 하나의 렌더링 누적 상태

각 픽셀(스레드)이 `rg_float()` 루프를 돌면서 유지하는 **running accumulator**.
이벤트를 처리할 때마다 `update()`를 거쳐 갱신된다.

| 필드 | 타입 | 역할 |
|------|------|------|
| `drgb` | float4 | **핵심**. x = 현재 겹친 Gaussian들의 σ 합계, yzw = σ·color 합계. entry/exit 이벤트마다 +/- 됨 |
| `logT` | float | 누적 log-transmittance = Σ(σ·Δt). `LOG_CUTOFF(5.54)` 초과 시 루프 종료 |
| `C` | float3 | 누적 색상. 최종 픽셀 색상 |
| `t` | float | 마지막으로 처리한 이벤트의 t. 다음 `optixTrace`의 `start_t`가 됨 |
| `distortion_parts` | float2 | distortion loss 계산용 두 항 분리 저장 |
| `cum_sum` | float2 | distortion loss 계산용 weight 누적합 |
| `padding[0]` | float | 누적 depth |

**`drgb`가 explicit active list를 대체한다**: Gaussian이 entry되면 `drgb += dirac`, exit되면 `drgb -= dirac`. 이 running sum이 항상 "지금 이 위치에서 겹쳐있는 Gaussian들의 밀도 합"을 나타낸다.

### ControlPoint — 하나의 이벤트

ray와 ellipsoid의 교점 한 개를 나타냄. entry(진입) 또는 exit(탈출).

```
tri = 2 × prim_id + 1  →  entry  →  dirac = (+σ, +σ·c)
tri = 2 × prim_id + 0  →  exit   →  dirac = (-σ, -σ·c)
```

`dirac.x` = ±σ (밀도), `dirac.yzw` = ±σ·color (밀도 가중 색상, premultiplied)

### make_empty_state() — 초기 상태

루프 시작 전 모든 필드를 0으로 초기화. `state.drgb = initial_drgb[idx.x]`로 ray origin이 Gaussian 내부에 있는 경우 처리.

---

## 전체 코드

```slang
// spline-machine.slang

#define EPS 1e-18
import safe_math;
#define PRE_MULTI 1000
#define LADDER_P -0.1

struct SplineState : IDifferentiable
{
  float2 distortion_parts;
  float2 cum_sum;
  float3 padding;
  // Spline state
  float t;
  float4 drgb;

  // Volume Rendering State
  float logT;
  float3 C;
};

SplineState make_empty_state() {
  return {
    float2(0.f),
    float2(0.f),
    float3(0.f),
    0.f,
    float4(0.0f),
    0.0,
    float3(0.0f),
  };
}

struct ControlPoint : IDifferentiable
{
  float t;
  float4 dirac;
}

SplineState to_dual(in SplineState state, in ControlPoint ctrl_pt)
{
  SplineState dual_state = state;
  return dual_state;
}

SplineState from_dual(in SplineState state, in ControlPoint ctrl_pt)
{
  SplineState dual_state = state;
  return dual_state;
}

SplineState inverse_update_dual(
    in SplineState new_state,
    in ControlPoint new_ctrl_pt,
    in ControlPoint ctrl_pt,
    in float t_min,
    in float t_max)
{
  const float t = ctrl_pt.t;
  const float dt = max(new_state.t - t, 0.f);

  SplineState state = {};
  state.drgb = new_state.drgb - new_ctrl_pt.dirac;
  state.t = t;

  float4 drgb = state.drgb;
  let avg = drgb;
  float area = max(avg.x * dt, 0.f);
  let rgb_norm = safe_div(float3(avg.y, avg.z, avg.w), avg.x);

  state.logT = max(new_state.logT - area, 0.f);
  const float weight = clip((1-safe_exp(-area)) * safe_exp(-state.logT), 0.f, 1.f);
  state.C = new_state.C - weight * rgb_norm;

  // depth 역산 (생략)
  // distortion 역산 (생략)

  return state;
}

[Differentiable]
SplineState update(
    in SplineState state,
    in ControlPoint ctrl_pt,
    no_diff in float t_min,
    no_diff in float t_max,
    no_diff in float max_prim_size)
{
  const float t = ctrl_pt.t;
  const float dt = max(t - state.t, 0.f);

  SplineState new_state;
  new_state.drgb = state.drgb + ctrl_pt.dirac;
  new_state.t = t;

  float4 drgb = state.drgb;
  let avg = drgb;
  let area = max(avg.x * dt, 0.f);
  let rgb_norm = safe_div(float3(avg.y, avg.z, avg.w), avg.x);

  new_state.logT = max(area + state.logT, 0.f);
  float alpha = -safe_expm1(-area);
  const float weight = clip(alpha * safe_exp(-state.logT), 0.f, 1.f);
  new_state.C = state.C + weight * rgb_norm;

  // depth, distortion 계산 생략

  return new_state;
}

struct SplineOutput: IDifferentiable {
  float3 C;
  float depth;
  float distortion_loss;
};

[Differentiable]
SplineOutput extract_color(in SplineState state, in float tmin) {
  return {
    state.C,
    state.padding[0],
    state.distortion_parts.x - state.distortion_parts.y
  };
}
```
