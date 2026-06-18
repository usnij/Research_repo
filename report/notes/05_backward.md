# backwards_kernel.slang

---

## 핵심 설명

### 전체 구조

forward에서 중간 SplineState 전체를 저장하지 않고, 최소한만 저장한다.

| 저장 데이터 | 용도 |
|------------|------|
| `tri_collection` | 각 ray가 처리한 이벤트 순서 (ray_id × iter 2D 배열) |
| `last_state` | 마지막 SplineState |
| `last_dirac` | 마지막 ControlPoint.dirac |
| `iters` | ray별 처리한 이벤트 수 |

backward는 `tri_collection`을 역순으로 순회하며 `inverse_update_dual()`로 state를 복원하고, `bwd_diff(update)`로 gradient를 계산한다.

### inverse_update_dual() — state 역산

forward `update()`를 해석적으로 역으로 풀어낸다. 저장 없이 `last_state`에서 역방향으로 모든 중간 state를 복원할 수 있다.

```
new_state.drgb = old_state.drgb + ctrl_pt.dirac
∴ old_state.drgb = new_state.drgb - ctrl_pt.dirac   (현재 이벤트 dirac을 빼면 됨)

new_state.logT = old_state.logT + area
∴ old_state.logT = new_state.logT - area             (area = old_state.drgb.x × dt)

new_state.C = old_state.C + weight × rgb_norm
∴ old_state.C = new_state.C - weight × rgb_norm
```

### bwd_diff(update) — Slang 자동미분

Slang이 `[Differentiable]` 어트리뷰트로부터 `bwd_diff(update)`를 자동 생성함.

```slang
bwd_diff(update)(old_deriv_state, deriv_ctrl_pt, tmin, tmax, max_prim_size, deriv_state.d);
```

- 입력: `deriv_state.d` = 현재 스텝의 adjoint (다음 스텝에서 carry된 값)
- 출력: `old_deriv_state.d` = 이전 state adjoint, `deriv_ctrl_pt.d` = ctrl_pt adjoint

`ctrl_pt.dirac`의 adjoint → `bwd_diff(safe_intersect)`로 전달 → mean, scale, quat, density, color의 gradient 계산.

### run_update() — 한 스텝의 역전파

```
1. bwd_diff(update)  → deriv_ctrl_pt.d 계산
2. bwd_diff(safe_intersect) → deriv_mean.d, deriv_scale.d, deriv_quat.d, deriv_density.d
3. bwd_diff(eval_color) → d_feat.d (SH 계수 gradient)
4. atomic_add → model.dL_d* 누적 (여러 ray 동시 접근 가능하므로 atomic 필수)
```

### gradient 누적 패턴

여러 ray가 같은 Gaussian에 동시에 gradient를 쓰므로 `InterlockedAdd`(atomic) 연산 필수.

```slang
atomic_add_float3(model.dL_dmeans,    prim_ind, deriv_mean.d);
atomic_add_float3(model.dL_dscales,   prim_ind, deriv_scales.d);
atomic_add_float4(model.dL_dquats,    prim_ind, deriv_quat.d);
model.dL_ddensities.InterlockedAdd(prim_ind, deriv_density.d, temp);
atomic_add_float3(model.dL_dfeatures, prim_ind, 0u, d_feat.d.f0);
```

### A의 gradient가 두 구간에서 모이는 이유

```
이벤트 순서: [A_entry] [B_entry] [A_exit] [B_exit]
구간:          S1:A만    S2:A+B    S3:B만
```

`drgb.x = σA` (S1구간), `drgb.x = σA + σB` (S2구간). A의 σA가 두 구간 avg.x에 모두 포함되므로, 역전파 시 S1과 S2 모두에서 dL/dσA가 누적된다.

| 역순 스텝 | 닫혔던 구간 | avg.x | A gradient 경로 |
|----------|------------|-------|----------------|
| B_exit   | S3: B만    | σB    | A 없음, adjoint carry |
| A_exit   | S2: A+B   | σA+σB | adj → drgb.x (σA+σB 합) |
| B_entry  | S1: A만    | σA    | adj → drgb.x (순수 σA) |
| A_entry  | 빈공간      | 0     | carry된 adj → dL_dσA 확정 |

---

## 전체 코드

```slang
// Copyright 2024 Google LLC
// (Apache 2.0 License)

#define tri_per_g 2
import spline_machine;
import optix;
import optix_intrinsics;
import tri_intersect;
import safe_math;
import sh;

// ─── 헬퍼 함수들 ───────────────────────────────────────────────────────────────

float3 get_float3(TensorView<float> view, uint ind) {
    return { view[ind, 0], view[ind, 1], view[ind, 2] };
}

float4 get_float4(TensorView<float> view, uint ind) {
    return { view[ind, 0], view[ind, 1], view[ind, 2], view[ind, 3] };
}

Features get_feats(in TensorView<float> features, in uint prim_ind, in uint sh_degree) {
    Features feat;
    feat.f0 = get_float3(features, prim_ind, 0);
    if (sh_degree > 0) {
        feat.f1 = get_float3(features, prim_ind, 1);
        feat.f2 = get_float3(features, prim_ind, 2);
        feat.f3 = get_float3(features, prim_ind, 3);
        if (sh_degree > 1) {
            feat.f4  = get_float3(features, prim_ind, 4);
            feat.f5  = get_float3(features, prim_ind, 5);
            feat.f6  = get_float3(features, prim_ind, 6);
            feat.f7  = get_float3(features, prim_ind, 7);
            feat.f8  = get_float3(features, prim_ind, 8);
            if (sh_degree > 2) {
                feat.f9  = get_float3(features, prim_ind, 9);
                feat.f10 = get_float3(features, prim_ind, 10);
                feat.f11 = get_float3(features, prim_ind, 11);
                feat.f12 = get_float3(features, prim_ind, 12);
                feat.f13 = get_float3(features, prim_ind, 13);
                feat.f14 = get_float3(features, prim_ind, 14);
                feat.f15 = get_float3(features, prim_ind, 15);
            }
        }
    }
    return feat;
}

void atomic_add_float3(TensorView<float> view, uint ind, float3 val) {
    float temp;
    view.InterlockedAdd(uint2(ind, 0u), val.x, temp);
    view.InterlockedAdd(uint2(ind, 1u), val.y, temp);
    view.InterlockedAdd(uint2(ind, 2u), val.z, temp);
}

void atomic_add_float4(TensorView<float> view, uint ind, float4 val) {
    float temp;
    view.InterlockedAdd(uint2(ind, 0u), val.x, temp);
    view.InterlockedAdd(uint2(ind, 1u), val.y, temp);
    view.InterlockedAdd(uint2(ind, 2u), val.z, temp);
    view.InterlockedAdd(uint2(ind, 3u), val.w, temp);
}

void atomic_add_float3(TensorView<float> view, uint ind, uint feat_ind, float3 val) {
    float temp;
    view.InterlockedAdd(uint3(ind, feat_ind, 0u), val.x, temp);
    view.InterlockedAdd(uint3(ind, feat_ind, 1u), val.y, temp);
    view.InterlockedAdd(uint3(ind, feat_ind, 2u), val.z, temp);
}

void atomic_add_float2(TensorView<float> view, uint ind, float2 val) {
    float temp;
    view.InterlockedAdd(uint2(ind, 0u), val.x, temp);
    view.InterlockedAdd(uint2(ind, 1u), val.y, temp);
}

SplineState get_state(TensorView<float> view, uint ind) {
    return {
        float2(view[ind, 0], view[ind, 1]),
        float2(view[ind, 2], view[ind, 3]),
        float3(view[ind, 4], view[ind, 5], view[ind, 6]),
        view[ind, 7],
        float4(view[ind, 8], view[ind, 9], view[ind, 10], view[ind, 11]),
        view[ind, 12],
        float3(view[ind, 13], view[ind, 14], view[ind, 15]),
    };
}

// ─── DualModel: 파라미터 + gradient 버퍼 ─────────────────────────────────────

struct DualModel {
    TensorView<float> means;
    TensorView<float> scales;
    TensorView<float> quats;
    TensorView<float> densities;
    TensorView<float> features;

    TensorView<float> dL_dmeans;
    TensorView<float> dL_dscales;
    TensorView<float> dL_dquats;
    TensorView<float> dL_ddensities;
    TensorView<float> dL_dfeatures;
    TensorView<float> dL_drayos;
    TensorView<float> dL_drayds;
    TensorView<float> dL_dmeans2D;
};

// ─── run_update: 한 스텝 역전파 ──────────────────────────────────────────────

DifferentialPair<SplineState>
run_update(in SplineState old_dual_state,
           in ControlPoint old_ctrl_pt,
           in ControlPoint ctrl_pt,
           in uint prim_ind,
           in uint face_id,
           in uint ray_ind,
           in DifferentialPair<SplineState> deriv_state,
           in float3 origin,
           in float3 direction,
           in float tmin,
           in float tmax,
           in uint sh_degree,
           in float max_prim_size,
           in float4x4 wct,
           in float4x4 inv_wct,
           inout DualModel model)
{
    var old_deriv_state = diffPair(from_dual(old_dual_state, old_ctrl_pt), {});
    var deriv_ctrl_pt = diffPair(ctrl_pt, {});
    bool skip_close = false;

    // 1. bwd_diff(update): update()의 adjoint → old_deriv_state.d, deriv_ctrl_pt.d 계산
    bwd_diff(update)(old_deriv_state, deriv_ctrl_pt, tmin, tmax, max_prim_size, deriv_state.d);

    // 2. Gaussian 파라미터 로드
    let mean    = get_float3(model.means, prim_ind);
    let scale   = get_float3(model.scales, prim_ind);
    let quat    = get_float4(model.quats, prim_ind);
    let density = model.densities[prim_ind];
    Features feat = get_feats(model.features, prim_ind, sh_degree);
    float3 color = eval_color(direction, feat, sh_degree);

    var deriv_origin    = diffPair(origin, {});
    var deriv_direction = diffPair(direction, {});
    var deriv_scales    = diffPair(scale, {});
    var deriv_mean      = diffPair(mean, {});
    var deriv_quat      = diffPair(quat, {});
    var deriv_color     = diffPair(color, {});
    var deriv_density   = diffPair(density, {});

    // 3. bwd_diff(safe_intersect): ctrl_pt adjoint → Gaussian 파라미터 adjoint
    bwd_diff(safe_intersect)(deriv_origin, deriv_direction,
        deriv_scales, deriv_mean, deriv_quat, deriv_color, deriv_density,
        face_id, skip_close, deriv_ctrl_pt.d);

    // 4. Gaussian 파라미터 gradient atomic 누적
    atomic_add_float3(model.dL_dmeans,  prim_ind, deriv_mean.d);
    atomic_add_float3(model.dL_dscales, prim_ind, deriv_scales.d);
    atomic_add_float4(model.dL_dquats,  prim_ind, deriv_quat.d);
    float temp;
    model.dL_ddensities.InterlockedAdd(prim_ind, deriv_density.d, temp);

    // 5. bwd_diff(eval_color): color adjoint → SH feature gradient
    float3 d_rayd = deriv_direction.d;
    deriv_direction = diffPair(direction, {});
    var d_feat = diffPair(feat, {});
    bwd_diff(eval_color)(deriv_direction, d_feat, sh_degree, deriv_color.d);
    d_rayd += deriv_direction.d;

    atomic_add_float3(model.dL_dfeatures, prim_ind, 0u, d_feat.d.f0);
    if (sh_degree > 0) {
        atomic_add_float3(model.dL_dfeatures, prim_ind, 1u, d_feat.d.f1);
        atomic_add_float3(model.dL_dfeatures, prim_ind, 2u, d_feat.d.f2);
        atomic_add_float3(model.dL_dfeatures, prim_ind, 3u, d_feat.d.f3);
        if (sh_degree > 1) {
            atomic_add_float3(model.dL_dfeatures, prim_ind, 4u, d_feat.d.f4);
            atomic_add_float3(model.dL_dfeatures, prim_ind, 5u, d_feat.d.f5);
            atomic_add_float3(model.dL_dfeatures, prim_ind, 6u, d_feat.d.f6);
            atomic_add_float3(model.dL_dfeatures, prim_ind, 7u, d_feat.d.f7);
            atomic_add_float3(model.dL_dfeatures, prim_ind, 8u, d_feat.d.f8);
            if (sh_degree > 2) {
                atomic_add_float3(model.dL_dfeatures, prim_ind,  9u, d_feat.d.f9);
                atomic_add_float3(model.dL_dfeatures, prim_ind, 10u, d_feat.d.f10);
                atomic_add_float3(model.dL_dfeatures, prim_ind, 11u, d_feat.d.f11);
                atomic_add_float3(model.dL_dfeatures, prim_ind, 12u, d_feat.d.f12);
                atomic_add_float3(model.dL_dfeatures, prim_ind, 13u, d_feat.d.f13);
                atomic_add_float3(model.dL_dfeatures, prim_ind, 14u, d_feat.d.f14);
                atomic_add_float3(model.dL_dfeatures, prim_ind, 15u, d_feat.d.f15);
            }
        }
    }

    atomic_add_float3(model.dL_drayos, ray_ind, deriv_origin.d);
    atomic_add_float3(model.dL_drayds, ray_ind, d_rayd);

    // 6. 2D mean gradient (projection 역전파)
    float3 xyd = project(mean, wct);
    var d_xy   = diffPair(float2(xyd.x, xyd.y), {});
    var d_dist = diffPair(xyd.z, {});
    var d_inv_wct = diffPair(inv_wct, {});
    bwd_diff(inv_project)(d_xy, d_dist, d_inv_wct, deriv_mean.d);
    atomic_add_float2(model.dL_dmeans2D, prim_ind, d_xy.d);

    return old_deriv_state;
}

// ─── load_ctrl_pt: tri_ind에서 ControlPoint 재구성 ──────────────────────────

ControlPoint load_ctrl_pt(in uint older_tri_ind, in DualModel model,
                           in float3 origin, float3 direction, uint sh_degree, bool skip_close)
{
    let older_prim_ind = (uint)floor(older_tri_ind / tri_per_g);
    let older_face_id  = mod(older_tri_ind, tri_per_g);

    let older_mean    = get_float3(model.means, older_prim_ind);
    let older_scale   = get_float3(model.scales, older_prim_ind);
    let older_quat    = get_float4(model.quats, older_prim_ind);
    let older_density = model.densities[older_prim_ind];

    Features older_feat = get_feats(model.features, older_prim_ind, sh_degree);
    float3 older_color  = eval_color(direction, older_feat, sh_degree);

    return safe_intersect(origin, direction,
        older_scale, older_mean, older_quat, older_color, older_density, older_face_id, skip_close);
}

// ─── 좌표 변환 헬퍼 ───────────────────────────────────────────────────────────

[Differentiable]
float3 project(in float3 xyz, in float4x4 wct) {
    float4 xyzw = float4(xyz, 1.f);
    let p_view = mul(xyzw, wct);
    float2 pix2d = { safe_div(p_view.x, p_view.z), safe_div(p_view.y, p_view.z) };
    return { pix2d.x, pix2d.y, p_view.z };
}

[Differentiable]
float3 inv_project(in float2 xy, in float dist, float4x4 inv_wvt) {
    let p_hom = float4(xy * dist, dist, 1.f);
    let out = mul(p_hom, inv_wvt);
    return { out.x, out.y, out.z };
}

// ─── backwards_kernel: 메인 backward 커널 ────────────────────────────────────

[AutoPyBindCUDA]
[CUDAKernel]
void backwards_kernel(
    TensorView<float> last_state,
    TensorView<float> last_dirac,
    TensorView<int>   iters,
    TensorView<int>   tri_collection,

    TensorView<float> ray_origins,
    TensorView<float> ray_directions,
    DualModel model,
    TensorView<float> initial_drgb,
    TensorView<float> dL_dinital_drgb,
    TensorView<int32_t> touch_count,

    TensorView<float> dL_doutputs,
    TensorView<float> wcts,

    float tmin,
    float tmax,
    float max_prim_size,
    uint max_iters)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint ray_ind = dispatchIdx.x;
    if (ray_ind >= ray_origins.size(0)) return;

    var dual_state  = get_state(last_state, ray_ind);
    let direction   = get_float3(ray_directions, ray_ind);
    let origin      = get_float3(ray_origins, ray_ind) + tmin * direction;
    bool skip_close = false;

    var deriv_state = diffPair(dual_state, {});

    // dL_doutput: PyTorch에서 전달된 최종 loss gradient
    let dL_dC              = get_float4(dL_doutputs, ray_ind);
    let dL_ddistortion_loss = dL_doutputs[ray_ind, 4];
    SplineOutput.Differential dL_doutput;
    dL_doutput.C              = { dL_dC.x, dL_dC.y, dL_dC.z };
    dL_doutput.depth          = dL_dC.w;
    dL_doutput.distortion_loss = dL_ddistortion_loss;

    uint num_iters = max(min(iters[ray_ind], max_iters), 0);
    if (iters[ray_ind] >= max_iters-1 || iters[ray_ind] <= 0) return;

    var dtmin = diffPair(tmin, {});
    var dtmax = diffPair(tmax, {});
    // extract_color의 역전파로 마지막 state의 adjoint 초기화
    bwd_diff(extract_color)(deriv_state, dtmin, dL_doutput);

    let feature_size = model.features.size(1);
    let sh_degree    = int(sqrt(feature_size)) - 1;

    uint start_id = ray_ind * max_iters;
    let wct_ind = (ray_ind < wcts.size(0)) ? ray_ind : 0;
    float4x4 wct = {
        wcts[wct_ind, 0, 0], wcts[wct_ind, 0, 1], wcts[wct_ind, 0, 2], wcts[wct_ind, 0, 3],
        wcts[wct_ind, 1, 0], wcts[wct_ind, 1, 1], wcts[wct_ind, 1, 2], wcts[wct_ind, 1, 3],
        wcts[wct_ind, 2, 0], wcts[wct_ind, 2, 1], wcts[wct_ind, 2, 2], wcts[wct_ind, 2, 3],
        wcts[wct_ind, 3, 0], wcts[wct_ind, 3, 1], wcts[wct_ind, 3, 2], wcts[wct_ind, 3, 3],
    };
    float4x4 inv_wct = inverse(wct);

    // tri_collection에서 마지막 이벤트 로드
    uint tri_ind = tri_collection[ray_ind + max(num_iters-1, 0) * ray_origins.size(0)];
    ControlPoint ctrl_pt = load_ctrl_pt(tri_ind, model, origin, direction, sh_degree, skip_close);

    // 역순 순회: num_iters-1 → 0
    for (int i=num_iters; i-->0; )
    {
        uint old_tri_ind;
        ControlPoint old_ctrl_pt;
        if (i-1 >= 0) {
            old_tri_ind = tri_collection[ray_ind + (i-1) * ray_origins.size(0)];
            old_ctrl_pt = load_ctrl_pt(old_tri_ind, model, origin, direction, sh_degree, skip_close);
        } else {
            old_ctrl_pt.t    = 0;
            old_ctrl_pt.dirac = { 0.f, 0.f, 0.f, 0.f };
        }

        // 직전 state 해석적 복원
        SplineState old_dual_state = inverse_update_dual(dual_state, ctrl_pt, old_ctrl_pt, tmin, tmax);

        // 한 스텝 역전파 + gradient 누적
        let old_deriv_state = run_update(
            old_dual_state, old_ctrl_pt, ctrl_pt,
            (uint)floor(tri_ind / tri_per_g),
            mod(tri_ind, tri_per_g),
            ray_ind, deriv_state,
            origin, direction, tmin, tmax, sh_degree, max_prim_size,
            wct, inv_wct, model);

        int itemp;
        touch_count.InterlockedAdd((uint)floor(tri_ind / tri_per_g), 1, itemp);

        tri_ind    = old_tri_ind;
        dual_state = old_dual_state;
        ctrl_pt    = old_ctrl_pt;
        deriv_state = diffPair(old_dual_state, old_deriv_state.d);
    }

    // initial_drgb의 gradient (ray origin이 Gaussian 내부인 경우용)
    dL_dinital_drgb[ray_ind, 0u] = deriv_state.d.drgb.x;
    dL_dinital_drgb[ray_ind, 1u] = deriv_state.d.drgb.y;
    dL_dinital_drgb[ray_ind, 2u] = deriv_state.d.drgb.z;
    dL_dinital_drgb[ray_ind, 3u] = deriv_state.d.drgb.w;
}

// ─── backwards_initial_drgb_kernel: ray origin 내부 Gaussian backward ────────

[Differentiable]
float4 mix_drgb(float density, float3 color) {
    return { density, density*color.x, density*color.y, density*color.z };
}

[AutoPyBindCUDA]
[CUDAKernel]
void backwards_initial_drgb_kernel(
    TensorView<float>   ray_origins,
    TensorView<float>   ray_directions,
    DualModel model,
    TensorView<float>   initial_drgb,
    TensorView<int32_t> initial_inds,
    TensorView<float>   dL_dinital_drgb,
    TensorView<int32_t> touch_count,
    float tmin)
{
    uint thread_j = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint thread_i = cudaThreadIdx().y + cudaBlockIdx().y * cudaBlockDim().y;
    if (thread_i >= initial_inds.size(0) || thread_j >= ray_directions.size(0)) return;

    uint prim_ind = initial_inds[thread_i];
    uint ray_ind  = thread_j;

    float3 mean   = get_float3(model.means, prim_ind);
    float4 quat   = get_float4(model.quats, prim_ind);
    float3 scales = get_float3(model.scales, prim_ind);
    float3 rayd   = get_float3(ray_directions, ray_ind);
    float3 rayo   = get_float3(ray_origins, 0) + tmin * rayd;
    float3 clip_scale = max(scales, 1e-8);

    let R = quat2mat(safe_div(quat, length(quat)));
    let Trayo = safe_div(mul(rayo - mean, R), clip_scale);

    if (length(Trayo) <= 1) {  // ray origin이 이 Gaussian 내부에 있음
        float temp;
        let density = model.densities[prim_ind];
        int sh_degree = 0;
        Features feat = get_feats(model.features, prim_ind, sh_degree);
        float3 color  = eval_color(rayd, feat, sh_degree);

        var deriv_color   = diffPair(color, {});
        var deriv_density = diffPair(density, {});
        float4 vdL_dinital_drgb = get_float4(dL_dinital_drgb, ray_ind);
        bwd_diff(mix_drgb)(deriv_density, deriv_color, vdL_dinital_drgb);

        model.dL_ddensities.InterlockedAdd(prim_ind, deriv_density.d, temp);

        var deriv_direction = diffPair(rayd, {});
        var d_feat = diffPair(feat, {});
        bwd_diff(eval_color)(deriv_direction, d_feat, sh_degree, deriv_color.d);
        float3 dfeat = { d_feat.d.f0.x, d_feat.d.f0.y, d_feat.d.f0.z };
        atomic_add_float3(model.dL_dfeatures, prim_ind, 0, dfeat);
    }
}
```
