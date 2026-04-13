# 3DGUT 렌더링 파이프라인 분석 보고서

---

## 1. Ray-Gaussian 교차: t1, t2 계산

### 1.1 개요

3DGUT에서 각 Gaussian은 3D 타원체(ellipsoid)로 표현된다. ray가 Gaussian에 hit 판정을 받은 뒤, 해당 Gaussian의 타원체와 ray의 **정확한 entry/exit depth** `t1`, `t2`를 구하는 과정이다.

- `t1`: ray가 타원체에 진입하는 시점 (world space t)
- `t2`: ray가 타원체에서 빠져나오는 시점 (world space t)

구현 위치: `threedgut_tracer/include/3dgut/kernels/cuda/renderers/gutKBufferRenderer.cuh`, line 821–844

---

### 1.2 수학적 원리

#### Canonical Space 변환

Gaussian 타원체는 position **μ**, rotation **R**, scale **s** (반경)로 정의된다. 이 타원체를 **단위 구(unit sphere)**로 변환하는 canonical space 변환을 적용한다:

```
T_canonical: x_c = S^{-1} R^T (x - μ)
```

여기서 `S = diag(s)`. 이 변환 하에 타원체는 단위 구 `|x_c| = 1`이 된다.

#### Ray의 canonical space 표현

world space의 ray `r(t) = o + t * d`를 canonical space로 변환:

```
d_c_unnorm = S^{-1} R^T d          (canonical 방향, 미정규화)
o_c        = S^{-1} R^T (o - μ)    (canonical 원점)
```

canonical space에서의 "ray 속도" (world t에 대한 canonical 거리 변화율):
```
grduLen = |d_c_unnorm|
d_c     = d_c_unnorm / grduLen      (정규화된 canonical 방향)
```

#### 단위 구 교차 방정식

정규화된 canonical ray `r_c(s) = o_c + s * d_c`와 단위 구 교차:

```
|o_c + s * d_c|² = 1
|o_c|² + 2s (d_c · o_c) + s² = 1
s² + 2hs + (|o_c|² - 1) = 0        where h = d_c · o_c
```

판별식:
```
disc = h² - (|o_c|² - 1)
```

canonical parameter `s`의 해:
```
s = -h ± sqrt(disc)
```

#### World space t로 변환

canonical parameter `s`와 world space `t`의 관계: `s = t * grduLen`, 따라서:

```
t1 = (-h - sqrt(disc)) / grduLen    (entry)
t2 = (-h + sqrt(disc)) / grduLen    (exit)
```

---

### 1.3 CUDA 구현

```cpp
// gutKBufferRenderer.cuh, line 826-844
const tcnn::vec3 giscl  = tcnn::vec3(1.f) / particles.scale(particleData.densityParameters);
const tcnn::mat3 rotT   = tcnn::transpose(particles.rotation(particleData.densityParameters));
const tcnn::vec3 grdu   = giscl * (rotT * ray.direction);
const tcnn::vec3 o_c    = giscl * (rotT * (ray.origin - particles.position(particleData.densityParameters)));
const float grduLen     = tcnn::length(grdu);
const tcnn::vec3 d_c    = grdu / grduLen;
const float h           = tcnn::dot(d_c, o_c);
const float disc        = h * h - (tcnn::dot(o_c, o_c) - 1.f);
hitParticle.grduLen     = grduLen;
hitParticle.disc        = fmaxf(disc, 0.f);
if (disc >= 0.f) {
    const float sq = sqrtf(disc);
    hitParticle.t1 = (-h - sq) / grduLen;
    hitParticle.t2 = (-h + sq) / grduLen;
} else {
    // density hit이지만 ray가 unit sphere를 지나지 않는 degenerate 케이스
    hitParticle.t1 = hitParticle.hitT;
    hitParticle.t2 = hitParticle.hitT;
}
```

---

### 1.4 Degenerate Case

`disc < 0`은 수학적으로 불가능하지만, floating point 오차 또는 2D Gaussian projection의 hit 판정과 3D 타원체 교차 판정 간의 불일치로 발생할 수 있다. 이 경우 `t1 = t2 = hitT`로 설정해 점(point) interval로 처리한다.

---

### 1.5 활성화 조건

t1, t2 계산은 비용이 있으므로 overlap 처리가 필요한 모드에서만 수행된다:

```cpp
if constexpr (Params::FragmentBlend || Params::ZThicknessMerge ||
              Params::GeomOverlapMerge || Params::SwapMerge ||
              Params::SoftOIBlend)
```

Plain k-buffer 모드(baseline)에서는 이 계산을 skip한다. `GaussianFragmentBlend` 모드가 추가된 이후에는 해당 모드도 이 조건에 포함된다.

---

## 2. Gaussian Alpha 프로파일과 구간 분할 Compositing 구현

### 2.1 개요

기존 FragmentBlend(SFM)는 Gaussian의 실제 밀도 분포를 무시하고 `[t1, t2]` 구간에 걸쳐 **균일 밀도(uniform density)** 를 가정한다. 본 섹션에서는 Gaussian의 **실제 bell-curve 밀도 프로파일**을 ray 위에서 정확히 적분하는 `GaussianFragmentBlend` 모드의 수학적 원리와 구현을 다룬다.

구현 위치: `threedgut_tracer/include/3dgut/kernels/cuda/renderers/gutKBufferRenderer.cuh` — `processNWayGaussianFB()`

---

### 2.2 Ray 위에서의 Gaussian 밀도 프로파일

3D Gaussian의 밀도 함수:
```
ρ(x) = exp(-0.5 * (x - μ)^T Σ^{-1} (x - μ))
```

Ray `r(t) = o + t * d`를 canonical space로 변환하면 밀도는 t의 함수로 표현된다:

```
ρ(t) = exp(-0.5 * |o_c + t * grdu|²)
     = exp(-0.5 * (grduLen² * (t - t*)² + perp_dist²))
     = C * exp(-0.5 * grduLen² * (t - t*)²)
```

여기서:
- `t* = -h / grduLen` — ray 위에서 밀도가 최대인 깊이 (= `hitT`)
- `grduLen` — canonical space에서의 ray 속도 (배율 인자)
- `C = exp(-0.5 * perp_dist²)` — 최대 밀도 스케일 (`perp_dist`: ray와 Gaussian 중심 간의 수직 거리)

**결론**: ray 위에서의 밀도는 **t\* 를 중심으로 하는 1D Gaussian** (bell curve). 비선형.

#### 누적 alpha 프로파일

`[t1, t]` 구간까지의 누적 alpha:
```
alpha(t) = 1 - exp(-∫_{t1}^{t} σ(s) ds)
```

Gaussian 적분은 **erf(error function)** 으로 정확히 계산 가능:
```
∫_{a}^{b} exp(-0.5 * grduLen² * (t - t*)²) dt
  = (√(2π) / grduLen) * [Φ((b - t*) * grduLen) - Φ((a - t*) * grduLen)]
```

where `Φ(x) = 0.5 * (1 + erf(x / √2))`. 따라서 누적 alpha는 **erf 기반 sigmoid** — 비선형.

---

### 2.3 기존 SFM과의 비교

| | FragmentBlend (SFM) | GaussianFragmentBlend (신규) |
|---|---|---|
| 밀도 모델 | `σ = const = -log(1-α) / (t2-t1)` | `σ(t) = σ_peak * exp(-0.5 * grduLen² * (t-t*)²)` |
| 구간 적분 | `τ = σ * Δt` (단순 곱) | `τ = σ₀ * erf_seg / erf_tot` (erf 적분) |
| t1/t2 경계 | 밀도가 갑자기 켜지고 꺼짐 | t1/t2에서 밀도 ≈ 0 (자연스러운 경계) |
| overlap 구간 | 두 uniform slab의 합산 | 두 bell curve의 독립적 적분 합산 |
| N=1 일관성 | processHitParticle과 동일 | 동일 (`erf_seg / erf_tot = 1`) |

---

### 2.4 핵심 수식 정의

각 Gaussian i에 대해 다음을 사전 계산한다:

```
σ0_i    = -log(1 - alpha_i)                     # 총 소광 계수
erf_tot_i = erf(sqrt(disc_i / 2))               # [t1,t2] 전체 구간의 erf 적분값
```

여기서 `disc_i`는 1번 항목에서 계산된 판별식. canonical space에서의 관계:
```
(t1 - t*) * grduLen = -sqrt(disc)
(t2 - t*) * grduLen = +sqrt(disc)
→ erf_tot = erf(sqrt(disc) / sqrt(2)) = erf(sqrt(disc/2))
```

구간 `[s_lo, s_hi]`에서의 Gaussian i 기여도:
```
erf_seg_i(s_lo, s_hi) = 0.5 * (erf((s_hi - t*_i) * grduLen_i / √2)
                               - erf((s_lo - t*_i) * grduLen_i / √2))

contrib_i = σ0_i * erf_seg_i / erf_tot_i
```

**정규화 보장**: `[t1, t2]` 전체 구간에서 `contrib_i = σ0_i` → 총 alpha = `1 - exp(-σ0_i) = alpha_i` (2D projected alpha 보존).

---

### 2.5 Forward Pass

N개의 Gaussian이 overlap하는 경우, endpoint sweep (2N개의 t1/t2 정렬) 후 각 구간 `[s_k, s_{k+1}]`에서:

```
tau_k = Σ_{i ∈ active} contrib_i(s_lo, s_hi)    # 구간 총 광학 두께
alpha_k = 1 - exp(-tau_k)                          # 구간 alpha
ck = Σ_{i ∈ active} (contrib_i / tau_k) * feat_i  # contribution-weighted color
dk = Σ_{i ∈ active} (contrib_i / tau_k) * hitT_i  # contribution-weighted depth

color += alpha_k * T * ck
depth += alpha_k * T * dk
T *= (1 - alpha_k)
```

---

### 2.6 Backward Pass

Back-to-front undo 방식으로 각 구간을 순방향으로 처리하며 gradient를 누적한다.

각 구간 k에서 Gaussian i의 `σ0_i`에 대한 gradient:

```
dalpha_k = dot(ck - C_rest, FG) - T_k * TG       # 색상 + transmittance 경로

d_tau_k = dalpha_k * (1 - alpha_k)               # d(alpha)/d(tau) = exp(-tau) = 1-alpha

ratio_i = erf_seg_i / erf_tot_i                   # on-the-fly 재계산 (메모리 절약)

# alpha 경로: tau_k ← contrib_i = σ0_i * ratio_i
d_σ0[i] += d_tau_k * ratio_i

# color mixing 경로: ck ← contrib_i / tau_k 가중치
d_σ0[i] += alpha_k * dot(feat_i - ck, FG) * ratio_i / tau_k

# feature gradient
d_feat[i] += alpha_k * (contrib_i / tau_k) * FG
```

최종 alpha gradient 변환:
```
# σ0_i = -log(1 - alpha_i)  →  dσ0/dalpha = 1/(1-alpha_i)
alphaGrad_i = d_σ0[i] / (1 - alpha_i)
```

---

### 2.7 구현 세부사항

#### 메모리 설계

Backward sweep에서 구간별 `contrib_i` 배열을 저장하지 않고, `s_lo / s_hi`만 `Seg` 구조체에 저장한 뒤 **erf를 on-the-fly 재계산**한다. 이는 `MaxSeg × NWayMax` 크기의 레지스터 배열(최대 15×16=240 floats)을 피하기 위한 선택이다.

```cpp
struct Seg {
    float T, alpha, tau;
    TFeaturesVec c;
    int mask;
    float s_lo, s_hi;  // erf 재계산용 구간 경계
};
```

#### Edge Case 처리

```cpp
// erf_tot이 0에 가까운 degenerate Gaussian (disc ≈ 0): division guard
erf_tot[i] = fmaxf(erff(sqrtf(disc * 0.5f)), 1e-6f);

// alpha ≈ 1인 Gaussian: alphaGrad 계산 시 division guard
alphaGrad_i = (alpha < 1 - 1e-6f) ? d_σ0[i] / (1 - alpha) : 0.f;
```

#### 오버랩 검출 및 Drain 로직

buffer-full 시 연속 chain 검출, drain 시 union-find connected component 방식 - `FragmentBlend`와 동일한 구조, dispatch 함수만 `processNWayGaussianFB`로 교체.

---
