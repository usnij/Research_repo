# Gaussian Fragment Blend: 설계 보고서

---

## 1. 설계 동기

### 1.1 기존 3DGUT의 Alpha 처리 방식

3DGUT의 표준 렌더링 파이프라인에서 Gaussian 하나는 ray 위에서 **단일 alpha 값** 하나로 처리된다. 이 값은 2D projection 기반으로 계산되며, hitT (밀도 최대점)에 alpha가 집중된 **점 질량(point mass)** 형태로 compositing에 참여한다.

```
standard compositing:
  color += alpha_i * T * feat_i
  T     *= (1 - alpha_i)
```

이 방식은 Gaussian의 3D 체적 특성을 무시하고, 깊이 순서에만 의존한다.

### 1.2 설계 목표

Ray가 Gaussian 타원체를 통과하는 구간 `[t1, t2]` 전체에 걸쳐 alpha를 **연속적으로 분포**시키는 compositing 방식을 설계한다. 설계 조건:

1. **끝점 조건**: `alpha(t1) = 0`, `alpha(t2) = alpha_i` (2D alpha 보존)
2. **단조 증가**: ray 진행 방향으로 누적 alpha가 단조 증가
3. **물리적 근거**: 실제 3D Gaussian 밀도 분포 반영
4. **N=1 일관성**: 단일 Gaussian일 때 기존 방식과 동일한 결과
5. **미분 가능**: backward pass를 통한 gradient 전파 가능

---

## 2. 수학적 설계

### 2.1 Ray 위에서의 1D Gaussian 밀도 프로파일

3D Gaussian의 밀도를 canonical space로 변환하면, ray `r(t) = o + t*d` 위에서의 밀도는 **t의 1D Gaussian 함수**가 된다:

```
ρ(t) = C · exp(-0.5 · grduLen² · (t - t*)²)
```

- `t* = (t1 + t2) / 2` — 밀도 최대점 (ray-ellipsoid 구간의 정확한 중점)
- `grduLen = |S⁻¹ Rᵀ d|` — canonical space에서의 ray 속도
- `C` — 최대 밀도 스케일 (수직 거리 기반)

이 프로파일은 `t*`를 중심으로 하는 **bell curve**이며, `t1`과 `t2`에서 자연스럽게 감소한다.

### 2.2 누적 Alpha 프로파일

`t1`부터 임의의 `t`까지의 누적 alpha:

```
alpha_cumul(t) = 1 - exp(-σ₀ · erf_seg(t1, t) / erf_tot)
```

구성 요소:

| 기호 | 정의 | 계산식 |
|------|------|--------|
| `σ₀` | 총 소광 계수 | `-log(1 - alpha_i)` |
| `erf_tot` | `[t1, t2]` 전체 erf 적분 | `erf(√(disc/2))` |
| `erf_seg(a, b)` | 구간 `[a, b]`의 erf 적분 | `0.5·(erf((b-t*)·grduLen/√2) - erf((a-t*)·grduLen/√2))` |

**끝점 조건 검증**:
```
t = t1:  erf_seg(t1, t1) = 0        → alpha_cumul(t1) = 0          ✓
t = t2:  erf_seg(t1, t2) = erf_tot  → alpha_cumul(t2) = alpha_i    ✓
```

`erf_tot`로 나누는 정규화가 끝점 조건을 수학적으로 보장한다.

### 2.3 프로파일 형태

누적 alpha 프로파일은 `t1 → t* → t2`로 진행하면서 erf sigmoid(S-curve) 형태를 가진다:

- `t1` 근처: 느린 증가 (bell curve 왼쪽 꼬리, 밀도 낮음)
- `t*` 근처: 빠른 증가 (밀도 최대 구간)
- `t2` 도달 시: 정확히 `alpha_i` (erf 정규화로 수학적 보장)

### 2.4 구간 분할 Compositing

N개의 Gaussian이 ray 위에서 overlap할 때, 각 Gaussian의 t1/t2를 이벤트 포인트로 사용해 ray를 **M개의 구간**으로 분할한다.

구간 `[s_k, s_{k+1}]`에서 활성화된 Gaussian i의 기여도:

```
contrib_i(s_k, s_{k+1}) = σ₀_i · erf_seg_i(s_k, s_{k+1}) / erf_tot_i
```

구간 총 광학 두께:
```
τ_k = Σ_{i ∈ active} contrib_i(s_k, s_{k+1})
```

구간 compositing:
```
alpha_k = 1 - exp(-τ_k)
c_k     = Σ_i (contrib_i / τ_k) · feat_i    (contribution-weighted color)
color  += alpha_k · T · c_k
T      *= (1 - alpha_k)
```

**핵심 성질**: 각 Gaussian의 기여도는 해당 구간 내 bell curve의 면적 비율로 결정 → peak(t*) 근처 구간에 alpha 집중.

---

## 3. 구현 설계

### 3.1 전체 구조

```
GaussianFragmentBlend
├── 플래그 시스템
│   ├── configs/render/3dgut.yaml          gaussian_fragment_blend: false
│   ├── threedgut_tracer/setup_3dgut.py    -DGAUSSIAN_GAUSSIAN_FRAGMENT_BLEND
│   └── threedgut.cuh                      TGUTRendererParams::GaussianFragmentBlend
│
├── t1/t2 계산 (기존 코드 활용)
│   └── gutKBufferRenderer.cuh             hit 판정 후 erf 계산 조건에 추가
│
└── processNWayGaussianFB()
    ├── Forward Pass
    │   ├── sigma0, erf_tot 사전 계산
    │   ├── 2N endpoint 정렬 (insertion sort)
    │   ├── endpoint sweep → 구간별 contrib 계산 (erff)
    │   └── tau 기반 alpha, color, depth 누적
    └── Backward Pass
        ├── 동일 sweep으로 segment list 구성
        ├── Back-to-front undo (C_rest 복원)
        ├── dalpha_k → d_tau_k → d_sigma0[i] 누적
        ├── Color mixing gradient 누적
        └── d_sigma0[i] → alphaGrad → densityProcessHitBwdToBuffer
```

### 3.2 핵심 자료구조

#### HitParticle (기존 구조체 활용)

```cpp
struct HitParticle {
    int   idx;      // Gaussian 인덱스
    float hitT;     // 2D projected peak depth (≈ t*)
    float alpha;    // 2D projected opacity
    float t1, t2;   // ray-ellipsoid entry/exit (GaussianFragmentBlend에서 사용)
    float grduLen;  // canonical ray 속도 (erf 계산에 사용)
    float disc;     // 판별식 (erf_tot 계산에 사용)
};
```

`t1, t2, grduLen, disc`는 기존에 overlap 모드에서 이미 계산되던 값들로, 추가 연산 없이 재사용.

#### Seg (Backward용 segment 기록)

```cpp
struct Seg {
    float      T;      // forward transmittance entering segment
    float      alpha;  // segment alpha
    float      tau;    // segment optical depth
    TFeaturesVec c;    // contribution-weighted color
    int        mask;   // active Gaussian bitmask
    float      s_lo;   // segment start (erf on-the-fly 재계산용)
    float      s_hi;   // segment end
};
```

`s_lo, s_hi`를 저장해 backward에서 `erf_seg`를 on-the-fly 재계산 → `contrib_i` 배열 전체 저장 불필요 (레지스터 절약).

### 3.3 t* 계산: hitT vs (t1+t2)/2

`gaussianSegErf`의 피크 위치로 `hitT` 대신 `(t1 + t2) / 2`를 사용:

```cpp
const float t_star = (hits[i].t1 + hits[i].t2) * 0.5f;  // 정확한 피크: -h/grduLen
```

**이유**: `t* = -h/grduLen`은 t1, t2의 정확한 중점과 동일. `hitT`는 2D projection 근사값이므로 미세한 오차 존재. `(t1+t2)/2` 사용 시:

```
erf_seg(t1, t2) = erf_tot    (정확히 일치)
→ N=1에서 contrib = σ₀
→ alpha = 1 - exp(-σ₀) = alpha_i  (수치적으로 보장)
```

### 3.4 Backward 설계

#### Gradient 흐름

```
loss
 ↓
ray.featuresGradient (dL/dC_final)
 ↓  back-to-front undo per segment
dalpha_k = dot(c_k - C_rest, FG) - T_k · TG    ← feature + transmittance 경로
 ↓  d(alpha)/d(tau) = exp(-tau) = (1-alpha)
d_tau_k = dalpha_k · (1 - alpha_k)
 ↓  d(contrib_i)/d(sigma0_i) = erf_seg_i / erf_tot_i = ratio_i
d_sigma0[i] += d_tau_k · ratio_i                ← alpha 경로
d_sigma0[i] += alpha_k · dot(feat_i-c_k, FG) · ratio_i / tau_k  ← color mixing 경로
 ↓  d(sigma0)/d(alpha) = 1/(1-alpha_i)
alphaGrad_i = d_sigma0[i] / (1 - alpha_i)
 ↓
densityProcessHitBwdToBuffer(alphaGrad_i)        ← Gaussian 파라미터로 전파
```

#### Back-to-front undo

```cpp
// 구간 k의 색상 기여를 취소해 이전 상태 C_rest 복원
C_rest = (C - alpha_k · c_k) / (1 - alpha_k)

// gradient 스케일 전파
FG *= (1 - alpha_k)    // 다음 구간의 T 배율 반영
TG *= (1 - alpha_k)
```

### 3.5 Dispatch 구조

#### Buffer-full 시 (연속 chain 검출)

```
k-buffer가 꽉 참
  ↓
index 0부터 연속 overlap chain 탐색
  (fmin(hc.t2, hn.t2) > fmax(hc.t1, hn.t1) 조건)
  ↓
chain 추출 → t1 기준 정렬
  ↓
processNWayGaussianFB(chain, chainLen)
  ↓
drainFront(chainLen)
```

#### Drain 시 (union-find 기반)

```
남은 k-buffer hits
  ↓
union-find로 connected component 탐색
  (t1/t2 interval overlap이 있으면 같은 component)
  ↓
component별로 t1 정렬 후 processNWayGaussianFB 호출
```

---

## 4. Edge Case 처리

| 상황 | 처리 방법 |
|------|-----------|
| `disc ≈ 0` (ray가 타원체를 겨우 스침) | `erf_tot = max(erff(...), 1e-6f)` — division guard |
| `alpha ≈ 1` (불투명 Gaussian) | `alphaGrad = 0` — division guard |
| `tau_k ≈ 0` (구간 기여 없음) | color weight `wi = 0` — division guard |
| `ak ≈ 1` (구간이 거의 불투명) | `C_rest = 0` — back-to-front undo 안정화 |
| N=1 (단일 Gaussian) | 자동으로 `erf_seg = erf_tot` → `alpha = alpha_i` (수치 보장) |
| degenerate `t1 = t2 = hitT` | 구간 길이 `1e-8f` 미만 → skip |

---

## 5. 설정 및 사용

### 5.1 설정 파일

```yaml
# configs/render/3dgut.yaml
splat:
  k_buffer_size: 16               # GaussianFragmentBlend는 k > 0 필요
  gaussian_fragment_blend: true   # 활성화
```

### 5.2 실험 설정

```bash
conda run -n 3dgrut python train.py \
  --config-name apps/colmap_3dgut \
  path=data/bonsai \
  dataset.downsample_factor=2 \
  render.splat.k_buffer_size=16 \
  render.splat.gaussian_fragment_blend=true
```

### 5.3 JIT 재컴파일

플래그 변경 시 JIT 캐시 삭제 필요:
```bash
rm -rf ~/.cache/torch_extensions/py311_cu118/lib3dgut_cc/
```

---

## 6. 구현 파일 목록

| 파일 | 변경 내용 |
|------|-----------|
| `threedgut_tracer/include/3dgut/kernels/cuda/renderers/gutKBufferRenderer.cuh` | `processNWayGaussianFB()` 추가, t1/t2 계산 조건 추가, buffer-full/drain dispatch 추가 |
| `threedgut_tracer/include/3dgut/threedgut.cuh` | `TGUTRendererParams::GaussianFragmentBlend` 추가 |
| `threedgut_tracer/setup_3dgut.py` | `-DGAUSSIAN_GAUSSIAN_FRAGMENT_BLEND` define 추가 |
| `configs/render/3dgut.yaml` | `gaussian_fragment_blend: false` 기본값 추가 |

---

## 7. 실험 결과

공통 조건: bonsai 씬, downsample_factor=2

| 방법 | iter | PSNR | 차이 |
|------|------|------|------|
| Baseline (k=0) | 30k | 32.352 dB | 기준 |
| GaussianFragmentBlend (k=16) | 30k | 28.688 dB | **-3.664 dB** |

GaussianFragmentBlend는 erf 기반 bell-curve 적분으로 물리적으로 더 정확한 compositing을 수행하지만, 학습이 수렴하지 못했다. 배경 영역의 노출 오차 및 세부 디테일 손실이 두드러진다.

### 정성적 비교 (frame 15)

**Ground Truth**

![GT frame 15](images/gt_f15.png)

**Baseline (k=0, 30k iter, PSNR 32.352 dB)**

![Baseline frame 15](images/baseline_f15.png)

**GaussianFragmentBlend (k=16, 30k iter, PSNR 28.688 dB)**

![GaussianFragmentBlend frame 15](images/gfb_f15.png)

### 정성적 비교 (frame 25)

**Ground Truth**

![GT frame 25](images/gt_f25.png)

**Baseline (k=0, 30k iter, PSNR 32.352 dB)**

![Baseline frame 25](images/baseline_f25.png)

**GaussianFragmentBlend (k=16, 30k iter, PSNR 28.688 dB)**

![GaussianFragmentBlend frame 25](images/gfb_f25.png)

Frame 25에서 GFB의 배경 벽면이 GT 대비 확연히 어둡고, 화분 디테일도 뭉개지는 경향이 관찰된다. Baseline은 GT와 시각적으로 거의 동일한 품질을 보인다.
