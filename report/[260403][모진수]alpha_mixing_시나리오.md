# GaussianFragmentBlend 파라미터 정의

시나리오: Gaussian A (앞), B (뒤) overlap

---

## 시나리오 설정

| | Gaussian A (빨강, 앞) | Gaussian B (파랑, 뒤) |
|--|--|--|
| t1 | 1.0 | 2.0 |
| t2 | 3.0 | 4.0 |
| alpha | 0.8 | 0.8 |
| grduLen | 1.0 (가정) | 1.0 (가정) |

---

## 파라미터 정의와 직관

### t1, t2
**정의**: ray가 Gaussian 타원체에 진입하는 시점(t1)과 빠져나오는 시점(t2).
ray `r(t) = origin + t·direction`에서 canonical space 변환 후 unit sphere와의 교점.

**직관**: Gaussian이 ray 위에서 "존재하는 구간". t1~t2 사이에서만 이 Gaussian이 color/opacity에 기여한다.

---

### alpha (2D projected alpha)
**정의**: UT(Unscented Transform) 기반 2D projection에서 계산된 Gaussian의 불투명도.
ray가 Gaussian 타원체를 완전히 통과했을 때 최종적으로 누적되는 총 opacity.

**직관**: "이 Gaussian이 완전히 통과되면 ray를 얼마나 막는가". alpha=0.8이면 80% 차단.

---

### grduLen
**정의**: Canonical space(S⁻¹Rᵀ 변환)에서의 ray 방향 벡터 크기.

```
grduLen = |S⁻¹Rᵀ d|
```

**직관**: Gaussian을 ray가 "얼마나 빨리 통과하는가". 값이 크면 좁은 구간(날카로운 bell-curve), 값이 작으면 넓은 구간(완만한 bell-curve).

grduLen=1.0이면 t1~t2 폭이 표준 정규분포 ±1σ에 해당.

---

### σ₀ (총 광학 두께, sigma-zero)
**정의**: alpha를 Beer-Lambert 법칙으로 변환한 광학 두께.

```
σ₀ = -ln(1 - alpha)
```

**직관**: "총 불투명도를 밀도로 환산한 값". alpha=0.8 → σ₀ = 1.6094.
alpha가 누적 opacity라면, σ₀는 그 opacity를 만들어내는 적분 밀도.

---

### t* (밀도 피크)
**정의**: ray 위에서 Gaussian 밀도가 최대인 지점. t1과 t2의 정확한 중점.

```
t* = (t1 + t2) / 2
```

**직관**: Bell-curve의 꼭짓점. 이 지점에서 Gaussian이 ray를 가장 강하게 차단.
A의 t*=2.0, B의 t*=3.0.

---

### erf_seg(s_lo, s_hi)
**정의**: 구간 [s_lo, s_hi]에서 bell-curve 밀도의 적분값.

```
erf_seg(a, b) = 0.5 · (erf((b - t*) · grduLen/√2) - erf((a - t*) · grduLen/√2))
```

**직관**: "구간 안에 bell-curve 면적이 얼마나 들어오는가". 피크(t*) 근처 구간일수록 값이 크다.

---

### erf_tot
**정의**: Gaussian 전체 구간 [t1, t2]에 대한 bell-curve 적분. erf_seg의 정규화 기준.

```
erf_tot = erf_seg(t1, t2)
```

**직관**: "이 Gaussian의 bell-curve 면적 전체". 어떤 구간의 erf_seg를 erf_tot로 나누면 → 해당 구간이 전체 중 몇 %를 차지하는지.

---

### contrib_i (구간 기여도)
**정의**: 구간 [s_lo, s_hi]에서 Gaussian i가 기여하는 광학 두께.

```
contrib_i = σ₀_i · erf_seg_i(s_lo, s_hi) / erf_tot_i
```

**직관**: "σ₀(총 광학 두께)를 bell-curve 면적 비율로 구간에 분배". 피크 근처 구간이 더 많은 광학 두께를 받는다.

---

### tau_k (구간 총 광학 두께)
**정의**: 구간 k에서 활성화된 모든 Gaussian의 contrib 합.

```
tau_k = Σ_i contrib_i
```

**직관**: "구간 전체의 밀도 합". 여러 Gaussian이 겹치면 각각의 밀도가 더해진다.

---

### alpha_k (구간 alpha)
**정의**: tau_k를 Beer-Lambert로 변환한 구간 불투명도.

```
alpha_k = 1 - exp(-tau_k)
```

**직관**: "이 구간을 지나면서 ray가 얼마나 차단되는가".

---

### T (transmittance)
**정의**: 현재까지 ray가 통과한 비율. 앞 구간들에서 얼마나 살아남았는가.

```
T_new = T_old · (1 - alpha_k)
```

**직관**: "ray가 아직 투과할 수 있는 여력". T=1이면 완전 투명, T=0이면 완전 차단.
앞쪽 구간에서 많이 차단될수록 뒤쪽 Gaussian의 기여가 감소.

---

## 수치 적용 (A: t*=2, B: t*=3, grduLen=1.0)

```
erf_tot (공통) = erf(1/√2) ≈ 0.6827
σ₀ (공통)     = -ln(0.2) = 1.6094
```

| 구간 | 활성 | erf_seg (A) | erf_seg (B) | contrib_A | contrib_B | tau | alpha_k | T_in |
|------|------|-------------|-------------|-----------|-----------|-----|---------|------|
| [1,2] | A | 0.3414 | — | 0.8047 | — | 0.8047 | 0.553 | 1.000 |
| [2,3] | A+B | 0.3414 | 0.3414 | 0.8047 | 0.8047 | 1.609 | 0.800 | 0.447 |
| [3,4] | B | — | 0.3414 | — | 0.8047 | 0.8047 | 0.553 | 0.089 |

**의심 포인트**: 구간 [2,3]에서 A와 B의 contrib가 정확히 동일 → color mixing이 50:50.
A가 먼저 진입했다는 사실(depth ordering)이 overlap 구간 내 color mixing에 반영되지 않음.
