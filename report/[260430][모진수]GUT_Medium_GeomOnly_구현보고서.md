# [260430][모진수] GUT-Medium Geometry-Only 재설계 구현 보고서

---

## 1. 배경 및 문제 제기

### 1.1 기존 GUT-Medium 모델의 설계

이전 구현에서 GUT-Medium은 다음 공식을 사용했다:

$$\sigma_{0,\text{peak}} = -\log(1 - \text{opacity})$$

$$\sigma_{0,\text{med}} = \sigma_{0,\text{peak}} \times \text{erf\_tot}$$

$$\alpha = 1 - \exp(-\sigma_{0,\text{med}})$$

- $\text{opacity}$ : Gaussian의 학습 파라미터 (2D 표면 모델에서 유래)
- $\text{erf\_tot}$ : ray-ellipsoid 교차 구간 $[t_1, t_2]$의 적분 비율 $= \mathrm{erf}(\sqrt{\text{disc}/2})$
- $\sigma_{0,\text{med}}$ : $\text{erf\_tot}$로 변조된 광학 깊이(optical depth)

### 1.2 교수님 미팅 결과 — 근본적 재설계 필요

> "Gaussian에서 ray가 지나는 $t_1$, $t_2$ 구간에서 벨커브 형태의 density가 나타난다. 그 구간의 밀도는 적분 가능한 형태이므로, opacity 파라미터 없이 geometry(scale, rotation)만으로 transmittance를 결정할 수 있다."

**핵심 지적:**
1. `opacity`는 2D 표면 렌더링 개념 — 3D 매질(medium) 모델에 불필요
2. Gaussian의 scale + rotation이 3D 형태를 완전히 결정하므로, ray가 통과하는 밀도 프로파일도 geometry로 충분히 표현 가능
3. Beer-Lambert 법칙에서 $\alpha = 1 - \exp(-\int \sigma\, dt)$이므로, $\int \sigma\, dt$를 geometry에서 직접 유도

---

## 2. 새로운 모델 설계

### 2.1 물리적 유도

3D 가우시안 매질에서 ray $t$ 방향으로의 밀도 프로파일은 **가우시안 벨커브** 형태다:

$$\sigma(t) = \sigma_\text{peak} \cdot \exp\!\left(-\frac{(t - t^*)^2}{2\,\sigma_\text{world}^2}\right)$$

- $t^*$ : ray가 Gaussian 중심에 가장 가까워지는 지점 (벨커브 피크)
- $\sigma_\text{world} = 1/\text{grduLen}$ : world 공간에서의 Gaussian 폭
- $\sigma(t)$를 $[t_1, t_2]$ 구간에서 적분하면 optical depth $\tau$를 얻는다

canonical 공간으로 변환하면 적분이 erf 형태로 닫힌해를 가진다:

$$\tau = \int_{t_1}^{t_2} \sigma(t)\, dt = \sigma_\text{peak} \cdot \sigma_\text{world} \cdot \sqrt{2\pi} \cdot \text{erf\_tot} = \frac{\sqrt{2\pi}\;\text{erf\_tot}}{\text{grduLen}}$$

- $\text{grduLen} = \|\Sigma^{-1/2} d\|$ : canonical 공간에서의 ray 속도 (scale + rotation에만 의존); $\sigma_\text{world} = 1/\text{grduLen}$
- $\text{erf\_tot} = \mathrm{erf}(\sqrt{\text{disc}/2})$ : $[t_1, t_2]$ 구간이 전체 가우시안 적분 중 차지하는 비율
- geometry-only 모델에서 $\sigma_\text{peak} = 1$로 고정 → opacity 파라미터 불필요

Beer-Lambert에 의한 **alpha**:

$$\alpha = 1 - \exp(-\tau)$$

### 2.2 핵심 특성

| 항목 | 기존 모델 | 새 모델 |
|------|-----------|---------|
| 학습 파라미터 | opacity, scale, rotation, pos | scale, rotation, pos (opacity 불사용) |
| alpha 결정 요인 | $\text{opacity} \times \text{erf\_tot}$ | geometry만 ($\sqrt{2\pi} \cdot \text{erf\_tot} / \text{grduLen}$) |
| 물리적 근거 | opacity = 2D surface 개념 | Beer-Lambert + 3D Gaussian density |
| 발산 위험 | opacity→1 시 $\sigma_{0,\text{peak}} \to \infty$ | $\tau \leq 20$으로 clamp (안정적) |

---

## 3. Forward 구현

**파일**: `gutKBufferRenderer.cuh` (forward eval block, ~line 2097)

```cpp
// GUT-Medium: replace 2D-projection alpha with 3D ray-ellipsoid Beer-Lambert integral.
// sigma_peak = exp(-(1-disc)/2)             : peak density (ray-center distance)
// tau        = sigma_peak * sqrt(2pi) * erf_tot / grduLen
// alpha      = 1 - exp(-tau)
if constexpr (Params::Medium) {
    hitParticle.alphaVanilla = hitParticle.alpha;
    static constexpr float SQRT_2PI = 2.5066282746310002f;
    const float erf_tot    = erff(sqrtf(hitParticle.disc * 0.5f));
    const float sigma_peak = expf(-(1.f - hitParticle.disc) * 0.5f);
    const float tau        = sigma_peak * SQRT_2PI * erf_tot / fmaxf(hitParticle.grduLen, 1e-6f);
    hitParticle.alpha      = 1.f - expf(-fminf(tau, 20.f));
}
```

변경점:
- `opacity` 파라미터 참조 제거
- $\sigma_\text{peak} = \exp(-(1-\text{disc})/2)$ 직접 계산 (geometry에서 유도)
- $\tau = \sigma_\text{peak} \cdot \sqrt{2\pi} \cdot \text{erf\_tot} / \text{grduLen}$
- $\tau \leq 20$ clamp 추가 (수치 안정성)

---

## 4. Backward 구현

### 4.1 Gradient 유도

$$\alpha = 1 - e^{-\tau} \quad\Rightarrow\quad \frac{\partial \mathcal{L}}{\partial \tau} = \frac{\partial \mathcal{L}}{\partial \alpha} \cdot (1 - \alpha)$$

$$\tau = \sigma_\text{peak} \cdot \frac{\sqrt{2\pi} \cdot \text{erf\_tot}}{\text{grduLen}}$$

$$\frac{\partial \mathcal{L}}{\partial \,\text{erf\_tot}} = \frac{\partial \mathcal{L}}{\partial \tau} \cdot \frac{\sigma_\text{peak}\,\sqrt{2\pi}}{\text{grduLen}}$$

$$\frac{\partial \mathcal{L}}{\partial \,\text{grduLen}} = \frac{\partial \mathcal{L}}{\partial \tau} \cdot \left(-\frac{\sigma_\text{peak}\,\sqrt{2\pi}\cdot\text{erf\_tot}}{\text{grduLen}^2}\right)$$

$$\frac{\partial \mathcal{L}}{\partial \,\sigma_\text{peak}} = \frac{\partial \mathcal{L}}{\partial \tau} \cdot \frac{\sqrt{2\pi}\cdot\text{erf\_tot}}{\text{grduLen}}$$

$$\sigma_\text{peak} = \exp\!\left(-\frac{1-\text{disc}}{2}\right) \quad\Rightarrow\quad \frac{\partial \mathcal{L}}{\partial \,\text{disc}}\bigg|_{\sigma} = \frac{\partial \mathcal{L}}{\partial \,\sigma_\text{peak}} \cdot \frac{\sigma_\text{peak}}{2}$$

### 4.2 Gradient 전파 경로

**$\partial\mathcal{L}/\partial\,\text{erf\_tot}$ → disc chain:**

$$\text{erf\_tot} = \mathrm{erf}(v),\quad v = \sqrt{\text{disc}/2} \quad\Rightarrow\quad \frac{\partial\mathcal{L}}{\partial\,\text{disc}}\bigg|_{\text{erf}} = \frac{\partial\mathcal{L}}{\partial\,\text{erf\_tot}} \cdot \frac{2}{\sqrt{\pi}} \cdot \frac{e^{-v^2}}{4v}$$

$$\text{disc} = h^2 - (\|o_c\|^2 - 1) \quad\Rightarrow\quad \partial o_c,\; \partial d_c$$

$$d_c = \frac{\text{grdu}}{\|\text{grdu}\|} \quad\Rightarrow\quad \partial\,\text{grdu} \text{ (via safe\_normalize\_bw)}$$

**$\partial\mathcal{L}/\partial\,\text{grduLen}$ → grdu norm chain:**

$$\text{grduLen} = \|\text{grdu}\| \quad\Rightarrow\quad \partial\,\text{grdu} \mathrel{+}= \frac{\partial\mathcal{L}}{\partial\,\text{grduLen}} \cdot \frac{\text{grdu}}{\text{grduLen}}$$

**공통 canonical chain:**

$$\text{grdu} = \text{giscl} \odot (R^\top d) \quad\Rightarrow\quad \partial\,\text{giscl},\; \partial(R^\top d)$$

$$o_c = \text{giscl} \odot (R^\top (\mathbf{o} - \mathbf{p})) \quad\Rightarrow\quad \partial\,\text{giscl},\; \partial(R^\top(\mathbf{o}-\mathbf{p}))$$

$$\partial\,\text{scale}_i = -\text{giscl}_i^2 \cdot \partial\,\text{giscl}_i, \qquad \partial\mathbf{p} = -R\,\partial(R^\top(\mathbf{o}-\mathbf{p})), \qquad \partial q \text{ via matmul\_bw\_quat}$$

**주요 변경점:**
- density gradient (`addDensityGradAtomic`) 완전 제거
- $\partial\,\text{erf\_tot}$, $\partial\,\text{grduLen}$, $\partial\,\sigma_\text{peak}$ 세 경로 추가
- $K_{2D}$ gradient 억제(`alphaGrad=0`, `transGrad=0` 전달)는 유지

### 4.3 구현 코드 (`gutKBufferRenderer.cuh`, ~line 265)

```cpp
if constexpr (Params::Medium) {
    if (hitParticle.disc > 1e-8f && medium_total_alpha_grad != 0.f) {
        static constexpr float SQRT_2PI = 2.5066282746310002f;
        const float alpha_med  = hitParticle.alpha;
        const float grduLen    = hitParticle.grduLen;
        const float v          = sqrtf(hitParticle.disc * 0.5f);
        const float erf_tot    = erff(v);
        const float sigma_peak = expf(-(1.f - hitParticle.disc) * 0.5f);

        const float d_tau        = medium_total_alpha_grad * (1.f - alpha_med);
        const float d_erf_tot    = d_tau * sigma_peak * SQRT_2PI / fmaxf(grduLen, 1e-6f);
        const float d_grduLen    = d_tau * sigma_peak * (-SQRT_2PI * erf_tot / fmaxf(grduLen*grduLen, 1e-12f));
        const float d_sigma_peak = d_tau * SQRT_2PI * erf_tot / fmaxf(grduLen, 1e-6f);
        const float d_disc_sigma = d_sigma_peak * sigma_peak * 0.5f;

        float d_disc = d_disc_sigma;
        if (v > 1e-4f)
            d_disc += d_erf_tot * 2.f * 0.56418958354775628f * expf(-v*v) / (4.f * v);

        if (d_disc != 0.f || d_grduLen != 0.f) {
            float3 d_grdu = safe_normalize_bw(grdu, d_d_c)
                          + d_grduLen * (grdu / fmaxf(grduLen, 1e-6f));
            // → d_scale, d_position, d_quat via canonical chain
            particles.addGeomGradientAtomic(hitParticle.idx, d_position, d_scale, d_quat);
        }
    }
}
```

---

## 5. 학습 결과

### 5.1 실험 설정

| 항목 | 값 |
|------|-----|
| 씬 | bonsai |
| `downsample_factor` | 2 |
| `k_buffer_size` | 4 |
| `n_iterations` | 30,000 |
| 비교 baseline | `k_buffer_size=0`, 30k iter |


### 5.2 정량 결과

| 모델 | iter | PSNR (dB) | vs Baseline |
|------|------|-----------|-------------|
| **Baseline** (k=0) | 30k | **32.352** | 기준 |
| Medium Geom-Only ($\sigma_\text{peak}$ 누락) | 30k | 29.628 | −2.724 |
| 기존 opacity-based Medium | 30k | 23.185 | −9.167 |

### 5.3 정성 비교

> **좌**: GT (Ground Truth) &nbsp;|&nbsp; **중**: Baseline (k=0, 30k, PSNR=32.352 dB) &nbsp;|&nbsp; **우**: Medium Geom-Only (k=4, 30k, PSNR=29.628 dB)

**View A**
![View A 비교](report_image_모진수/medium_compare_00000.png)

**View B**
![View B 비교](report_image_모진수/medium_compare_00013.png)

**View C**
![View C 비교](report_image_모진수/medium_compare_00027.png)

---

