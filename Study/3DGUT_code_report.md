# 3DGRUT 프로젝트 핵심 리포트

> 이 문서는 [3DGUT.md](./3DGUT.md)의 이론적 내용을 바탕으로, 실제 코드 구현과의 연결점을 분석합니다.

---

## 프로젝트 개요

**3DGRUT**는 NVIDIA 연구팀이 개발한 3D Gaussian 기반 렌더링 플랫폼으로, 세 가지 핵심 기술을 통합합니다:

| 기술 | 방식 | 발표 |
|------|------|------|
| **3DGRT** | 볼륨 가우시안 레이 트레이싱 | SIGGRAPH Asia 2024 |
| **3DGUT** | Unscented Transform 래스터화 | CVPR 2025 (구두) |


### 핵심 알고리즘

### 1. 3DGRT (Ray Tracing)
- NVIDIA OptiX 기반 BVH 가속구조
- 반사/굴절/그림자 등 세컨더리 레이 효과 지원
- 구면 조화(SH) 기반 방향성 색상 표현

### 2. 3DGUT (Rasterization)
- **Unscented Transform**: 카메라 왜곡을 가우시안 분포로 모델링
- 타일 기반 래스터화로 메모리 효율성
- K-Buffer 렌더링 지원


## 핵심 디렉토리 구조

```
3dgrut/
├── threedgrut/              # 메인 Python 패키지
│   ├── model/model.py       # MixtureOfGaussians (핵심 모델)
│   ├── trainer.py           # 훈련 루프
│   ├── datasets/            # 데이터셋 로더 (NeRF, COLMAP, ScanNet++)
│   └── strategy/            # 최적화 전략 (GS, MCMC)
│
├── threedgrt_tracer/        # OptiX 레이 트레이싱 엔진
├── threedgut_tracer/        # 래스터화 렌더링 엔진
├── threedgrut_playground/   # 인터랙티브 시각화 도구
└── configs/                 # Hydra 설정 파일들
```


## 1. Introduction

### 기존 3DGS의 한계 (이론적 배경)

3DGS는 3D Gaussian을 2D로 투영할 때 **EWA (Elliptical Weighted Average) Splatting**을 사용한다. 이는 비선형 투영 함수의 **Jacobian(야코비안)**을 이용한 1차 테일러 근사에 기반한다.

```math
\Sigma_{2D} = J \cdot \Sigma_{3D} \cdot J^{\top}
```

- $J$ : 투영 함수의 야코비안 (평균점에서 계산)
- $\Sigma_{3D}$ : 3D 공분산 행렬
- $\Sigma_{2D}$ : 2D 공분산 행렬

**문제점:**
- 야코비안은 **평균점 한 곳**에서만 계산되므로, 큰 Gaussian이나 왜곡이 심한 카메라에서 오차가 커진다.
- Rolling shutter처럼 시간에 따라 카메라가 변하면 야코비안 기반 모델이 정의 불가능하다.

### 3DGUT의 해결책

3DGUT는 야코비안 대신 **Unscented Transform (UT)** 을 사용한다:
- Gaussian → 7개의 sigma points로 샘플링
- 각 점을 **정확한 투영함수**로 변환
- 다시 2D Gaussian으로 재조합

즉, **비선형 투영을 근사하는 것이 아니라, Gaussian 자체를 근사해서 정확하게 투영**한다.

---

## 2. Unscented Transform: 이론과 구현

### 2.1 Sigma Points 생성 (이론)

3D Gaussian 하나를 7개의 sigma points로 근사한다:

```math
x_i =
\begin{cases}
\mu, & i = 0 \\
\mu + \sqrt{(3 + \lambda)\,\Sigma_{[i]}}, & i = 1, 2, 3 \\
\mu - \sqrt{(3 + \lambda)\,\Sigma_{[i-3]}}, & i = 4, 5, 6
\end{cases}
```

- $\mu$ : Gaussian의 중심 (평균)
- $\Sigma$ : 3D 공분산 행렬
- $\lambda = \alpha^2(3 + \kappa) - 3$ : 스케일링 파라미터

### 2.2 UT 파라미터 (코드 구현)

**파일**: `configs/render/3dgut.yaml`

```yaml
splat:
  ut_alpha: 1.0      # α: sigma points의 분산 스케일
  ut_beta: 2.0       # β: 분포 prior (Gaussian일 때 β=2가 최적)
  ut_kappa: 0.0      # κ: 보조 스케일링 파라미터
```

| 파라미터 | 의미 | 기본값 |
|----------|------|--------|
| `ut_alpha` | 평균 주위로 sigma points의 분포 제어 | 1.0 |
| `ut_beta` | 분포에 대한 prior 통합 (Gaussian=2) | 2.0 |
| `ut_kappa` | 스케일링 파라미터 | 0.0 |

### 2.3 가중치 계산 (이론)

각 sigma point의 **평균 가중치** $w_i^{\mu}$:

```math
w_i^{\mu} =
\begin{cases}
\dfrac{\lambda}{3 + \lambda}, & i = 0 \\
\dfrac{1}{2(3 + \lambda)}, & i = 1, \ldots, 6
\end{cases}
```

**공분산 가중치** $w_i^{\Sigma}$:

```math
w_i^{\Sigma} =
\begin{cases}
\dfrac{\lambda}{3 + \lambda} + (1 - \alpha^2 + \beta), & i = 0 \\
\dfrac{1}{2(3 + \lambda)}, & i = 1, \ldots, 6
\end{cases}
```

### 2.4 2D Gaussian 재구성 (이론)

투영된 sigma points로부터 2D Gaussian의 **평균**:

```math
\nu_\mu = \sum_{i=0}^{6} w_i^{\mu} \, v_{x_i}
```

2D Gaussian의 **공분산**:

```math
\nu_\Sigma = \sum_{i=0}^{6} w_i^{\Sigma} \, (v_{x_i} - \nu_\mu)(v_{x_i} - \nu_\mu)^{\top}
```

- $v_{x_i}$ : sigma point $x_i$를 실제 카메라 모델로 투영한 2D 위치

### 2.5 코드 구현 (gutProjector.cuh)

**파일**: `threedgut_tracer/include/3dgut/kernels/cuda/renderers/gutProjector.cuh`

```cpp
// === 1. Lambda 및 가중치 계산 ===
constexpr float Lambda = Alpha * Alpha * (D + Kappa) - D;  // λ = α²(D+κ) - D
constexpr float weight0_mean = Lambda / (D + Lambda);      // w_0^μ
constexpr float weightI = 1.f / (2.f * (D + Lambda));      // w_i^μ (i=1..6)
constexpr float weight0_cov = weight0_mean + (1.f - Alpha*Alpha + Beta);  // w_0^Σ

// === 2. Sigma Points 생성 및 투영 ===
vec2 projectedSigmaPoints[7];  // 2*D + 1 = 7개

// Point 0: 평균점
project(particleMean, projectedSigmaPoints[0]);
particleProjCenter = projectedSigmaPoints[0] * weight0_mean;

// Points 1~6: ±delta 방향
for (int i = 0; i < D; ++i) {
    vec3 delta = Delta * particleScale[i] * particleRotation[i];

    project(particleMean + delta, projectedSigmaPoints[i + 1]);
    project(particleMean - delta, projectedSigmaPoints[i + 1 + D]);

    particleProjCenter += weightI * (projectedSigmaPoints[i+1] + projectedSigmaPoints[i+1+D]);
}

// === 3. 2D 공분산 계산 ===
vec2 c0 = projectedSigmaPoints[0] - particleProjCenter;
particleProjCovariance = weight0_cov * vec3(c0.x*c0.x, c0.x*c0.y, c0.y*c0.y);

for (int i = 0; i < 2*D; ++i) {
    vec2 ci = projectedSigmaPoints[i+1] - particleProjCenter;
    particleProjCovariance += weightI * vec3(ci.x*ci.x, ci.x*ci.y, ci.y*ci.y);
}
```

### 2.6 이론-코드 대응 관계

| 이론 (수식) | 코드 (변수) | 설명 |
|-------------|-------------|------|
| $\lambda = \alpha^2(D + \kappa) - D$ | `Lambda` | 스케일링 파라미터 |
| $\sqrt{(D + \lambda) \Sigma}$ | `UTParams::Delta * particleScale[i] * particleRotation[i]` | Sigma point 오프셋 |
| $w_0^{\mu} = \frac{\lambda}{D + \lambda}$ | `Lambda / (UTParams::D + Lambda)` | 평균점 가중치 |
| $w_i^{\mu} = \frac{1}{2(D + \lambda)}$ | `weightI` | 나머지 점 가중치 |
| $w_0^{\Sigma} = w_0^{\mu} + (1 - \alpha^2 + \beta)$ | `weight0` | 공분산 가중치 (평균점) |
| $\nu_\mu$ | `particleProjCenter` | 투영된 2D 평균 |
| $\nu_\Sigma$ | `particleProjCovariance` | 투영된 2D 공분산 [xx, xy, yy] |
| $v_{x_i}$ | `projectedSigmaPoints[i]` | 투영된 sigma point |

---

## 3. Particle Response 평가: 이론과 구현

### 3.1 3DGS vs 3DGUT의 Response 평가 차이

| 항목 | 3DGS | 3DGUT |
|------|------|-------|
| 평가 공간 | 2D 이미지 평면 | 3D 공간 (ray 기반) |
| 방식 | 투영된 2D Gaussian과 픽셀 거리 | Ray와 3D Gaussian의 관계 |
| 왜곡 처리 | 불안정 | 안정적 |

### 3.2 Ray-Gaussian 최대 응답점 (이론)

Gaussian의 response가 ray 위에서 최대가 되는 지점 $r_{max}$:

```math
r_{\max} = \frac{(\mu - o)^{\top} \Sigma^{-1} d}{d^{\top} \Sigma^{-1} d}
```

- $o$ : 카메라 중심 (ray origin)
- $d$ : ray 방향
- $\mu$ : Gaussian 중심
- $\Sigma$ : 공분산 행렬

**의미**: Gaussian 중심 μ에서 ray 방향으로 "수직으로 가장 가까운 지점"을 구하는 것이다.

### 3.3 코드 구현

**파일**: `threedgut_tracer/src/gutRenderer.cu`

```cpp
// 1단계: 가우시안을 타일에 투영 (UT 기반)
::projectOnTiles<<<...>>>(
    tileGrid,
    numParticles,
    params.sensorModel,          // 카메라 모델 (핀홀/어안렌즈)
    particlesProjectedPosition,  // ν_μ (2D 평균)
    particlesProjectedConicOpacity,  // ν_Σ (2D 공분산) + opacity
    particlesGlobalDepth,        // r_max (전역 깊이)
    ...
);
```

---

## 4. 렌더링 파이프라인 비교

### 4.1 기존 3DGS 파이프라인

```
1. 3D Gaussian → Jacobian 기반 2D 투영 (EWA Splatting)
2. 타일 기반 컬링
3. 타일 내 깊이 정렬 (로컬)
4. 알파 블렌딩
```

### 4.2 3DGUT 파이프라인

```
1. 3D Gaussian → Unscented Transform 기반 2D 투영
2. 타일 기반 컬링 (tight_opacity_bounding)
3. Global Z-order 정렬
4. K-Buffer 또는 정렬되지 않은 블렌딩
```

### 4.3 코드 구현 (gutRenderer.cu)

```cpp
// Forward 렌더링 파이프라인
Status GUTRenderer::renderForward(...) {
    // 1. UT 기반 투영
    ::projectOnTiles<<<...>>>(...);

    // 2. 누적 합계로 타일 오프셋 계산
    cub::DeviceScan::InclusiveSum(...);

    // 3. 타일-가우시안 매핑 확장
    ::expandTileProjections<<<...>>>(...);

    // 4. Radix 정렬 (타일 + 깊이 기준)
    cub::DeviceRadixSort::SortPairs(...);

    // 5. 타일별 렌더링
    ::render<<<...>>>(...);
}
```

---

## 5. 카메라 모델 지원

### 5.1 지원 카메라 유형

3DGUT는 UT 덕분에 다양한 카메라 모델을 **네이티브로** 지원한다:

| 카메라 모델 | 3DGS | 3DGUT |
|-------------|------|-------|
| Pinhole | O | O |
| OpenCV Fisheye | X | O |
| Rolling Shutter | X | O |
| Radial Distortion | X | O |

### 5.2 코드 구현

**파일**: `threedgrut/datasets/camera_models.py`

```python
@dataclass
class OpenCVFisheyeCameraModelParameters:
    resolution: np.ndarray          # [width, height]
    shutter_type: ShutterType       # GLOBAL / ROLLING_*
    principal_point: np.ndarray     # [cx, cy]
    focal_length: np.ndarray        # [fx, fy]
    radial_coeffs: np.ndarray       # [k1, k2, k3, k4]
    max_angle: float                # 최대 FOV 각도
```

### 5.3 Rolling Shutter 지원

```yaml
# configs/render/3dgut.yaml
splat:
  n_rolling_shutter_iterations: 5  # 롤링 셔터 보정 반복 횟수
```

Rolling shutter는 각 픽셀 행/열마다 다른 시간에 촬영되므로, 시작/끝 포즈 사이를 **보간**하여 처리한다.

---

## 6. 정렬 방식 비교

### 6.1 이론적 배경

3DGS는 타일 내에서만 깊이 정렬을 수행하므로 타일 경계에서 불연속이 발생할 수 있다. 3DGUT는 **Global Z-order**를 사용하여 이 문제를 해결한다.

### 6.2 코드 구현

**파일**: `configs/render/3dgut.yaml`

```yaml
splat:
  global_z_order: true   # 전역 깊이 정렬 활성화
  k_buffer_size: 0       # 0 = 정렬되지 않은 모드
```

**정렬 키 구조** (64비트):
```
[상위 32비트: 타일 인덱스] [하위 32비트: 깊이값]
```

```cpp
// gutRenderer.cu
uint64_t sortKey = (uint64_t(tileIdx) << 32) | depthBits;
```


## 7. 요약: 핵심 차이점

| 항목 | 기존 3DGS | 3DGUT |
|------|-----------|-------|
| **투영 방식** | Jacobian (1차 근사) | Unscented Transform (7 sigma points) |
| **Response 평가** | 2D 이미지 평면 | 3D 공간 (ray 기반) |
| **카메라 왜곡** | 사후 보정 필요 | 네이티브 지원 |
| **Rolling Shutter** | 미지원 | 네이티브 지원 |
| **정렬** | 타일 내 로컬 | Global Z-order |
| **수학적 기반** | $\Sigma_{2D} = J \Sigma_{3D} J^{\top}$ | $\nu_\Sigma = \sum w_i^{\Sigma} (v_{x_i} - \nu_\mu)(v_{x_i} - \nu_\mu)^{\top}$ |

---

