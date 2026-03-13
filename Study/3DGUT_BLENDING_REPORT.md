# GUT 렌더링에서의 블렌딩 구현 분석

## 1. 개요

GUT(Gaussian Unscented Transform)의 핀홀 카메라 렌더링은 **ray 기반 front-to-back alpha compositing**을 사용한다.
블렌딩 공식 자체는 3DGS와 동일하지만, alpha를 구하는 방식이 근본적으로 다르다.

- **3DGS**: 2D splatting 기반 (Gaussian을 이미지 평면에 투영 후 합성)
- **GUT**: Ray-based 볼륨 렌더링 (각 픽셀에서 ray를 쏘아 3D Gaussian과 직접 교차)

## 2. 전체 파이프라인

```
픽셀별 ray 초기화 → 타일별 Gaussian 순회 → hit 판정 및 hit buffer 저장 → ray-particle density 평가 및 alpha 계산 → front-to-back 색상 블렌딩 → 최종 출력
```

### 2.1 Ray 초기화

각 픽셀마다 ray를 생성하고, transmittance를 1.0으로 초기화한다.

**파일**: `threedgut_tracer/include/3dgut/kernels/cuda/common/rayPayload.cuh:76-108`

```cuda
RayPayloadT ray;
ray.hitT          = 0.0f;
ray.transmittance = 1.0f;                         // T = 1 (완전 투명 상태)
ray.features      = tcnn::vec<FeatDim>::zero();    // 색상 누적값 = 0

ray.origin    = sensorToWorldTransform * vec4(sensorRayOriginPtr[ray.idx], 1.0f);
ray.direction = mat3(sensorToWorldTransform) * sensorRayDirectionPtr[ray.idx];
```

### 2.2 Gaussian Hit 및 Alpha 계산

ray는 Gaussian과의 hit 여부를 판정한다.
hit가 발생하면 해당 Gaussian의 인덱스와 깊이 정보가 ray의 hit buffer에 저장되며, 이후 이 buffer에 저장된 Gaussian들에 대해 density response를 평가하여 alpha 값을 계산한다.

**파일**: `threedgut_tracer/include/3dgut/kernels/slang/models/gaussianParticles.slang:186-222`
```c
// ray와 Gaussian의 교차 판정
alpha = min(MaxParticleAlpha, maxResponse * parameters. );
const bool acceptHit = ((maxResponse > MinParticleKernelDensity) && (alpha > MinParticleAlpha));

if (acceptHit) {
    depth = canonicalRayDistance(canonicalRayOrigin, canonicalRayDirection, parameters.scale);
}
```
여기서 `maxResponse`는 ray가 Gaussian의 canonical space를 통과할 때 얻는 최대 커널 응답값으로,
ray와 Gaussian 중심 사이의 상대적 위치 관계에 의해 결정된다.

#### maxResponse 계산 과정
hit 판정 전에 먼저 ray를 Gaussian의 **canonical space**로 변환한다.

**canonical space**란 Gaussian의 공분산 행렬(scale, rotation)을 단위 구(unit sphere)로 정규화한 공간이다.
이 변환을 통해 임의의 타원체 형태를 가진 Gaussian을 반지름 1의 구로 취급할 수 있어, 이후 거리 계산이 단순해진다.

canonical space로 변환된 ray에 대해 `canonicalRayMaxKernelResponse`를 호출하여 `maxResponse`를 계산한다.
```c
const float maxResponse = gaussianParticle.canonicalRayMaxKernelResponse
    (
    canonicalRayOrigin,
    canonicalRayDirection);
```


이렇게 계산된 maxResponse를 Gaussian의 density 파라미터와 곱하여 최종 alpha가 결정된다.

여기서 density 파라미터는 3DGS에서 gaussian의 파라미터 중 하나인 opacity와 기능적으로 동일하다. 

명칭이 다른 이유는 렌더링 방식의 차이에서 비롯된다. 3DGS는 Gaussian을 2D 이미지 평면에 투영한 뒤 픽셀 단위로 불투명도를 계산하는 **splatting** 방식이므로 `opacity`라는 표현이 직관적이다. 반면 3DGUT는 ray가 3D 공간의 Gaussian 볼륨을 직접 통과하며 해당 지점의 **밀도(density)** 를 샘플링하는 볼륨 렌더링 방식이므로, 물리적으로 더 정확한 표현인 `density`를 사용한다.

### 2.3 블렌딩 (핵심)
 
buffer에 저장된 Gaussian hit들은 깊이 순서로 정렬된 뒤 front-to-back alpha compositing으로 합성된다.

**파일**: `threedgut_tracer/include/3dgut/kernels/cuda/renderers/gutKBufferRenderer.cuh:153-165`

```cuda
// Forward pass: 각 hit된 Gaussian에 대해
const float hitWeight =
    particles.densityIntegrateHit(hitParticle.alpha,    // 이 Gaussian의 alpha
                                  ray.transmittance,     // 현재 누적 transmittance
                                  hitParticle.hitT,      // hit 깊이
                                  ray.hitT);             // 누적 깊이

particles.featureIntegrateFwd(hitWeight,                 // weight로 색상 합성
                              particleFeatures[hitParticle.idx],
                              ray.features);
```

#### `integrateHit` - weight 계산 및 transmittance 갱신

**파일**: `threedgut_tracer/include/3dgut/kernels/slang/models/gaussianParticles.slang:224-253`

```c
float integrateHit<let backToFront : bool>(
    in float alpha,
    inout float transmittance,
    in float depth,
    inout float integratedDepth, ...)
{
    // Front-to-back: weight = alpha * T
    const float weight = backToFront ? alpha : alpha * transmittance;

    // 깊이 누적
    integratedDepth += depth * weight;

    // transmittance 갱신: T *= (1 - alpha)
    transmittance *= (1 - alpha);

    return weight;
}
```

#### `integrateRadiance` - 색상 누적

**파일**: `threedgut_tracer/include/3dgut/kernels/slang/models/shRadiativeParticles.slang:84-99`

```c
void integrateRadiance<let backToFront : bool>(
    float weight,
    in vector<float, Dim> radiance,
    inout vector<float, Dim> integratedRadiance)
{
    if (weight > 0.0f)
    {
        // Front-to-back: C += color * weight
        integratedRadiance += radiance * weight;
    }
}
```

### 2.4 조기 종료

transmittance가 임계값 이하로 떨어지면 ray를 종료한다.

**파일**: `gutKBufferRenderer.cuh:167-169`

```cuda
if (ray.transmittance < Particles::MinTransmittanceThreshold) {  // 기본값 0.0001
    ray.kill();
}
```

### 2.5 최종 출력

**파일**: `threedgut_tracer/include/3dgut/kernels/cuda/common/rayPayload.cuh:159`

```cuda
// RGB = 누적 색상, Alpha = 1 - T (최종 불투명도)
radianceDensityPtr[ray.idx] = {ray.features[0], ray.features[1], ray.features[2],
                               (1.0f - ray.transmittance)};
```

## 3. 블렌딩 공식 요약

각 픽셀에서 ray를 따라 만나는 Gaussian $i = 1, 2, ..., N$ 에 대해 (깊이 순):

$$w_i = \alpha_i \cdot T_i, \quad T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)$$

$$C = \sum_{i=1}^{N} w_i \cdot c_i$$

| 기호 | 의미 |
|------|------|
| $\alpha_i$ | i번째 Gaussian의 불투명도 (ray-particle density에서 계산) |
| $T_i$ | i번째 Gaussian까지의 누적 투과율 (transmittance) |
| $w_i$ | i번째 Gaussian의 최종 기여 weight |
| $c_i$ | i번째 Gaussian의 색상 (SH로 디코딩) |
| $C$ | 최종 픽셀 색상 |

