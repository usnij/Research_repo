# Z-Thickness Blending: Effective Fragment Merging for Multi-Fragment Rendering
Dongjoon Kim, Heewon Kye | Pacific Graphics 2021 (CGF Vol.40, No.7) | DOI: 10.1111/cgf.14409

해당 논문에 대한 이해를 목적으로 논문을 읽고 정리함

---

## 1. Introduction

### Multi-Fragment Rendering의 문제

- **Multi-Fragment Rendering**이란 하나의 픽셀 위치에 여러 fragment가 존재할 때 이를 처리하는 렌더링 기법이다.
  - GPU 파이프라인에서 3D 기하 정보를 framebuffer에 저장하고 최종 이미지를 생성할 때 동일 픽셀에 여러 fragment가 동시에 들어올 수 있다.
  - Order-independent transparency(OIT), dynamic photorealistic rendering, screen-space global illumination 등 다양한 응용에 사용된다.

- 대표적인 두 가지 문제:
  1. **Fragment Overflow**: 복잡한 씬에서 fragment의 수가 k-buffer 용량을 초과해 정보가 손실 → 잘못된 occlusion과 flickering 아티팩트 발생
  2. **Z-Fighting**: 깊이 값이 비슷한 coplanar fragment 간 부동소수점 반올림 오류로 인해 깜빡임/얼룩 발생

### 기존 방법들의 한계

- **k-buffer 계열**: 고정 수 k의 fragment만 저장 → overflow 시 기하 정보 손실, coplanar fragment를 별도 처리해야 함
- **Transmittance 함수 근사 방식 (SML11, MKKP18)**: coplanar/z-fighting 처리가 불완전하고 픽셀 간 일관성 없는 투명도 결과 발생
- **Tail-handling 방식 (MCTB13)**: 멀리 있는 fragment를 tail fragment로 근사 → 겹치는 fragment의 부분 가시성 미반영

### Z-Thickness Blending의 핵심 아이디어

- 각 fragment에 **z-방향의 가상 두께(z-thickness)**를 부여해, 표면을 반투명 매질 밴드로 모델링한다.
- 이를 기반으로 두 가지 **fragment merging** 방식을 제안한다.
  - **SFM (Smooth Fragment Merging)**: 인접 fragment를 세분·합성해 부드러운 가시성 전환 제공 (depth-sorted 환경)
  - **OFM (Order-independent Fragment Merging)**: 순서와 무관하게 겹치는 fragment를 하나로 병합 (k-buffer 환경)

- 주요 기여 요약:
  - Z-fighting 처리: z-resolution 기반 z-thickness 값 결정 → coplanar layer 자연스럽게 블렌딩
  - 메모리 효율적 다중 레이어 표현: 인접 fragment 병합, 먼 fragment 분리 저장
  - 부드러운 가시성 전환: 겹침 정도에 따른 시각적 효과
  - 다양한 screen-space 렌더링 알고리즘 적용 가능

---

## 2. Related Work

- **A-buffer / k-buffer** (Car84, BCL*07): 픽셀당 linked list 또는 고정 크기 GPU buffer로 k개의 fragment를 캡처. 무제한 메모리 요구 또는 fragment overflow 문제 존재.
- **k^+ buffer** (VPF15): dynamic k 값과 최적 GPU 구현을 갖춘 효율적인 k-buffer. 본 논문에서 bounded memory 구현의 기반.
- **MBT** (MKKP18): transmittance 함수를 moment로 이론적 근사. 교차 기하 처리 불가.
- **Weighted Blended OIT** (MB13): opacity와 깊이 기반 weighted blending operator. 정렬 불필요하나 coplanar 처리 불완전.
- **Layered weighted blended OIT** (FEE20): 일정 간격의 레이어를 로컬 블렌딩해 weighted blending operator 개선. 교차 기하 처리가 시각적으로 구분되지 않는 한계.
- **Core-tail 방식** (MCTB13, SV14): foremost fragment를 정렬 보존하고 나머지를 tail로 누적. 겹치는 fragment 고려 부족.
- **Binary z-test 기반 방법** (VF12): coplanar speckling 억제. 별도의 coplanar fragment 추출 처리 필요.

---

## 3. Method

- 핵심 아이디어: z-방향 가상 두께를 각 fragment에 부여하고, 이를 기반으로 인접 fragment를 병합해 multi-fragment rendering의 제한된 buffer 문제를 해결한다.

### 3.1 Z-Thickness Surface Model

- 각 fragment에 z-방향의 두께(t)를 할당하여, 원래 표면 앞쪽에 **반투명 매질 밴드(translucent band)**를 형성하는 것으로 모델링한다.

![Figure 2: z-thickness 모델. d_{1,2,3}은 ray의 깊이값, t_{1,2,3}은 각 깊이에서의 z-thickness 값]

- z-thickness 모델의 두 가지 이점:
  1. **Z-Fighting 해결**: speckling·noisy 가시성 대신 연속적인 blending 가시성 표시 (Figure 3)
  2. **메모리 효율성**: 가까이 위치한 모든 겹친 layer를 저장하는 대신, 인접 layer를 병합하고 먼 layer는 분리 저장 (Figure 4)

### 3.2 Fragment Merging

- fragment merging은 인접한 N개의 fragment를 M개(M ≤ N)로 병합하는 근사 모델이다.
- 병합된 fragment는 다음 두 가지로 정의:
  - (i) 인접 fragment들의 최대 깊이 값
  - (ii) 인접 fragment들을 아우르는 z-thickness 값

#### 3.2.1 Smooth Fragment Merging (SFM)

- 두 fragment가 부분적으로 겹칠 때, fragment를 **세분(subdivide)**하고 각 세분 fragment의 가시성을 합성한다.
- 깊이 정렬 순서가 보장될 때 사용한다.
- 두 fragment가 겹치면 최대 3개의 세분 fragment 생성 후 병합
- visibility는 **over operator** (front-to-back blending)로 결정
- 전제 조건: 같은 픽셀에 들어오는 fragment의 깊이 순서가 확정되어야 함 (미보장 시 픽셀 간 일관성 없는 결과)

#### 3.2.2 Order-independent Fragment Merging (OFM)

- 겹치는 fragment를 **순서와 무관하게** 하나의 fragment로 병합한다.
- k-buffer처럼 fragment 순서가 불확정적인 store pass에서 사용한다.
- visibility는 **mix-operator** (Section 3.3.2)로 결정 → order-independent 가시성
- fragment 세분 없이 단순히 겹치는지 여부만 판단해 병합 → SFM보다 단순하지만 덜 정밀

#### 3.2.3 SFM과 OFM의 시각적 차이

- **SFM**: 겹침 정도(degree of partial overlap)를 고려 → 교차 지점 주변 픽셀에서 부드러운 색상 전환 (anti-aliasing 효과)
- **OFM**: 겹치는지 여부만 판단 → 단조로운 가시성 전환이지만 순서 독립적

### 3.3 Visibility Decision

- z-thickness 모델에서 겹치는 fragment의 가시성을 분할·합성하는 광학 연산자를 제시한다.

#### 3.3.1 Visibility Subdivision

- 각 fragment는 homogeneous medium으로 가정하므로, **소광 계수(extinction coefficient) τ**가 상수이다.
- 깊이 z까지의 누적 불투명도:

```math
A(z) = 1 - e^{-\tau z}
```

- fragment 두께 d에서의 불투명도를 A_d = A(d)라 할 때:

```math
A(z) = 1 - (1 - A_d)^{z/d} \quad (0 \le z \le d)
```

- 방출-흡수 광학 모델(Max95)에 기반해, 깊이 z에서의 누적 색상:

```math
C(z) = C_d \cdot \frac{A(z)}{A_d}
```

- β 파라미터로 선형 근사 추가 (시각적 제어):

```math
A(z) = \beta \frac{z}{d} A_d + (1-\beta)\left(1-(1-A_d)^{z/d}\right) \quad (0 \le \beta \le 1)
```

  | β 값 | 효과 |
  |------|------|
  | 0 | 순수 지수 감쇠 (물리적으로 정확) |
  | 1 | 선형 근사 (교차 지점에서 더 부드러운 전환) |
  | 크게 (≈1) | 내부 구조가 보이는 **ghosted 효과** 가능 |

- 앞쪽 세분 fragment (C_d, A_d)에서 뒤쪽 세분 fragment의 가시성 (C_b, A_b):

```math
(C_b, A_b) = \frac{(C_d, A_d) - (C(z), A(z))}{1 - A(z)}
```

#### 3.3.2 Mix Operator

- 완전히 겹치는 fragment들이 순서와 무관하게 동등하게 기여해야 할 때 사용한다.
- OFM에서 사용하며, opacity와 color를 순서 독립적으로 누적한다.

- **누적 opacity** A_acc:

```math
A_{acc} = 1 - \prod_i (1 - A_i)
```

- **누적 color** c_acc (각 fragment의 opacity로 정규화):

```math
c_{acc} = \frac{\sum_i c_i A_i}{\sum_i A_i}, \quad C_{acc} = c_{acc} \cdot A_{acc}
```

- 새 fragment가 들어올 때 incremental 업데이트:

```math
\tilde{A}_{acc} = 1 - (1 - A_{acc})(1 - A_{new})
```
```math
\tilde{C}_{acc} = \frac{\frac{C_{acc}}{A_{acc}} A_{sum} + C_{new}}{A_{sum} + A_{new}} \cdot \tilde{A}_{acc}
```

---

## 4. Implementation

### 4.1 공통 연산

#### 4.1.1 Fragment Merging

- N개의 fragment를 M개로 병합 (M ≤ N), 선형 시간 복잡도 O(N)
- **Store pass**: OFM 사용 (fragment 순서 불확정)
- **Resolve pass**: SFM 사용 (fragment 깊이 정렬 후)
- Store pass에서 OFM 적용 후 Resolve pass에서 SFM을 추가로 적용하는 것은 **불필요** (store pass에서 이미 인접 fragment가 병합됨)

#### 4.1.2 Tail-handling

- multi-fragment rendering은 복잡한 씬에서 fragment overflow를 완전히 피하기 어려움
- 가장 먼 k-front fragment 너머의 fragment를 **tail fragment**로 누적해 처리
- Store pass: mix-operator로 tail 처리 (순서 불확정)
- Resolve pass: over-operator로 tail 처리 (순서 확정)

#### 4.1.3 Z-Thickness 값 결정

- z-fighting 해결을 위해 **z-resolution**을 기반으로 z-thickness 값을 결정한다.
- z-buffer의 비선형성으로 인해, 깊이 z에서의 z-resolution P(z):

```math
P(z) = \frac{b}{\frac{b}{z} - \frac{1}{n}} - z, \quad \left(b = \frac{z_n z_f}{z_n - z_f}\right)
```

  - z_n: near plane 깊이, z_f: far plane 깊이, n: 단정밀도 부동소수점 정밀도 비트 수

- 기본 z-thickness 값: **2 × P(z)** → z-fighting 해결에 충분한 여유 확보
- 사용자가 시각화 목적에 따라 수동 조절 가능

### 4.2 Bounded Memory 방식 (k-buffer)

- k^+ buffer (VPF15) 기반으로 고정된 수 M = k의 fragment만 저장
- Store pass에서 **rasterizer-ordered-view (atomic 처리)** 로 data race 방지
- **max array buffer** + **fragment culling**: 최대 깊이 fragment를 tail에 할당하는 방식으로 k-front fragment 효율적 교체
- OFM은 store pass에서 순서 독립적으로 직접 구현 가능

### 4.3 Unbounded Memory 방식

- **Dynamic framebuffer** (MCTB12) 기반: atomic count instruction으로 data race 없이 모든 fragment 저장
- Count pass → Store pass → Resolve pass 3단계 구성
- Resolve pass에서 SFM 적용
- 최대 1024개/픽셀까지 사전 할당 → k-buffer 대비 느리지만 더 정확한 결과

---

## 5. Experiments

- 실험 환경: Intel Core i7700 3.6GHz, 32GB RAM, NVIDIA GeForce GTX 1080, Windows 10 x64
- 출력 fragment 수: M = 8 (Algorithm 1 기준)
- 비교 대상:
  - **DFB** (Dynamic Fragment Buffer): unbounded memory, 레퍼런스 기준
  - **MBT** (Moment-based Transmittance): 교차 기하 처리 불가
  - **SKB** (Static k^+ Buffer): bounded memory, fragment overflow 취약
  - **SKB+OFM**: 제안 방법 (bounded + OFM)
  - **DFB+SFM**: 제안 방법 (unbounded + SFM)

### 5.1 Order Independent Transparency

#### 5.1.1 시각적 분석 (Visual Analysis)

- **SKB+OFM**: SKB 대비 limited memory에서 OIT 품질 향상, fragment overflow 문제 완화
- **DFB+SFM**: DFB 대비 negligible한 품질 손실로 동등한 결과 제공
- 제안 방법은 복잡한 교차 기하(surfel, 포인트 클라우드)에서도 absorbance function이 DFB와 유사하게 나타남 (Figure 13)

#### 5.1.2 성능 분석 (Performance Analysis)

- SFM의 Resolve pass 오버헤드: sorting 오버헤드 대비 **무시할 수준**
- 병합 fragment는 픽셀 동기화 시 read-write-modify 연산 횟수를 변화시켜 store pass 성능에 영향을 줄 수 있으나 시각적 이득 대비 미미
- 이미지 해상도 1024×1024 및 2048×2048에서 비교 (Figure 12)

### 5.2 Z-Thickness 효과 변화 (Varying Z-Thickness Effects)

#### 5.2.1 Local Depth Blending

- SFM은 깊이 순서와 겹침 정도를 기반으로 이미지를 블렌딩 → **거리 단서(distance cue)** 시각화
- 포인트 클라우드(surfel) 렌더링 시 z-fighting speckling 제거
- z-thickness = 10 × P(z)로 설정 시, 인접 surfel 간 자연스러운 색상 전환 (Figure 14)
- Comparative visualization: color mapping 대신 smooth visibility transition으로 표면 간 거리 정보 표현 가능

#### 5.2.2 Ghosted Illustration

- 외부 표면의 가시성 대비를 낮춰 **내부 구조를 투시**하는 효과
- β = 1로 설정 + 큰 z-thickness 값 → 오브젝트들이 서로 겹치도록 해 ghosted 효과 강조
- 내부 오브젝트가 낮은 대비로 깊이 단서와 함께 보임 (Figure 9, 15)

### 5.3 Volumetric Object와의 Hybrid Rendering

- 얇은 투명 다각형 표면(polygonal surface)과 두꺼운 반투명 볼륨 표면(volumetric surface)의 교차 처리
- 기존 normal sampling: banding artifact 발생 (Figure 16(b))
- **DFB+SFM**: smooth visibility transition → volume ray-casting의 sampling discontinuity 아티팩트 제거, supersampling 결과와 유사한 고품질 이미지 (Figure 16(c))
- 실제 volume 데이터에 superimposed plane 렌더링 시에도 SFM 효과 확인 (Figure 16(d)(e))

### 5.4 Screen-Space Rendering으로의 확장

- Screen-space 렌더링(AO, DOF 등)은 framebuffer에 저장된 screen-space 기하 정보를 활용
- 단일 레이어만으로는 복잡한 기하 교차 처리 불충분 → multi-layered 방식 필요
- z-thickness 모델 기반 병합 fragment의 **front/back 경계**를 screen-space 기하로 활용
  - **Virtual front surface**: 원래 geometry shape 유지
  - **Back boundary**: z-thickness 모델 기반 깊이 값 사용
- **Dynamic Ambient Occlusion(AO)** + **Depth-of-Field(DOF)** 에 적용 (Figure 17)
  - 반투명 primitive를 포함한 씬에서 성공적으로 screen-space 렌더링 효과 적용

---

## 6. Conclusions

- z-thickness 모델 기반의 fragment merging 기법을 제안하였다.
- 핵심 성과:
  1. Fragment overflow의 조기 발생을 완화해 SKB의 OIT 성능 향상
  2. Coplanar/z-fighting 처리를 blending visibility로 자연스럽게 해결
  3. SFM을 통한 smooth visibility transition (다양한 시각화 응용)
  4. Multi-layered screen-space rendering에 적용 가능
- 두 가지 구현 방식 제공:
  - **DFB+SFM**: unbounded memory, 정확한 결과
  - **SKB+OFM**: bounded memory, 효율적 성능

- **한계 및 향후 연구**:
  - z-thickness 값(4 bytes float) + 누적 opacity(4 bytes float) 추가 저장 → 겹치는 layer가 없는 씬에서 불필요한 메모리 낭비
  - Subdivided fragment의 가시성 결정을 higher-order approximation (MKKP18)과 결합 시 추가 개선 가능
  - Pixel area에 따른 attention-based level-of-detail 방식으로 z-thickness 값 적응적 결정 가능

---

## 참고

- 논문 파일: `논문/CGF_PG21_zThickness.pdf`
