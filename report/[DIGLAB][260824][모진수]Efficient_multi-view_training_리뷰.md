# Efficient Multi-View Training for 3D Gaussian Splatting 리뷰 — densification 통계량과 Jensen inflation의 관계

작성일 2026-08-24 · 모진수

## 요약

arXiv 2506.12727을 검토했다. 이 논문은 densification 통계량으로 각 gradient의
L2 norm을 먼저 계산한 뒤 합산하는 방식과, gradient vector를 먼저 합산한 뒤 L2
norm을 계산하는 방식을 제시한다. 그러나 두 통계량의 차이를 좌표계 문제로만
설명하고, sampling에서 발생하는 편향은 분석하지 않는다. 이 차이는 우리가 정의한
Jensen inflation과 직접 관련된다. 핵심 발견은 세 가지다.

### 발견 1 — 우리가 사용하는 두 통계량이 논문의 식 13과 14에 제시되어 있다

논문은 densification 기준으로 두 값을 정의한다.

- `E₁(𝒢) = Σₖ Σ ‖∇ℒ‖₂` — Gaussian과 교차한 픽셀별 gradient의 L2 norm을 합산한다.
- `E₂(𝒢) = Σₖ ‖Σ ∇ℒ‖₂` — view 안에서 gradient vector를 합산한 뒤 L2 norm을 계산하여 view별로 누적한다.

두 통계량은 각각 norm의 평균(mean of norms)과 평균 gradient의 norm(norm of mean)에
해당한다. L2 norm은 볼록 함수이므로 sampling 상황에서 `E‖g‖ ≥ ‖E g‖`가 성립한다.

논문은 둘 중 무엇을 쓸지를 이렇게 정한다.

> "We empirically found that E₁ is good at splitting, and E₂ is good at cloning."

논문은 두 기준을 실험 결과에 따라 선택했으며, 두 값이 달라지는 통계적 원인이나
sampling 비율에 따른 차이의 변화는 분석하지 않았다.

### 발견 2 — 논문은 두 통계량의 차이를 좌표계 문제로만 설명한다

논문이 원본 3DGS의 기준으로 제시한 식은 다음과 같다.

> `E_old(𝒢) = ‖Σₖ Σ ∇ℒ‖₂`

논문은 이 식의 문제를 다음과 같이 설명한다.

> "ADC involves the addition of 2D positional gradients, namely vectors. And
> vectors in different spaces cannot be added."

Figure 5에서는 서로 반대편에 있는 두 view를 예로 든다. 동일한 3D 방향의 gradient가
각 view의 2D image plane에 투영되면 x 성분이 상쇄되어 합이 0이 된다.

> "two 2D positional gradients are nullified even though two gradients represent
> the same 3D space gradient."

이에 따라 gradient vector 대신 L2 norm을 합산한다.

> "Compared to adding vectors, adding norm remains valid even though vectors are
> placed in different spaces."

이 설명은 서로 다른 image plane에 정의된 vector를 직접 합산할 수 없다는 기하학적
문제를 다룬다. L2 norm은 좌표계에 무관한 scalar이므로 view 간 합산이 가능하다.
그러나 좌표계는 E₁과 E₂의 차이를 발생시키는 원인 중 하나이며, sampling variance에
따른 차이는 별도로 분석해야 한다.

### 발견 3 — Jensen, 볼록성, 편향은 논문 어디에도 나오지 않는다

식 13과 14를 제안하면서 각 통계량이 추정하는 값이나 estimator의 bias는 논하지
않는다. 부록 8.2에서 좌표계 사이의 gradient 변환을 유도하지만 제안한 지표에 대한
이론적 보장은 다루지 않는다.

논문에 나오는 통계 논거는 Lemma 1 하나다.

> "For two cases which are 1) draw K sample from each 𝒟ᵢ and 2) choose i first,
> then draw NK i.i.d samples from 𝒟ᵢ, the former has a smaller variance of the
> sample mean than the latter."

Variance는 식 5에 다음과 같이 정의되어 있다.

> `𝕍(𝐱) := 𝔼[‖𝐱−𝝁‖²₂]`

이 식은 vector sample mean의 variance를 정의하며, norm을 적용한 통계량의 variance를
나타내지 않는다. 논문은 새로 제안한 densification 지표가 기존 지표의 variance
특성을 보존하는지 분석하지 않았다.

---

## 1. 논문이 무엇을 하는가

논문은 3DGS와 NeRF의 mini-batch 구성 차이를 문제로 제시한다. NeRF는 여러 이미지의
Ray로 mini-batch를 구성하지만, 3DGS는 한 장의 이미지로 mini-batch를 구성한다.
저자들은 단일 view 학습에서 stochastic gradient의 variance가 증가한다고 설명한다.

> "single-view training can lead to suboptimal optimization due to increased
> variance in mini-batch stochastic gradients"

제안 방법은 세 부분으로 구성된다.

**부분 렌더링.** 하나의 tile에 여러 view의 픽셀을 배치하고 rendering할 픽셀을 index
array로 지정한다. 전체 픽셀 수는 단일 view 학습과 동일하게 유지된다.

> "the model can be trained with the same number of pixels compared to the
> single-view training"

**3D 거리 인지 D-SSIM.** 다중 view 상황에 맞춘 손실 항이다.

**Multi-View Adaptive Density Control.** 앞서 설명한 E₁과 E₂를 densification 기준으로 사용한다.

MipNeRF-360에서 PSNR은 29.28 dB에서 29.74 dB로, SSIM은 0.878에서 0.886으로
증가했고, LPIPS는 0.167에서 0.154로 감소했다. 3DGS-MCMC에 적용했을 때 PSNR은
29.83 dB에서 30.42 dB로 증가했다. 학습 시간은 전체 다중 view rendering이 127분,
기존 부분 rendering이 105분, 제안 방법이 50분이었다. 각 iteration에 사용하는
이미지 수는 4개로 고정했다.

---

## 2. 다중 view에서 원본 densification 기준에 발생하는 문제

논문의 `E_old = ‖Σₖ Σ ∇ℒ‖₂`는 여러 view의 gradient vector를 모두 합산한 뒤 L2
norm을 계산하는 형태다. 이는 원본 3DGS의 계산 방식과 다르다.

단일 view 3DGS는 step마다 해당 view의 gradient L2 norm을 계산하여 누적한다. 한
step에 하나의 view만 사용하므로 서로 다른 view의 gradient vector가 합산되지 않는다.
따라서 이 계산은 `E₁`에 가깝다.

다중 view에서는 N개 view의 loss를 한 번에 backward하면 각 view의 gradient가 같은
buffer에 누적된 후 L2 norm이 한 번만 계산된다. 이 계산은 `E_old`에 해당하며,
Figure 5에서 설명한 gradient cancellation이 발생할 수 있다.

따라서 논문이 지적한 gradient cancellation은 다중 view 학습에서 발생한다. 단일
view 학습에는 없던 문제이므로, 다중 view 학습을 적용하려면 densification 기준도
함께 변경해야 한다.

MVGS(arXiv 2410.02103)는 이 문제를 직접 해결하지 않는다. 공개된 `train.py`에서는
`pipe.mv`개의 view에서 loss를 합산한 뒤 한 번 backward하는데,
`add_densification_stats`를 반복문 밖에서 호출한다. 이 시점의
`viewspace_point_tensor`는 **마지막 view의 값**이므로, 여러 view의 통계량을 합산하지
않고 마지막 view의 값만 사용한다.

---

## 3. Sampling에 따른 Jensen inflation

논문은 E₁과 E₂의 차이를 좌표계로 설명하지만, 동일한 좌표계에서도 sampling
variance가 존재하면 두 통계량에 차이가 발생한다.

Gaussian 하나의 gradient를 확률변수 `g`, 그 평균과 분산을 각각 `μ`, `σ²`라고 하자.
L2 norm은 볼록 함수이므로

```
E‖g‖ ≥ ‖E g‖ = ‖μ‖
```

이며, 등호는 분산이 0일 때 성립한다. 따라서 norm을 먼저 계산한 뒤 평균하면 평균
gradient의 norm보다 큰 값이 나오며, 그 차이는 sampling variance에 의해 발생한다.

E₁은 픽셀별 gradient norm을 합산하므로 부등식의 왼쪽에 해당한다. E₂는 view 안에서
gradient vector를 먼저 합산하므로 오른쪽에 가깝다. 논문은 두 통계량을 모두
사용하지만 이 관계는 분석하지 않는다.

Dense 학습에서는 한 view의 모든 픽셀을 사용하므로 픽셀 sampling으로 인한 variance가
발생하지 않는다. 이 조건에서는 E₁과 E₂의 차이를 좌표계 문제로 설명할 수 있다.
반면 sparse Ray 학습에서는 픽셀 sampling variance도 두 통계량의 차이에 포함된다.

우리 설정은 전체 픽셀의 1/16에 해당하는 Ray만 캐스팅한다. 따라서 step별 gradient는
full-Ray gradient의 추정값이며 variance가 0이 아니다. 이때 E₁은 평균 gradient의
norm을 과대 추정한다. 과대 추정 비율은 `φ = E‖g‖ / ‖E g‖`로 정의한다.

측정할 Ray를 같은 view에서 독립적인 두 표본 A와 B로 나누면

```
E[g_A · g_B] = ‖μ‖²
```

이므로 full-Ray gradient의 squared L2 norm을 추정할 수 있다. 기존에 캐스팅할 Ray를
두 표본으로 나누므로 추가 Ray casting 비용이 없으며, dense rendering 결과도 필요하지 않다.

sampling 비율 1/16에서 측정한 φ는 Gaussian당 step별 Ray hit 수 `h`에 따라 달라졌다.
`h`가 약 1.4일 때 φ는 2.568이고, 약 162일 때는 1.275였다. 씬 간 차이는
10~12% 범위였다.

이 과대 추정은 최종 Gaussian 개수에도 반영된다. counter의 15000 step에서 full-Ray
학습 결과는 1,254,878개였고, Ray를 1/16만 캐스팅하면서 별도 보정을 적용하지 않은
결과는 2,154,760개였다. Sparse Ray 학습의 Gaussian 수가 **1.72배 많았다.**

---

## 4. Lemma 1의 적용 범위

Lemma 1은 여러 분포에서 각각 sample을 추출한 경우가 하나의 분포를 먼저 선택한 뒤
같은 수의 sample을 추출한 경우보다 **sample mean의 variance가 작다**는 내용이다.
논문에서 사용한 variance도 vector에 대해 정의된다.

이 보조정리는 다중 view가 씬 전체 gradient를 더 낮은 variance로 추정한다는 근거가
된다. 그러나 densification은 씬 전체 gradient가 아니라 **Gaussian별 gradient norm
통계량**을 사용하므로 적용 대상이 다르다.

counter의 15000 step checkpoint에서 전체 2,154,760개 Gaussian 중 모든 설정에서
공통으로 관측된 554,762개를 분석했다. step당 Ray 예산을 약 10만 개로 고정하고
view 수만 변경하여 inflation 계수 φ를 측정했다.

| step당 view 수 | block 수 | inflation 계수 φ 중앙값 | 단일 view 대비 |
|---:|---:|---:|---:|
| 1 | 4 | 4.878 | 1.000 |
| 2 | 6 | 5.32 | 1.091 |
| 4 | 8 | 5.70 | 1.168 |
| 8 | 11 | 5.92 | 1.213 |
| 16 | 16 | 5.92 | 1.214 |

예산을 고정하고 view 수를 1개에서 16개로 늘리면 inflation 계수의 중앙값이
4.878에서 5.92로 **21.4% 증가했다.** View 수가 증가하면 view당 픽셀 sampling
density가 감소한다. 각 Gaussian은 일부 view에서만 관측되므로 Gaussian당 Ray hit
수도 감소하며, φ는 Ray hit 수가 적을수록 증가한다.

이 결과는 Lemma 1과 모순되지 않는다. Lemma 1은 씬 전체 gradient의 sample mean을
다루지만, densification은 Gaussian별 gradient norm을 사용한다. 또한 논문의 dense
학습 설정은 모든 픽셀을 사용하므로 Gaussian당 Ray hit 수 감소에 따른 inflation이
나타나지 않는다.

---

## 5. 우리 설정과의 관계

고정 Ray 예산을 여러 view에 배분하는 학습 구조는 이 논문에서 이미 제안했다.
따라서 고정 예산 다중 view 학습 자체는 우리 연구의 기여로 보기 어렵다.

두 연구의 차이는 세 가지다. 첫째, 이 논문은 rasterization 기반이므로 부분 rendering
모듈이 필요하지만, Ray tracing에서는 casting할 Ray를 선택하는 방식으로 구현할 수
있다. 둘째, 이 논문은 다중 view 학습을 전체 학습 구간에 적용하므로 densification
기준도 변경한다. 우리는 densification이 종료된 15000 step 이후에만 다중 view를
적용하므로 해당 통계량을 변경하지 않는다. 셋째, 이 논문은 dense 학습을 전제로
하므로 픽셀 sampling에 따른 Jensen inflation을 다루지 않는다.

---

## 참고 문헌

- Efficient multi-view training for 3D Gaussian Splatting — arXiv 2506.12727
- MVGS: Multi-view Regulated Gaussian Splatting for Novel View Synthesis — arXiv 2410.02103
- On Scaling Up 3D Gaussian Splatting Training — arXiv 2406.18533
- Stochastic Ray Tracing for the Reconstruction of 3D Gaussian Splatting — arXiv 2603.23637
