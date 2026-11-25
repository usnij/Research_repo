# Counter: 고정 체크포인트 Dense 300회 / Sparse 300회 분모 검증

**목적:** Sparse의 densification 점수가 커지는 현상을, **gradient가 발생한 step 수를 분모로 쓰는 영향**과 **분모를 Dense와 같게 맞춰도 남는 차이**로 나누어 확인한다.



## 요약

기존에 실측한 jensen에 의한 부풀림현상은 sparse로 인한 gaussian의 관측 부족이 핵심 원인은 아닌 것을 확인했다. sparse에서 평균적으로 gaussian의 관측step이 더 적어 densify를 진행하는 thresholding 값에 영향을 주는 것은 맞으나 이것이 주원인아니다.

### 발견 1 - 실험을 진행한 핵심 원인이자 가설인 **"sparse에서 통계량 부품 현상은 관측 step수를 분모로 가져가며 이 숫자가 작아서 통계량이 커지는 단순한 착시일 것"** 는 기각이다. 
좀 더 명확하게 설명하면 Sparse에서 gaussian의 관측 step 수 자체는 어느 정도 감소하기 때문에 통계량 부풀림의 일부에 기여한다. 그러나 분모를 맞춘 뒤에도 Dense 대비 1.254–1.623배가 남아, 관측 step 수 차이만으로는 부풀림을 설명할 수 없다.

### 발견 2 - dense와 sparse에서 gaussian의 관측되는 step 수의 차이는 크지 않다. 

실험 전 가설은 gaussian의 관측 step수 자체가 차이가 커서 생기는 영향이 클 것으로 판단했으나 학습이 종료된 모델(30K) 300step기준으로 dense 86.70회 sparse 69.15회 로 큰 차이는 아닌 것으로 보인다.

또한 300step 대비 관측 수 자체가 적은 이유는 한 view에서 렌더링되는 gaussian자체가 모든 gaussian을 포함하지 않기 때문이라고 추측된다. 


### 결론 : 
**정리하면 sparse로 인해 gaussian의 관측 step이 작아지고 이로인해 densify 판단값이 커지는 것은 영향은 있지만 주원인은 아니다. 추측하건데 주원인은 gaussian의 총 hit ray수가 적어 gradient의 norm이 편향되는 것으로 생각된다.** 



## 1. 고정 체크포인트 준비 — 학습 OFF

8월 Counter dense 실행(`counter-1508_142158`)의 7k·30k 모델을 각각 고정했다. **Dense와 Sparse를 각각 300회 직접 측정했으며, 추가 학습은 하지 않았다.** 매 측정 step에서 같은 모델·같은 view로 Dense와 Sparse의 forward → backward를 각각 수행했다.



| 고정 모델 | 복원한 파일 | 전체 Gaussian 수 | Dense 측정 | Sparse 측정 |
|---|---|---:|---:|---:|
| 7k | `ckpt_7000.pt` | 820,791개 | 300회 | 300회 |
| 30k | `export_last.ingp` | 1,255,427개 | 300회 | 300회 |

- **고정 조건:** optimizer 업데이트·Gaussian 생성·삭제 OFF. 두 모델 모두 측정 전후 파라미터 해시가 동일했다.
- **View:** Counter DS2의 학습 view 210개에서 중복을 허용해 300회 선택. Dense와 Sparse는 같은 순서를 사용했다(고유 view 159개, view seed 11, sparse seed 5).
- **Ray:** Dense 1,617,204개/회. Sparse는 4×4 block마다 독립적으로 1픽셀을 뽑아 101,400개/회(약 1/16).
- **손실:** `0.8 L1`과 `0.8 L1 + 0.2 SSIM`을 각각 측정. L1 비교는 경계 block 면적을 가중해 Dense와 같은 목적함수로 맞췄다. 혼합 손실은 Dense에서 전체 이미지 SSIM, Sparse에서 compact SSIM을 사용했다.

## 2. 기준값 측정 — Full ray(Dense) 300회

각 Gaussian에 대해 다음을 저장했다. 사진의 `N_full`은 아래의 `N_dense`에 해당한다.

- **`S_dense`**: 300회에 걸친 위치-gradient 크기의 합. 원 3DGRT ADC와 동일하게 매회 **`위치-gradient norm × 카메라와 Gaussian 사이 거리 / 2`**를 누적했다.
- **`N_dense`**: 300회 중 위치 gradient가 0이 아니었던 step 수.
- **`A_dense = S_dense / N_dense`**: Gaussian별 Dense 기준 평균 점수.

아래 표는 **`N_dense > 0`인 Gaussian**을 집계했다(7k 814,745개, 30k 1,102,862개). ‘평균 step 수’는 **Gaussian 하나당 300회 중 평균 몇 회 gradient가 발생했는가**를 뜻한다.

| 고정 모델 | 손실 | S_dense 중앙값 | N_dense 평균 step 수 | A_dense 중앙값 |
|---|---|---:|---:|---:|
| 7k | 0.8 L1 | 3.9840 × 10⁻⁴ | 81.94회 | 9.7145 × 10⁻⁶ |
| 7k | 0.8 L1 + 0.2 SSIM | 8.4557 × 10⁻⁴ | 81.94회 | 2.0574 × 10⁻⁵ |
| 30k | 0.8 L1 | 6.6347 × 10⁻⁴ | 86.70회 | 1.1560 × 10⁻⁵ |
| 30k | 0.8 L1 + 0.2 SSIM | 1.3000 × 10⁻³ | 86.70회 | 2.4302 × 10⁻⁵ |

`A_dense`는 Gaussian마다 먼저 `S_dense / N_dense`를 계산한 뒤 중앙값을 구했다. 표의 S 중앙값을 N 평균으로 나눈 값이 아니다.
Dense에서 300회 모두 위치 gradient가 0인 Gaussian은 7k **6,046개**, 30k **152,565개**이며 위 집계에서 제외했다.

## 3. 비교값 측정 — Sparse ray(1/16), 동일한 300회

Dense와 **동일한 view 순서·동일한 고정 Gaussian**에 대해 Sparse ray로 직접 forward → backward를 수행했다. 각 Gaussian의 **`S_sparse`(거리 보정 gradient 크기의 합)**와 **`N_sparse`(gradient가 0이 아니었던 step 수)**를 저장했다.

`N_hit`은 여기서는 **`N_sparse`**로 표기한다. 이 값은 ray 개수가 아니라 **위치 gradient가 0이 아니었던 step 수**이며, 실제 hit 기록 `H`와는 별도로 측정했다.

아래 표는 2절과 **같은 `N_dense > 0` 집단**을 사용했다. Sparse에서 300회 모두 gradient가 0인 Gaussian도 S와 N에 0으로 포함했다.

| 고정 모델 | 손실 | S_sparse 중앙값 | N_sparse 평균 step 수 |
|---|---|---:|---:|
| 7k | 0.8 L1 | 5.5842 × 10⁻⁴ | 73.50회 |
| 7k | 0.8 L1 + 0.2 SSIM | 1.0468 × 10⁻³ | 73.50회 |
| 30k | 0.8 L1 | 1.0725 × 10⁻³ | 69.15회 |
| 30k | 0.8 L1 + 0.2 SSIM | 1.7700 × 10⁻³ | 69.15회 |

**실제 hit 기록 비교.** `H`는 렌더러가 해당 Gaussian을 한 ray 이상에서 합성 대상으로 받아들였다고 기록한 step 수다. 한 step에서 여러 ray가 hit해도 1회로 센다. 아래 평균 역시 2절과 같은 `N_dense > 0` 집단이다.

| 고정 모델 | H_dense 평균 step 수 | H_sparse 평균 step 수 | 평균 감소 step 수 |
|---|---:|---:|---:|
| 7k | 82.09회 | 73.77회 | 8.32회 |
| 30k | 86.95회 | 69.73회 | 17.22회 |

**30k 모델의 실제 hit 빈도 분포.** 이 표는 **전체 Gaussian 1,255,427개**를 분모로 한다. Dense는 `H_dense`, Sparse는 `H_sparse`로 각각 구간을 나눴다.

| hit된 step 수 (총 300 step) | Dense Gaussian 비율 | Sparse Gaussian 비율 |
|---|---:|---:|
| 1–30회 | 31.04% | 35.91% |
| 31–100회 | 24.00% | 25.82% |
| 101–200회 | 22.28% | 20.21% |
| 201–300회 | 10.58% | 5.16% |



## 4. 분모 두 방식으로 나누어 Sparse / Dense 배율 비교

같은 Gaussian의 **Sparse 분자 `S_sparse`를 그대로 두고 분모만 바꿨다.** 비교 기준은 두 경우 모두 `A_dense = S_dense / N_dense`이다.

| 비교 방식 | Sparse 점수 | Dense 대비 배율 |
|---|---|---|
| **0 제외 — 기존 3DGRT 분모** | `S_sparse / N_sparse` | `(S_sparse / N_sparse) / (S_dense / N_dense)` |
| **0 포함 — Dense 분모로 고정** | `S_sparse / N_dense` | `(S_sparse / N_dense) / (S_dense / N_dense)` |

여기서 ‘0 포함’은 **Sparse에서도 Dense의 관측 횟수 N_dense로 나누는 통제 계산**이다. 

**Gaussian별 배율의 중앙값.** 분모의 영향만 비교하도록 **두 열 모두 `N_dense > 0`이고 `N_sparse > 0`인 같은 Gaussian 집단**으로 맞췄다.

| 고정 모델 | 손실 | 0 제외: N_sparse로 나눔 | 0 포함: N_dense로 나눔 |
|---|---|---:|---:|
| 7k | 0.8 L1 | **1.448×** | **1.340×** |
| 7k | 0.8 L1 + 0.2 SSIM | **1.369×** | **1.254×** |
| 30k | 0.8 L1 | **1.995×** | **1.623×** |
| 30k | 0.8 L1 + 0.2 SSIM | **1.693×** | **1.392×** |


분모를 Dense와 동일하게 맞추면 배율이 낮아지므로, Sparse의 관측 step 수 감소가 통계량 부풀림의 일부에 기여한다. 그러나 분모를 맞춘 뒤에도 Dense 대비 1.254–1.623배가 남아, 관측 step 수 차이만으로는 부풀림을 설명할 수 없다



**Gaussian 크기 구간별 관측 step 비율.**  각 값은 **Gaussian별 비율의 중앙값**이다.

| 고정 모델 | Gaussian 크기 구간 | Gradient N_sparse / N_dense | 실제 hit H_sparse / H_dense |
|---|---|---:|---:|
| 7k | 하위 25% — 작음 | **90.0%** | **90.3%** |
| 7k | 25–50% | **95.6%** | **95.7%** |
| 7k | 50–75% | **97.3%** | **97.4%** |
| 7k | 상위 25% — 큼 | **98.8%** | **98.9%** |
| 30k | 하위 25% — 작음 | **70.6%** | **71.4%** |
| 30k | 25–50% | **86.2%** | **86.9%** |
| 30k | 50–75% | **91.9%** | **92.3%** |
| 30k | 상위 25% — 큼 | **96.1%** | **96.3%** |

