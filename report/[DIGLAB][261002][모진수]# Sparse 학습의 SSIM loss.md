# Sparse 학습의 SSIM loss

작성일 2026-10-02 · 모진수 

이 리포트는 sparse Ray 학습에서 SSIM loss가 dense와 다른 양을 계산하고 gradient가 다른 범위에만 도달하는 문제를 정의하고, 그 두 측면을 각각 고치려고 설계한 대안의 전체 학습 결과를 보고한다. 모든 비교는 같은 체크포인트에서 분기한 paired 실험이고 전체 test 30장으로 평가했다.

---

## 요약

이번 리포트의 핵심 발견은 세 가지다.

### 발견 1 — sparse SSIM loss는 dense와 다른 양을 계산하고, 계산에 쓴 범위의 6.25%에만 gradient를 전달한다

dense 3DGS는 이미지의 11×11 window, 즉 121개 pixel의 국소 평균, variance, covariance로 SSIM을 계산하고 그 121개 pixel 전부에 gradient가 도달한다. sparse 학습의 SSIM은 표집된 Ray를 sparse 2차원 격자로 모은 뒤 그 격자에 같은 11×11 window를 적용한다. block size 4에서 sparse 격자의 1 pixel은 4×4 stratum에서 추출한 Ray 1개이므로, sparse 11×11 window는 44×44 = 1,936개 pixel 범위에 대응한다.

여기서 두 가지가 동시에 달라진다. 첫째, 국소 moment를 계산하는 공간 규모가 121개 pixel에서 1,936개 pixel 범위(44x44)로 16배 증가하므로 dense와 같은 국소 통계량이 아니다. 둘째, 그 1,936개 pixel 중 Ray가 배정된 121개, 즉 6.25%에만 gradient가 도달하고 나머지 1,815개 pixel이 덮는 내용은 구조 supervision을 전혀 받지 못한다. sparse 격자는 서로 인접하지 않은 pixel들을 인접한 것으로 취급하므로, window 안에서 등간격 가정도 성립하지 않는다.

이로 인해 단일스텝(동일한 view)를 반복해서 학습할 때 L1 loss는 dense의 gradient양으로 수렴하는데 ssim-loss는 수렴하지 않는다

### 발견 2 — dense 학습에서 SSIM 항을 제거하면 PSNR은 0.012103 dB만 감소하지만 SSIM은 0.011554, LPIPS는 0.022456 나빠진다. SSIM 항의 기여는 pixel 오차가 아니라 구조·지각 품질에 있다

counter를 전체 ray로 초기화부터 native MCMC로 30,000 step 학습하고, 목적함수만 `0.8 L1 + 0.2 (1 - SSIM_11x11)`에서 `1.0 L1`으로 바꿔 비교했다. SSIM 제거로 영상 손실 전체 규모가 20% 줄어드는 것을 피하기 위해 L1 가중을 1.0으로 두었다. Gaussian cap 1,131,968개, 전체 test 30장 평가이고 두 arm의 최종 Gaussian 수가 같다.

| 목적함수 | PSNR | SSIM | LPIPS | MSE |
|---|---:|---:|---:|---:|
| 0.8 L1 + 0.2 dense SSIM | 29.001662 | 0.914902 | 0.243588 | 0.001377435 |
| 1.0 L1 (SSIM 제거) | 28.989559 | 0.903348 | 0.266044 | 0.001369790 |
| 차이 | +0.012103 dB | +0.011554 | -0.022456 | +0.000007645 |

PSNR과 MSE는 view마다 방향이 갈려 각각 17/30, 17/30에서만 SSIM 조건이 좋았다. SSIM은 29/30 view, LPIPS는 30/30 view에서 SSIM 조건이 좋았다. 즉 dense 학습에서도 SSIM 항은 pixel 오차를 줄이는 항이 아니라, 비슷한 pixel 오차 수준에서 구조와 지각 품질을 유지하는 항으로 작동한다.

결론적으로 SSIM-loss의 영향은 단일씬 단일반복학습이지만 크지 않다고 볼 수 있다. 

### 발견 3 — L1과 MCMC 관측은 sparse 1/16으로 두고 SSIM만 전체 이미지로 계산하면 dense 전체 학습과 최종 품질이 dense와 유사해진다. sparse L1 자체가 품질 병목이라는 근거는 약하다

주 경로는 4×4 sampling으로 약 1/16 ray만 렌더링해 L1, visibility, MCMC 관측을 담당하게 하고, 같은 view를 전체 해상도로 한 번 더 미분 가능하게 렌더링해 11×11 dense SSIM을 계산했다. 발견 2와 같은 pure MCMC 설정, 같은 cap 1,131,968개, 30,000 step, 전체 test 30장이다.

| 조건 | PSNR | SSIM | LPIPS | MSE |
|---|---:|---:|---:|---:|
| dense ray + dense SSIM | 29.001662 | 0.914902 | 0.243588 | 0.001377435 |
| sparse L1 + 전체 이미지 dense SSIM | 29.007465 | 0.914000 | 0.245544 | 0.001356411 |
| dense ray + L1 단독 | 28.989559 | 0.903348 | 0.266044 | 0.001369790 |



이 조건은 sparse ray만으로 dense SSIM을 추정한 방법이 아니라 상한 실험이다. dense SSIM용 전체 렌더를 실제로 수행하므로 step당 렌더 ray가 약 1,718,604개이고 단일 dense 렌더보다 6.27% 많다. 따라서 남은 과제는 dense 11×11 SSIM이 주는 공간 gradient를 1/16 ray 예산 안에서 추정하는 것이며, 발견 1의 두 가지 차이를 어떻게 줄이는지가 그 추정의 설계 공간이다.


결론적으로 ssim loss를 잘 추정하는 것은 의미가 있다고 판단된다. 




---







## 1. sparse SSIM loss가 계산하는 양과 gradient가 닿는 범위

### 1.1 dense와 sparse에서 SSIM의 산술 비교

dense는 이미지에서 11×11 window를 이동시키며 각 window의 국소 평균, variance, covariance로 SSIM을 계산한다. 한 window는 pixel 121개를 사용하고, 그 121개 pixel 전부가 loss에 들어가므로 전부 gradient를 받는다.

sparse 학습의 SSIM은 표집된 Ray를 sparse 2차원 격자로 모은 뒤 그 격자에 같은 11×11 window를 적용한다. block size 4에서 sparse 격자의 1 pixel은 4×4 stratum에서 추출한 Ray 1개에 대응한다. 따라서 sparse 11×11 window는 dense 이미지 좌표에서 44×44 범위에 대응한다.

| 항목 | dense 11×11 | sparse 11×11 (block 4) |
|---|---:|---:|
| window가 덮는 pixel | 121 | 1,936 |
| loss에 들어가는 pixel | 121 | 121 |
| gradient가 도달하는 pixel 비율 | 100% | 6.25% |
| 인접 pixel 사이의 거리 | 1 | 4 (stratum 내부 jitter에 따라 1에서 7) |

세 가지가 달라진다. 첫째, 국소 moment를 계산하는 공간 규모가 16배 증가하므로 dense와 같은 국소 통계량이 아니다. 둘째, window가 덮는 1,936개 pixel 중 1,815개가 덮는 내용은 구조 supervision을 받지 않는다. 셋째, sparse 격자는 dense 이미지에서 4 pixel 이상 떨어진 pixel들을 인접한 것으로 취급하므로 window 내부의 등간격 가정이 성립하지 않는다.

3DGRT에서 gradient는 pixel이 아니라 Gaussian으로 전달된다. Ray가 배정되지 않은 pixel이 덮는 영역의 Gaussian도 다른 Ray와 교차하면 gradient를 받을 수 있다. 다만 그 gradient는 해당 Gaussian이 포함된 다른 window의 구조 항에서 온 것이므로, 1,815개 pixel이 나타내는 국소 구조 자체는 어떤 window에서도 직접 평가되지 않는다.

### 1.2 sparse SSIM gradient와 dense SSIM gradient의 차이

counter step 4,601, Gaussian 985,239개, 고정된 train view 4장에서 같은 view의 dense SSIM gradient를 기준으로 sparse SSIM의 gradient를 비교했다. 방법마다 동일한 4×4 stratum당 1 Ray 추출을 8회 반복했다.

| 측정 항목 | sparse SSIM |
|---|---:|
| 단일 추출 density gradient cosine | 0.713779 |
| 8회 평균 density gradient cosine | 0.760530 |
| 8회 평균 상대 L2 | 1.092794 |
| 8회 평균 norm 비율 | 1.629347 |
| 표집된 image gradient cosine | 0.805565 |
| scalar loss 절대 오차 | 0.004848 |

단일 추출에서 8회 평균으로 가도 cosine이 0.713779에서 0.760530으로만 증가한다. 추출 횟수를 늘려 표본 variance를 줄여도 dense SSIM gradient로 수렴하지 않는다. norm 비율이 1.629347이므로 gradient 크기도 dense보다 크다.

### 1.3 dense SSIM 항이 기여하는 축

sparse SSIM이 무엇을 놓치는지 판단하려면 dense SSIM 항이 원래 무엇을 기여하는지 기준이 필요하다. counter를 전체 ray로 초기화부터 native MCMC로 30,000 step 학습하고 목적함수만 바꿔 비교했다. RADC와 멀티뷰는 사용하지 않고 step당 view 1장이며, Gaussian cap은 1,131,968개다. SSIM 제거로 영상 손실 전체 규모가 20% 줄어드는 것을 피하기 위해 L1 단독 arm의 L1 가중을 1.0으로 두었다. 두 arm의 최종 Gaussian 수는 같다.

| 목적함수 | PSNR | SSIM | LPIPS | MSE |
|---|---:|---:|---:|---:|
| 0.8 L1 + 0.2 dense SSIM | 29.001662 | 0.914902 | 0.243588 | 0.001377435 |
| 1.0 L1 | 28.989559 | 0.903348 | 0.266044 | 0.001369790 |


| 지표 | 평균 차이 | 중앙값 | 최소에서 최대 | dense SSIM 우세 view |
|---|---:|---:|---:|---:|
| PSNR | +0.012103 dB | +0.089057 | -0.665916에서 +0.685411 | 17/30 |
| SSIM | +0.011554 | +0.011604 | -0.000666에서 +0.019507 | 29/30 |
| LPIPS | -0.022456 | -0.023648 | -0.035307에서 -0.010723 | 30/30 |
| MSE | +0.000007645 | -0.000017755 | -0.000229809에서 +0.000336441 | 17/30 |

PSNR과 MSE는 view마다 부호가 갈리고 SSIM과 LPIPS는 거의 모든 view에서 같은 방향이다. dense 학습에서도 SSIM 항은 pixel 오차를 줄이는 항이 아니라 비슷한 pixel 오차 수준에서 구조와 지각 품질을 유지하는 항으로 작동한다.



### 1.4 전체 이미지 dense SSIM을 sparse L1에 결합한 상한

1.1절의 두 가지 차이를 모두 제거했을 때 도달할 수 있는 품질을 측정했다. 주 경로는 4×4 층화 표집으로 약 1/16 ray만 렌더링해 L1, visibility와 hit 통계, MCMC 관측을 담당하게 하고, 같은 view를 전체 해상도로 한 번 더 미분 가능하게 렌더링해 11×11 dense SSIM을 계산했다. 1.3절과 같은 pure MCMC 설정, 같은 cap, 30,000 step, 전체 test 30장이다.

| 조건 | sparse L1 ray | dense SSIM ray | step당 총 ray |
|---|---:|---:|---:|
| dense ray + dense SSIM | 0 | 1,617,204 | 1,617,204 |
| sparse L1 + 전체 이미지 dense SSIM | 101,400 | 1,617,204 | 1,718,604 |

dense SSIM용 full forward가 sparse visibility와 hit buffer를 덮어쓰지 않도록 renderer를 직접 호출했고, MCMC의 관측 경로는 sparse forward만 사용했다. step당 렌더 ray가 단일 dense 렌더보다 6.27% 많으므로 이 조건은 sparse 가속 방법이 아니라 상한이다.

| 조건 | PSNR | SSIM | LPIPS | MSE |
|---|---:|---:|---:|---:|
| dense ray + dense SSIM | 29.001662 | 0.914902 | 0.243588 | 0.001377435 |
| sparse L1 + 전체 이미지 dense SSIM | 29.007465 | 0.914000 | 0.245544 | 0.001356411 |
| dense ray + L1 단독 | 28.989559 | 0.903348 | 0.266044 | 0.001369790 |







---

## 2. adaptive 9×9 멀티뷰 sampler

### 2.0 동기

가장 핵심은 dense대비 ssin을 구하는 window size의 차이와 ssim을 구하는 픽셀들간의 거리가 너무 멀다는 점으로 생각했다. 

그래서 표집된 ray의 ssim gradient를 다음 step에서 재사용하는 방법을 생각했다. 

이 경우 단일 뷰일 경우 학습이미지가 210장 일경우 평균 210step마다 한 번씩 그 view를 학습하기 때문에 효율이 떨어지기 때문에 멀티뷰로 학습하면 같은 view를 더 자주 학습하게 되어 효율적이라고 판단된다. 



### 2.1 설계

view당 Ray 수를 기존 block-8 예산과 같게 유지하면서 배분 방식을 변경한다. 모든 9×9 pixel block에 base Ray 1개를 배정하고, 남은 Ray로 residual exponential moving average 점수가 높은 block과 균등 선택한 block을 50 대 50으로 승격한다. 승격된 block은 3×3 pixel subcell마다 Ray 1개를 받아 9×9 pixel에 대한 중심 정렬 3×3 구조 표본을 만든다. 멀티뷰 4장을 동시에 학습한다.

이 구현은 과거 EVER 배분 연구에서 확인한 세 결론을 재사용한다. 모든 block에 base 관측이 필요하고, 배분 점수는 느린 residual exponential moving average가 유리하며 (alpha 0.1이 0.3보다 좋았다), base Ray와 bonus Ray는 화면 순서로 병합해야 traversal 일관성이 유지된다. 멀티뷰 구조 손실에 사용하는 것은 검증되지 않은 부분이다.

### 2.2 Ray 예산 회계

counter 1040×1560, view 1장 기준이다.

| 항목 | 값 |
|---|---:|
| block-8 기준 예산 | 25,350 Ray |
| block-9 base | 20,184 Ray |
| adaptive bonus | 5,166 Ray |
| 완전한 3×3 구조 block | 645개 |
| 구조 Ray | 5,805 Ray |
| priority 선택 block | 323개 |
| 균등 선택 block | 323개 |
| 중복 없는 좌표 | 25,350개 |
| 멀티뷰 4장 합계 | 101,400 Ray |



### 2.3 멀티뷰 sparse SSIM과의 차이와 전체 학습 결과

멀티뷰 4-view는 view당 sparse 격자 130×195 = 25,350개 Ray 전체를 SSIM sparse 격자 이미지에 사용하여 중첩 window를 만든다. 유효 sparse window 중심은 view당 약 120×185 = 22,200개다. adaptive 9×9는 view당 25,350개 Ray 중 5,805개, 22.9%만 645개의 비중첩 3×3 구조 patch에 참여한다. 구조 중심 수를 비교하면 약 34.4배 적다.

세 조건 모두 같은 counter step 4,601 체크포인트에서 시작하고 Gaussian 985,239개를 유지했으며, downsampling 2, test split interval 8의 전체 test 30장으로 평가했다. loss는 0.8 L1 + 0.2 SSIM이다.


### 2.4 실험결과

현재 설계하며 학습 중에 있다. 




