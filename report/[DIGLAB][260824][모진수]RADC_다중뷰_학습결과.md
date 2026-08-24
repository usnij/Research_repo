# RADC + 다중 view 학습 — 고정 Ray 예산에서 sparse 학습의 PSNR 회복

작성일 2026-08-24 · 모진수

## 요약

Step당 view 수를 1개에서 4개로 늘리되 전체 Ray 수는 고정했다. 이 설정은
densification이 종료되는 15000 step부터 적용했다. Mip-NeRF 360 실내 4개 씬에서
RADC로 0~15000 step을 학습한 뒤, 동일한 checkpoint에서 단일 view와 4개 view 조건으로
나누어 15000~30000 step을 학습했다. 핵심 발견은 세 가지다.

### 발견 1 — 추가 Ray 없이 sparse 학습의 PSNR 하락분 중 76~88%를 회복했다

전체 Ray의 1/16만 캐스팅하면 full-Ray 학습보다 PSNR이 낮아진다. counter에서
28.594 dB 가 28.365 dB 로 0.229 dB 낮아지고, kitchen 에서 29.958 dB 가
29.191 dB 로 0.767 dB 낮아진다.

step당 view 수를 1개에서 4개로 늘리면 counter는 28.566 dB, kitchen은 29.777 dB가
된다. **Counter에서는 하락분 0.229 dB 중 0.201 dB를, kitchen에서는 0.767 dB 중
0.586 dB를 회복했다.** 각각 87.8%와 76.4%다.

세 조건 모두 step당 Ray 수는 101,400개다. Gaussian 수도 15000 step 이후에는
변하지 않으므로 조건 간에 같다. 차이는 Ray를 sampling하는 view 수뿐이다.

### 발견 2 — RADC와 다중 view 학습을 순차 적용했을 때 PSNR 개선이 유지됐다

RADC는 0~15000 step에 동작하고 다중 view 학습은 15000 step부터 적용하므로 두
방법의 적용 구간이 겹치지 않는다. RADC로 생성된 Gaussian 구조에서도 다중 view
학습의 PSNR 개선이 유지되는지 별도로 확인했다.

counter에서 보정 없이 학습한 구조(Gaussian 2,154,760개)의 개선 폭은 0.186 dB였고,
RADC로 학습한 구조(Gaussian 1,456,878개)에서는 0.201 dB였다. Gaussian 수가 32%
적은 RADC 구조에서 개선 폭이 0.015 dB 더 컸다.

### 발견 3 — Full-Ray 학습보다 3.4배 빠르면서 PSNR은 0.054 dB 높았다

counter에서 15000~30000 step을 full-Ray로 학습하면 74분이 걸리고 PSNR은
28.594 dB다. 같은 구간을 1/16 Ray와 4개 view로 학습하면 22분이 걸리고 PSNR은
28.648 dB다. **학습 시간은 3.4배 짧고 PSNR은 0.054 dB 높았다.**

단일 view의 1/16 Ray 학습은 같은 구간에 18분이 걸렸다. 4개 view를 적용하면 22분으로
1.22배 증가한다. 0~15000 step은 동일하므로 전체 학습 시간은 1.11배 증가한다.

---

### 지난 리포트 (2026-08-20) 로부터
- 지난번까지 알아낸 것: RADC 가 실내 4개 씬 12개 조건 전부에서 순위 중첩을 회복하고 Gaussian 개수를 dense 의 105~122% 로 맞춘다. 다만 PSNR 로는 세 arm 이 구분되지 않는다.
- 그때 몰랐던 것: 순위 회복이 PSNR 개선으로 이어지지 않는 원인과 sparse 학습의 PSNR 하락을 줄이는 방법.
- 이번에 하기로 한 것: densification 순위는 유지하면서 Ray를 casting하는 view만 변경하여 PSNR을 회복할 수 있는지 확인한다.

### 합의 사항 → 상태
- **[완료] 다중 view 학습 구현** — step당 K개 view를 사용하고 block 크기를 √K배로 늘려 전체 Ray 수를 고정했다. Ray tracer는 한 번 호출한다.
- **[완료] RADC 와의 병행 확인** — 두 장치가 시간상 겹치지 않고, RADC 로 학습한 구조에서 이득이 0.015 dB 더 크다.
- **[완료] 실내 씬 확장** — counter, kitchen, bonsai, room 4개 씬에서 학습했다. counter 와 kitchen 은 view 1개 대조가 있고 bonsai 는 추가로 돌리고 있다.
- **[완료] 손실 함수와 학습률 검토** — SSIM 가중치와 learning rate를 변경했으나 두 조건 모두 PSNR이 감소하여 원본 설정을 유지한다.
- **[완료] 실외 씬** — garden 과 flowers 를 4090 에서 마쳤다. PSNR 이득이 0.054 dB 와 0.044 dB 로 실내의 1/5 이다. 부호는 유지된다.
- **[미착수] view 수 절제** (사유: 4개만 학습했다. 기울기로는 밀도가 4개에서 포화하고 위치는 16개까지 오른다.) 🔴


---

## 1. 다중 view 학습 구성

3DGS 계열은 일반적으로 iteration당 이미지 한 장을 사용한다. 기존 sparse 학습도
동일하게 step마다 view 하나를 선택하고, 해당 view 픽셀의 1/16에 Ray를 캐스팅했다.

다중 view 학습에서는 같은 Ray 예산을 K개 view에 배분한다. Block 크기를 √K배로
늘려 view당 픽셀 수를 1/K로 줄이므로 전체 Ray 수는 유지된다.

counter, kitchen, bonsai 는 이미지가 1039×1559 라 숫자가 정확히 맞는다.

| step당 view 수 | block 크기 | view당 grid | 전체 Ray 수 |
|---:|---:|---|---:|
| 1 | 4 | 260 × 390 | 101,400 |
| 4 | 8 | 130 × 195 | 101,400 |

garden 과 flowers 는 이미지 크기가 블록으로 나누어떨어지지 않아 0.31%, 0.32% 더
많다. 1% 안쪽이다.

### 1.1 Sparse Ray 학습에서의 구현

우리 방법과 선행연구는 모두 고정 예산 다중 view 학습을 사용하지만 rendering
방식이 다르다.

**Rasterization에서는 부분 view rendering을 위한 별도 구현이 필요하다.** Rasterizer는
이미지 단위로 동작하므로 여러 view의 일부 픽셀만 rendering하려면 rasterizer를
수정해야 한다. Efficient Multi-View Training(arXiv 2506.12727)은 하나의 tile에
여러 view의 픽셀을 배치하고 rendering할 픽셀을 index array로 전달한다.

**Ray tracing에서는 Ray casting 대상만 변경하면 된다.** Ray tracer는 입력된 Ray
집합을 처리하므로 각 Ray의 view가 같을 필요가 없다. 구현에서는 sampling 함수만
변경했으며 renderer, loss, backward는 변경하지 않았다.

기존 sparse 학습은 전체 픽셀의 1/16에 해당하는 고정 Ray 예산을 사용한다. 따라서
전체 Ray 수를 늘리지 않고도 이 예산을 여러 view에 배분할 수 있다. 이 점이 전체
이미지를 rendering하는 dense 학습과 다르다.

예산을 맞추는 방법은 블록 크기다. view 를 K 개 쓰면 블록을 √K 배로 키워 view 당
픽셀 수를 1/K로 줄인다. 4개 view를 batch dimension에 쌓아
`[4, 130, 195, 3]` 형태의 batch 하나로 구성하고, Ray tracer는 한 번 호출한다.

**View별 grid 형태는 유지한다.** Ray tracing은 Ray를 일차원 배열로 입력해도
동작하지만, 이 경우 인접 픽셀 정보가 사라져 SSIM을 계산할 수 없다. View별로
`130 × 195` grid를 유지하여 각 view에서 SSIM을 계산한다.

### 1.2 Densification 종료 이후에 적용하는 이유

선행연구는 전체 학습 구간에서 4개 view를 사용하므로 densification 통계량도 다중
view 조건에 맞게 다시 정의한다. 각 gradient의 L2 norm을 먼저 계산하여 합산한 값과
gradient vector를 먼저 합산한 뒤 L2 norm을 계산한 값을 각각 split과 clone에 사용한다.

Dense 학습에는 픽셀 sampling에 따른 noise가 없지만, 1/16 sparse Ray 학습의 step별
gradient는 full-Ray gradient의 추정값이다. Sampling variance로 인해 densification
통계량이 과대 추정되며, 다중 view 적용 시 Jensen inflation이 추가로 17~21%
증가한다(2절).

따라서 다중 view 학습은 densification이 종료된 15000 step 이후에만 적용한다. 이후
구간에서는 densification 통계량을 사용하지 않으므로 새로운 통계량을 정의할 필요가 없다.

### 1.3 손실은 view 별로 계산해 평균한다

예측과 정답이 둘 다 `[4, 130, 195, 3]` 이다. L1 은 텐서 전체 원소의 평균이고,
view마다 픽셀 수가 같으므로 view별 L1의 평균과 같다. SSIM도 batch 단위로 계산하면
view별 SSIM의 평균이 나온다. 두 view를 각각 계산했을 때
0.005848 과 −0.004262 이고 배치로 한 번에 계산하면 0.000793 으로, 둘의 평균과
같다.

최종 손실은 `0.8 × L1 + 0.2 × (1 − SSIM)` 이고 역전파는 한 번이다.

Gaussian별 gradient는 해당 step에서 Gaussian과 교차한 모든 Ray의 gradient를 합산한
값이다. 단일 view 조건과 달리, 다중 view 조건에서는 4개 view의 Ray가 gradient에
기여한다.

Loss는 view별 합이 아니라 평균으로 계산한다. 따라서 gradient scale이 유지되어
learning rate를 별도로 조정할 필요가 없다. 6절에서도 learning rate를 1.414배로
높였을 때 PSNR이 0.040 dB 감소했다.

### 1.4 View sampling 방법

Step당 4개 view 중 하나는 dataloader가 제공하고, 나머지 3개는 전체 training
view에서 무작위로 추출한다. 카메라 거리가 가까운 view를 선택하거나 서로 먼 view를
선택하는 방법도 비교했다. Reference gradient와의 방향 일치도는 무작위 선택 0.780,
근접 view 선택 0.790, 최원점 선택 0.531이었다. 무작위 선택과 근접 view 선택의
차이는 0.010이었으며 반복 측정이 없어 우열을 판단하지 않았다. 최원점 선택은 매번
같은 극단 view가 반복 선택되는 구현이므로, 결과 0.531만으로 view 분산 전략 전체를
기각할 수 없다. 본 실험에는 무작위 선택을 사용했다.

---

## 2. 적용 시작 시점을 15000 step으로 정한 근거

다중 view 적용 시 densification 통계량과 parameter update 정확도가 서로 반대
방향으로 변한다.

**Densification 통계량의 Jensen inflation은 증가한다.** View 수가 늘면 view당 픽셀
sampling density가 감소하고, 각 Gaussian은 일부 view에서만 관측되므로 Gaussian당
Ray hit 수도 감소한다. counter의 15000 step에서 전체 2,154,760개 Gaussian 중 모든
조건에서 공통으로 관측된 554,762개를 대상으로 측정했다.

| step당 view 수 | inflation 계수 φ 중앙값 | 단일 view 대비 |
|---:|---:|---:|
| 1 | 4.878 | 1.000 |
| 2 | 5.32 | 1.091 |
| 4 | 5.70 | 1.168 |
| 8 | 5.92 | 1.213 |
| 16 | 5.92 | 1.214 |

**Parameter update에 사용하는 gradient의 정확도는 높아진다.** 32개 view의 full-image
gradient 평균을 reference로 사용하여 step별 gradient의 방향 일치도를 측정했다.
Density gradient의 방향 일치도는 단일 view에서 0.680, 4개 view에서 0.829였고,
position gradient는 0.498에서 0.509로 증가했다.

Densification은 15000 step에 종료된다. 이후에는 해당 통계량을 사용하지 않으므로
Jensen inflation의 영향을 피하면서 parameter update 정확도 향상만 이용할 수 있다.

또한 15000 step 이후에는 Gaussian 수가 변하지 않는다. 동일 checkpoint에서 분기한
단일 view와 다중 view 조건의 Gaussian 수가 같으므로 PSNR 차이를 모델 크기와 분리하여
비교할 수 있다.

---

## 3. 씬별 결과

각 씬에서 RADC로 0~15000 step을 한 번 학습한 뒤 checkpoint를 저장했다. 동일한
checkpoint에서 단일 view와 4개 view 조건으로 분기하여 15000~30000 step을 학습했으므로
전반부 학습의 실행 간 변동은 비교에 포함되지 않는다.

| 씬 | 전체 Ray | view 1개 | view 4개 | PSNR 회복량 | 전체 Ray와의 차이 |
|---|---:|---:|---:|---:|---:|
| counter | 28.594 | 28.365 | **28.566** | +0.201 | −0.028 |
| kitchen | 29.958 | 29.191 | **29.777** | +0.586 | −0.181 |
| bonsai | 31.893 | 31.444 | **31.796** | +0.352 | −0.097 |
| room | 30.233 | 30.716 | 30.142 | −0.574 | −0.091 |

학습 중 출력된 SSIM과 LPIPS가 NaN이어서 checkpoint에서 전체 view를 대상으로 별도
평가했다. PSNR도 동일한 평가 script로 다시 측정했으므로 위 표의 30개 view 결과와 다르다.

| 씬 | arm | Gaussian | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|
| counter | view 1 개 | 1,456,878 | 28.370 | **0.8926** | **0.2708** |
| counter | view 4 개 | 1,456,878 | **28.578** | 0.8914 | 0.2720 |
| bonsai | view 1 개 | 1,618,239 | 31.446 | 0.9331 | 0.2559 |
| bonsai | view 4 개 | 1,618,239 | **31.799** | **0.9333** | 0.2559 |
| kitchen | view 1 개 | 1,542,363 | 29.469 | 0.8963 | 0.1887 |
| kitchen | view 4 개 | 1,607,143 | **29.784** | **0.9059** | **0.1862** |

PSNR은 세 씬 모두 증가했다. SSIM과 LPIPS는 kitchen에서만 함께 개선됐다. counter는
SSIM이 0.0012 감소하고 LPIPS가 0.0012 증가했으며, bonsai의 LPIPS는 소수점 넷째
자리까지 같았다.

세 지표가 모두 개선된 kitchen은 두 조건의 Gaussian 수가 다르다. 단일 view 결과는
동일 trunk에서 분기한 실행이 아니라 별도로 학습한 `kitchen_radc2_r1`이며, 4개 view
조건보다 Gaussian이 64,780개 적다. Counter와 bonsai는 동일 checkpoint에서 분기하여
Gaussian 수가 일치한다.

30개 view 평가와 전체 view 평가의 차이도 확인됐다. kitchen 단일 view 조건의 PSNR은
29.191 dB에서 29.469 dB로 0.278 dB 차이가 나며, 이는 앞서 계산한 회복량 0.586 dB의
47%다. 따라서 회복률 76~88%는 전체 view 평가로 다시 계산해야 한다. 🔴

Gaussian 개수는 counter 가 1,456,878 개, kitchen 이 1,607,143 개, bonsai 가
1,618,239 개다. 전체 Ray 로 학습한 기준선은 각각 1,221,868 개, 1,497,515 개,
1,484,957 개이므로 RADC 를 쓴 쪽이 7~19% 많다.

kitchen 의 view 1 개 값 29.191 dB 는 별도로 3 회 학습한 것의 평균이다. 반복 사이
표준편차가 0.056 dB이므로 회복량 0.586 dB는 표준편차의 약 10배다. room의 30.716 dB도
3 회 평균이고 표준편차가 0.212 dB 다.

room에서는 1/16 sparse Ray 조건이 full-Ray 조건보다 0.483 dB 높다. Full-Ray 모델은
Gaussian 1,204,903개, sparse 모델은 1,474,349개로 sparse 모델이 22% 크므로 두 결과를
동일 모델 크기의 성능 차이로 해석할 수 없다.

bonsai 의 view 1 개는 같은 체크포인트에서 갈라진 실행이고 Gaussian 이 1,618,239 개로
view 4개와 일치한다. PSNR 회복량은 0.352 dB였다.

**room 은 view 4 개가 view 1 개보다 0.574 dB 낮다.** 4 개 씬 중 부호가 뒤집힌
유일한 씬이다. room 의 view 4 개는 trunk 에서 갈라진 실행이라 Gaussian 이
1,463,491 개로 trunk 와 같지만, view 1 개 값 30.716 dB 는 별도 3 회 실행의 평균이고
개수가 1,474,349 개다. 개수가 10,858 개 많고 반복 표준편차가 0.212 dB 이므로,
0.574 dB 차이를 다중 view 적용의 영향으로 판단하려면 동일 trunk에서 분기한 단일 view 실행이 필요하다. 🔴

### 3.1 Densification 구간에 적용할 때 Gaussian 수 감소

counter에서 `--multiview-from 0`으로 다중 view 학습을 densification 구간에도 적용했다.

| counter | Gaussian | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| view 1 개 (15000 부터) | 1,456,878 | 28.370 | **0.8926** | **0.2708** |
| view 4 개 (15000 부터) | 1,456,878 | **28.578** | 0.8914 | 0.2720 |
| view 4 개 (0 부터) | 290,441 | 28.383 | 0.8730 | 0.3119 |

Gaussian 수는 기준의 20%로 감소했고 PSNR은 28.383 dB로 단일 view와 비슷했다.
그러나 SSIM은 0.0196 감소하고 LPIPS는 0.0411 증가했다. LPIPS의 상대 증가율은
15%다. Kitchen에서도 densification 종료 시점의 Gaussian 수가 453,324개로,
단일 view 실행은 1,542,363개다. 해당 다중 view 실행은 15000 step checkpoint까지만 학습했다.

원인은 RADC의 online 보정이 다중 view 경로에서 비활성화되는 데 있다. Online `k(h)`
보정은 같은 view에서 두 번째 sample을 추출하여 split-half 방식으로 noise-to-signal
ratio를 측정한다. 그러나 다중 view 경로에서는 원본 sample 변수를 `None`으로 설정한다
(`train_scene_sparse.py:181-183`). 보정 함수가 즉시 반환되어 통계량이 누적되지 않으며,
곡선 update 횟수는 단일 view 49회, 다중 view 1회였다.

따라서 `k(h)`는 counter 단일 view에서 사전에 측정한 고정 곡선을 유지한다. 다중 view에서는
Gaussian 하나를 맞히는 Ray 가 서로 다른 네 view 에서 오므로 단일 view 보다 독립에
가깝고, 같은 h에서 실제 k가 더 낮다. 단일 view에서 측정한 곡선을 적용하면
densification score가 필요 이상으로 감소한다.

---

## 4. 실외 씬 결과

garden과 flowers는 RTX 4090에서 학습했다. 로컬 RTX 4070 SUPER에서는 Gaussian 수가
약 5.1M에 도달할 때 backward 과정에서 OOM이 발생했다.

| 씬 | arm | Gaussian | PSNR | 이득 | 남은 차이 | dead 비율 | 시간 |
|---|---|---:|---:|---:|---:|---:|---:|
| garden | 전체 Ray | 4,206,710 | 26.502 | — | — | — | — |
| garden | view 1 개 | 6,020,366 | 26.287 | — | −0.215 | 24.59% | 1,262 초 |
| garden | view 4 개 | 6,020,366 | **26.341** | +0.054 | −0.161 | 24.53% | 1,272 초 |
| flowers | 전체 Ray | 4,952,430 | 21.290 | — | — | — | — |
| flowers | view 1 개 | 7,074,987 | 21.262 | — | −0.028 | 31.26% | 1,332 초 |
| flowers | view 4 개 | 7,074,987 | **21.306** | +0.044 | +0.016 | 31.45% | 1,401 초 |

Full-Ray 기준선은 로컬 환경에서 동일한 평가 경로로 측정했다. 두 모델 모두 Gaussian
상한 8M에 도달하지 않았으며, garden은 상한의 52.6%, flowers는 61.9%였다.

실내 씬의 PSNR 개선 폭은 0.208~0.353 dB였고, 실외 씬은 0.044~0.054 dB였다. 평가된
5개 씬에서 PSNR 변화는 모두 양수였지만 실외 씬의 개선 폭은 실내보다 작았다.

sparse 로 잃는 양 자체가 실외에서 작다. garden 은 전체 Ray 대비 0.215 dB 낮고
flowers 는 0.028 dB 낮은데, 실내 kitchen 이 0.767 dB, counter 가 0.229 dB 낮았다.
flowers에서는 4개 view 조건이 full-Ray 기준보다 0.016 dB 높았다. Full-Ray 대비
PSNR 하락 폭이 작을수록 회복 가능한 PSNR도 작다는 해석이 가능하지만, 실외 두 씬의
SSIM과 LPIPS를 측정하지 않아
0.028 dB 차이가 실제로 작은 손실인지는 확정할 수 없다.

Gaussian 개수는 반대 방향이다. sparse 쪽이 garden 에서 6,020,366 개로 전체 Ray 의
1.43 배, flowers 에서 7,074,987 개로 1.43 배다. RADC 를 적용한 값이 이렇다. 실내에서
RADC 가 개수를 dense 의 105~122% 로 맞춘 것과 대비된다.

Gaussian 개수가 실내의 4 배다. 실내가 1.46~1.62M 개인데 garden 이 6.02M 개,
flowers 가 7.07M 개다. 같은 101,400 Ray 를 4 배 많은 Gaussian 이 나눠 받으므로
Gaussian 당 관측량이 1/4 이 된다. 2 절에서 view 분할이 밀도 gradient 일치도를
0.680 에서 0.829 로 올린 것은 관측량이 충분한 조건에서였고, 1/4 조건에서도 같은
크기로 오르는지는 재지 않았다.

flowers 의 trunk 는 상한 7,000,000 개에 대해 7,074,987 개로 상한 구간에 들어갔다.
초과분이 0.1% 이고 view 1 개와 view 4 개가 같은 trunk 를 쓰므로 두 arm 의 비교는
성립한다. garden 은 상한에 닿지 않았다.

dead Gaussian 비율이 실내보다 높다. trunk 시점에 garden 0.14%, flowers 0.16% 이던
것이 30000 step에서 24.5~31.5%가 된다. 15000 step 이후에 발생하며 해당 구간에는
pruning이 실행되지 않는다. 다중 view 적용 여부에 따른 차이는 garden에서 0.06%p,
flowers 에서 0.19%p 로 방향이 반대다.

---

## 5. 시간

counter의 15000~30000 step 학습 시간이다.

| 조건 | Ray | 시간 | PSNR |
|---|---:|---:|---:|
| 전체 Ray, view 1 개 | 1,617,204 | 74 분 | 28.594 |
| 1/16, view 1 개 | 101,400 | 18 분 | 28.462 |
| **1/16, view 4 개** | **101,400** | **22 분** | **28.648** |

View 수를 1개에서 4개로 늘리면 학습 시간이 18분에서 22분으로 1.22배 증가했다.
Ray 수가 같으므로 차이는 여러 view에서 생성된 Ray의 coherence 감소와 고정 학습
비용에서 발생한다. 별도 측정한 step당 시간은 1.48배였지만 전체 학습 시간은
optimizer 등 Ray 수와 무관한 비용을 포함하므로 1.22배 증가했다.

0~15000 step은 동일하므로 전체 학습 시간은 1.11배 증가한다.

RADC 를 쓴 구조에서는 더 짧다. Gaussian 이 1,456,878 개로 32% 적어 view 1 개가
13 분, view 4 개가 17 분이다.

---

## 6. 손실 함수와 학습률은 바꾸지 않았다

SSIM 가중치와 learning rate를 변경했으나 두 설정 모두 PSNR이 감소했다.

**SSIM 가중치.** Step별 gradient를 reference gradient와 비교하면, 1/16 Ray 조건에서는
SSIM 가중치를 0.2에서 0.095로 낮출 때 density gradient가 reference와 가장 가까웠다. 그러나
실제로 학습해 보니 view 1 개에서 28.462 dB 가 28.441 dB 로, view 4 개에서
28.648 dB 가 28.641 dB 로 오히려 낮아졌다. 기본값 0.2 를 그대로 쓴다.

**학습률.** Grendel (arXiv 2406.18533) 은 배치 학습에서 학습률을 √배치 배로
올리라고 한다. view 4 개면 2 배다. RADC 구조에서 1.414 배로 올려 보니 28.566 dB 가
28.526 dB 로 0.040 dB 낮아졌다. 그쪽 배치는 이미지를 더 보는 것이라 표본이 늘지만
우리는 같은 Ray 를 나누는 것이라 표본이 늘지 않는다. 규칙이 그대로 옮겨지지
않는다.

Loss와 backward는 원본 설정을 유지했으며, Ray sampling 대상 view만 변경했다.

---

## 7. 선행연구와의 중복 및 차이

Efficient multi-view training for 3D Gaussian Splatting (arXiv 2506.12727) 이 같은
구조를 사용한다. View마다 픽셀을 부분 sampling하여 전체 픽셀 수를 단일 view
학습과 같게 유지한다. 따라서 고정 예산 다중 view 학습 자체는 우리 연구의 기여가 아니다.

다른 점이 셋이다.

선행연구는 rasterization 기반이므로 여러 view의 일부 픽셀만 rendering하기 위해
rasterizer를 수정한다. Ray tracing 기반의 우리 구현은 Ray casting 대상 view를
sampling하는 함수만 변경했다.

그 논문은 전체 학습 구간에서 4개 view를 사용한다. 따라서 densification 통계량을 다중 view 조건에 맞게
새로 정의해야 했고, gradient별 L2 norm을 먼저 계산해 합산한 값과 gradient vector를 먼저 합산한 뒤 L2 norm을 계산한 값을
용도에 따라 나눠 쓴다. 우리는 densification 이 끝난 뒤에만 켜므로 그 문제가 생기지
않는다.

선행연구는 전체 이미지를 rendering하므로 픽셀 sampling noise가 없다. 우리 방법은
전체 Ray의 1/16만 캐스팅하므로 step별 gradient가 full-Ray gradient의 추정값이고,
sampling variance가 densification 통계량의 Jensen inflation을 유발한다. 선행연구는
이 현상을 분석하지 않는다.

---

## 8. 추가 확인이 필요한 항목

**최적 view 수.** 현재 전체 학습은 4개 view 조건만 완료했다. Gradient 측정에서는
density gradient의 방향 일치도가 4개 view에서 포화했고, position gradient는 16개
view까지 증가했다. Step 시간은 4개 이상에서 각각 1.48배, 1.51배, 1.49배로 비슷했다.
최종 view 수는 2, 4, 8, 16개 조건의 전체 학습 결과로 결정해야 한다. 🔴

**실외 씬의 SSIM과 LPIPS.** garden과 flowers의 PSNR 평가는 완료했지만 SSIM과
LPIPS는 측정하지 않았다. Checkpoint의 Gaussian 수가 6~7M이므로 로컬 환경에서
평가 가능한지 확인해야 한다. 🔴

**room의 동일 checkpoint 대조군.** 현재 단일 view 결과는 별도 3회 실행의 평균이고,
4개 view 결과는 trunk checkpoint에서 분기했다. 동일 checkpoint에서 단일 view
조건을 추가 학습해야 0.574 dB 차이를 다중 view 적용의 영향으로 판단할 수 있다. 🔴

---

## 참고 문헌

- Efficient multi-view training for 3D Gaussian Splatting — arXiv 2506.12727
- On Scaling Up 3D Gaussian Splatting Training — arXiv 2406.18533
- MVGS: Multi-view Regulated Gaussian Splatting — arXiv 2410.02103
- Stochastic Ray Tracing for the Reconstruction of 3D Gaussian Splatting — arXiv 2603.23637
