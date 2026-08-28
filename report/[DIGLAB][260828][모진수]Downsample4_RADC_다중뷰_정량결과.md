# Downsampling 4에서 block2 RADC와 다중 view 학습의 정량 결과

작성일 2026-08-28 · 모진수

## 요약

이번 리포트의 핵심 발견은 네 가지다.

### 발견 1 — block2 RADC와 15000 step 이후 다중 view 학습은 8개 씬 중 7개에서 dense보다 높은 PSNR을 기록했다

Mip-NeRF 360 데이터를 downsampling 4로 학습했다. Dense는 한 view의 전체 픽셀에
Ray를 캐스팅했고, sparse 조건은 0~15000 step에서 block2로 전체 픽셀의 1/4에
Ray를 캐스팅하면서 RADC를 적용했다. 15000~30000 step에서는 Gaussian을 고정하고
동일한 Ray 수를 4개 view에 배분했다.

Dense 대비 PSNR 변화는 counter +0.141 dB, room -0.068 dB, bonsai +0.098 dB,
kitchen +0.247 dB, garden +0.260 dB, flowers +0.029 dB, stump +0.103 dB,
treehill +0.099 dB였다. 

**8개 씬 평균은 +0.114 dB이며 room을 제외한 7개 씬에서 PSNR이 증가했다.** 

다만 각 씬을 한 번씩 학습했으므로 완전한 품질차이를 신뢰할 수는 없지만 8개씬에서의 일관된 결과이기 때문에 의미가 있다고 판단된다.

### 발견 2 — 실내 4개 씬에서는 품질과 학습 시간이 함께 개선됐지만, 대규모 실외 씬에서는 Gaussian 수와 시간이 증가했다

Counter, room, bonsai, kitchen에서 sparse 모델의 Gaussian 수는 dense의
1.032~1.092배였고 학습 시간은 1.47~1.86배 짧았다. 네 씬의 PSNR 변화 평균은
+0.105 dB다.

Garden은 Gaussian 수가 dense의 0.909배로 감소했지만 학습 시간은 42.6분에서 56.3분으로 증가했다. 

Stump와 treehill은 Gaussian 수가 각각 dense의 1.531배와 1.428배였고, 학습 시간은 dense와 4% 이내였다. **Downsampling 4에서 품질 개선은 실외 씬까지 유지됐지만, 시간 단축은 대규모 실외 씬에서 재현되지 않았다.**

### 발견 3 — counter에서 RADC+MV4는 block2 무보정보다 PSNR이 0.215 dB 높고 Gaussian 수가 26.5% 적었다

Downsampling 4의 counter에서 전체 픽셀의 1/4에 Ray를 캐스팅하는 block2 조건을
비교했다. 보정 없이 단일 view로 30000 step을 학습하면 PSNR은 28.647 dB이고
Gaussian 수는 1,959,957개다. 0~15000 step에 RADC를 적용하고 이후 4개 view로
학습하면 PSNR은 28.862 dB이고 Gaussian 수는 1,441,479개다.

두 조건의 PSNR 차이는 +0.215 dB이고 Gaussian 수는 518,478개 감소했다. 학습 시간도
약 35.7분에서 31.1분으로 감소했다. **Block2만 적용한 결과보다 RADC와 MV4를 결합한
조건의 PSNR은 높고 Gaussian 수와 학습 시간은 적었다.** PSNR 차이에는
RADC와 MV4의 효과가 함께 포함되며, Gaussian 수는 MV4 적용 전에 확정되므로 개수
차이는 densification 구간의 RADC 적용에서 발생한다.

### 발견 4 — densify 임계값을 0.0002에서 0.00028로 높이면 garden block4 RADC의 Gaussian 수가 49.5% 감소했다

Downsampling 4 이미지로 COLMAP을 다시 실행한 garden 데이터에서 block4 RADC를
학습했다. Densify 임계값 0.0002에서는 15000 step에 Gaussian이 6,244,813개였고,
0.00028에서는 3,156,328개였다. 다른 densification 설정과 초기 point는 같으며,
Gaussian 수가 3,088,485개 감소했다.

0.00028 조건에서 15000 step 이후 4개 view로 학습한 최종 PSNR은 26.162 dB였다.
0.0002 조건의 단일 view 최종 PSNR 26.035 dB보다 0.127 dB 높지만, 이 비교에는
임계값과 다중 view 적용 여부가 함께 바뀐다. 따라서 Gaussian 수 감소는 임계값
변경 결과로 비교할 수 있지만, PSNR 차이를 임계값의 단독 효과로 해석할 수는 없다.

---

### 지난 리포트 (2026-08-27) 로부터

- 지난번까지 알아낸 것: densify 임계값 0.0002는 해상도에 따라 다른 강도로 작동한다. 같은 모델에서 임계값을 넘는 Gaussian 비율이 bonsai 기준 downsampling 4에서 0.741%, downsampling 2에서 0.205%였다.
- 그때 몰랐던 것: downsampling 4에서 block2 RADC와 densification 이후 다중 view 학습을 결합했을 때 dense 대비 품질·Gaussian 수·학습 시간이 여러 씬에서 어떻게 변하는지.
- 이번에 하기로 한 것: 실내와 실외 씬을 같은 downsampling 4 조건으로 학습하고 dense와 정량 비교한다. Garden에서는 densify 임계값 0.00028 조건도 별도로 확인한다.

### 합의 사항 → 상태

- **[완료] 실내 4개 씬 비교** — counter, room, bonsai, kitchen에서 PSNR 변화 평균이 +0.105 dB이고 학습 시간이 1.47~1.86배 짧다.
- **[완료] 실외 4개 씬 비교** — garden, flowers, stump, treehill에서 PSNR이 각각 +0.260, +0.029, +0.103, +0.099 dB 변했다.
- **[완료] counter block2 무보정 비교** — RADC+MV4는 무보정보다 PSNR이 0.215 dB 높고 Gaussian 수가 26.5% 적으며 학습 속도가 1.15배 빠르다.
- **[완료] Stump·treehill 자연 수렴 확인** — Gaussian 상한 15,000,000개에 대해 dense는 각각 7,682,007개와 9,739,164개, sparse는 11,761,154개와 13,910,942개에서 densification을 마쳤다.
- **[부분] densify 임계값 0.00028 비교** — Gaussian 수 감소를 sparse에서는 확인했고 이 임계값이 dense에서는 어떤 결과로 나타나는지 확인하지 않았다. 🔴
- **[미착수] 반복 실행** (사유: 각 조건을 한 번 학습했다.) 🔴

### 다음

- Counter, kitchen, stump에서 동일 설정을 반복하여 +0.103~+0.247 dB 차이가 실행 변동 범위를 넘는지 확인한다.
- EVER에서도 ds=4에서의 결과가 재현되는지 확인한다.

---

## 1. 실험 설정

### 1.1 Dense와 sparse 조건

두 조건 모두 Mip-NeRF 360 데이터를 downsampling 4로 학습했고 전체 학습 길이는
30000 step이다. Clone과 split에 사용하는 densify 임계값은 0.0002이고,
densification은 500~15000 step에서 실행된다.

| 항목 | Dense | Block2 RADC + MV4 |
|---|---|---|
| 0~15000 step Ray sampling | 한 view의 전체 픽셀 | 한 view에서 block2, 전체 픽셀의 약 1/4 |
| Densification 보정 | 없음 | RADC |
| 15000~30000 step view 수 | 1개 | 4개 |
| 15000~30000 step Ray 예산 | 전체 픽셀 | block2와 동일한 총 Ray 수를 4개 view에 배분 |
| Densification 종료 | 15000 step | 15000 step |
| 전체 학습 길이 | 30000 step | 30000 step |

Sparse 조건은 15000 step 이후 view 수를 4개로 늘리면서 view당 sampling grid를
줄인다. 따라서 step당 총 Ray 수는 block2 단일 view와 같다. 이 구간에는 densification이
없으므로 다중 view 학습이 Gaussian 수를 변경하지 않는다.

이번 sparse 실행에서 counter, room, bonsai, kitchen, garden, flowers의 Gaussian
상한은 7,000,000개다. Stump와 treehill은 Gaussian 수가 크므로 15,000,000개를
사용했다. Dense 기준선을 포함한 최종 모델은 모두 각 실행에 설정된 상한에 도달하지
않았다.

### 1.2 비교 지표와 실행 환경

정량 비교에는 30000 step의 PSNR, Gaussian 수, 학습 시간을 사용했다. 이번 실행에서
SSIM과 LPIPS가 NaN으로 기록되어 두 지표는 표에서 제외했다.

Counter, room, bonsai, kitchen, garden은 로컬 RTX 4070 SUPER에서 학습했다. Stump와
treehill은 RTX 4090에서 dense와 sparse를 모두 학습했다. 

Flowers는 0~15000 step checkpoint를 RTX 4090에서 만들고 15000~30000 step을 RTX 4070 SUPER에서 학습했으므로 전체 학습 시간을 dense와 비교하지 않았다.

## 2. Dense 대비 block2 RADC + MV4 결과

### 2.1 PSNR과 Gaussian 수

| 씬 | Dense PSNR | RADC+MV4 PSNR | PSNR 변화 | Dense Gaussian | RADC+MV4 Gaussian | 개수 비율 |
|---|---:|---:|---:|---:|---:|---:|
| counter | 28.721 | 28.862 | +0.141 | 1,320,446 | 1,441,479 | 1.092 |
| room | 30.880 | 30.812 | -0.068 | 1,329,486 | 1,386,476 | 1.043 |
| bonsai | 32.164 | 32.263 | +0.098 | 1,539,056 | 1,658,249 | 1.077 |
| kitchen | 30.206 | 30.453 | +0.247 | 1,484,268 | 1,532,044 | 1.032 |
| garden | 26.502 | 26.762 | +0.260 | 4,206,710 | 3,824,596 | 0.909 |
| flowers | 21.290 | 21.319 | +0.029 | 4,952,430 | 5,981,642 | 1.208 |
| stump | 26.300 | 26.403 | +0.103 | 7,682,007 | 11,761,154 | 1.531 |
| treehill | 22.039 | 22.138 | +0.099 | 9,739,164 | 13,910,942 | 1.428 |
| 평균 | — | — | +0.114 | — | — | — |

PSNR은 room을 제외한 7개 씬에서 증가했다. 가장 큰 변화는 garden +0.260 dB와
kitchen +0.247 dB이고, 가장 작은 양의 변화는 flowers +0.029 dB다. 각 씬을 한 번씩
학습했으므로 +0.029~+0.103 dB 범위의 차이는 반복 실행 후 실행 변동과 비교해야 한다.

실내 4개 씬에서 Gaussian 수는 dense의 1.032~1.092배다. Garden은 0.909배로 감소했으며,
flowers는 1.208배로 증가했다. Stump와 treehill은 각각 1.531배와 1.428배로 증가했다.
따라서 downsampling 4의 block2 RADC가 Gaussian 수를 dense 수준으로 유지한 결과는
실내 4개 씬과 garden에는 해당하지만, flowers·stump·treehill에는 해당하지 않는다.

### 2.2 학습 시간

Sparse 시간은 0~15000 step의 block2 RADC와 15000~30000 step의 MV4 시간을 합산했다.
속도 배율은 `dense 시간 / sparse 시간`이며, 1보다 크면 sparse가 빠르다.

| 씬 | Dense | RADC+MV4 | 속도 배율 | 결과 |
|---|---:|---:|---:|---|
| counter | 약 49.3분 | 약 31.1분 | 1.59배 | sparse가 빠름 |
| room | 38.0분 | 25.9분 | 1.47배 | sparse가 빠름 |
| bonsai | 55.4분 | 34.5분 | 1.61배 | sparse가 빠름 |
| kitchen | 71.2분 | 38.3분 | 1.86배 | sparse가 빠름 |
| garden | 42.6분 | 약 56.3분 | 0.76배 | sparse가 1.32배 느림 |
| stump | 90.4분 | 93.9분 | 0.96배 | sparse가 1.04배 느림 |
| treehill | 103.0분 | 102.0분 | 1.01배 | 차이 1% |

실내 4개 씬에서는 PSNR 변화 평균이 +0.105 dB인 동시에 학습 시간이 1.47~1.86배
짧았다. Garden에서는 Gaussian 수가 감소했지만 sparse 시간이 13.7분 증가했다.
Stump와 treehill은 sparse에서 Gaussian 수가 4백만개 이상 증가하여 Ray sampling으로
줄인 시간이 Gaussian 처리 비용으로 상쇄됐다.

Flowers는 전반부와 후반부를 서로 다른 GPU에서 학습했으므로 시간 표에서 제외했다.
Counter와 garden의 시간은 실행 파일 시각으로 계산한 근삿값이고, 나머지는 학습 로그의
`training_time`을 사용했다.

## 3. 다중 view 적용의 단독 변화

2절은 dense와 `RADC+MV4` 결합 조건을 비교하므로 PSNR 차이를 MV4의 단독 효과로
해석할 수 없다. 동일한 15000 step RADC checkpoint에서 단일 view와 4개 view로
분기한 최종 결과가 모두 있는 세 씬을 별도로 비교했다. 두 조건은 Gaussian 수가 같다.

| 씬 | 단일 view PSNR | 4개 view PSNR | 변화 | Gaussian |
|---|---:|---:|---:|---:|
| counter | 28.681 | 28.862 | +0.181 | 1,441,479 |
| garden | 26.754 | 26.762 | +0.009 | 3,824,596 |
| flowers | 21.366 | 21.319 | -0.047 | 5,981,642 |

Counter에서는 4개 view 조건이 0.181 dB 높고, garden에서는 0.009 dB 높으며,
flowers에서는 0.047 dB 낮다. 세 결과의 방향이 같지 않으므로 현재 결과만으로
다중 view 학습이 모든 씬의 PSNR을 높인다고 결론 내릴 수 없다. 2절에서 확인한 것은
downsampling 4의 전체 학습 파이프라인이 dense와 비교해 8개 씬 중 7개에서 높은
PSNR을 기록했다는 사실이다.

## 4. Block2 무보정과 RADC+MV4 비교

Counter를 downsampling 4로 학습하고 두 조건 모두 block2를 사용했다. Densify 임계값은
0.0002, Gaussian 상한은 7,000,000개, 전체 학습 길이는 30000 step으로 같다.
무보정 조건은 전체 구간에서 단일 view를 사용했다. RADC+MV4 조건은 0~15000 step에
RADC를 적용하고 15000~30000 step에 동일한 총 Ray 수를 4개 view에 배분했다.

| 조건 | 0~15000 step | 15000~30000 step | PSNR | Gaussian | 학습 시간 |
|---|---|---|---:|---:|---:|
| block2 무보정 | 단일 view, 보정 없음 | 단일 view | 28.647 | 1,959,957 | 약 35.7분 |
| block2 RADC+MV4 | 단일 view, RADC | 4개 view | 28.862 | 1,441,479 | 약 31.1분 |
| 변화 | — | — | +0.215 | -518,478 (-26.5%) | 12.8% 감소 |

RADC+MV4 조건은 block2 무보정보다 PSNR이 0.215 dB 높고 Gaussian 수는 26.5% 적다.

Gaussian 처리량이 감소하여 4개 view를 사용했는데도 전체 학습 시간은 약 4.6분 감소했다. 이는 RADC와 멀티뷰 학습의 호환성이 좋다고 판단할 수도 있다. 

최종 PSNR의 상승은 RADC로 생성한 Gaussian 구조와 이후 MV4 학습이 모두 반영된 값이므로 두 방법의 각 효과를 정확히 파악할 수는 없지만 RADC와 멀티뷰 학습이 유의미한 결과를 준다고 해석할 수 있다. 

## 5. Densify 임계값 0.00028 결과

### 5.1 실험 설정

전체 픽셀의 1/16에 해당하는 block4 Ray를 사용했고 0~15000 step에는 RADC를 적용했다. Gaussian 상한은 7,000,000개다.

비교 조건은 clone과 split의 densify 임계값만 0.0002에서 0.00028로 변경했다.
0.00028 조건은 15000 step checkpoint에서 4개 view로 분기하여 30000 step까지
학습했다. 다중 view 구간에는 densification이 없으므로 0.00028 모델의 Gaussian 수는
15000 step과 30000 step에서 같다.

### 5.2 정량 결과

| densify 임계값 | 15000 step 조건 | Gaussian | 15000 step PSNR | 30000 step 조건 | 최종 PSNR | 시간 |
|---:|---|---:|---:|---|---:|---:|
| 0.0002 | block4 RADC | 6,244,813 | — | 단일 view | 26.035 | 42분 |
| 0.00028 | block4 RADC | 3,156,328 | 25.390 | 4개 view | 26.162 | 약 37.5분 |

임계값을 40% 높였을 때 Gaussian 수는 6,244,813개에서 3,156,328개로 49.5% 감소했다.
0.00028 모델은 0.0002 모델의 50.5% 크기다. 두 조건은 같은 초기 point와 RADC를
사용했고 densification 구간의 차이는 임계값이다.

최종 PSNR은 0.00028 조건이 0.127 dB 높다. 그러나 0.0002의 최종 결과는 단일 view이고
0.00028의 최종 결과는 4개 view다. 또한 0.0002는 RTX 4090, 0.00028은 RTX 4070
SUPER에서 학습했으므로 시간도 직접 비교하지 않는다.

임계값의 PSNR 효과에 대해서도 15000 이후의 구간에서 single view, multi view의 차이로 인해 파악 할 수 없으나, RADC를 통한 통제가 잘 이뤄지지 않았던 환경에서 densify 임계값을 통해 통제를 할 수 있다는 점을 파악했다는 점에서 의미가 있다고 생각한다. 

## 6. 결과의 해석 범위

첫째, 각 씬과 조건을 한 번 학습했다. Room의 -0.068 dB와 flowers의 +0.029 dB처럼
작은 차이는 실행 변동 범위일 수 있다. 

둘째, dense 대비 표는 RADC와 다중 view 학습을 결합한 결과다. RADC는 densification
구간의 Gaussian 구조를 정하고, 다중 view 학습은 이후 고정된 구조의 parameter를
업데이트한다. 두 방법의 개별 기여를 확인하기 위해서는 동일 checkpoint 분기 실험으로 따로 비교해야 한다.

셋째, downsampling 4에서 품질 변화와 계산 효율은 분리된다. 실내 4개 씬은 PSNR과
학습 시간이 함께 개선됐지만, stump와 treehill은 PSNR이 약 0.1 dB 증가하는 동안
Gaussian 수도 43~53% 증가했다. 


