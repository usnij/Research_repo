# 해상도에 따른 sparse 학습의 속도 및 Gaussian 수 변화

작성일 2026-08-25 · 갱신 2026-08-26 · 모진수

## 요약

Downsample 4에서 sparse 학습의 속도 향상과 Gaussian 수 통제가 동시에 성립하는지
sampling 비율을 변경하며 동일 GPU에서 측정했다. 평가한 조건에서는 두 목표를 동시에
만족하는 sampling 비율을 찾지 못했다.

- **1/16 sampling에서는 Gaussian 수가 dense의 1.43배였다.** RADC를 적용해도 개수
  통제가 충분하지 않았다.
- **1/4 sampling에서는 Gaussian 수가 dense의 0.91배였다.** 그러나 학습 시간은 dense
  42.6분, sparse 50.9분으로 sparse가 8.3분 더 길었다.
- 측정한 두 조건에서는 Gaussian 수 통제와 학습 시간 단축이 동시에 나타나지 않았다.

Downsample 2에서는 garden의 Gaussian 수가 dense의 0.95배였고 학습 시간은 3.8배
짧았다. Flowers도 3.4배 빨랐지만 Gaussian 수는 dense의 1.48배로 증가했다. 따라서
속도 향상은 두 씬에서 재현됐지만 개수 통제는 garden에서만 확인됐다. Downsample 1의
bonsai에서는 1/64 sampling 조건의 PSNR이 dense보다 0.030 dB 높았다.

### 발견 1 — Downsample 4에서는 1/16과 1/4 sampling이 서로 다른 조건을 만족하지 못했다

garden downsample 4, 같은 4070, 같은 RADC 설정, 같은 임계값이다.

| garden ds=4 | step 당 Ray | Gaussian | dense 대비 개수 | 0→30000 시간 | dense 대비 속도 |
|---|---:|---:|---:|---:|---:|
| dense | 1,089,480 | 4,206,710 | 1.00 | **42.6 분** | 1.00× |
| 1/16 RADC | 68,040 | 6,020,366 | **1.43** | 36 분 🔴 | 1.18× |
| **1/4 RADC** | 272,160 | **3,824,596** | **0.91** | **50.9 분** | **0.84×** |

1/16 조건은 학습 시간이 짧았지만 Gaussian 수가 dense의 1.43배였다. 

1/4 조건은 Gaussian 수를 0.91배로 유지했지만 dense보다 **8.3분 오래 걸렸다.** 🔴 1/16의 36분은
0→15000 구간을 RTX 4070에서, 15000→30000 구간을 RTX 4090에서 측정한 합이므로 동일 GPU 비교가 아니다. 

1/4의 50.9분은 두 구간 모두 RTX 4070에서 측정했다.

### 발견 2 — Downsample 4의 dense Ray/Gaussian 비는 0.26이었다

downsample 4 에서 garden 의 dense 한 장은 Ray 1,089,480 개에 Gaussian 4,206,710 개다.
Ray/Gaussian 비는 0.26이었다. 

이 조건에서는 acceleration structure update, optimizer,densification 등 Gaussian 수에 의존하거나 Ray 수와 무관한 연산의 비중이 크다. 

따라서 Ray 수 감소에 따른 전체 학습 시간 단축이 제한된다.

같은 씬의 downsample 2 조건에서 Ray/Gaussian 비는 1.02였고, sparse 학습 시간은 dense보다 3.8배 짧았다.

### 발견 3 — Downsample 2에서는 두 씬 모두 학습 시간이 단축됐지만 Gaussian 수 통제는 씬에 따라 달랐다

RTX 4090에서 dense와 sparse를 동일한 설정으로 학습한 결과다.

| ds=2, 1/16 RADC | Gaussian | dense 대비 | PSNR | dense 대비 | 시간 | 속도 |
|---|---:|---:|---:|---:|---:|---:|
| **garden** dense | 4,271,544 | 1.00 | 26.100 | — | 133 분 | 1.00× |
| **garden** sparse | 4,056,975 | **0.95** | 25.947 | −0.153 | **35 분** | **3.8×** |
| **flowers** dense | 3,385,719 | 1.00 | 20.818 | — | 113 분 | 1.00× |
| **flowers** sparse | 5,022,182 | **1.48** | 20.853 | **+0.035** | **33 분** | **3.4×** |


---

### 지난 리포트 (2026-08-25 초판) 로부터
- 초판의 결론: garden의 실외 개수 통제 실패는 sampling 비율과 관련됐으며, 1/4 조건에서 dense의 0.91배가 됐다.
- 그때 없던 것: **1/4 조건의 전체 학습 시간.** 초판에는 0→15000 구간의 20분만 기록됐다.
- 이번에 추가한 것: 동일 RTX 4070에서 1/4 조건의 15000→30000 구간을 측정했다. 전체 시간은 50.9분으로 dense보다 8.3분 길었다. Downsample 2에서는 두 씬의 dense·sparse 결과를 비교했다.

### 합의 사항 → 상태
- **[완료] downsample 4에서 1/4 조건의 전체 시간 측정** — 50.9분으로 dense의 42.6분보다 8.3분 길었다. 초판의 “최적 sampling 비율이 1/16과 1/4 사이에 있다”는 결론은 Gaussian 수만 고려한 것이며 학습 시간까지 포함하면 성립하지 않는다.
- **[완료] downsample 2 에서 dense·sparse 쌍** — garden 3.8 배, flowers 3.4 배.
- **[완료] downsample 1 조건** — bonsai에서 1/64 sampling의 PSNR이 dense보다 0.030 dB 높았다(§4).
- **[미착수] 실내 downsample 4 학습** (사유: 해상도와 씬 종류를 분리하는 조건이나, downsample 4 를 쓰지 않기로 하면 필요가 줄어든다.) 🔴

### 다음
- Downsample 2의 flowers에서 Gaussian 수가 dense의 1.48배가 된 원인을 확인한다. 동일 sampling 비율의 garden과 결과가 다르므로 씬별 특성을 분석해야 한다.
- Downsample 1에서 dense와 sparse를 동일 GPU로 전체 학습하여 시간 비율을 측정한다. 현재는 step당 시간만 측정했다.

---

## 1. Downsample 4의 sampling 비율별 결과

### 1.1 시간

모든 값은 동일 RTX 4070 SUPER에서 실행 directory 생성 시각과 checkpoint 저장 시각의
차이로 계산했다.

| 실행 | 구간 | 시작 | checkpoint | 시간 |
|---|---|---|---|---:|
| garden dense | 0→30000 | 08-18 23:54:08 | 08-19 00:36:41 | **42.6 분** |
| garden 1/4 RADC 분기 | 0→15000 | 08-25 18:10:58 | 08-25 18:31:02 | 20.1 분 |
| garden 1/4 분기 (K=1) | 15000→30000 | 08-25 18:43:05 | 08-25 19:13:55 | 30.8 분 |
| | | | **합** | **50.9 분** |

1/4 sparse 조건의 학습 시간은 dense보다 8.3분 길었다.

두 원인이 관련될 수 있다. 첫째, downsample 4에서는 Ray 수와 무관하거나 Gaussian 수에
의존하는 연산의 비중이 크다. 둘째, RADC의 online-k 추정은 densification step마다
split-half용 sample을 추가로 rendering한다. 그러나 이 rendering이 없는 15000 이후


### 1.2 개수

garden의 downsample을 4로 고정하고 sampling 비율만 변경했다. RADC 설정
(`--hit-shrink --shrink-alpha 1 --online-k`)과 임계값 0.0002 는 동일하다.

| garden ds=4 | step 당 Ray | step 당 Ray/G | Gaussian | dense 대비 |
|---|---:|---:|---:|---:|
| dense | 1,089,480 | 0.259 | 4,206,710 | 1.00 |
| 1/16 RADC | 68,040 | 0.0162 | 6,020,366 | **1.43** |
| **1/4 RADC** | 272,160 | 0.0647 | **3,824,596** | **0.91** |

학습 중 측정한 `k(h)`도 차이가 있었다. h=5에서 k는 1/16 조건 1.53, 1/4 조건
0.83이었다. Shrink factor는 각각 0.629와 0.739였으며, 1/4 조건에서 RADC가
densification score를 26% 감소시켰다. Gaussian 수 0.91배에 대한 RADC의 기여는
무보정 대조군이 없어 분리되지 않았다. 🔴

### 1.3 Flowers의 Gaussian 수

flowers는 dense와 sparse를 서로 다른 GPU에서 학습했으므로 학습 시간은 비교하지 않았다.
Gaussian 수만 비교했다.

| flowers ds=4 | Gaussian | dense 대비 |
|---|---:|---:|
| dense (4070) | 4,952,430 | 1.00 |
| 1/4 RADC (4090) | 5,981,642 | **1.21** |

1/4 조건의 Gaussian 수는 garden에서 dense의 0.91배, flowers에서 1.21배였다. 따라서
동일한 sampling 비율과 RADC 설정에서도 씬별로 결과가 달랐다. §5에서 이 차이를
정리한다.

## 2. 해상도에 따른 Ray/Gaussian 비

Dense 조건의 image당 Ray 수를 해당 씬의 Gaussian 수로 나눈 값이다.

| 씬 | downsample | dense Ray | Gaussian | Ray/G |
|---|---:|---:|---:|---:|
| counter | 1 | 6,466,740 | 1,064,001 | 6.08 |
| bonsai | 1 | 6,479,204 | 1,265,883 | 5.12 |
| room | 2 | 1,614,609 | 1,214,096 | 1.33 |
| counter | 2 | 1,616,166 | 1,221,868 | 1.32 |
| **flowers** | **2** | **4,161,528** | **3,385,719** | **1.23** |
| bonsai | 2 | 1,619,801 | 1,484,957 | 1.09 |
| kitchen | 2 | 1,617,723 | 1,497,515 | 1.08 |
| **garden** | **2** | **4,360,514** | **4,271,544** | **1.02** |
| bicycle | 4 | 1,014,756 | 3,159,475 | 0.32 |
| **garden** | **4** | **1,089,480** | **4,206,710** | **0.26** |
| **flowers** | **4** | **1,039,968** | **4,952,430** | **0.21** |

Ray/Gaussian 비는 downsample 1에서 5.12~6.08, downsample 2에서 1.02~1.33,
downsample 4에서 0.21~0.32였다. 측정 범위에서는 씬보다 해상도에 따른 차이가 컸다.

garden과 flowers는 downsample 2와 4에서 모두 측정했다. garden의 Ray/Gaussian 비는
0.26에서 1.02로 3.9배, flowers는 0.21에서 1.23으로 5.9배 증가했다. 

Flowers의 Gaussian 수는 해상도가 높아질 때 4,952,430개에서 3,385,719개로 감소했지만, garden은
4,206,710개에서 4,271,544개로 비슷했다. 이 씬별 차이의 원인은 확인되지 않았다. 🔴

## 3. Jensen inflation과 Ray/Gaussian 비의 관계

7개 씬(실내 4개, 실외 3개), block 2/4/6/8, downsample 1/2/4로 구성한 27개
조건에서 densification 통계량의 Jensen inflation을 측정했다. Dense 수렴 checkpoint에서
200~300 step 구간을 사용했다.

| step 당 Ray/G 구간 | 조건 수 | inflation |
|---|---:|---|
| 0.14 ~ 0.38 | 4 | 1.24 ~ 1.58 |
| 0.065 ~ 0.095 | 7 | 1.43 ~ 1.84 |
| 0.017 ~ 0.037 | 8 | 2.09 ~ 3.56 |
| 0.006 ~ 0.013 | 6 | 2.53 ~ 4.11 |

실내외 씬은 Ray/Gaussian 비에 따라 같은 범위에 포함됐다. bicycle block 2는
Ray/Gaussian 0.0802, inflation 1.478이었고 counter block 4는 각각 0.0825, 1.734였다.
Kitchen block 8은 0.0167, 3.559로 실외 씬과 같은 범위에 포함됐다.

회귀식에 씬 category를 indicator variable로 추가했을 때 설명력은 0.909에서
0.918로 증가했다. Block 크기만 사용한 회귀식의 설명력은 0.656이었다. 따라서 측정한
조건에서는 step당 Ray/Gaussian 비가 inflation을 가장 잘 설명했다.

Downsample 4의 1/16 조건은 Ray/Gaussian 0.0162로 inflation 2.53~4.11 범위에 해당한다.
이 조건에서 Gaussian 수는 dense의 1.43배였다. 1/4 조건은 Ray/Gaussian 0.0647,
inflation 1.43이었다.

## 4. Downsample 1의 sparse 학습 결과

bonsai를 downsample 1(3118×2078, 6,479,204 pixels)로 학습한 결과다. Dense 기준은
RTX 4090에서 다시 평가했다.

| bonsai ds=1 | Gaussian | dense 대비 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| dense | 1,265,883 | 1.00 | 31.706 | 0.9339 | 0.3548 |
| **sparse 1/64 + RADC + 다중 view** | 1,567,412 | 1.24 | **31.736** | 0.9281 | 0.3596 |

Sparse의 PSNR은 31.736 dB로 dense보다 0.030 dB 높았다. Ray 예산은
101,400/6,479,204로 약 1/64이다.

속도는 Gaussian model을 1,265,883개로 고정하고 sampling 비율만 변경하여 step당 시간으로
측정했다. 전체 학습 시간에는 학습 중 Gaussian 수 변화가 포함되지만, 이 측정은 동일한
model과 RTX 4090을 사용하여 Ray 수에 따른 시간 차이를 분리했다.

| block | Ray/step | sampling 비율 | ms/step | dense 대비 |
|---:|---:|---:|---:|---:|
| 1 | 6,479,204 | 1/1 | 383.2 | 1.00× |
| 2 | 1,619,801 | 1/4 | 94.2 | 4.07× |
| 4 | 404,301 | 1/16 | 29.6 | 12.95× |
| **8** | **100,751** | **1/64** | **17.4** | **22.07×** |

1/4 조건의 step당 시간은 dense보다 4.07배 짧아 Ray 감소 비율과 비슷했다. 1/64에서는
22.07배로, 이론적인 64배에 도달하지 않았다. 

Acceleration structure와 optimizer 등 Ray 수와 무관한 비용이 남기 때문이다. 

🔴 이 표는 step당 시간이고 downsample 2와 4는 전체 학습 시간이므로 직접 비교할 수 없다. 

Downsample 2에서도 동일한 step당 시간 측정이 필요하다.

## 5. Ray/Gaussian 비만으로 설명되지 않는 결과

Gaussian 수는 해상도만으로 결정되지 않았다. Downsample 2, 1/16, RADC 조건에서
garden은 dense의 0.95배, flowers는 1.48배였다. Downsample 4, 1/4 조건에서도 각각
0.91배와 1.21배였다. 동일한 sampling 비율과 해상도에서도 씬별 차이가 남았다.

| | ds=4, 1/4 | ds=2, 1/16 |
|---|---:|---:|
| garden | 0.91 | 0.95 |
| flowers | 1.21 | 1.48 |

Densification ranking 개선 폭도 Ray/Gaussian 비만으로 설명되지 않았다. 비슷한 Ray/G에서
상위 1% overlap 개선은 counter +0.165, bicycle +0.041로 약 4배 차이가 났다. Bicycle은
같은 Ray/G에서도 h 중앙값이 3.71로 counter의 4.65보다 낮았고, 큰 Gaussian에 Ray hit가
집중되는 분포 차이가 남았다.

Downsample 4에서 sparse 학습의 memory 이점은 별도로 측정하지 않았다. Garden의 dense
학습은 RTX 4070에서 완료됐으므로 이 씬에서는 sparse 학습이 memory capacity를
확장한다는 근거도 없다.

1/4 sparse가 dense보다 느린 원인은 확인되지 않았다. Densification이 종료된
15000→30000 구간에서도 sparse 30.8분, dense 약 21.3분이었으며, 이 구간에는 RADC의
split-half 추가 rendering이 없다.

