# Fisheye씬 EVER vs 3DGUT 비교 실험을 통해 향후 연구방향 보고서


## 1. 3DGUT 논문의 EVER 언급

### 1.1 EVER를 비교 대상으로 포함
3DGUT 논문(CVPR 2025)은 실험 비교표(pinhole)에 EVER를 직접 포함시킴:

**3DGUT 논문 원본 비교표 (CVPR 2025)**

![3DGUT 논문 비교표 — EVER / 3DGRT / StopThePop / 3DGS 수치 비교](report_image_모진수/EVER_vs_3DGUT/3dgut_paper_table.png)

즉 3DGUT 저자들은 EVER를 "distorted camera(fisheye)를 native하게 지원하는 volumetric ray tracing 방법"으로 설명함.

**그러나 fisheye data에 대해 비교하는 부분에서는 ever와 비교하지 않음**

### 1.2 핀홀 데이터에서는 EVER 포함, Fisheye 데이터에서는 제외

3DGUT 논문의 비교 구조를 보면 명확한 비대칭이 존재한다.

| 평가 데이터 | EVER 포함 여부 |
|------------|--------------|
| Table 1 — MipNeRF360 + Tanks&Temples (핀홀) | **포함** |
| Table 3 — ScanNet++ (Fisheye) | **미포함** |

정작 fisheye 데이터셋(ScanNet++) 평가에서는 EVER와 3DGRT 모두 제외되었으며, 논문 내 어디에도 제외 이유가 명시되어 있지 않다.


---

## 2. 실험 설정 (공통)

| 항목 | 값 |
|------|-----|
| 데이터셋 | alameda, london, nyc (모두 OPENCV_FISHEYE) |
| 해상도 | downsample_factor=4, images_4 |
| 학습 이터레이션 | 30,000 |
| train/test 분할 | test_split_interval=8 (8프레임마다 1개 test) |
| spawn_cap | 3,000,000 (gradient top-K 기반 densification 제한) |
| GPU | RTX 3060 Ti 12GB, 동일 머신 |

---

## 3. 씬별 상세 결과

### 3.1 Alameda ✅

| 지표 | EVER | 3DGUT |
|------|------|-------|
| **PSNR** | **21.288 dB** | 20.733 dB |
| **SSIM** | **0.772** | 0.736 |
| **LPIPS** ↓ | **0.286** | 0.368 |
| 학습 시간 | 74.20 min | **15.21 min** |
| 추론 속도 | ~323 ms/frame | **4.68 ms/frame** |

### 3.2 London ✅

| 지표 | EVER | 3DGUT |
|------|------|-------|
| **PSNR** | **24.131 dB** | 23.718 dB |
| **SSIM** | **0.801** | 0.777 |
| **LPIPS** ↓ | **0.331** | 0.417 |
| 학습 시간 | 65.40 min | **13.66 min** |
| 추론 속도 | ~309 ms/frame  | **3.99 ms/frame** |

### 3.3 NYC ✅

| 지표 | EVER | 3DGUT |
|------|------|-------|
| **PSNR** | **27.593 dB** | 25.850 dB |
| **SSIM** | **0.895** | 0.860 |
| **LPIPS** ↓ | **0.194** | 0.295 |
| 학습 시간 | 75.83 min | **14.48 min** |
| 추론 속도 | ~317 ms/frame  | **4.55 ms/frame** |

---

## 4. 전체 비교 요약

| 씬 | EVER PSNR | 3DGUT PSNR | **EVER 우위** |
|---|---|---|---|
| alameda | 21.288 | 20.733 | **+0.56 dB** |
| london | 24.131 | 23.718 | **+0.41 dB** |
| nyc | 27.593 | 25.850 | **+1.74 dB** |
| **평균** | **24.337** | **23.434** | **+0.90 dB** |

| 항목 | EVER | 3DGUT |
|------|------|-------|
| PSNR 평균 | **24.337 dB** | 23.434 dB |
| SSIM 평균 | **0.823** | 0.791 |
| LPIPS 평균 | **0.270** | 0.360 |
| 학습 시간 평균 | 71.8 min | **14.5 min** |
| 추론 속도 | ~323 ms/frame | **~4.4 ms/frame** |

---

## 5. 해석 및 결론

### 품질
- **EVER가 전 씬, 전 지표에서 일관되게 우세**
- PSNR 평균 +0.90 dB (씬별 +0.41 ~ +1.74 dB)
- LPIPS 평균 -0.090 (perceptual 품질 특히 우수)
- nyc처럼 복잡한 도심 씬일수록 EVER 이점이 커지는 경향

### 속도
- 학습: 3DGUT이 **약 5배** 빠름 (14.5 min vs 71.8 min)
- 추론: 3DGUT이 **약 70배** 빠름 (4.4 ms vs ~323 ms) → 실시간 가능

### 결론
- 오프라인 품질 우선 → **EVER** (BVH ray tracing, 정확한 fisheye 처리)
- 실시간/빠른 학습 필요 → **3DGUT** (Unscented Transform 근사, 훨씬 빠름)
- fisheye 지원 자체는 두 방법 모두 가능. 품질 차이가 유의미한 수준(~1 dB)

---

## 6. 시각적 비교

이미지 경로: `report_image_모진수/EVER_vs_3DGUT/`  
각 씬별 3개 대표 프레임 (GT / 3DGUT / EVER 순)

---

### 6.1 Alameda

**Frame 030**

| GT | 3DGUT | EVER |
|:--:|:-----:|:----:|
| ![](report_image_모진수/EVER_vs_3DGUT/alameda_gt_00030.png) | ![](report_image_모진수/EVER_vs_3DGUT/alameda_3dgut_00030.png) | ![](report_image_모진수/EVER_vs_3DGUT/alameda_ever_00030.png) |

**Frame 100**

| GT | 3DGUT | EVER |
|:--:|:-----:|:----:|
| ![](report_image_모진수/EVER_vs_3DGUT/alameda_gt_00100.png) | ![](report_image_모진수/EVER_vs_3DGUT/alameda_3dgut_00100.png) | ![](report_image_모진수/EVER_vs_3DGUT/alameda_ever_00100.png) |

**Frame 180**

| GT | 3DGUT | EVER |
|:--:|:-----:|:----:|
| ![](report_image_모진수/EVER_vs_3DGUT/alameda_gt_00180.png) | ![](report_image_모진수/EVER_vs_3DGUT/alameda_3dgut_00180.png) | ![](report_image_모진수/EVER_vs_3DGUT/alameda_ever_00180.png) |

---

### 6.2 London

**Frame 030**

| GT | 3DGUT | EVER |
|:--:|:-----:|:----:|
| ![](report_image_모진수/EVER_vs_3DGUT/london_gt_00030.png) | ![](report_image_모진수/EVER_vs_3DGUT/london_3dgut_00030.png) | ![](report_image_모진수/EVER_vs_3DGUT/london_ever_00030.png) |

**Frame 120**

| GT | 3DGUT | EVER |
|:--:|:-----:|:----:|
| ![](report_image_모진수/EVER_vs_3DGUT/london_gt_00120.png) | ![](report_image_모진수/EVER_vs_3DGUT/london_3dgut_00120.png) | ![](report_image_모진수/EVER_vs_3DGUT/london_ever_00120.png) |

**Frame 220**

| GT | 3DGUT | EVER |
|:--:|:-----:|:----:|
| ![](report_image_모진수/EVER_vs_3DGUT/london_gt_00220.png) | ![](report_image_모진수/EVER_vs_3DGUT/london_3dgut_00220.png) | ![](report_image_모진수/EVER_vs_3DGUT/london_ever_00220.png) |

---

### 6.3 NYC (PSNR 차이 가장 큰 씬, EVER +1.74 dB)

**Frame 020**

| GT | 3DGUT | EVER |
|:--:|:-----:|:----:|
| ![](report_image_모진수/EVER_vs_3DGUT/nyc_gt_00020.png) | ![](report_image_모진수/EVER_vs_3DGUT/nyc_3dgut_00020.png) | ![](report_image_모진수/EVER_vs_3DGUT/nyc_ever_00020.png) |

**Frame 060**

| GT | 3DGUT | EVER |
|:--:|:-----:|:----:|
| ![](report_image_모진수/EVER_vs_3DGUT/nyc_gt_00060.png) | ![](report_image_모진수/EVER_vs_3DGUT/nyc_3dgut_00060.png) | ![](report_image_모진수/EVER_vs_3DGUT/nyc_ever_00060.png) |

**Frame 100**

| GT | 3DGUT | EVER |
|:--:|:-----:|:----:|
| ![](report_image_모진수/EVER_vs_3DGUT/nyc_gt_00100.png) | ![](report_image_모진수/EVER_vs_3DGUT/nyc_3dgut_00100.png) | ![](report_image_모진수/EVER_vs_3DGUT/nyc_ever_00100.png) |

---



## 7. EVER 논문의 Zip-NeRF 핀홀 결과 (비교 맥락)

아래는 EVER 논문 Supplementary에 수록된 Zip-NeRF 4씬(alameda, berlin, london, nyc) 씬별 결과다. 이 실험은 **undistorted 핀홀 이미지** 기준이며, 3DGS·StopThePop과 동일 조건 비교다.

#### PSNR ↑

| 방법 | berlin | nyc | alameda | london | 평균 |
|------|--------|-----|---------|--------|------|
| 3DGS | 26.83 | 26.90 | 24.14 | 25.48 | 25.84 |
| StopThePop | 26.81 | 27.14 | 24.12 | 25.61 | 25.92 |
| SMERF | 28.52 | 28.21 | 25.35 | 27.05 | 27.28 |
| **EVER** | **27.24** | **27.93** | **24.72** | **26.49** | **26.60** |
| ZipNeRF (offline) | 28.59 | 28.42 | 25.41 | 27.06 | 27.37 |

#### SSIM ↑

| 방법 | berlin | nyc | alameda | london | 평균 |
|------|--------|-----|---------|--------|------|
| 3DGS | .899 | .861 | .776 | .830 | .842 |
| StopThePop | .885 | .844 | .748 | .801 | .819 |
| SMERF | .887 | .844 | .758 | .829 | .830 |
| **EVER** | **.900** | **.863** | **.779** | **.837** | **.845** |
| ZipNeRF (offline) | .891 | .850 | .767 | .835 | .836 |

#### LPIPS ↓

| 방법 | berlin | nyc | alameda | london | 평균 |
|------|--------|-----|---------|--------|------|
| 3DGS | .406 | .380 | .441 | .446 | .418 |
| StopThePop | .402 | .373 | .433 | .438 | .411 |
| SMERF | .391 | .361 | .416 | .390 | .389 |
| **EVER** | **.371** | **.337** | **.389** | **.374** | **.368** |
| ZipNeRF (offline) | .378 | .331 | .387 | .360 | .364 |

핀홀 기준에서도 EVER는 SSIM 전 씬 1위, LPIPS에서 offline ZipNeRF와 동등 수준이다.


### 8. 결론: 3DGUT의 한계로 EVER를 언급
3DGUT 논문 Conclusion에서 overlapping Gaussian 처리 한계를 인정하며 EVER를 미래 방향으로 명시:

> *"as our method still uses a **single point to evaluate each primitive**, it is currently unable to render overlapping Gaussians accurately. Approaches such as **EVER [30] may offer promising directions for addressing this limitation**."*

### 8.1 3DGUT에 EVER렌더링 수식 이식

3DGUT의 파이프라인에 EVER의 exact overlap 렌더링 수식을 이식하면, fisheye 데이터에 대해 두 가지 시나리오가 가능하다:

1. **속도 무관, 품질 극대화**: EVER의 exact volumetric rendering을 유지하되 3DGUT의 fisheye camera model(Unscented Transform)을 결합 → EVER 단독보다 fisheye 왜곡 처리가 개선되어 현재 EVER 품질(PSNR +0.90 dB)을 추가로 뛰어넘을 가능성

2. **EVER보다 빠르고, 3DGUT보다 높은 품질**: GUT의 속도 이점은 어느정도 유지해 EVER(~323 ms)보다 빠른 추론/학습 속도 확보하며 3DGUT 보다 품질 gap(+0.90 dB)을 줄임


3DGUT 저자 스스로 "single point evaluation로 overlapping Gaussian을 정확히 렌더링할 수 없다"고 인정했고, EVER를 해결 방향으로 명시한 만큼 이 이식은 자연스러운 다음 단계다.
