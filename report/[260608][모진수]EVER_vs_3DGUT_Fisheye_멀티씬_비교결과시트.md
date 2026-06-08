# EVER vs 3DGUT — Fisheye 멀티씬 비교 실험 결과 시트

**작성일**: 2026-06-08  
**목적**: EVER(ray tracing)와 3DGUT(rasterization)의 fisheye 데이터셋 3씬에서의 품질·속도 공정 비교

---

## 1. 실험 설정 (공통)

| 항목 | 값 |
|------|-----|
| 데이터셋 | alameda, london, nyc (모두 OPENCV_FISHEYE) |
| 해상도 | downsample_factor=4, images_4 |
| 학습 이터레이션 | 30,000 |
| train/test 분할 | test_split_interval=8 (8프레임마다 1개 test) |
| spawn_cap | 3,000,000 (gradient top-K 기반 densification 제한) |
| GPU | RTX 3060 Ti 12GB, 동일 머신 |

---

## 2. 씬별 상세 결과

### 2.1 Alameda ✅

| 지표 | EVER | 3DGUT |
|------|------|-------|
| **PSNR** | **21.288 dB** | 20.733 dB |
| **SSIM** | **0.772** | 0.736 |
| **LPIPS** ↓ | **0.286** | 0.368 |
| 학습 시간 | 74.20 min | **15.21 min** |
| 추론 속도 | ~323 ms/frame | **4.68 ms/frame** |

### 2.2 London ✅

| 지표 | EVER | 3DGUT |
|------|------|-------|
| **PSNR** | **24.131 dB** | 23.718 dB |
| **SSIM** | **0.801** | 0.777 |
| **LPIPS** ↓ | **0.331** | 0.417 |
| 학습 시간 | 65.40 min | **13.66 min** |
| 추론 속도 | — | **3.99 ms/frame** |

### 2.3 NYC ✅

| 지표 | EVER | 3DGUT |
|------|------|-------|
| **PSNR** | **27.593 dB** | 25.850 dB |
| **SSIM** | **0.895** | 0.860 |
| **LPIPS** ↓ | **0.194** | 0.295 |
| 학습 시간 | 75.83 min | **14.48 min** |
| 추론 속도 | — | **4.55 ms/frame** |

---

## 3. 전체 비교 요약

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

## 4. 해석 및 결론

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

## 5. 시각적 비교

이미지 경로: `report_image_모진수/EVER_vs_3DGUT/`  
각 씬별 3개 대표 프레임 (GT / 3DGUT / EVER 순)

---

### 5.1 Alameda

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

### 5.2 London

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

### 5.3 NYC (PSNR 차이 가장 큰 씬, EVER +1.74 dB)

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

## 6. 3DGUT 논문의 EVER 언급

### 6.1 EVER를 비교 대상으로 포함
3DGUT 논문(CVPR 2025)은 실험 비교표에 EVER를 직접 포함시킴:

> *"we limit our comparison to the original 3DGS and StopThePop as the representative splatting methods, along with **3DGRT and EVER** as volumetric particle tracing methods that natively support distorted cameras and secondary lighting effects."*

즉 3DGUT 저자들은 EVER를 "distorted camera(fisheye)를 native하게 지원하는 volumetric ray tracing 방법"으로 분류하고 직접 비교했음.

### 6.2 3DGUT의 한계로 EVER를 언급
3DGUT 논문 Conclusion에서 overlapping Gaussian 처리 한계를 인정하며 EVER를 미래 방향으로 명시:

> *"as our method still uses a **single point to evaluate each primitive**, it is currently unable to render overlapping Gaussians accurately. Approaches such as **EVER [30] may offer promising directions for addressing this limitation**."*

### 6.3 우리 연구와의 연결

| 포인트 | 의미 |
|--------|------|
| 3DGUT 저자가 overlap 처리 한계를 직접 인정 | 우리가 연구하는 overlap 인식 렌더링이 논문에서도 공인된 문제 |
| EVER를 해결 방향으로 지목 | EVER의 exact volumetric rendering이 overlap 처리의 정답에 가까운 접근 |
| 우리 실험(+0.90 dB)과 일치 | EVER 우위가 특히 복잡한 씬(nyc +1.74 dB)에서 크게 나타남 → overlap이 많은 씬일수록 EVER 이점 |

---

## 7. 코드 수정 이력 (EVER `ever_training/`)

| 파일 | 수정 내용 |
|------|----------|
| `gaussian_renderer/ever.py` | `from utils import camera_utils_zipnerf` import 추가 |
| `arguments/__init__.py` | `spawn_cap = -1` 파라미터 추가 |
| `scene/gaussian_model.py` | `densify_and_clone/explore/prune`에 gradient top-K spawn_cap 구현 |
| `train.py` | `densify_and_prune()` 호출 시 `spawn_cap=opt.spawn_cap` 전달 |
| `scene/__init__.py` | `skip_train=True` 시 train 카메라 로딩 skip (render eval OOM 방지) |
| `metrics.py` | `readImages()`에서 이미지 CPU 유지 (GPU OOM 방지) |
