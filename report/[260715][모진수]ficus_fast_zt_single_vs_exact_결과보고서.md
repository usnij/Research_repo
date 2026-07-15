# ficus 근사구조 vs exact 렌더링 비교 보고서

**실험 대상:** Ficus, 근사구조 vs exact  
**이미지 폴더:** `report_image_모진수/260715/`  
**핵심 질문:** 근사구조 방식과 exact 방식이 동일 Ficus 씬에서 어떤 차이를 보이는가

---

## 1. 실험 조건

| 항목 | 내용 |
|---|---|
| 데이터셋 | **Ficus** |
| 비교 방법 A | **근사구조** (`ficus_fast_zt_single`) |
| 비교 방법 B | **exact** (`ficus_exact`) |
| 시점 수 | 각 방법 8 pose (pose_1 ~ pose_8), 800×800 |


---

## 2. 핵심 이미지 비교

### 2.1 컨택트 시트 (근사구조)

![근사구조 8-pose 컨택트 시트](report_image_모진수/260715/ficus_no_mask_fast_zt_single/contact_sheet_fast_zt_single.png)

### 2.2 컨택트 시트 (exact)

![exact 8-pose 컨택트 시트](report_image_모진수/260715/ficus_no_mask_exact/contact_sheet_exact.png)

### 2.3 pose_1

| 근사구조 | exact |
|---|---|
| ![근사구조 pose_1](report_image_모진수/260715/ficus_no_mask_fast_zt_single/pose_1/render_fast_zt_single.png) | ![exact pose_1](report_image_모진수/260715/ficus_no_mask_exact/pose_1/render_exact.png) |

### 2.4 pose_2

| 근사구조 | exact |
|---|---|
| ![근사구조 pose_2](report_image_모진수/260715/ficus_no_mask_fast_zt_single/pose_2/render_fast_zt_single.png) | ![exact pose_2](report_image_모진수/260715/ficus_no_mask_exact/pose_2/render_exact.png) |

### 2.5 pose_3

| 근사구조 | exact |
|---|---|
| ![근사구조 pose_3](report_image_모진수/260715/ficus_no_mask_fast_zt_single/pose_3/render_fast_zt_single.png) | ![exact pose_3](report_image_모진수/260715/ficus_no_mask_exact/pose_3/render_exact.png) |

### 2.6 pose_4

| 근사구조 | exact |
|---|---|
| ![근사구조 pose_4](report_image_모진수/260715/ficus_no_mask_fast_zt_single/pose_4/render_fast_zt_single.png) | ![exact pose_4](report_image_모진수/260715/ficus_no_mask_exact/pose_4/render_exact.png) |

### 2.7 pose_5

| 근사구조 | exact |
|---|---|
| ![근사구조 pose_5](report_image_모진수/260715/ficus_no_mask_fast_zt_single/pose_5/render_fast_zt_single.png) | ![exact pose_5](report_image_모진수/260715/ficus_no_mask_exact/pose_5/render_exact.png) |

### 2.8 pose_6

| 근사구조 | exact |
|---|---|
| ![근사구조 pose_6](report_image_모진수/260715/ficus_no_mask_fast_zt_single/pose_6/render_fast_zt_single.png) | ![exact pose_6](report_image_모진수/260715/ficus_no_mask_exact/pose_6/render_exact.png) |

### 2.9 pose_7

| 근사구조 | exact |
|---|---|
| ![근사구조 pose_7](report_image_모진수/260715/ficus_no_mask_fast_zt_single/pose_7/render_fast_zt_single.png) | ![exact pose_7](report_image_모진수/260715/ficus_no_mask_exact/pose_7/render_exact.png) |

### 2.10 pose_8

| 근사구조 | exact |
|---|---|
| ![근사구조 pose_8](report_image_모진수/260715/ficus_no_mask_fast_zt_single/pose_8/render_fast_zt_single.png) | ![exact pose_8](report_image_모진수/260715/ficus_no_mask_exact/pose_8/render_exact.png) |
