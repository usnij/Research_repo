# Counter: SSIM 손실 유무에 따른 Gaussian 개수

**SSIM 포함 기록에서 Gaussian 수가 크게 증가했다. 아래 배율은 기존 기록 간 비교도 섞여있다.**

## 1. Gaussian 개수 비교

Counter DS2(1558×1038), SfM 초기 155,767개, ADC 기반 학습(MCMC 아님). 아래 표는 **15,000 step 시점의 실제 Gaussian 수**다.

| 렌더러 | 학습 픽셀 | 손실 | Gaussian 수 | 해당 L1 기록 대비 |
|---|---|---|---:|---:|
| 3DGS | Dense | L1 only | **411,523** | 기준 |
| 3DGS | Dense | L1 + SSIM | **1,169,328** | **2.84배** |
| 3DGS | Sparse · 명목 1/16 | L1 only | **512,165** | 기준 |
| 3DGS | Sparse · 명목 1/16 | L1 + 0.2 compact SSIM | **1,404,255** | **2.74배** |
| 3DGRT | Dense | L1 + SSIM | **1,255,427** | — |
| 3DGRT | Sparse · 명목 1/16 | 0.8 L1 only | **675,652** | 기준 |
| 3DGRT | Sparse · 명목 1/16 | 0.8 L1 + 0.2 compact SSIM | **2,126,054** | **3.15배** |

- sparse 학습에 넣은 ssim은 샘플링된 픽셀을 11x11 격자로 구성해 구하는 방식으로 구현했다. 여기서는 품질 비교보단 일단 갯수의 변화를 확인하기 위함이므로 15,000 stpe까지만 학습했고 품질비교는 하지 않는다.
- 
**결론:** SSIM 포함 여부와 큰 개수 차이가 관측된다. 이는 sparse학습만의 현상은 아니고 dense에서도 나타나는 현상이며 원인을 생각하면 그 이유는 명확하긴 하다. 

## 2. SSIM을 넣었는데 왜 Gaussian이 더 생기는가?

**Gaussian을 학습시키는 신호가, Gaussian을 더 만들지 결정하는 데도 쓰이기 때문이다.** L1은 각 픽셀의 색을 맞추고, SSIM은 주변 픽셀 사이의 대비·무늬까지 맞추도록 요구한다. 이 요구가 Gaussian의 위치를 수정하려는 신호인 **위치 gradient**를 바꾼다. ADC는 그 신호가 일정 크기를 넘으면 해당 Gaussian을 복제하거나 분할한다. 따라서 SSIM을 추가하면 **기존 Gaussian의 학습 방향뿐 아니라, 새 Gaussian의 생성 여부까지 함께 바뀐다.**

예를 들어 생성 기준이 `0.0002`일 때, L1만으로는 점수가 `0.00015`였던 Gaussian이 SSIM 추가 후 `0.00030`이 되면 생성 대상으로 바뀐다. *(설명용 예시이며, SSIM이 모든 Gaussian의 점수를 높이는 것은 아니다.)*

**GRT는 드물게 관측되는 큰 신호를 더 강하게 반영할 수 있다.** 화면 안에 있는 Gaussian의 신호가 네 step 동안 `0, 0, 0, 4`였다고 가정하면, 원 GS 방식은 네 번을 모두 세어 평균 **1**, GRT 방식은 0을 제외해 평균 **4**로 집계한다. Sparse에서는 픽셀이 뽑히지 않아 0이 되는 경우가 생기므로 이 차이가 중요하다. *(같은 신호를 가정한 분모 설명이며, 실제 GS·GRT는 gradient 계산 자체도 다르다.)* 또한 원 GS에는 크게 퍼진 Gaussian을 삭제하는 규칙이 있지만 이번 GRT 설정에서는 꺼져 있다. **이 차이들이 최종 개수 격차를 얼마나 설명하는지는 아직 확인하지 않았다.**

[고정 모델 진단 및 코드 비교 근거](../research_resume_260918/grt_counter_sparse_l1_adc_261123/loss_adc_probe/README.md)

근거: [GS dense L1 로그](/home/jinsu/Documents/Github/gaussian-splatting/logs_thin/f15_dense_l1.log) · [GS dense SSIM 15k/30k 기록](../research_resume_260918/gs_counter_ds2_30000_261014/run/steps.jsonl) · [GS sparse L1 15k PLY](/home/jinsu/Documents/Github/gaussian-splatting/output/f15_tile11k10_l1/point_cloud/iteration_15000/point_cloud.ply) · [GS compact SSIM 로그](/home/jinsu/Documents/Github/gaussian-splatting/logs_thin/f15_tile16k16_compact.log) · [GRT dense 실행](/home/jinsu/Documents/Github/3dgrut/runs/grt/counter_grt_ds2/counter-1508_142158) · [GRT sparse 비교](../research_resume_260918/grt_counter_sparse_l1_adc_261123/RESULTS.md).
