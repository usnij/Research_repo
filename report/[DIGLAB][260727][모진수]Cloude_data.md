# WDAS Cloud 데이터 — 논문에서의 언급과 우리가 만든 데이터셋

## 요약

**지난 미팅 (2026-07-26)** — 키워드 3줄
- Celarek et al.(CGF 2025) 실험이 대부분 solid object 장면
- 저밀도 볼류메트릭 구간의 수치가 없어 직접 확인 필요
- 논문이 쓴 WDAS Cloud 데이터의 성격 파악

**합의 사항 → 상태**
- [완료] 논문에서의 WDAS Cloud 언급 정리
- [완료] 3DGS 학습용 데이터셋 구성

**이번 결과 / 막힌 것 / 다음**
- 데이터 성격과 우리가 만든 데이터셋 구성은 본문 참조
- 다음: 저밀도 장면에서 근사 방식 간 차이 측정

---

## 1. 논문에서의 언급

대상 논문은 Celarek et al., *Does 3D Gaussian Splatting Need Accurate Volumetric Rendering?*, CGF 44(2) / Eurographics 2025.

### 1.1 어떤 데이터인가

원본은 **Walt Disney Animation Studios Cloud Data Set**이다. 단일 OpenVDB 밀도장 파일이며, 다시점 이미지 데이터셋이 아니다.

- 공식 페이지: <https://disneyanimation.com/resources/clouds/>
- 라이선스: CC BY-SA 3.0
- 구성: `wdas_cloud.vdb` (전체 해상도) + half/quarter/eighth/sixteenth 축소본 + Hyperion 참고 렌더 + Mitsuba 예제 씬

논문은 표준 벤치마크인 NeRF-synthetic 만으로는 볼륨 렌더링의 효과를 평가할 수 없다고 판단했다. 그 데이터셋의 장면들이 대부분 불투명한 solid object 이기 때문이다. 그래서 **볼류메트릭 장면을 직접 제작**했다.

> these scenes focus mostly on solid objects: to further assess the impact of correct volumetric rendering in detail, we created additional volumetric datasets with **varying parameters for transparency, frequency, scattering coefficients, and color**.

논문에 실린 볼류메트릭 장면 사진이다.

![논문에 실린 볼류메트릭 장면 네 개](./report_image_모진수/260726/Celarek_Figure8_VolumetricScenes.png)

*왼쪽부터 EXPLOSION, WDAS Cloud 2, WDAS Cloud 3, Colored WDAS Cloud.*

**이 사진이 데이터의 성격을 그대로 드러낸다.** 가운데 두 장(Cloud 2, Cloud 3)과 오른쪽 Colored Cloud 는 **실루엣이 완전히 동일하다.** 밀도와 색만 다를 뿐 같은 구름이다. 즉 WDAS Cloud 1 / 2 / 3 / Colored 는 서로 다른 구름 네 개가 아니라 **하나의 Disney VDB 를 투명도·산란 계수·색만 바꿔 렌더링한 변종**이며, 앞서 인용한 "varying parameters" 라는 표현이 정확히 이 뜻이다. 맨 왼쪽 EXPLOSION 만 다른 애셋이다.

이것은 우리 실험 설계에 직접적인 함의를 준다. **논문 스스로 "같은 볼륨, 다른 매질 파라미터" 라는 통제 실험 구조를 이미 쓰고 있었다는 뜻**이기 때문이다. 다만 논문은 그 파라미터 축을 결과 분석에 쓰지 않고 장면 이름으로만 남겨두었다(§1.5 ④ 참조).

**중요한 제약:** 각 변종의 구체적 파라미터 값(밀도 스케일, 카메라 배치, 뷰 개수, 해상도)은 논문에도 공개 저장소에도 없다. 저장소 README 는 MipNeRF360 · Tanks&Temples 만 안내한다. 따라서 **논문의 Cloud 1/2/3 을 수치까지 그대로 재현할 수는 없다.** 우리가 같은 VDB 에서 직접 렌더링해야 하는 이유이며, 그 과정은 2 장에 정리했다.

### 1.2 본문에서의 언급 — "예외"

논문의 전체 결론은 **"Gaussian 수가 10만 개를 넘으면 근사를 쓰는 opacity 기반이 이긴다"** 이다. 그런데 WDAS Cloud 1 은 본문에 딱 한 번, **그 결론이 성립하지 않는 예외**로 등장한다.

> the volumetric, **low-density WDAS CLOUD 1 scene is an exception to this trend**: here, the more principled, EWA-based variants show an advantage **regardless of model size**.

"regardless of model size" 가 핵심이다. 다른 모든 장면에서는 primitive 를 늘리면 medium 의 이점이 사라지는데, 이 장면에서는 사라지지 않는다.

논문이 그 근거로 제시한 사진이다.

![논문에 실린 MATERIALS 와 WDAS Cloud 1 비교](./report_image_모진수/260726/Celarek_Figure13_Materials_WdasCloud.png)

*(a–c) MATERIALS, (d–g) WDAS Cloud 1. 둘 다 Gaussian 100만 개 조건이며, 좌하단 삽입 이미지는 GT 대비 제곱 오차다. 어두울수록 오차가 작다.*

오른쪽 세 장(e–g)이 WDAS Cloud 1 이다. **(f) 3DGS 의 오차 삽입 이미지에는 구름 실루엣을 따라 밝은 윤곽선이 뚜렷한 반면, (g) OTS Marcher 는 그 윤곽이 흐릿하다.** 즉 오차가 구름 **경계**에 집중되어 있고, 물리적으로 옳은 모델이 그 경계를 더 잘 잡는다. 논문 캡션의 표현도 같다.

> Similar to MATERIALS, **silhouettes are better captured with the more physically-inspired model.**

왼쪽 (a–c)의 MATERIALS 도 같은 성격의 예외다. 반사 물체의 grazing angle 실루엣에서 EWA 계열이 유리하다. **두 예외가 모두 "실루엣/경계"** 라는 공통점을 갖는다는 점은 눈여겨볼 만하다 — 3.2 에서 본 시점 의존 불투명도가 경계 표현에 관여하기 때문으로 보이지만, 논문은 이 연결을 명시하지 않는다.

**본문의 언급은 여기서 끝난다.** 왜 이 장면만 예외인지, 밀도가 얼마나 낮아야 예외가 되는지에 대한 분석은 없다. 정량 수치조차 본문 표에는 없고 supplemental 에만 실려 있어, 사실상 각주 취급을 받은 셈이다. 다음 두 절에서 그 수치를 꺼내 본다.

### 1.3 supplemental 의 수치 — 볼류메트릭 장면 평균

**PSNR ↑** (supplemental Table 2)

| Gaussian 수 | 4k | 12k | 36k | 108k | 324k | 972k |
|---|---|---|---|---|---|---|
| 3DGS | 39.37 | 40.78 | 42.15 | **43.29** | **44.01** | **44.33** |
| 3DGS + STP | 39.44 | 40.85 | 42.23 | 43.43 | 44.12 | 44.43 |
| OTS | **39.59** | **40.98** | 42.06 | 42.91 | 43.43 | 43.71 |
| OTS + SAtn | **39.66** | **41.04** | 42.15 | 42.93 | 43.38 | 43.71 |
| 3DGS marcher | 39.51 | 41.03 | **42.31** | 43.15 | 43.59 | 43.66 |
| OTS marcher | **39.66** | **41.04** | 42.20 | 42.98 | 43.34 | 43.48 |

**LPIPS ↓** (supplemental Table 8)

| Gaussian 수 | 4k | 12k | 36k | 108k | 324k | 972k |
|---|---|---|---|---|---|---|
| 3DGS | .1200 | .1023 | .0853 | .0721 | .0641 | .0600 |
| 3DGS + STP | .1192 | .1015 | .0849 | .0720 | .0641 | .0602 |
| **OTS** | **.1146** | **.0954** | **.0803** | **.0685** | **.0613** | **.0574** |
| **OTS + SAtn** | **.1146** | .0955 | **.0800** | **.0683** | **.0611** | **.0573** |
| 3DGS marcher | .1173 | .0986 | .0832 | .0719 | .0647 | .0609 |
| OTS marcher | .1148 | .0959 | .0809 | .0691 | .0623 | .0586 |

**PSNR 은 10만 개 부근에서 역전하지만 LPIPS 는 전 구간에서 extinction 기반이 앞선다.** 전 장면 평균에서는 LPIPS 도 1M 에서 역전되는데(.0467 vs .0471), 볼류메트릭 장면만 보면 역전이 없다.

### 1.4 supplemental 의 수치 — WDAS Cloud 1 단독

가장 저밀도인 장면이다.

**PSNR ↑**

| Gaussian 수 | 4k | 12k | 36k | 108k | 324k | 972k |
|---|---|---|---|---|---|---|
| 3DGS | 48.87 | 49.05 | 49.11 | 49.17 | 49.17 | 49.23 |
| 3DGS + STP | 48.87 | 49.11 | 49.28 | 49.38 | 49.36 | 49.37 |
| OTS | 48.82 | 49.04 | 49.36 | 49.59 | 49.71 | 49.99 |
| OTS + SAtn | 48.96 | 49.12 | 49.43 | 49.51 | 49.70 | 49.95 |
| 3DGS marcher | 48.88 | 48.91 | 49.17 | 49.14 | 49.17 | 49.22 |
| **OTS marcher** | **49.03** | **49.31** | **49.51** | **49.68** | **50.00** | **50.17** |
| **OTS marcher − 3DGS** | **+0.16** | **+0.26** | **+0.40** | **+0.51** | **+0.83** | **+0.94** |

**LPIPS ↓**

| Gaussian 수 | 4k | 12k | 36k | 108k | 324k | 972k |
|---|---|---|---|---|---|---|
| 3DGS | .0741 | .0723 | .0703 | .0680 | .0670 | .0675 |
| OTS | .0734 | .0709 | .0690 | .0651 | .0623 | **.0611** |
| OTS + SAtn | .0733 | .0708 | .0689 | **.0650** | **.0623** | **.0608** |
| 3DGS marcher | .0739 | .0719 | .0695 | .0667 | .0646 | .0647 |
| OTS marcher | .0734 | .0709 | .0692 | **.0649** | .0624 | .0610 |

**SSIM ↑** — 차이가 3~4 째 자리에서만 나타난다 (3DGS .9951 / OTS .9954 / OTS marcher .9954, 전 구간 거의 평평).

### 1.5 읽어야 할 지점

**① 경향이 반대로 뒤집힌다.** 전체 평균에서는 Gaussian 이 많아질수록 opacity 가 이기는데, 이 장면에서는 medium 이 항상 이기고 **격차가 오히려 커진다** (+0.16 → +0.94 dB).

**② 이득의 출처가 overlap 이 아니라 파라미터화다.** 3DGS marcher(= opacity 유지 + overlap·투영 정확)는 3DGS 와 사실상 동일하다(49.22 vs 49.23). 반면 OTS(= extinction + splatting)는 3DGS 를 크게 앞선다(49.99 vs 49.23). **overlap 을 정확히 처리하는 것보다 extinction 파라미터화가 결정적이다.**

**③ 절대 PSNR 이 48~50 dB 로 매우 높다.** 극단적으로 옅은 구름이라 절대 오차 자체가 작기 때문이다. 따라서 +0.94 dB 라는 격차의 체감 크기를 해석할 때 이 점을 감안해야 한다.

**④ 논문은 이 예외를 설명하지 않으며, 스스로 반례도 갖고 있다.** BURNING FICUS 장면의 연기에 대해서는 오히려 **"Volumetric effects like smoke are unaffected"** 라고 적었다. 즉 볼류메트릭 요소가 있다고 해서 항상 medium 이 유리한 것이 아니다.

정리하면 논문 안에 이런 대비가 있다.

| 장면 | 볼류메트릭인가 | medium 이 유리한가 |
|---|---|---|
| WDAS Cloud 1 (저밀도 구름) | ○ | **○ (크기 무관하게)** |
| WDAS Cloud 2 / 3 (더 짙은 같은 구름) | ○ | 부분적 (LPIPS 만) |
| BURNING FICUS 의 연기 | ○ | **× (차이 없음)** |

**"볼류메트릭이냐"가 아니라 "얼마나 옅으냐"가 갈림선으로 보인다.** 그런데 §1.1 에서 확인했듯 논문은 이미 같은 구름의 밀도 변종을 여러 개 갖고 있었으면서도, 그것을 밀도 축으로 정렬해 분석하지 않고 서로 다른 장면 이름으로만 다뤘다. **밀도를 통제 변수로 둔 분해가 통째로 비어 있는 실험이며, 2 장의 데이터셋 설계가 정확히 이 빈칸을 겨냥한다.**

---

## 2. 우리가 만든 데이터셋

논문의 Cloud 1/2/3 을 재현할 수 없으므로, 같은 Disney VDB 에서 **밀도를 통제 변수로 두고 직접 렌더링**했다. 생성 도구는 [`make_cloud_dataset.py`](./make_cloud_dataset.py), 상세 문서는 [`CLOUD_DATASET_README.md`](./CLOUD_DATASET_README.md).

### 2.1 원본 VDB

| 파일 | 크기 | 복셀 해상도 | 활성 복셀 |
|---|---|---|---|
| `wdas_cloud.vdb` | 2794 MB | — | — |
| `wdas_cloud_half.vdb` | 469 MB | 994 × 676 × 1225 | 188,358,293 |
| `wdas_cloud_quarter.vdb` | 65 MB | 498 × 338 × 613 | 24,063,202 |
| **`wdas_cloud_eighth.vdb`** ← 사용 | **9.6 MB** | 250 × 170 × 307 | 3,122,022 |
| `wdas_cloud_sixteenth.vdb` | 1.6 MB | 126 × 86 × 154 | 415,642 |

폴더 전체 3.3 GB. 월드 크기 420 × 287 × 513 단위. 400×400 렌더에서는 eighth 이상 올려도 차이가 잘 보이지 않아 eighth 를 택했다.

### 2.2 생성된 데이터셋 — `datasets/cloud_d025`

NeRF-synthetic(Blender) 형식이며 3DGS · EVER 가 바로 읽는다.

```
datasets/cloud_d025/          총 8.3 MB
├── train/   100장   6.4 MB
├── test/     20장   1.3 MB
├── val/       8장   528 KB
├── transforms_train.json / _test.json / _val.json
└── render_config.json
```

| 항목 | 값 |
|---|---|
| 이미지 | 400 × 400 RGBA PNG, **검정 배경** |
| 뷰 | 128 장, 구면 균일 분포(Fibonacci), 고도각 −60° ~ +60° |
| 카메라 | 반지름 4.0 고정, 수평 FoV 39.6° |
| spp | 256 (+ OptiX 디노이저) |
| 렌더러 | Mitsuba 3.9.0 `volpath`, `cuda_ad_rgb` |
| 생성 시간 | 약 15 분 (RTX 4070 SUPER) |

### 2.3 매질 파라미터

기본값은 원본 `wdas_cloud_mitsuba_scene.xml` 에서 가져왔다.

| 파라미터 | 값 | 비고 |
|---|---|---|
| **density scale** | **0.25** | 원본 xml 은 4.0. **핵심 실험 변수** |
| albedo | 1.0 | 흡수 없음 |
| phase | HG, g = 0.877 | 강한 전방 산란 |
| sun irradiance | (2.6, 2.5, 2.3) | directional |
| sun direction | (−0.5826, −0.7660, −0.2717) | |
| ambient | 0.0 | 배경을 정확히 검정으로 유지 |
| max_depth / rr_depth | 512 / 512 | RR 비활성화 |

### 2.4 설계상 중요한 결정 세 가지

**① 씬 정규화.** 3DGS 의 Blender 로더는 초기 포인트를 `[-1.3, 1.3]^3` 에 **고정 크기로** 배치한다. 원본 구름은 420 × 287 × 513 월드 단위라 그대로 쓰면 학습이 되지 않는다. 구름 중심을 원점으로 옮기고 최대 half-extent 를 1.0 으로 맞췄으며, 광학 두께 보존을 위해 밀도 스케일에 역배율을 곱한다.

**② Russian roulette 비활성화.** 고알베도 매질에서 RR 을 켜면 긴 경로가 조기 종료되며 fireflies 가 생긴다. Disney README 도 같은 문제를 지적한다. 실측(200×200, spp 128, density 4.0):

| 설정 | 렌더 시간 | 최대 라디언스 | 1.5 초과 픽셀 |
|---|---|---|---|
| `rr_depth 256` (RR 켬) | 32.9 s | **44.87** | 0.04 % |
| `rr_depth = max_depth` (RR 끔) | 43.6 s | **1.02** | 0.00 % |

33 % 느려지는 대신 fireflies 가 완전히 사라진다.

**③ 검정 배경.** 환경광 없이 directional light 만 쓰므로 배경이 정확히 검정이다. 3DGS 를 기본 배경으로 학습시키면 빈 영역이 정확히 일치하므로 alpha 채널이 불필요하다.

### 2.5 밀도별 렌더 비용

400×400, spp 256, RTX 4070 SUPER 기준.

| `--density-scale` | 프레임당 | 128 뷰 총 | 겉보기 |
|---|---|---|---|
| 4.0 (Disney 원본값) | 357 s | ~12.7 시간 | 두껍고 불투명한 적운 |
| 1.0 | 93 s | ~3.3 시간 | 중간 |
| **0.25 (현재)** | **8.2 s** | **~15 분** | 옅고 반투명 |

광학 두께가 커지면 산란 횟수가 폭증해 비용이 비선형으로 증가한다. 저밀도가 빠를 뿐 아니라 **논문에서 medium 이 이긴 장면이 "low-density" 였다는 점에서 연구 목적에도 부합한다.**

### 2.6 현재까지의 학습 결과

**3DGS** (`gs_orig`, 기본 설정, densification 켬, `densify_until_iter = 15000`)

| 지표 | 7k | 30k |
|---|---|---|
| test PSNR | **32.77** | 32.04 |
| train PSNR | 33.98 | 36.42 |
| Gaussian 수 | 43,232 | **50,639** |

초기 100,000 개에서 시작해 7k 시점에 43,232 개로 줄었다가 30k 에 50,639 개로 끝났다. **densification 보다 pruning 이 우세**했으며, 옅은 매질이라 불투명도 임계값 아래로 떨어진 Gaussian 이 다수 정리된 것으로 보인다. 학습 2 분, 렌더링 46 FPS.

(주의: 3DGS 는 iteration 별 개수를 로그로 남기지 않으므로 위 값은 저장된 체크포인트 두 개에서 센 것이다. densification 이 15k 까지 도는 만큼 **중간 최대치는 50,639 보다 클 수 있으며 기록되지 않았다.**)

**EVER** (`--eval --sh_degree 2`, 진행 중)

| 조건 | 7k test PSNR | primitive 수 |
|---|---|---|
| densification 무제한 | 31.93 | 2,290,283 (10.8k 시점) |
| `--spawn_cap 100000` | 측정 중 | 110,000 고정 |

무제한 조건에서 primitive 가 **229 만 개**까지 폭증했다. 3DGS 최종치의 45 배다. 상수밀도 타원체는 부드러운 falloff 가 없어 개수로 메우려는 경향이 있는 것으로 보이며, 그럼에도 test PSNR 은 3DGS 보다 낮았다.

`--spawn_cap 100000` 은 실제로는 **densification 을 완전히 끄는 효과**를 낸다. EVER 는 초기 110,000 개로 시작하는데 `slots = max(0, spawn_cap − 현재 개수)` 이므로 slots 가 0 이 되기 때문이다. 결과적으로 pruning 만 동작한다.

### 2.7 비교의 공정성에 대한 단서

현재 두 실행은 조건이 다르다.

| | 3DGS | EVER (capped) |
|---|---|---|
| densification | 켬 (15k 까지) | 사실상 꺼짐 |
| 최종 primitive | 50,639 | 110,000 이하 |

**파라미터화·적분 방식의 순수 비교를 하려면 Celarek 프로토콜대로 양쪽 모두 densification 을 끄고 개수를 맞춰야 한다.**

또한 현재 데이터셋의 GT 자체에 잔여 몬테카를로 노이즈가 있다. 3DGS 의 test PSNR 이 7k(32.77) → 30k(32.04) 로 **떨어지고** train 은 계속 오른 것이 그 증거다. 시점마다 무작위인 노이즈를 학습이 외우는 방향으로 과적합한 것이다. 방법 간 비교의 해상도를 높이려면 **spp 를 1024 정도로 올려 GT 를 더 깨끗하게 만들어야 한다** (ds 0.25 기준 프레임당 8 초 → 약 33 초, 128 뷰에 약 1 시간).

### 2.8 다음 단계

밀도 시리즈를 만들어 **"어느 광학 두께부터 medium 가정이 opacity 가정을 앞서는가"** 를 측정할 수 있다. 논문이 하지 않은 실험이다.

```bash
for d in 0.1 0.25 0.5 1.0 2.0; do
  python3 make_cloud_dataset.py \
      --out datasets/cloud_d${d} --density-scale $d \
      --n-train 100 --n-test 20 --n-val 8 --res 400 --spp 256
done
```
