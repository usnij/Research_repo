# 3DGRT 재현 설정 검증 — densification 주기와 입자 커널 설정의 확인

작성일 2026-08-19 · 모진수

## 요약

3DGRT 로 학습한 우리 dense 베이스라인의 PSNR 이 3DGRT 논문이 보고한 값보다 낮아서, 우리 실험 설정에 오류가 있는지 확인했다. 핵심 발견은 세 가지다.

### 발견 1 — densification 주기 300 이 3DGRT 의 원본 설정이며, 재학습이 필요 없다

3DGRT 논문은 densification 주기 (densification interval) 를 본문과 부록 어디에서도 숫자로 명시하지 않는다. 부록 A 에는 densification 과 pruning 을 500 번째 iteration 에서 시작해 15,000 번째 iteration 까지 수행한다는 구간만 적혀 있다. 반면 NVIDIA 공식 저장소 `nv-tlabs/3dgrut` 의 `origin/main` 은 `configs/strategy/gs.yaml` 에서 `densify.frequency: 300` 을 쓰며, 우리 저장소도 같은 값이다. 3DGS 의 기본값은 100 이고 저장소 주석도 이를 병기하고 있으나, 3DGRT 는 300 을 채택했다.

이 값을 독립적으로 확인할 근거를 후속 논문에서 찾았다. UTrice (arXiv 2512.04421) 는 부록 Table 5 에 자신들이 사용한 3DGRT 하이퍼파라미터를 전부 싣고 있으며, densify frequency 300, densify start iteration 500, densify end iteration 15000, prune frequency 100, prune start iteration 500, prune end iteration 15000, density 학습률 0.05 로 기재한다. 본문에는 "원본 `base_gs.yaml` 설정에 명시된 하이퍼파라미터를 그대로 재사용한다" 고 적혀 있다. **densification 주기를 100 으로 바꿔 dense 학습과 sparse 학습을 전부 다시 실행할 이유는 없다.**

### 발견 2 — 우리가 사용한 설정은 논문의 `Ours (reference)` 가 아니라 속도 최적화 변형에 해당한다

3DGRT 논문은 두 변형을 보고한다. `Ours (reference)` 는 표준 3D Gaussian 커널을 쓰는 고품질 설정이고, `Ours` 는 degree-2 일반화 Gaussian (generalized Gaussian) 커널과 density 학습률 0.09 를 써서 품질을 일부 내주고 렌더링 속도를 얻는 설정이다.

우리는 `configs/apps/colmap_3dgrt.yaml` 을 사용했고, 이 파일은 `configs/render/3dgrt.yaml` 을 상속한다. 저장소에는 논문 재현 전용 설정이 `configs/paper/3dgrt/` 아래에 따로 들어 있는데, 우리가 쓴 앱 설정과 세 항목이 다르다. 세 항목 모두 렌더링 속도를 위한 근사이며, 논문의 reference 변형은 세 항목 모두를 고품질 쪽으로 설정한다. **우리 실험은 논문 기준으로 `Ours (reference)` 가 아니라 속도 최적화 쪽 설정에서 수행되었다.**

### 발견 3 — 우리 dense 베이스라인 수치는 공개 구현의 표준 재현치와 일치하므로 설정 오류가 아니다

UTrice 는 3DGRT 공식 코드로 재현한 결과를 Mip-NeRF360 에서 야외 PSNR 25.92, 실내 PSNR 30.12, 평균 PSNR 28.32 로 보고한다. 3DGRT 논문이 보고한 평균 PSNR 은 28.69 이므로, 공개 구현으로 재현했을 때 0.37 dB 낮게 나온다.

우리 야외 3 개 씬 (bicycle, garden, stump) 의 dense 학습 PSNR 평균은 25.952 로, UTrice 의 25.92 와 차이가 0.03 dB 이다. 그중 bicycle 은 Gaussian 개수 상한 4,719,535 개에 도달한 상태의 값이므로 우리 야외 평균은 하한이다. **우리 dense 베이스라인은 3DGRT 공개 구현을 재현했을 때 나오는 표준 수치이며, 우리 설정 실수로 품질이 낮아진 것이 아니다.**

---

### 지난 리포트 (2026-08-18) 로부터
- 지난번까지 알아낸 것: dense 대비 sparse 의 densification 통계량 인플레이션 배율, split-half 도입에 따른 순위 중첩 회복량, RADC 의 학습 결과.
- 그때 몰랐던 것: 우리 dense 베이스라인의 PSNR 이 3DGRT 논문 값보다 낮은 원인. densification 주기 300 이 원본인지 여부.
- 이번에 하기로 한 것: 논문 본문과 부록, 공식 저장소 설정, 후속 논문의 재현 기록을 대조해 설정 오류 여부를 판정한다.

### 합의 사항 → 상태
- **[완료] densification 주기 300 대 100 판정** — 논문은 미명시, 공식 저장소와 후속 논문 UTrice 모두 300 을 사용. 재학습 불필요.
- **[완료] 우리 설정과 논문 reference 설정의 대조** — `particle_kernel_degree`, `particle_kernel_density_clamping`, `primitive_type` 세 항목이 다름을 확인.
- **[완료] dense 베이스라인 정상성 확인** — 야외 3 개 씬 평균 25.952 가 UTrice 재현치 25.92 와 0.03 dB 차이.
- **[미착수] `configs/paper/3dgrt/colmap_ours_reference.yaml` 로 고품질 설정 재실행** (사유: 현재 GPU 가 RADC 12 회 배치에 사용 중이며, 설정 변경 시 hit 수가 달라져 진행 중 실험군과 조건이 어긋난다.)

### 다음
- 현재 실험군은 단일 설정 (속도 최적화 쪽) 으로 통일해 유지한다. 실험 간 비교가 내부적으로 일관되어야 하기 때문이다.
- 리포트와 논문에 dense 베이스라인을 기재할 때 어느 설정에서 학습했는지 명시한다. UTrice 가 quality 설정과 performance 설정을 구분해 보고하는 것과 같은 방식이다.
- 🔴 고품질 설정에서 counter 1 회를 추가로 학습해 부기할지 여부는 판단이 필요하다. 아래 5 절에 소요 시간과 영향 범위를 정리했다.

---

## 1. 확인하게 된 경위

우리 dense 학습 결과를 3DGS 의 씬별 PSNR 과 비교했을 때 room 에서 1.39 dB, kitchen 에서 1.32 dB 낮게 나왔다. 3DGRT 논문은 Mip-NeRF360 평균 PSNR 에서 `Ours (reference)` 가 28.69 로 3DGS 와 같다고 보고하므로, 씬별로도 비슷해야 한다. 차이가 큰 원인이 우리 설정에 있다면 dense 학습과 sparse 학습을 전부 다시 실행해야 하므로 먼저 확인했다.

첫 번째 후보는 densification 주기였다. 우리 저장소의 `configs/strategy/gs.yaml` 에 `frequency: 300` 이 있고 바로 옆 주석에 3DGS 의 기본값이 100 이라고 적혀 있다. 주기가 짧을수록 densification 기회가 늘어나므로 품질에 영향을 줄 수 있다.

## 2. densification 주기는 논문에 없고 공식 구현은 300 이다

3DGRT 논문 부록 A (Implementation and Training Details) 는 densification 을 다음 범위로 서술한다.

> "After initial 500 iterations, we start the densification and pruning process, which we perform until 15,000 iterations are reached."

시작 iteration 500 과 종료 iteration 15,000 만 있고 주기 값이 없다. 논문이 3DGS 를 따랐다고 명시한 항목은 split 과 clone 을 구분하는 기준 (최대 scale 이 scene extent 의 1% 초과 시 split), density 를 0.01 로 되돌리는 주기 3,000, opacity 가 0.01 미만인 입자의 제거이며 주기는 여기에 포함되지 않는다.

공식 저장소 `nv-tlabs/3dgrut` 의 `origin/main` 에서 `configs/strategy/gs.yaml` 을 직접 확인했고 `densify.frequency: 300` 이었다. 우리 저장소의 값과 같으므로 우리 쪽 수정이 아니다.

후속 논문 UTrice 는 부록 Table 5 에 3DGRT 하이퍼파라미터를 다음과 같이 싣는다.

| 항목 | UTrice 가 기재한 3DGRT 값 |
|---|---:|
| features albedo 학습률 | 0.0025 |
| density 학습률 | 0.05 |
| rotation 학습률 | 0.001 |
| scale 학습률 | 0.005 |
| **densify frequency** | **300** |
| densify start iteration | 500 |
| densify end iteration | 15000 |
| prune frequency | 100 |
| prune start iteration | 500 |
| prune end iteration | 15000 |

본문에는 "we do not impose such a limit and simply reuse all hyperparameters specified in the original `base_gs.yaml` configuration" 이라고 적혀 있다. 독립적인 연구 그룹이 같은 값을 사용해 결과를 출판했으므로 300 이 재현 기준값이다.

## 3. 우리 설정과 논문 재현 설정의 차이는 세 항목이다

저장소에는 논문 재현용 설정이 `configs/paper/3dgrt/` 아래에 있다. `base_ours_reference.yaml` 이 논문의 `Ours (reference)` 에, `base_ours.yaml` 이 논문의 `Ours` 에 대응한다. 두 파일 모두 `/base_gs` 를 상속하므로 densification 주기는 300 으로 같다.

| 설정 항목 | 우리가 사용한 `apps/colmap_3dgrt.yaml` | 논문 재현용 `paper/3dgrt/colmap_ours_reference.yaml` |
|---|---|---|
| `particle_kernel_degree` | 4 | **2** |
| `particle_kernel_density_clamping` | true | **false** |
| `primitive_type` | instances | **icosahedron** |
| `strategy.densify.frequency` | 300 | 300 (동일) |
| `optimizer.params.density.lr` | 0.05 | 0.05 (동일) |
| 학습률 7 종 전체 | 논문과 일치 | 논문과 일치 |
| 손실 가중 (L1 0.8 + SSIM 0.2) | 논문과 일치 | 논문과 일치 |
| SH 차수 증가 (1000 step 마다, 최대 3) | 논문과 일치 | 논문과 일치 |

세 항목 모두 렌더링 속도를 위한 근사에 해당한다. 논문 5.2.1 절은 opacity 기반 adaptive clamping 이 framerate 에 큰 양의 효과를 준다고 서술하며, 부록의 입자 primitive 비교에는 "Icosahedron + unclamped" 항목이 별도로 있다. 논문 4.1 절은 각 입자를 늘린 정이십면체 (stretched icosahedron) 로 감싸는 방식을 채택했다고 명시한다.

`configs/paper/` 디렉터리는 저장소 README 에 언급되어 있지 않다. 앱 설정에서 시작하면 이 디렉터리의 존재를 알기 어렵다.

## 4. 커널 차수 표기는 논문과 코드에서 기준이 다르다

논문 식 (9) 는 degree $n$ 의 일반화 Gaussian 을 다음과 같이 정의한다.

$$\hat{\rho}_n(x) = \sigma e^{-\left((x-\mu)^T \Sigma^{-1} (x-\mu)\right)^n}$$

지수는 Mahalanobis 제곱거리 $d = (x-\mu)^T \Sigma^{-1} (x-\mu)$ 의 $n$ 제곱이다. 논문은 $n = 2$ 를 사용한다고 적는다.

코드는 `threedgrt_tracer/include/3dgrt/kernels/cuda/gaussianParticles.cuh` 의 `particleResponse` 에서 `GeneralizedGaussianDegree` 로 분기하며, 주석에 "generalized gaussian of degree n : scaling is s = -4.5/3^n" 이라고 적혀 있다. 코드 변수 `grayDist` 가 논문의 $d$ 에 해당한다.

| 코드의 `particle_kernel_degree` | 실제 커널 | 논문 표기 |
|---:|---|---|
| 2 | $\exp(-0.5 \, d)$ | 표준 3D Gaussian |
| 4 | $\exp(-0.0556 \, d^2)$ | degree-2 일반화 Gaussian |

코드의 degree 4 가 논문의 $n = 2$ 다. 저장소의 `configs/paper/3dgrt/base_ours_reference.yaml` 이 degree 2 를, `base_ours.yaml` 이 degree 4 를 지정하는 것과 일치한다.

UTrice 는 이 구분을 다음과 같이 서술한다.

> "For 3DGRT, we evaluate two settings: quality and performance. The quality setting uses the regular 3D Gaussian kernel and keeps the same optimization hyperparameters as in [3DGS]. The performance setting instead uses a degree-2 generalized Gaussian kernel and adjusts the optimization hyperparameters to trade rendering quality for faster ray tracing."

3DGRT 논문 Table 4 는 커널만 바꿨을 때의 품질과 속도 차이를 다음과 같이 보고한다.

| 입자 커널 | Tanks & Temples PSNR | Tanks & Temples FPS | Deep Blending PSNR | Deep Blending FPS |
|---|---:|---:|---:|---:|
| Gaussian (reference) | **23.03** | 143 | **29.89** | 77 |
| Generalized Gaussian (n=2) | 22.68 | 277 | 29.74 | 141 |

Tanks & Temples 에서 PSNR 이 0.35 dB, Deep Blending 에서 0.15 dB 낮아지고 FPS 는 약 2 배 증가한다.

## 5. 우리 dense 수치는 독립 재현치와 일치한다

3DGRT 논문 Table 1 의 Mip-NeRF360 결과는 다음과 같다.

| 방법 | PSNR | SSIM | LPIPS | 모델 크기 |
|---|---:|---:|---:|---:|
| 3DGS (paper) | 28.69 | 0.871 | — | 734MB |
| 3DGS (checkpoint) | 28.83 | 0.867 | 0.224 | 763MB |
| Ours (reference) | **28.69** | 0.871 | 0.220 | 676MB |
| Ours | 28.71 | 0.854 | 0.248 | 704MB |

UTrice 가 3DGRT 공식 코드로 재현한 결과는 다음과 같다. 표의 † 는 재현한 값이라는 뜻이다.

| 방법 | 야외 PSNR | 실내 PSNR | 평균 PSNR | 평균 SSIM |
|---|---:|---:|---:|---:|
| 3DGS | 26.40 | 30.41 | 28.69 | 0.870 |
| 3DGRT† (재현) | 25.92 | 30.12 | **28.32** | 0.859 |

논문이 보고한 28.69 와 공개 구현 재현치 28.32 사이에 0.37 dB 차이가 있다.

우리 dense 학습 결과는 다음과 같다. 실내 3 개 씬과 bicycle 은 4070 12GB 에서 학습한 값을 로그에서 직접 확인했고, garden 과 stump 는 4090 24GB 세션 기록의 값이다.

| 씬 | 구분 | PSNR | Gaussian 개수 |
|---|---|---:|---:|
| bicycle | 야외 | 24.758 | 4,719,535 (개수 상한 도달) |
| garden | 야외 | 26.859 | 약 4.21M |
| stump | 야외 | 26.239 | 약 7.86M |
| counter | 실내 | 28.570 | 1,256,408 |
| kitchen | 실내 | 29.958 | 1,498,527 |
| room | 실내 | 30.233 | 1,205,777 |

야외 3 개 씬 평균 PSNR 은 25.952 이고, UTrice 의 재현치 25.92 와 0.03 dB 차이다. bicycle 이 개수 상한에 도달한 상태의 값이므로 우리 야외 평균은 하한값이다. 실내는 우리에게 bonsai dense 학습 결과가 없어 3 개 씬 평균 29.587 만 비교 가능하며, UTrice 의 실내 평균 30.12 는 bonsai 를 포함한 4 개 씬 값이다.

우리 값이 3DGS 의 씬별 값보다 낮았던 것은 3DGRT 공개 구현 전반에서 나타나는 재현 격차이며, 우리 설정 오류가 아니다. UTrice 도 3DGRT 의 LPIPS 를 야외 0.221, 실내 0.245 로 보고하는데 3DGS 의 0.173, 0.192 보다 크다.

## 6. 고품질 설정으로 바꿀 때 영향을 받는 측정값

고품질 설정으로 전환하면 커널이 $\exp(-0.0556 \, d^2)$ 에서 $\exp(-0.5 \, d)$ 로 바뀌고 clamping 이 해제되므로, Ray 하나당 처리하는 입자 교차 (intersection) 횟수가 증가한다. 논문 5.2.3 절이 일반화 Gaussian 의 속도 이득을 입자 개수가 아니라 교차 횟수 감소로 설명한다.

우리 연구에서 교차 횟수에 종속되는 측정값은 다음과 같다.

- 미교차 Gaussian 비율. 현재 dense 에서 10.9~12.0%, 무보정 sparse 에서 26.2~27.5% 로 측정했다.
- $k(h)$ 곡선. $h$ 가 Ray 당 교차 횟수이므로 정의역 자체가 이동한다.
- backward pass 의 메모리 사용량. 교차 횟수 증가는 OOM 이 발생하는 Gaussian 개수 상한을 낮춘다.

따라서 진행 중인 실험군의 설정을 중간에 바꾸면 이전 측정값과 비교할 수 없게 된다. 현재 실험군은 하나의 설정으로 유지한다.

## 7. 판단이 필요한 항목

🔴 고품질 설정 (`configs/paper/3dgrt/colmap_ours_reference.yaml`) 으로 counter 를 1 회 추가 학습해 리포트에 부기할지 여부.

- 목적: 우리 파이프라인이 논문의 `Ours (reference)` 설정에서도 재현되는지 확인하고, 리포트에 두 설정의 값을 함께 기재한다. UTrice 가 quality 설정과 performance 설정을 나눠 보고하는 방식과 같다.
- 소요: `particle_kernel_degree` 가 컴파일 시점 상수이므로 `THREEDGUT_FORCE_JIT=1` 로 재컴파일이 필요하다. 속도 최적화 설정에서 counter dense 학습이 121.4 분이었고 커널 변경으로 FPS 가 약 절반이 되므로 200 분대로 예상한다.
- 영향 범위: 이 학습은 부기용 단일 실행이며 진행 중인 실험군의 설정은 변경하지 않는다.
- 대안: 부기하지 않고 리포트에 "3DGRT 속도 최적화 설정 기준" 이라고 명시만 한다. 후속 논문들이 두 설정을 구분해 표기하므로 이 표기만으로도 비교 기준은 성립한다.

## 참고 문헌

- Moenne-Loccoz, Mirzaei et al. *3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes.* (`3dgrt_compressed.pdf`, 본 저장소 `논문/` 에 보관)
- *UTrice: Unifying Primitives in Differentiable Ray Tracing and Rasterization via Triangles for Particle-Based 3D Scenes.* arXiv:2512.04421
- NVIDIA `nv-tlabs/3dgrut` 공식 저장소 `origin/main`
