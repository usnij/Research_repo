# Sparse Ray Sampling for Gaussian Splatting — 선행연구 재조사

## 요약

**지난 미팅 (2026-07-28)** — 키워드 3줄
- GS ray-tracing denoising 선행연구 1차 조사 완료
- 이후 신규 문헌 반영 필요
- 우리 축(primary pixel ray 생략 + GS 정보 기반 복원)의 선행연구 존재 여부 확인

**합의 사항 → 상태**
- [완료] 2024–2026 GS 계열 재조사, 줄이는 대상 기준 5개 축으로 분류
- [완료] 직접 선행연구 유무 판정

**이번 결과 / 막힌 것 / 다음**
- 우리 축의 직접 선행연구는 확인되지 않음
- 막힌 것: GRay(2026-06)가 sparse 샘플링 없이 3DGRT를 4배 가속 — 기여점의 전제를 위협 🔴
- UpscaleGS(ICCV 2025)는 같은 계열이며 ray tracing 적용을 향후 과제로 명시
- 다음: GRay와 우리 구조의 직교성 확인, 결합 가능 여부 판단

---

> 작성일: 2026-08-05
> 범위: 2024–2026, GS 계열 중심. [260728] 조사 이후 신규 문헌 반영
> 결론 요약: **우리 축(primary pixel ray 생략 + GS 정보 기반 학습 복원)의 직접 선행연구는 여전히 확인되지 않음.** 다만 **GRay(2026-06)가 sparse 샘플링 없이 3DGRT를 4배 가속**하여 기여점의 전제를 위협한다.

## 1. 판정 기준 — 무엇을 줄이는가로 축을 나눈다

"빠르게 한다"는 논문이 많으나 **줄이는 대상이 다르면 선행연구가 아니다.** 다섯 축으로 분류했다.

| 축 | 줄이는 것 | 우리와의 관계 |
|---|---|---|
| **A** | **화면 primary pixel ray 자체** | **우리 축** |
| B | ray 하나가 만나는 Gaussian hit 수 | 우리 §48(hit threshold)과 같은 축 |
| C | ray/hit 하나당 처리 비용 (BVH, 정렬, 하드웨어) | 직교. 결합 가능 |
| D | 렌더 해상도 (저해상도 후 확대) | 우리 제어 변량 $B$의 대안 |
| E | 프리미티브 수 (프루닝, LOD) | 우리 §44에서 배제 |

## 2. 축 A — 우리 축. 직접 선행연구 없음

### 2.1 가장 가까운 것은 여전히 GS 밖에 있다

**SSR: High-Quality Real-Time Rendering Using Subpixel Sampling Reconstruction** (AAAI 2024)
2×2 블록당 1픽셀만 ray trace(1/4-spp)하고 나머지는 full-resolution G-buffer + 이전 프레임으로 복원. **샘플링 구조가 우리와 거의 동일하나 mesh 기반 하이브리드 렌더러이며 GS가 아니다.** [260728] 조사의 판정을 재확인했다.

### 2.2 GS에서 A축을 시도한 논문은 이번에도 찾지 못했다

검색한 표현: sparse primary ray, subsampled pixels, fraction of pixels, 1/N spp + Gaussian, ray budget, adaptive pixel sampling, foveated + ray tracing GS 등.

가장 근접해 보였던 후보 둘은 모두 다른 축이었다.

- **Speedy-Splat: Fast 3DGS with Sparse Pixels and Sparse Primitives** (CVPR 2025)
  제목의 "sparse pixels"는 **출력 픽셀을 건너뛰는 것이 아니라 Gaussian 하나가 덮는 픽셀 범위를 정밀하게 좁히는 것**(SnugBox/AccuTile 계열)이다. 여기에 프루닝을 더해 6.71× 가속. **축 C+E이다.**
- **VR-Splatting: Foveated Radiance Field Rendering via 3DGS and Neural Points** (ACM PACMCGIT / I3D 2025)
  주변시를 저품질로 렌더. **축 D의 시선 의존 변형**이며 ray 생략이 아니다.

## 3. 축 B — 우리 §48과 같은 축. 선행연구 존재

우리는 `particle_kernel_min_response`를 올려 기여가 약한 hit을 버렸다(런타임 파라미터, 재학습 불필요). 같은 목표를 다른 방법으로 달성한 연구가 둘 있다.

**Stochastic Ray Tracing for the Reconstruction of 3D Gaussian Splatting** (arXiv 2603.23637, 2026-03)
ray당 Gaussian 부분집합만 확률적으로 평가하는 **불편 몬테카를로 추정량**. 정렬을 제거하고 relightable 장면까지 다룬다. 우리 hit threshold의 확률적·불편 버전에 해당한다. **우리 방식은 편향되지만 결정론적이고, 이들은 불편하지만 분산이 있다.** 인용하고 대비해야 할 가장 가까운 축 B 연구다.

**Speeding Up the Learning of 3D Gaussians with Much Shorter Gaussian Lists** (arXiv 2603.09277)
Gaussian 크기 축소 + alpha blending에 엔트로피 제약을 걸어 **학습 중에** 리스트를 짧게 만든다. 여기에 progressive resolution scheduler를 결합. 우리처럼 추론 시점 파라미터가 아니라 **학습 정규화**다.

**RaySplats** — ray당 누적 Gaussian 수에 상한을 두어 고정 크기 로컬 메모리에 담는다. 축 B의 하드 캡 버전.

## 4. 축 C — 활발하고, 우리 기여를 가장 위협한다

**GRay: Ray Tracing 3D Gaussians Near the Speed of Splats** (arXiv 2606.30869, 2026-06)
실제로 교차하는 Gaussian만 평가하여 프리미티브 수에 대해 선형이 아닌 **로그 스케일링**을 달성. 보고 수치가 결정적이다.

| | GRay |
|---|---|
| vs **3DGRT** | 렌더 **약 4배**, 최적화 **약 10배** 빠름, 품질 유사 |
| vs 3DGS | 렌더 속도 대등, 품질은 "다소 낮음" |

**이것이 본 연구의 전제를 직접 위협한다.** 우리는 3DGRT를 4.46배 가속하되 0.111 dB를 잃는데, GRay는 **품질 손실 없이 4배**를 얻는다. 3DGRT를 baseline으로 "느린 레이트레이서를 빠르게 했다"는 프레이밍은 GRay 이후 성립하기 어렵다.

**GRTX: Efficient Ray Tracing for 3D Gaussian-Based Rendering** (arXiv 2601.20429, 2026-01)
비등방 Gaussian을 ray-space 변환으로 단위구로 다루어 BVH 크기·순회 부담을 줄이고, ray tracing unit에 **traversal checkpointing 하드웨어**를 제안. 아키텍처 연구이며 소프트웨어만으로는 재현 불가.

**SHARP-GS** (SIGGRAPH 2026) — 8K 래스터화 파이프라인(EWA 투영, 비닝, 전역 정렬) 최적화. 래스터라이저 축.

## 5. 축 D — 정식 게재됨. 우리 제어 변량의 대안

**Mobile3DGS³: Accelerate Mobile 3DGS Rendering via Gradient-Aware Super-Sampling and Frame Interpolation** (SIGGRAPH 2026 Conference Papers, doi 10.1145/3799902.3811198)

[260728] 조사 시점에는 arXiv v1(2605.11489)이라 "정식 게재로 간주하면 안 된다"고 유보했으나, **SIGGRAPH 2026에 게재되었다.** 구조는 저해상도 3DGS 출력 + analytical image gradient를 GRU 기반 refinement로 초해상도화하고, 경량 U-Net으로 프레임 보간. 4K 96 FPS를 보고한다.

**GS에서 "저해상도로 렌더하고 학습 복원한다"는 축은 이제 정식 선행연구가 있다.** 우리 4.5절의 대조군(저해상도 base 단독)이 곧 이 계열이며, 우리가 그것을 이겨야 하는 이유가 더 분명해졌다.

**UpscaleGS: Lightweight Gradient-Aware Upscaling of 3DGS Images** (ICCV 2025) — 동일 계열. 결론에서 ray tracing 적용을 향후 과제로 명시.

## 6. 축 E — 우리가 §44에서 배제한 축

LOD·프루닝 계열(A LoD of Gaussians, Octree-GS, LightGaussian 등)은 활발하나, 우리 측정에서 같은 비용의 저해상도 렌더 대비 6.9 dB 열세였다.

## 7. 제어 변량 명명의 선행연구 — 확인됨

[260805] 리포트 §51 항목 6에서 미확인으로 남겨두었던 인용을 확정한다.

**Rousselle, Jarosz, Novák. "Image-space control variates for rendering." ACM TOG 2016 (SIGGRAPH Asia)**
설명이 우리 구조와 거의 일치한다 — 제어 변량이 **저주파 영역을 재구성**하고, 그것이 놓친 **고주파 디테일은 잔차의 몬테카를로 적분으로 회복**한다.

후속으로 두 편이 더 있다.

- **Imperfect Image-Space Control Variates for Monte Carlo Rendering** (ACM TOG, doi 10.1145/3763335) — 제어 변량이 부정확할 때를 다룬다. **$B$가 근사인 우리 상황과 정확히 일치**하므로 반드시 확인해야 한다.
- **Spatio-Temporal Control Variates with ReSTIR for Real-Time Rendering** (SIGGRAPH 2026, doi 10.1145/3799902.3811113) — 실시간 + 시간축 제어 변량. 우리가 §5.4에서 남긴 "base의 시간적 재사용"과 직결된다.

**용어는 정당하다. 다만 이 계보는 전부 경로추적 대상이며 GS에 적용된 사례는 확인되지 않았다.**

## 8. 종합 판정

### 8.1 공백은 유지된다

다음 넷을 동시에 만족하는 연구는 이번 조사에서도 없었다.

1. 고정 조명 GS novel-view synthesis
2. ray tracer(3DGRT/EVER 계열)에서 Gaussian을 3D ray로 평가
3. **primary pixel ray 자체를 생략**
4. GS 유래 정보로 미추적 픽셀을 학습 복원

### 8.2 그러나 기여점의 근거가 약해졌다

| 위협 | 내용 | 대응 |
|---|---|---|
| **GRay** | 품질 손실 없이 3DGRT 4배 | baseline을 GRay로 바꾸거나, GRay 위에 결합해 추가 이득을 보여야 함 |
| **Mobile3DGS³ 게재** | 저해상도+초해상도가 정식 선행연구 | 우리 대조군이 이것이며, 이겨야 함 (현재 4.3절에서 이기고 있음) |
| **Stochastic Ray Tracing 3DGS** | 축 B의 불편 추정 버전 | 우리 hit threshold와 대비 실험 필요 |

### 8.3 권고

1. **GRay를 최우선으로 확인한다.** 코드 공개 여부, 3DGRT 대비 4배의 조건, 우리 구조와의 결합 가능성. 결합 가능하면 오히려 기회다 — GRay 4배 × 우리 4.46배가 곱해지는지 확인할 가치가 있다.
2. **기여 프레이밍을 "레이트레이서 가속"에서 "sparse ray + 학습 복원의 품질-속도 곡선"으로 옮긴다.** 축 C 연구들이 이미 절대 속도를 크게 올렸으므로, 절대 속도 경쟁보다 **같은 비용에서 우리가 더 나은 품질을 낸다**(4.3절)는 형태가 방어 가능하다.
3. **Imperfect Image-Space Control Variates를 확인한다.** 우리 §48(제어 변량을 의도적으로 거칠게 만들면 기여가 커진다)이 그 논문에서 이미 이론적으로 다뤄졌을 가능성이 있다.

## 9. 조사의 한계

- 본 조사는 웹 검색과 초록 확인에 근거하며 전문을 읽지 않았다. 특히 GRay와 Stochastic Ray Tracing은 전문 확인이 필요하다.
- Speedy-Splat의 "sparse pixels" 해석은 초록과 알려진 기법명(SnugBox/AccuTile)에 근거한 추정이다.
- 2026년 7–8월 공개분은 색인이 불완전할 수 있다.

## 부록. 확인한 문헌

| 문헌 | 출처 | 축 | 우리와의 관계 |
|---|---|---|---|
| SSR | AAAI 2024 | A | 구조 동일, GS 아님 |
| Speedy-Splat | CVPR 2025 | C+E | "sparse pixels"는 footprint 축소 |
| VR-Splatting | PACMCGIT 2025 | D | foveated |
| UpscaleGS | ICCV 2025 | D | 저해상도+복원 |
| **Mobile3DGS³** | **SIGGRAPH 2026** | D | **정식 게재됨** |
| SHARP-GS | SIGGRAPH 2026 | C | 래스터화 8K |
| GRTX | arXiv 2601.20429 | C | BVH+하드웨어 |
| Stochastic Ray Tracing 3DGS | arXiv 2603.23637 | B | 불편 hit 표본 |
| Shorter Gaussian Lists | arXiv 2603.09277 | B | 학습 정규화 |
| **GRay** | **arXiv 2606.30869** | **C** | **3DGRT 4배, 최대 위협** |
| Image-space Control Variates | ACM TOG 2016 | — | 용어 근거 |
| Imperfect Image-space CV | ACM TOG | — | $B$가 근사인 경우 |
| Spatio-Temporal CV + ReSTIR | SIGGRAPH 2026 | — | 시간축 제어 변량 |
