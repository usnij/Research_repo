# 수식과 학습 정책을 유지하는 성능 점검

## 2026-10-30 GPU 검증 결과

새 RTX 4090 24GB 인스턴스에서 Counter DS2·MV4·7,000스텝 체크포인트로 검사했다.
Gaussian 1,131,968개, 스텝당 101,400 ray, L1 가중치 1·좌표 SSIM 가중치 0.2다.
각 실행은 동일한 파라미터·Adam 상태에서 시작하며, 일반 스텝과 재배치 스텝을 각각
동일한 10개 뷰 조합으로 측정했다. 입력부터 Adam·MCMC·BVH까지 포함하며 복원·빌드·I/O는 제외한다.
아래 수치는 일반 스텝 중앙값 99%와 재배치 스텝 중앙값 1%를 합친 대표 시간이다.

| 변경 | 대표 스텝 시간 | 판정 |
|---|---:|---|
| 기존 코드, 3회 실행 평균 | 50.75 ms | 대조군 |
| 사용하지 않는 진단 통계 쓰기 생략 | 50.37 ms | 작은 이득 |
| 선택한 uint8 픽셀만 float 변환·격자 캐시 | 50.08 ms | 작은 이득, 전체 GPU 이미지 캐시 불필요 |
| 위 입력 처리 + 전체 uint8 이미지 GPU 캐시 | 49.88 ms | 약 0.95 GiB 추가 사용 |
| SSIM의 정확히 0인 가중치 항 생략 | 50.69 ms | 속도 이득 확인 못 함 |
| 진단 통계 생략 + GPU 입력 캐시, 원래 SSIM 커널 유지 | **49.62 ms** | **약 2.2% 시간 감소** |

전체 후보를 함께 적용한 별도 2회 실행도 49.47·49.91 ms였다. 이 변경들로 큰 폭의 가속을
기대할 근거는 없다. 0 가중치 항 생략은 손실만 측정해도 4.911→4.916 ms로 이득이 없었다.

실제 비교한 좌표·입력 RGB·ray·렌더 RGB·loss·6종 Gaussian gradient·Adam 업데이트량·
Adam 모멘트·재배치와 noise 적용 후 파라미터는 모두 **비트 단위로 일치**했다.
진단 통계 비활성화도 확인했다. 기존에는 574,679개 Gaussian의 통계가 0이 아니었고,
비활성화하면 세 통계가 모두 0이었다. 이 진단 출력 자체는 의도적으로 달라진다.
CUDA 손실 단위 검사 25개에서도 기존 커널과 loss·RGB gradient 차이가 0이었다.

이는 고정된 상태의 수치·속도 검증이다. 3만 스텝을 다시 학습한 결과나 최종 PSNR·SSIM·LPIPS의
일치를 측정한 결과가 아니다. 기본 학습 코드에 최적화 패치를 자동 적용하지 않았다.
원자료: `research_resume_260918/speed_equivalence_261030/final_comparison.json`,
`results/*/result.json`, `results/*/parity.json`, `loss_cuda.json`.

## 우선순위

| 순위 | 코드에서 확인한 비용 | 변경 | 상태 |
|---|---|---|---|
| 1 | GRT accepted hit마다 사용하지 않는 진단 통계 3개 atomicAdd | reference 순전파에서 통계 포인터만 비활성화 | 별도 빌드 검증 완료; 이 상태에서는 약 0.7% 이득 |
| 2 | 전체 RGB의 매 스텝 전송과 float 변환 | uint8 GPU 캐시, 선택 후 변환 | CPU·CUDA 일치 확인; 약 1.7% 이득 |
| 3 | 순전파·역전파의 rayToWorld.cpu() | 이미 world-space인 입력의 identity를 명시하는 API/호스트 캐시 | 경로 확인, 미구현 |
| 4 | 매 뷰 층화 격자·면적 가중치 재생성 | 고정 격자 캐시, 난수 두 번은 동일하게 소비 | 입력 처리에 포함해 CUDA 일치 확인 |
| 5 | SSIM 49개 후보의 0 가중치에도 RGB/통계 로드 | 정확히 0인 가중치만 건너뜀 | 수치 일치, 측정 가능한 속도 이득 없음 |
| 6 | 뷰별 SSIM 커널 실행·PyTorch/CuPy 변환 반복 | 동일 크기 4뷰를 batch kernel로 실행 | 제안; reduction 순서의 수치 차이 검증 필요 |
| 7 | 매 스텝 비활성 probe/candidate/area 버퍼 재설정, 빈 텐서 생성 | backend 구성 시 한 번 설정 | 제안; 실행 중 기본 코드 유지 |
| 8 | scalar 로그 변환과 반복 synchronize | 로그 묶음 전송, CUDA event와 wall time 분리 | 제안; 기존 시간 계측 유지 |

## 1. 남아 있는 진단 통계

대상: 기존 `area_tracer512/src/optixTracer.cpp`와 `src/kernels/cuda/referenceOptix.cu`.
순전파가 Gaussian별 visibility, Σ(Tα), Σ(Tα)² 버퍼를 할당하고 적중마다 atomicAdd를 수행한다.
현재 학습 실행기는 renderer output 중 pred_rgb만 사용한다. MCMC는 opacity를 사용하고,
현재 `pre_backward`와 `post_backward`는 빈 hook이다. 통계로 gradient를 조정하지 않는다.
backward에서는 원래부터 이 세 포인터가 null이다.

`patches/disable_unused_diagnostics.patch`는 순전파도 환경변수
`SPARSE_GRT_DISABLE_DIAGNOSTICS=1`일 때만 포인터를 null로 설정한다.
기본 동작·버퍼 크기·반환값 형태는 유지한다. 통계 반환값은 0이 되므로 이 통계를 쓰는 진단에는
적용하면 안 된다. reference 외 pipeline에서는 활성화를 거부한다.
OptiX RGB/gradient kernel의 수식과 순회, 수락 조건, precision flag는 바꾸지 않는다.

현재 원본에는 적용하지 않았다. 별도 빌드 디렉터리와 별도 확장 이름을 사용해야 한다.
컴파일 산출물을 실행 중인 학습과 공유하지 않는다.

## 2. 입력 처리

Counter DS2는 뷰당 1,617,204픽셀, 현재 표집은 뷰당 25,350픽셀이다.
기존 경로는 디코딩·호스트 pinned 캐시를 이미 사용한다. 카메라 ray grid도 이미 캐시되어 있다.
따라서 JPEG를 매번 디코딩한다거나 ray grid를 매 스텝 새로 계산한다고 설명하면 틀리다.

남아 있는 비용은 다음과 같다.

- 매 스텝 4뷰의 uint8 RGB 19,406,448 bytes를 GPU로 전송한다.
- 6,468,816픽셀을 float으로 바꾼 뒤 101,400픽셀만 선택한다.
- 210개 학습뷰의 uint8 RGB 전체는 1,018,838,520 bytes, 약 972 MiB다.

선택 후 변환은 RGB 변환 픽셀 수를 약 63.8배 줄인다. 이것은 **해당 변환량의 감소**이며
전체 학습의 가속 배율이 아니다. GPU 캐시가 들어가는 경우 반복 이미지 전송도 제거한다.
캐시가 부족하면 pinned-host 전송은 유지하되 변환만 줄인다. 여러 scene에 대해 무조건
모든 이미지를 GPU로 올리지 않도록 상한을 둔다.

캐시 대상은 이미지·카메라·고정 층화 격자다. 실제 표집 좌표가 달라지면 이웃 가중치와
그 가중치로 계산하는 GT 평균·분산도 달라지므로, 이전 스텝의 값을 그대로 재사용하지 않는다.

## 3. 카메라 변환의 CPU 복사

현재 MV4 입력은 이미 world-space로 변환되어 있어 renderer에 identity를 넘긴다.
그런데 C++ trace/trace_bwd가 `rayToWorld.cpu()`로 호스트 파라미터를 만든다.
작은 행렬이어도 GPU→CPU 동기화가 생긴다. 일반 카메라 경로까지 identity로 간주하면 오동작하므로,
world-space 입력임을 명시하는 별도 API가 필요하다. 아직 적용하지 않았다.

## 4. 손실 커널만 바꿔서 얻는 이득의 한계

기존 측정에서 loss-only 계산은 9.79→4.13 ms였지만, frozen 모델의 입력+GRT+손실+backward+BVH는
46.90→45.18 ms였다. 서로 다른 계측 범위이므로 차이를 직접 빼서 병목 비중을 역산하지 않는다.
손실 커널의 배속을 전체 학습 배속으로 보고하지 않는다.

현재 σ=6/block=8에서 49개 후보 중 양의 가중치는 평균 약 21.4개다.
0인 항을 건너뛰는 커널은 양의 항의 합산 순서를 유지한다. 다만 GPU 분기 비용 때문에
빨라질지는 아직 모른다. finite RGB가 전제이며 NaN 검출을 대신하는 기능이 아니다.

## 검증 기준

1. 입력 RGB·좌표·ray direction·view 순서·generator 상태가 같은지 확인한다.
2. 같은 checkpoint와 RGB cotangent에서 위치·회전·scale·opacity·SH gradient를 비교한다.
3. 동일 Adam 상태에서 일반 스텝 및 relocation/ADD 스텝 이후의 파라미터·모멘트를 비교한다.
4. 같은 GPU에서 두 경로의 실행 순서를 번갈아 측정한다. warm-up·빌드·캐시 준비는 별도 기록한다.
5. renderer 단계, loss 단계, 입력 처리, Adam/MCMC/BVH, complete-step wall time을 구분한다.
6. 수치 검증을 통과한 조합만 기존 품질 대조군과 같은 설정으로 전체 학습한다.

같은 최적화는 L1-only 대조군과 구조손실 실험 양쪽에 적용한다. 입력 캐시나 진단 통계 제거의
속도 이득을 구조손실 자체의 이득으로 합쳐서 보고하지 않는다.

CUDA 검증은 시간 측정 중인 기존 학습이 끝난 뒤 수행한다. 로컬 GPU에서 자동 실행하지 않는다.
BVH 갱신 주기, FP64 gradient 누적, MCMC 주기, ray budget, 수락 임계값, SSIM 주기는
이번의 ‘같은 계산 최적화’ 범위에서 바꾸지 않는다.
