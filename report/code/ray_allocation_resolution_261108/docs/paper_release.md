# 논문용 코드 기준 — paper_v1_20261102

> 이 문서는 `paper-v1-20261102` 태그의 로컬 런타임 기준이다. 현재 개발 트리의 독립 실행 경로는 [standalone.md](standalone.md)를 따른다. 과거 source lock을 현재 개발 트리에 그대로 적용하면 변경 감지로 중단되는 것이 정상이다.

최신 Counter·Bonsai·Bicycle 6개 풀학습의 알고리즘을 그대로 고정한다. 이 릴리스의 변경은 **실행 전 검사, 명시적 설정, 재현 검증**이다. 손실·표집·MCMC·Adam·렌더링 CUDA 연산을 변경하지 않는다.

## 고정한 설정

| 항목 | 값 |
|---|---|
| 학습 | COLMAP 초기화, 30,000 update, MV4 |
| loss | 0.8 L1 + 0.2 좌표 구조손실 + opacity/scale 정규화 각각 0.01 |
| L1 대조군 | 구조손실 계수만 0, 나머지는 동일 |
| 구조손실 | 실제 관측 좌표, sigma 6, 축별 ±18픽셀, boundary=all, 1−SSIM |
| 예산 | fine_tracks_v1, **전체 train 이미지**, train SfM incidence |
| 성장 | index 600~5000의 45개 이벤트에서 log 성장, 이후 고정 N |
| Counter | DS2, block8, cap 1,229,851 |
| Bonsai | DS2, block8, cap 1,209,886 |
| Bicycle | DS4, block6, cap 3,021,939 |
| MCMC | relocation 500~25000/100, noise 0~27500/1, 기존 처리 유지 |
| 렌더러 | GRT reference/referenceBwd, degree4, k-buffer16, min T=0.001 |
| 정밀도 | 기존 FP64 gradient 누적, TF32 비활성 |
| 평가 | native GRT, 전체 test 뷰, per-image PSNR/SSIM/VGG-LPIPS |

MCMC 반복·종료 경계의 정확한 의미는 고정된 backend 코드와 `resolved_reference.json`을 따른다. 마지막 ADD index5000은 **5001번째 update**다. cap은 최적 용량을 보장하는 값이 아니다.

이전 `configs/counter_ds2_mv4.json`과 `MethodConfig` 기본값은 과거 재현용이다. 논문 실험에서는 `tools/train_paper.py`만 사용한다. 기본값에 기대어 옛 valid-boundary 목적함수를 실행하지 않는다.

## 실행 전 검사

```bash
env GRT_GRAD_ACCUM_DOUBLE=1 \
  PYTHONPATH=/tmp/irregular-ssim-cupy13 \
  CUDA_HOME=/home/jinsu/miniconda3/envs/3dgrut \
  CUDA_PATH=/home/jinsu/miniconda3/envs/3dgrut \
  /home/jinsu/miniconda3/envs/3dgrut/bin/python tools/train_paper.py \
  --scene-name counter --arm structure --mode check
```

`check`가 기본 모드이며 학습을 시작하지 않는다. source lock의 Python/CUDA/config, 기존 연구 런타임, 로컬 compiled extension, 주요 패키지 버전과 환경변수를 비교한다. 하나라도 다르면 중단한다.

실제 학습 명령은 다음과 같다. 출력 경로는 아직 없는 디렉터리여야 한다.

```bash
env GRT_GRAD_ACCUM_DOUBLE=1 \
  PYTHONPATH=/tmp/irregular-ssim-cupy13 \
  CUDA_HOME=/home/jinsu/miniconda3/envs/3dgrut \
  CUDA_PATH=/home/jinsu/miniconda3/envs/3dgrut \
  /home/jinsu/miniconda3/envs/3dgrut/bin/python tools/train_paper.py \
  --scene-name counter --arm structure \
  --scene /home/jinsu/Documents/Github/3dgrut/data/counter \
  --output /path/to/new/run --mode train
```

`--mode smoke`는 610 update로 끝나며 재배치와 첫 ADD를 포함한다. 풀학습이나 최종 품질 평가가 아니다. `--arm l1`은 동일 0.8 L1 대조군이다. 재계산한 cap이 등록값과 다르면 중단한다.

학습 시작 전 실제로 import된 backend 모듈/바이너리와 최종 resolved config도 다시 확인한다. `paper_release.json`에 release ID·lock hash·검사 결과를 저장한다. 검사는 timed step 밖에서만 수행한다.

## 재현 증거

검증 결과: CPU 14개, 최종 all-centre CUDA 20개가 통과했다. Counter에서 기존/새 실행기를 각각 610 update 실행한 결과, **매 step의 손실·N, 최종 Gaussian 파라미터·Adam·난수 상태가 비트 단위로 일치**했다. 155,767개에서 첫 ADD 후 163,087개가 됐다. 이는 30k 재학습이나 새 속도 측정 결과가 아니다.

- [소스 및 환경 lock](../releases/paper_v1_20261102/source_lock.json)
- [실험 프로토콜](../releases/paper_v1_20261102/protocol.json)
- [최신 6개 실행과 소스 대조](../releases/paper_v1_20261102/compatibility.json)
- [검증 결과](../releases/paper_v1_20261102/validation/summary.json)
- [로컬/기존 원격 바이너리 구분](../releases/paper_v1_20261102/binary_builds.json)

기존 실행과 달라진 파일은 `tools/train_scene.py`의 검사 hook뿐이다. 기존 경로에는 hook을 적용하지 않으며, 논문용 경로도 모델 업데이트 전에 검사할 뿐이다. 새 release 모듈·실행기·검증 코드가 추가됐다.

검증 도구는 `python -m unittest discover -s tests -v`, `tools/validate_paper_cuda.py`,
`tools/compare_smoke.py`다. CUDA 검증은 GPU가 비어 있을 때 실행한다.
`compare_smoke.py`에는 `--original`, `--current`, `--output`을 지정하며,
읽는 checkpoint는 직접 생성하고 신뢰하는 파일만 사용한다.

## 변경 및 이식 원칙

1. 손실·표집·부모 선택·성장·정밀도·평가 변경은 새 release와 대조 실험으로 분리한다. 오류를 발견하면 버전과 영향받는 결과를 명시하고 수정한다.
2. 아직 등록하지 않은 scene/seed/ablation은 데이터 해상도·Ray 예산·난수 정책을 명시한 protocol revision으로 추가한다. 세 씬 preset을 다른 scene 이름으로 돌려 본표에 넣지 않는다.
3. 현재 lock은 **이 머신의 연구 런타임**을 포함한다. 다른 머신에서 컴파일한 `.so`의 byte hash는 다를 수 있다. 재빌드 출처·설정·검증을 기록한 machine manifest를 추가한 뒤 실행한다. 단순히 검사를 끄지 않는다.
4. 기존 RTX4090 결과와 로컬 RTX4070 SUPER 검증은 바이너리·하드웨어가 다르다. 로컬 재현 검증이 원격 타이밍 재현까지 증명하지는 않는다.
5. 공개 배포에는 renderer/MCMC 의존성 정리, 독립 설치, clean-machine 검증, 라이선스·개인 경로 정리가 더 필요하다. 내부 소스 묶음을 익명 공개 ZIP으로 그대로 사용하지 않는다.

이번 freeze에서 새 풀학습이나 인스턴스 실행은 시작하지 않는다. 코드·프로토콜 검사와 짧은 재현 검증을 먼저 끝낸다.
