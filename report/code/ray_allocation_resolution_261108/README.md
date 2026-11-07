# Sparse GRT

희소 멀티뷰 Gaussian 학습과 관측 좌표 기반 SSIM을 정리하는 독립 저장소입니다.
기존 실험을 덮어쓰지 않으며, 아직 공개 배포판은 아닙니다.

**현재 독립 실행 경로는 `python -m sparse_grt.training`입니다.**
설치와 검증 절차는 [독립 설치 문서](docs/standalone.md)에 있습니다.
GRT·MCMC 소스를 포함하고, OptiX 헤더는 공식 고정 버전에서 받습니다.
기존 연구 폴더와 이미 빌드된 `.so`를 가져올 필요가 없습니다.

`paper-v1-20261102` 태그는 기존 로컬 런타임 기준의 비교 대상으로 보존했습니다.
현재 `0.2.0.dev0`은 새 RTX 4090 인스턴스에서 설치·610스텝 학습·체크포인트 복원과
전체 테스트뷰 평가를 확인한 독립 실행 후보입니다. 새 코드의 30k 풀학습과 최종 품질
재현은 별도 검증입니다. 아래 과거 재현 명령은 새 설치 경로와 구분합니다.
로컬 결과: [검증 요약](releases/standalone_candidate_20261102/summary.json).
새 인스턴스 결과: [독립 설치·학습 검증](releases/standalone_remote_20261102/summary.json).

새 환경에서는 [설치 스크립트](tools/setup_reference_env.sh)와
`python -m sparse_grt.training`을 사용합니다. 현재 논문 설정은
`releases/paper_v1_20261102/configs/*_structure.json`의 **0.8 L1 + 0.2 구조손실,
boundary=all**입니다. 과거 재현용 `configs/counter_ds2_mv4.json`과 혼용하지 않습니다.

실사 13씬의 고정 설정, 8대 배정, 자동 배포·백업·결과 취합은
[벤치마크 실행 문서](docs/benchmark.md)에 정리했습니다.

## 구현 상태

| 구성 | 상태 |
|---|---|
| 좌표 SSIM 기본 Python/CUDA 구현 | 검증된 실험 코드를 바이트 단위로 그대로 분리 |
| 표집·멀티뷰 순서·손실 조합·업데이트 순서 | 독립 모듈로 분리, CPU 검증 완료 |
| uint8 GPU 이미지 캐시 + 선택 후 RGB 변환 | 입력·Gaussian·Adam·MCMC 일치 검사 통과 |
| SSIM의 0 가중치 계산 생략 | 별도 실험 커널로 구현, 기본 경로 미적용, GPU 검증 대기 |
| 사용하지 않는 GRT 진단 통계 기록 중단 | 별도 패치, 기본값 유지, 빌드·GPU 검증 대기 |
| 경계를 포함한 좌표 SSIM | 별도 구현, CUDA 미분 검사와 Counter 2,000스텝 비교 두 번 통과 |
| 개수 추정·성장 일정 | 구간식/연속식 분리, 기존 입력 통계 재현, 공통 도달 시점 설정 지원 |
| 학습 실행 | Counter 재현용 실행기와 씬 인자를 받는 COLMAP 실행기 분리 |
| 다중 씬 풀학습 | Counter·Bonsai·Bicycle의 방법/대조군 6개 완료; 논문용 실행 전 검사 추가 |
| 독립 GRT 빌드·학습 | 새 RTX 4090 환경에서 설치·소스 빌드·610스텝·저장 후 100스텝 비교·전체 30뷰 평가 통과 |

진행 중인 기존 학습에는 이 저장소의 변경을 적용하지 않았습니다.

## 디렉터리

```text
src/sparse_grt/
  config.py           # 학습 방법의 설정, 머신 경로와 분리
  sampling.py         # 층화 표집과 멀티뷰 순서
  inputs.py           # 선택된 RGB만 변환하는 입력 경로
  objective.py        # L1/구조손실/정규화 항의 명시적 조합
  engine.py           # backward → Adam → scheduler → MCMC → SH → BVH
  provenance.py       # 실제 import된 모듈·바이너리의 해시 기록
  budget.py           # 학습 이미지·SfM 통계로 계산하는 고정된 두 개수 산정식
  growth.py           # 목표 개수와 도달 시점을 분리한 성장 일정
  standalone_backend.py # 포함된 소스로 빌드하는 독립 초기화·평가
  native_build.py     # GRT·MCMC CUDA 빌드
  training.py         # 독립 학습 CLI
  research_backend.py # 과거 실행기 비교를 위한 연구 런타임 연결부
  losses/             # 검증된 기본 구현과 별도 최적화 후보
configs/              # 실행 설정
tests/                # 수식·입력·표집·가중치의 검증
tools/                # 진단, CUDA 검증, 기존 런타임 호환 실행
patches/              # 기존 렌더러용 최소 변경
docs/                 # 설계·속도·이식 범위
provenance/           # 원본 코드 해시와 검증 기록
```

## CPU 검증

PyTorch와 NumPy가 있는 환경에서 저장소 루트 기준으로 실행합니다. GPU는 사용하지 않습니다.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONPATH=src \
  python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  python tools/audit_structure.py --output runs/structure_audit.json
```

설치가 필요하면 `pip install -e . --no-deps`를 사용합니다. CUDA 커널에는 현재 환경의 CUDA와
호환되는 CuPy가 별도로 필요합니다. 도구가 임의로 CUDA/PyTorch를 설치하거나 교체하지 않습니다.

## 과거 연구 런타임 재현 명령

시간 비교 중인 학습과 동시에 실행하지 않습니다. 로컬 GPU에서 자동 실행하지 않습니다.

```bash
python tools/validate_cuda.py --output runs/cuda_check.json
python tools/train_repro.py --help
python tools/train_scene.py --help
python tools/train_repro.py \
  --config configs/counter_ds2_mv4.json \
  --legacy-entry /path/to/existing/area_full/train.py \
  --output /path/to/new/smoke_run --mode smoke
```

`train_repro.py`의 기본 입력 경로는 기존 구현입니다. 새 입력 경로는
`--input-mode selective_host` 또는 `--input-mode resident_uint8`로 명시합니다.
GPU 캐시 기본 한도는 1,200 MiB이며, 모두 들어가지 않으면 호스트 경로를 사용합니다.
실험용 0 가중치 생략 커널은 아직 학습 실행기의 기본 손실에 연결하지 않았습니다.

기존 renderer bootstrap은 연구 폴더에 의존합니다. 따라서 위 명령은 현재 연구 환경에서의
이식 검증용이고, 외부 사용자가 이 저장소만 clone해서 학습할 수 있다는 의미가 아닙니다.
완전 독립화는 [이식 계획](docs/migration.md)에 구분해 두었습니다.

## 과거 재현 경로에서 고정한 의미

- 추가 레이·보간 이미지·면적 렌더링·GUT 보조 렌더링을 사용하지 않습니다.
- 각 뷰의 실제 관측 좌표로 이웃 가중치를 만들고, 겹치는 모든 이웃 관계를 미분합니다.
- 기존 L1의 표집 평균과 SSIM의 valid-centre 규칙을 유지합니다. 경계 면적 보정을 새로 넣지 않습니다.
- 기본 설정은 **L1=1, 구조손실=0.2, opacity/scale 정규화 각각 0.01**입니다.
  L1=0.8 실험은 별도 설정이며, 계수를 자동 정규화하지 않습니다.
- Counter 비교 실험의 cap은 **1,131,968**입니다. 이를 입력 이미지 추정식의 출력이라고 쓰지 않습니다.
- 공개 전 레퍼런스 renderer·MCMC 수정·평가 코드·라이선스를 각각 고정해야 합니다.

`train_scene.py`는 이 재현 설정과 별도입니다. `boundary=all`은 경계에서도 실제 이웃만
정규화해 모든 관측 위치를 중심으로 사용합니다. 기존 `boundary=valid` 결과를 덮어쓰지 않습니다.
`--budget fine_tracks_v1`은 기존 9개 씬에 보정한 계수를 그대로 사용하며, RGB만의 공식이나
최소 필요 용량의 추정이라고 주장하지 않습니다. SfM 정보가 없으면 실행을 중단합니다.
`--budget haar_v1`은 기존 100만/200만/300만 구간식입니다.

`--growth log_horizon --growth-end 5000`은 600스텝부터 100스텝마다 추가해 5,000스텝에서
목표에 도달하도록 증가율을 계산하는 명시적 실험 설정입니다. 5,000이라는 값의 품질 최적성을
가정하지 않습니다. `--growth native_5percent`는 기존 성장 일정을 유지합니다.
두 경로는 동일한 MCMC 부모 선택과 opacity·scale 보정을 사용합니다.

자세한 내용: [손실 설계](docs/structural_loss.md), [성능 점검](docs/performance.md),
[코드 이식](docs/migration.md).
