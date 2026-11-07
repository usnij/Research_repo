# 독립 설치와 학습 검증

`0.2.0.dev0`은 기존 연구 폴더를 import하지 않고, 포함된 GRT·MCMC 소스에서
CUDA 확장을 빌드하는 경로다. 학습 알고리즘은 `paper_v1_20261102`와 같다.
기존 태그는 로컬 연구 런타임 기준의 비교 대상으로 보존한다.

2026-11-02 로컬 검증에서 CPU 16개, 설치된 wheel의 새 CUDA 빌드, Counter 610 update,
checkpoint 재로딩 및 test 30장 평가가 통과했다. 610 update의 손실·N과 최종
파라미터·Adam·Torch/NumPy/Python 난수 상태가 기존 실행과 비트 단위로 일치했다.
근거는 [검증 요약](../releases/standalone_candidate_20261102/summary.json)과
[학습 비교](../releases/standalone_candidate_20261102/validation/training_parity.json)에 있다.
이후 새 RTX 4090 인스턴스에서 의존성 설치·소스 빌드·610 update·저장 후 100 update
이어학습 비교·전체 test 30장 평가를 수행했다. 이어학습의 파라미터·Adam·난수 상태는
중단 없이 계속한 결과와 비트 단위로 같았다. 원격 검증 기록은
[검증 요약](../releases/standalone_remote_20261102/summary.json)에 보관한다.
최종 설치 스크립트로 두 번째 환경을 만든 뒤 일반 학습 CLI를 다시 실행했으며,
두 환경의 610개 step 값과 checkpoint 16개 비교 항목이 모두 정확히 일치했다.
30k 풀학습 검증과는 구분한다.

## 새 서버 설치

검증 환경은 Ubuntu 24.04, RTX 4090 24GB, NVIDIA driver 560.35.03이다.
기본 Python 3.12/CUDA 12.6 환경을 사용하지 않고 별도의 Python 3.11.14/CUDA 11.8
환경을 만들었다. 호스트 드라이버는 바꾸지 않는다.

`uv`, `mamba`, `gcc-11`, `g++-11`, `make`, `curl`, `git`, zlib 개발 헤더가 필요하다.
Ubuntu에서 컴파일러·zlib 헤더가 없다면 관리자 권한으로 설치한다.

```bash
apt-get update
apt-get install -y --no-install-recommends gcc-11 g++-11 make curl git zlib1g-dev
# Vast PyTorch 이미지의 mamba 경로 예시. 다른 환경에서는 실제 경로를 지정한다.
MAMBA_EXE=/opt/miniforge3/bin/mamba \
  bash tools/setup_reference_env.sh /workspace/sparse-grt-env
source /workspace/sparse-grt-env/activate.sh
```

설치 스크립트는 이미 존재하는 환경 디렉터리를 덮어쓰지 않는다. 저장소를 `pip install`하고
공식 OptiX 고정 commit을 받으며, CUDA 개발 헤더와 재현용 JPEG 디코더까지 설치한다.
기존 연구 환경·바이너리·인증정보를 복사할 필요가 없다.

**Pillow 버전만 고정하면 입력 픽셀까지 고정되지는 않는다.** 기존 환경은 IJG JPEG 9f,
PyPI Pillow 12.0.0 wheel은 libjpeg-turbo를 사용했다. 같은 Counter 이미지에서 목표 개수가
1,229,851 → 1,229,647로 달라졌다. 위 스크립트는 SHA256을 확인한 IJG 9f 소스로
Pillow를 빌드한다. 이 환경에서는 240장 전체의 디코딩 RGB와 입력 목표 1,229,851이
기존 기록과 정확히 일치했다. JPEG 버전의 `9.0` 표시는 9f의 API 버전이다.

## 설치 대상

- Linux, NVIDIA RTX GPU와 정상 동작하는 드라이버
- Python 3.11, CUDA Toolkit 11.8 (`nvcc`와 cuBLAS/cuSPARSE/cuSOLVER 등 개발 헤더 포함), GCC/G++ 11, Git
- PyTorch 2.1.2/cu118, torchvision 0.16.2, CuPy 13.6.0/cuda11x
- 나머지 Python 의존성은 `pyproject.toml`에 고정

CUDA Toolkit과 드라이버는 Python 패키지에 포함되지 않는다. OptiX 7.5 헤더는
공식 저장소의 고정 commit에서 설치하며 원래 라이선스를 유지한다.
기존 연구 환경의 GUI·GUT·Slang은 필요하지 않다. COLMAP 카메라 로더가 사용하는
OpenCV는 GUI가 없는 `opencv-python-headless`로 설치한다.
새 환경의 입력 일치도와 설치 호환성은 별도로 확인해야 한다.

위 설치 스크립트를 썼으면 아래 수동 설치는 생략한다. 이미 준비된 **동일한 JPEG 9f 및
CUDA 개발 환경**에서 Python 패키지만 설치할 때는 저장소 루트를 기준으로 실행한다.

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=2
python -m pip install 'setuptools>=68,<81' wheel
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install .
python -m pip check
python -m sparse_grt.fetch_optix
```

기존 CUDA 바이너리나 개인 캐시를 복사하지 않는다. 처음 실행할 때 GRT·MCMC가
자동 빌드된다. `TORCH_EXTENSIONS_DIR`와 `CUPY_CACHE_DIR`를 빈 경로로 지정하면
기존 캐시 사용 여부를 분리해서 확인할 수 있다.

## Counter 검증 명령

데이터에는 `images_2`와 COLMAP의 `sparse/0`이 필요하다. 데이터 경로와 출력 경로를 지정한다.

```bash
export GRT_GRAD_ACCUM_DOUBLE=1
python -m sparse_grt.training \
  --scene /path/to/counter --downsample 2 \
  --config releases/paper_v1_20261102/configs/counter_structure.json \
  --budget fine_tracks_v1 --budget-views all \
  --growth log_horizon --growth-end 5000 \
  --output /path/to/new/smoke --mode smoke
```

`smoke`는 처음부터 610 update를 실행하고 checkpoint를 저장한다. 첫 ADD와 MCMC
재배치까지 포함한다. `--mode train`은 30,000 update 및 7k·15k·30k 전체 test 평가다.
VGG-LPIPS 평가는 공개 VGG16 가중치가 필요하며, 캐시가 없으면 torchvision이 내려받는다.
학습 종료 후 checkpoint를 다시 불러오는 검사도 별도로 수행한다.

```bash
python tools/check_standalone_checkpoint.py \
  --checkpoint /path/to/new/smoke/checkpoint_latest.pt \
  --output /path/to/new/checkpoint_evaluation.json
```

`tools/check_standalone_continuation.py`는 같은 학습 인자를 `--` 뒤에 받아 610 update를
실행한 뒤, 메모리 상태와 저장된 checkpoint에서 각각 100 update를 진행해 비교한다.
다음 ADD/재배치를 포함하며, 검사 결과에는 파라미터·Adam·RNG 해시를 남긴다.
이는 checkpoint의 복원 가능성을 검사하는 도구다. 현재 일반 학습 CLI에는
`--resume` 옵션이 없으므로 임의의 중간 스텝에서 재시작하는 사용자용 기능과 구분한다.

## 판정 범위

1. 설치된 wheel의 소스만 사용하고 기존 연구 코드 접근을 차단한다.
2. 빈 캐시에서 GRT·MCMC·좌표 SSIM CUDA를 빌드한다.
3. 초기 상태·표집 순서·각 step의 손실·N·최종 파라미터·Adam·난수 상태를 기존 실행과 비교한다.
4. 저장한 checkpoint를 다시 불러와 full-ray 렌더와 평가 경로를 확인한다.
5. 별도 인스턴스의 새 Python 환경에 의존성을 설치하고, 입력 디코딩까지 일치하는지 검사한다.

로컬의 기존 Python 패키지를 사용한 소스 격리 검사는 5번을 대신하지 않는다.
짧은 검사는 30k 최종 품질이나 다른 GPU의 시간 재현을 보장하지 않는다.
RTX 4070 SUPER와 RTX 4090 사이의 학습 결과는 비트 단위로 같지 않았다. 입력 디코딩·
뷰 순서·성장 일정의 일치와, 같은 GPU에서의 재실행·이어학습 일치는 별도로 판정한다.
