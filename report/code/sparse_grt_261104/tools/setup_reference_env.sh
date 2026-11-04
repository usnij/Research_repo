#!/usr/bin/env bash
# Build the validated Python/CUDA/JPEG environment without changing the host driver.
# Prerequisites: uv, mamba, gcc-11, g++-11, make, curl, git, and zlib development headers.
# Usage: MAMBA_EXE=/path/to/mamba bash tools/setup_reference_env.sh /absolute/new/env-root
set -Eeuo pipefail
if [[ $# != 1 || "$1" != /* ]]; then
  echo 'Usage: bash tools/setup_reference_env.sh /absolute/new/env-root' >&2
  exit 2
fi
INSTALL_ROOT=$1
if [[ -e "$INSTALL_ROOT" ]]; then
  echo "Refusing to overwrite existing environment: $INSTALL_ROOT" >&2
  exit 2
fi
CODE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MAMBA_EXE=${MAMBA_EXE:-mamba}
for required in uv "$MAMBA_EXE" gcc-11 g++-11 make curl git; do
  command -v "$required" >/dev/null
done
mkdir -p "$INSTALL_ROOT"
exec > >(tee "$INSTALL_ROOT/install.log") 2>&1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
uv python install 3.11.14
uv venv --python 3.11.14 --seed "$INSTALL_ROOT/venv"
"$MAMBA_EXE" create -y --no-rc --override-channels -c nvidia/label/cuda-11.8.0 \
  -p "$INSTALL_ROOT/cuda118" cuda-nvcc=11.8 cuda-cudart-dev=11.8 \
  cuda-nvrtc-dev=11.8 cuda-driver-dev=11.8 cuda-cccl=11.8 \
  libcublas-dev libcusparse-dev libcusolver-dev libcurand-dev libcufft-dev
{
  printf 'export CUDA_HOME=%q\n' "$INSTALL_ROOT/cuda118"
  printf 'export PATH=%q:%q:$PATH\n' "$INSTALL_ROOT/venv/bin" "$INSTALL_ROOT/cuda118/bin"
  printf 'export LD_LIBRARY_PATH=%q:%q:${LD_LIBRARY_PATH:-}\n' "$INSTALL_ROOT/cuda118/lib" "$INSTALL_ROOT/cuda118/lib64"
  printf 'export CC=gcc-11 CXX=g++-11 CUDAHOSTCXX=g++-11\n'
  printf 'export MAX_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2\n'
  printf 'export TORCH_EXTENSIONS_DIR=%q\n' "$INSTALL_ROOT/build-cache/torch"
  printf 'export CUPY_CACHE_DIR=%q\n' "$INSTALL_ROOT/build-cache/cupy"
  printf 'export TORCH_HOME=%q\n' "$INSTALL_ROOT/model-cache"
  printf 'export PYTHONNOUSERSITE=1 GRT_GRAD_ACCUM_DOUBLE=1 PYTHONUNBUFFERED=1\n'
  printf 'unset PYTHONPATH\n'
} > "$INSTALL_ROOT/activate.sh"
source "$INSTALL_ROOT/activate.sh"
python -m pip install 'setuptools>=68,<81' wheel
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install "$CODE_ROOT"

# Pillow's version alone does not fix JPEG pixels: the PyPI wheel uses turbo,
# while the established training inputs were decoded with IJG libjpeg 9f.
mkdir -p "$INSTALL_ROOT/downloads" "$INSTALL_ROOT/codec-build"
JPEG_ARCHIVE="$INSTALL_ROOT/downloads/jpegsrc.v9f.tar.gz"
curl --fail --location --retry 3 https://www.ijg.org/files/jpegsrc.v9f.tar.gz -o "$JPEG_ARCHIVE"
printf '04705c110cb2469caa79fb71fba3d7bf834914706e9641a4589485c1f832565b  %s\n' "$JPEG_ARCHIVE" | sha256sum --check
tar -xzf "$JPEG_ARCHIVE" -C "$INSTALL_ROOT/codec-build"
(
  cd "$INSTALL_ROOT/codec-build/jpeg-9f"
  ./configure --prefix="$INSTALL_ROOT/jpeg9f"
  make -j2
  make install
)
CFLAGS="-I$INSTALL_ROOT/jpeg9f/include" \
LDFLAGS="-L$INSTALL_ROOT/jpeg9f/lib -Wl,-rpath,$INSTALL_ROOT/jpeg9f/lib" \
python -m pip install --no-binary=pillow --force-reinstall --no-cache-dir Pillow==12.0.0
python -m sparse_grt.fetch_optix
python -m pip check
python -m pip freeze > "$INSTALL_ROOT/requirements-installed.txt"
python - <<'PY'
import sys
from PIL import features
import torch, cv2, sparse_grt
assert features.version('jpg') == '9.0' and not features.check('libjpeg_turbo')
assert torch.__version__ == '2.1.2+cu118'
print('Python:', sys.version)
print('Installed package:', sparse_grt.__file__)
print('JPEG:', features.version('jpg'), 'OpenCV:', cv2.__version__)
PY
printf 'Environment ready. Activate with: source %q\n' "$INSTALL_ROOT/activate.sh"
