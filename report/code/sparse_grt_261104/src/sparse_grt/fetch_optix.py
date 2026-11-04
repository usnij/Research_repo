"""Fetch the pinned external OptiX headers, retaining NVIDIA's license."""
from pathlib import Path
from importlib.util import find_spec
import subprocess
from .native_build import OPTIX_COMMIT, OPTIX_URL


def main():
    # Locate the package without importing the renderer and its runtime dependencies.
    spec = find_spec('sparse_grt_tracer')
    if spec is None or spec.origin is None:
        raise RuntimeError('The sparse-grt renderer package is not installed')
    target = Path(spec.origin).resolve().parent/'dependencies/optix-dev'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        head = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        if head != OPTIX_COMMIT:
            raise RuntimeError(f'Existing OptiX checkout has unexpected revision {head}; it was not changed')
        if subprocess.check_output(['git', '-C', str(target), 'status', '--porcelain'], text=True).strip():
            raise RuntimeError('Existing OptiX checkout is modified; it was not changed')
    else:
        subprocess.run(['git', 'clone', '--no-checkout', OPTIX_URL, str(target)], check=True)
        subprocess.run(['git', '-C', str(target), 'checkout', '--detach', OPTIX_COMMIT], check=True)
    if not (target/'include/optix.h').is_file():
        raise RuntimeError('Pinned checkout does not contain OptiX headers')
    print(f'OptiX {OPTIX_COMMIT}: {target}\nLicense: {target / "LICENSE.txt"}')


if __name__ == '__main__':
    main()
