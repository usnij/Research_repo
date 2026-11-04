"""Build the headless reference renderer and MCMC from repository sources."""
import hashlib
import os
from pathlib import Path

OPTIX_COMMIT = '4bcbb37d69476e3be620c1453b4ea729fb5e8f7a'
OPTIX_URL = 'https://github.com/NVIDIA/optix-dev.git'
BUILD_RECORDS = {}


def _record(name, module, source_root):
    files = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(source_root.rglob('*')) if p.is_file()
             and p.suffix in ('.cpp', '.cu', '.cuh', '.h', '.hpp')}
    binary = Path(module.__file__).resolve()
    BUILD_RECORDS[name] = dict(binary=str(binary),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), sources=files)
    return module


def build_renderer(conf):
    from sparse_grt_backend.utils import jit
    from torch.utils.cpp_extension import CUDA_HOME
    import sparse_grt_tracer
    root = Path(sparse_grt_tracer.__file__).resolve().parent
    if (conf.render.pipeline_type, conf.render.backward_pipeline_type) != ('reference', 'referenceBwd'):
        raise ValueError('This package supports the frozen reference/referenceBwd pipeline only')
    if os.environ.get('GRT_GRAD_ACCUM_DOUBLE', '1') != '1':
        raise ValueError('The paper runtime requires FP64 gradient accumulation')
    if 'GRT_ALPHA_MIN' in os.environ and float(os.environ['GRT_ALPHA_MIN']) != conf.render.particle_kernel_min_alpha:
        raise ValueError('GRT_ALPHA_MIN would override the declared renderer threshold')
    os.environ['GRT_GRAD_ACCUM_DOUBLE'] = '1'
    sdk = root / 'dependencies/optix-dev/include'
    if not (sdk/'optix.h').is_file():
        raise RuntimeError('OptiX headers missing. Run python -m sparse_grt.fetch_optix first.')
    if not CUDA_HOME or not (Path(CUDA_HOME)/'bin/nvcc').is_file():
        raise RuntimeError('A CUDA toolkit with nvcc is required; set CUDA_HOME before Python starts')
    os.environ.setdefault('MAX_JOBS', '2')
    name = '_sparse_grt_ray_fp64_v1'
    module = jit.load(name=name,
        sources=[str(root/p) for p in ('src/optixTracer.cpp', 'src/particlePrimitives.cu', 'bindings.cpp')],
        extra_cflags=['-DGRT_GRAD_ACCUM_DOUBLE=1'],
        extra_cuda_cflags=['-DGRT_GRAD_ACCUM_DOUBLE=1'],
        extra_include_paths=[str(root/'include'), str(sdk)])
    return _record(name, module, root)


def build_mcmc():
    from sparse_grt_backend.utils import jit
    import sparse_grt_backend.strategy
    root = Path(sparse_grt_backend.strategy.__file__).resolve().parent/'src'
    os.environ.setdefault('MAX_JOBS', '2')
    name = '_sparse_grt_mcmc_v1'
    module = jit.load(name=name, sources=[str(root/p) for p in ('gaussian_mcmc.cu', 'bindings.cpp')],
                      extra_cuda_cflags=['-use_fast_math', '-O3'], extra_include_paths=[str(root)])
    return _record(name, module, root)
