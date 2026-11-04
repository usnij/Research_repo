"""Capture loaded implementations, not the folder a launcher hoped to use."""
import hashlib
import importlib.metadata
import os
from pathlib import Path
import sys


def loaded_runtime():
    modules = {}
    for name,module in sorted(sys.modules.items()):
        if not name.startswith(("sparse_grt", "_sparse_grt", "threedgrut", "threedgrt_tracer", "area_tracer512",
                                "lib3dgrt", "lib_mcmc")):
            continue
        filename = getattr(module,"__file__",None)
        if filename and Path(filename).is_file():
            path = Path(filename).resolve()
            modules[name] = dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    versions={}
    for name in ("torch","numpy","cupy","cupy-cuda11x","cupy-cuda12x","cupy-cuda13x","torchmetrics","hydra-core"):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    root=Path(__file__).resolve().parents[2]
    sources={}
    for folder in ('src','tools','configs','patches'):
        for path in sorted((root/folder).rglob('*')):
            if path.is_file() and path.suffix in ('.py','.cu','.json','.patch'):
                sources[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    torch_module=sys.modules.get('torch')
    cuda_version=getattr(getattr(torch_module,'version',None),'cuda',None)
    builder=sys.modules.get('sparse_grt.native_build')
    native_builds=getattr(builder,'BUILD_RECORDS',{})
    return dict(modules=modules,versions=versions,source_sha256=sources,native_builds=native_builds,torch_cuda=cuda_version,flags={k:os.environ.get(k) for k in
                ("GRT_GRAD_ACCUM_DOUBLE","GRT_ALPHA_MIN","SPARSE_GRT_DISABLE_DIAGNOSTICS")})
