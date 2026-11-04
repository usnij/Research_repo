"""Scene-independent adapter for the pinned, already-built research runtime.

This is deliberately not advertised as a standalone upstream GRT installation.
It retains the paired experiments' CUDA binary, optimizer, and MCMC fork.
"""
import importlib
import importlib.util
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace


def load(entry):
    entry = Path(entry).resolve()
    sys.path.insert(0, str(entry.parent))
    spec = importlib.util.spec_from_file_location('explicit_research_backend', entry)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    return runtime


def configure(runtime, path, downsample, method):
    cfg = runtime.config()
    cfg.path = str(Path(path).resolve())
    cfg.dataset.downsample_factor = downsample
    cfg.strategy.add.max_n_gaussians = method.cap
    cfg.n_iterations = method.steps
    cfg.loss.lambda_l1 = method.l1_weight
    cfg.loss.lambda_ssim = method.structure_weight
    cfg.loss.lambda_opacity = method.opacity_weight
    cfg.loss.lambda_scale = method.scale_weight
    assert cfg.initialization.method == 'colmap' and not cfg.resume
    return cfg


def dataset(runtime, cfg, split='train'):
    return runtime.S.ColmapDataset(cfg.path, split=split,
                downsample_factor=cfg.dataset.downsample_factor,
                test_split_interval=cfg.dataset.test_split_interval)


def initialize(runtime, cfg, method, train_data, growth_factory=None):
    T = runtime
    S, torch = T.S, T.torch
    S.setup()
    torch.cuda.set_per_process_memory_fraction(.82)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from torch.utils.cpp_extension import _get_build_directory
    name = 'lib3dgrt_area_subset512_261029_1'
    sys.path.insert(0, _get_build_directory(name, False))
    import area_tracer512, threedgrt_tracer
    area_tracer512.tracer._3dgrt_plugin = importlib.import_module(name)
    threedgrt_tracer.Tracer = area_tracer512.Tracer
    sys.path.insert(0, _get_build_directory('lib_mcmc_cc', False))
    S.mc._mcmc_plugin = importlib.import_module('lib_mcmc_cc')
    random.seed(method.initialization_seed)
    T.np.random.seed(method.initialization_seed)
    torch.manual_seed(method.initialization_seed)
    torch.cuda.manual_seed_all(method.initialization_seed)
    model = S.MixtureOfGaussians(cfg, scene_extent=train_data.get_scene_extent())
    strategy = S.mc.MCMCStrategy(cfg, model)
    trainer = SimpleNamespace(strategy=strategy, device='cuda')
    S.Trainer3DGRUT.setup_training(trainer, cfg, model, train_data)
    assert trainer.global_step == 0 and not model.optimizer.state
    if model.num_gaussians > method.cap:
        raise ValueError("Estimated cap is smaller than initialization; no implicit pruning is allowed")
    plan = None
    if growth_factory is not None:
        from .growth import install
        plan = growth_factory(model.num_gaussians)
        install(strategy, plan)
    T.B.keyed_mcmc(strategy)
    runner = T.M.Runner(model)
    return model, strategy, runner, plan


def evaluate(runtime, model, cfg):
    """Same per-image metrics as the established Counter evaluator."""
    torch, S = runtime.torch, runtime.S
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    validation = dataset(runtime, cfg, 'val')
    ssim = StructuralSimilarityIndexMeasure(data_range=1.).cuda()
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).cuda()
    before = S.digest({k: getattr(model, k) for k in S.KEYS})
    rows = []
    with torch.no_grad():
        for i in range(len(validation)):
            batch = validation.get_gpu_batch_with_intrinsics(
                torch.utils.data.default_collate([validation[i]]))
            pred = model(batch, train=False)['pred_rgb'].permute(0, 3, 1, 2).clamp(0, 1)
            gt = batch.rgb_gt.permute(0, 3, 1, 2).clamp(0, 1)
            row = dict(view=i, image=str(validation.image_paths[i]),
                       PSNR=float(-10*torch.log10((pred-gt).square().mean().clamp_min(1e-12))),
                       SSIM=float(ssim(pred, gt)), LPIPS=float(lpips(pred, gt)))
            assert all(math.isfinite(row[k]) for k in ('PSNR', 'SSIM', 'LPIPS'))
            rows.append(row)
    assert S.digest({k: getattr(model, k) for k in S.KEYS}) == before
    return dict(views=len(rows), N=model.num_gaussians,
                **{k:sum(r[k] for r in rows)/len(rows) for k in ('PSNR','SSIM','LPIPS')}, per_view=rows)
