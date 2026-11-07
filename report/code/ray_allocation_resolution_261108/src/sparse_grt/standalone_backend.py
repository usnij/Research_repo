"""Headless, source-built GRT backend without imports from research directories."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from sparse_grt_backend.datasets.dataset_colmap import ColmapDataset
from sparse_grt_backend.datasets.protocols import Batch
from sparse_grt_backend.model.model import MixtureOfGaussians
import sparse_grt_backend.strategy.mcmc as mc
from sparse_grt_backend.utils.misc import check_step_condition
from sparse_grt_backend.utils.logger import logger
from .cached_dataset import CachedDataset

KEYS = ['positions', 'rotation', 'scale', 'density', 'a_raw', 'features_albedo', 'features_specular']

def cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: cpu(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return type(x)(cpu(v) for v in x)
    return x


def digest(x):
    h = hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            h.update(str((v.dtype, tuple(v.shape))).encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            for k, w in sorted(v.items(), key=lambda kv: str(kv[0])):
                h.update(str(k).encode())
                visit(w)
        elif isinstance(v, (tuple, list)):
            for w in v:
                visit(w)
        else:
            h.update(repr(v).encode())
    visit(x)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def clear(runner):
    tensors = [torch.empty(0, device='cuda', dtype=d) for d in
               [torch.int32, torch.int32, torch.int64, torch.float32, torch.uint8]]
    runner.wrapper.set_area_buffers(*tensors)


class Runner:
    def __init__(self, model):
        self.wrapper = model.renderer.tracer_wrapper
        self.probe_empty = tuple(torch.empty(0, device='cuda', dtype=d) for d in
                                 (torch.int64, torch.int32, torch.float32, torch.int32))
        self.candidate_empty = tuple(torch.empty(0, device='cuda', dtype=d) for d in
                                     (torch.int64, torch.int32, torch.int32, torch.int32))
        self.native()
        clear(self)

    def native(self):
        self.wrapper.set_candidate_buffers(*self.candidate_empty, 0)
        self.wrapper.set_probe_buffers(*self.probe_empty)


def keyed_mcmc(strategy):
    for name, base in [('relocate_gaussians', 810000000), ('add_new_gaussians', 820000000),
                       ('perturb_gaussians', 830000000)]:
        original = getattr(strategy, name)
        def wrapped(*args, _fn=original, _base=base, **kwargs):
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(_base + strategy.paired_step)
                torch.cuda.manual_seed_all(_base + strategy.paired_step)
                return _fn(*args, **kwargs)
        setattr(strategy, name, wrapped)


def load():
    # Keep the trainer interface stable while removing legacy imports entirely.
    support = SimpleNamespace(cpu=cpu, digest=digest, write=write, KEYS=KEYS,
                              OmegaConf=OmegaConf, Batch=Batch, ColmapDataset=ColmapDataset)
    return SimpleNamespace(torch=torch, np=np, S=support,
                           B=SimpleNamespace(CachedDataset=CachedDataset,
                                             check_step_condition=check_step_condition),
                           M=SimpleNamespace(Runner=Runner, clear=clear))


def configure(runtime, path, downsample, method):
    cfg = OmegaConf.create(json.loads(Path(__file__).with_name('backend_config.json').read_text()))
    cfg.path = str(Path(path).resolve())
    cfg.dataset.downsample_factor = downsample
    cfg.strategy.add.max_n_gaussians = method.cap
    cfg.n_iterations = method.steps
    cfg.loss.lambda_l1 = method.l1_weight
    cfg.loss.lambda_ssim = method.structure_weight
    cfg.loss.lambda_opacity = method.opacity_weight
    cfg.loss.lambda_scale = method.scale_weight
    return cfg


def dataset(runtime, cfg, split='train'):
    return ColmapDataset(cfg.path, split=split, downsample_factor=cfg.dataset.downsample_factor,
                         test_split_interval=cfg.dataset.test_split_interval)


def initialize(runtime, cfg, method, train_data, growth_factory=None):
    if cfg.initialization.method != 'colmap' or cfg.resume or cfg.optimizer.type != 'adam':
        raise ValueError('This entry point supports from-scratch COLMAP initialization and Adam')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.50)
    torch.manual_seed(261001)
    if not OmegaConf.has_resolver('int_list'):
        OmegaConf.register_new_resolver('int_list', lambda v: list(map(int, v)))
    original_sample = mc._multinomial_sample
    mc._multinomial_sample = lambda probabilities,n,replacement=True: original_sample(
        probabilities.double(), n, replacement=replacement)
    torch.cuda.set_per_process_memory_fraction(.82)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(method.initialization_seed)
    np.random.seed(method.initialization_seed)
    torch.manual_seed(method.initialization_seed)
    torch.cuda.manual_seed_all(method.initialization_seed)
    model = MixtureOfGaussians(cfg, scene_extent=train_data.get_scene_extent())
    strategy = mc.MCMCStrategy(cfg, model)
    # Preserve the original trainer's initialization log as well: Rich's link
    # formatting consumes Python RNG state, although training uses Torch/NumPy.
    logger.info("🤸 Initiating new 3dgrut training..")
    observer_points = torch.tensor(train_data.get_observer_points(), dtype=torch.float32, device='cuda')
    model.init_from_colmap(cfg.path, observer_points)
    strategy.init_densification_buffer()
    model.build_acc()
    model.setup_optimizer()
    assert not model.optimizer.state
    if model.num_gaussians > method.cap:
        raise ValueError('Estimated cap is smaller than initialization; no implicit pruning is allowed')
    plan = None
    if growth_factory is not None:
        from .growth import install
        plan = growth_factory(model.num_gaussians)
        install(strategy, plan)
    keyed_mcmc(strategy)
    runner = Runner(model)
    return model, strategy, runner, plan


def evaluate(runtime, model, cfg, residual_map_dir=None):
    """Same per-image metrics as the established Counter evaluator."""
    torch, S = runtime.torch, runtime.S
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    validation = dataset(runtime, cfg, 'val')
    ssim = StructuralSimilarityIndexMeasure(data_range=1., kernel_size=11, sigma=1.5).cuda()
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).cuda()
    before = S.digest({k: getattr(model, k) for k in S.KEYS})
    rows = []
    if residual_map_dir is not None:
        residual_map_dir = Path(residual_map_dir)
        residual_map_dir.mkdir(parents=True, exist_ok=False)
    with torch.no_grad():
        for i in range(len(validation)):
            batch = validation.get_gpu_batch_with_intrinsics(
                torch.utils.data.default_collate([validation[i]]))
            pred = model(batch, train=False)['pred_rgb'].permute(0, 3, 1, 2).clamp(0, 1)
            gt = batch.rgb_gt.permute(0, 3, 1, 2).clamp(0, 1)
            row = dict(view=i, image=str(validation.image_paths[i]),
                       width=int(pred.shape[-1]), height=int(pred.shape[-2]),
                       PSNR=float(-10*torch.log10((pred-gt).square().mean().clamp_min(1e-12))),
                       SSIM=float(ssim(pred, gt)), LPIPS=float(lpips(pred, gt)))
            assert all(math.isfinite(row[k]) for k in ('PSNR', 'SSIM', 'LPIPS'))
            if residual_map_dir is not None:
                import torch.nn.functional as F
                residual=(pred-gt).abs().mean(1,keepdim=True)
                h,w=residual.shape[-2:]; pad=(0,(-w)%8,0,(-h)%8)
                mass=F.avg_pool2d(F.pad(residual,pad),8,8)*64
                area=F.avg_pool2d(F.pad(torch.ones_like(residual),pad),8,8)*64
                means=mass/area
                order=torch.argsort(means.flatten(),descending=True)
                k=math.ceil(order.numel()*.2)
                denominator=float(mass.sum())
                row['residual_top20_fraction']=(float(mass.flatten()[order[:k]].sum())/denominator
                                               if denominator>0 else 0.)
                np.savez_compressed(residual_map_dir/f'view_{i:04d}.npz',
                                    mean=means.cpu().numpy(),mass=mass.cpu().numpy(),
                                    pixels=area.cpu().numpy())
            rows.append(row)
    assert S.digest({k: getattr(model, k) for k in S.KEYS}) == before
    return dict(views=len(rows), N=model.num_gaussians,
                **{k:sum(r[k] for r in rows)/len(rows) for k in ('PSNR','SSIM','LPIPS')}, per_view=rows)
