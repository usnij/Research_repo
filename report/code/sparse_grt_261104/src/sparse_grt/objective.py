"""Explicit loss coefficients; no hidden 0.8/0.2 normalization."""
import torch
from .losses import coordinate_ssim as coordinate


def structural_loss(rgb, plans, backend=coordinate.loss):
    return torch.stack([
        backend(rgb[p["start"]:p["end"]].reshape(*p["shape"], 3),
                p["gt"][0], p["weights"], p["k"])
        for p in plans
    ]).mean()


def objective(pred, gt, plans, density, scale, config, backend=coordinate.loss):
    l1 = (pred-gt).abs().mean()
    structure = (structural_loss(pred.reshape(-1, 3), plans, backend)
                 if config.structure_weight else pred.new_zeros(()))
    regularization = (config.opacity_weight*density.abs().mean()
                      + config.scale_weight*scale.abs().mean())
    total = config.l1_weight*l1 + config.structure_weight*structure + regularization
    return total, dict(L1=l1, coordinate_SSIM=structure, regularization=regularization)
