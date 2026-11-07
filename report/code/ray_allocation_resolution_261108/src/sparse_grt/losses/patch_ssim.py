"""SSIM on actual observed 2x2 patches, with uniform population moments.

Every pixel in a centre-symmetric 2x2 Gaussian has equal weight. Constants and
the residual algebra match coordinate SSIM: C1=.0001, C2=.0009 and 1-SSIM,
without a half factor, Bessel correction, variance clamp, or extra GT pixels.
"""
import torch


def loss(pred, gt, patch_indices):
    if pred.shape != gt.shape or pred.shape[-1] != 3 or not pred.is_floating_point():
        raise ValueError('Expected matching observed RGB tensors')
    if (gt.requires_grad or gt.device != pred.device or gt.dtype != pred.dtype
            or patch_indices.device != pred.device or patch_indices.dtype != torch.long
            or patch_indices.ndim != 2 or patch_indices.shape[1] != 4
            or patch_indices.shape[0] == 0):
        raise ValueError('Expected nonempty observed 2x2 patches and fixed GT')
    x, y = pred.reshape(-1, 3)[patch_indices], gt.reshape(-1, 3)[patch_indices]
    mx, my = x.mean(1), y.mean(1)
    dx, dy = x-mx[:, None], y-my[:, None]
    vx, vy = dx.square().mean(1), dy.square().mean(1)
    ve = (dx-dy).square().mean(1)
    a = (mx-my).square()/(mx.square()+my.square()+.0001)
    b = ve/(vx+vy+.0009)
    return (a+b-a*b).mean()
