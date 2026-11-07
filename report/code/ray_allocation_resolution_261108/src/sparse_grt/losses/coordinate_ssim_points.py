"""Observed-point SSIM for arbitrary sampled pixels, including repeated rays.

The objective is the same all-centre, population-moment SSIM as
``coordinate_ssim_all``. The physical square support is +/-3*sigma and each
centre normalizes its Gaussian weights over observed samples only. Repeated
coordinates retain their multiplicity both as neighbours and as centres.

CUDA uses a sorted spatial index, not pairwise NxN storage. Repeated pixels are
compressed to count/mean/within-pixel variance before neighbourhood traversal;
this is exact even if repeated samples have different RGB values. Backward
gathers all contributing window centres, retaining a gradient for every input
sample. It neither deduplicates the rendered ray budget nor uses GT elsewhere.
"""
from pathlib import Path
import math
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
_kernels = {}


def _validate(pred, gt, y, x, sigma, height, width):
    if pred.ndim != 2 or pred.shape[1] != 3 or gt.shape != pred.shape or not len(pred):
        raise ValueError('pred and gt must be nonempty [N,3] arrays')
    if y.shape != (len(pred),) or x.shape != y.shape:
        raise ValueError('y and x must have one entry per RGB sample')
    if y.dtype not in (torch.int32, torch.int64) or x.dtype not in (torch.int32, torch.int64):
        raise TypeError('pixel coordinates must be integer tensors')
    if any(t.device != pred.device for t in (gt, y, x)) or gt.dtype != pred.dtype:
        raise ValueError('all tensors must share a device; pred/gt must share dtype')
    if gt.requires_grad:
        raise ValueError('GT is a fixed observation, not an optimization variable')
    if type(height) is not int or type(width) is not int or min(height, width) < 1:
        raise ValueError('positive integer image dimensions are required')
    if max(height, width, len(pred)) >= 2**31:
        raise ValueError('image dimensions and sample count must fit int32')
    if not math.isfinite(float(sigma)) or sigma <= 0:
        raise ValueError('sigma must be positive and finite')


def validate_coordinates(y, x, height, width):
    """Explicit bounds check for sampler validation, outside a timed hot path.

The CUDA loss trusts coordinates produced by the validated sampler to avoid a
host/device synchronization every update. Call this when accepting new inputs.
"""
    if not bool(((y >= 0) & (y < height) & (x >= 0) & (x < width)).all()):
        raise ValueError('sample coordinate outside the image')


def reference_loss(pred, gt, y, x, sigma, height, width):
    """Small-test O(N^2) reference with stable centred empirical moments."""
    _validate(pred, gt, y, x, sigma, height, width)
    validate_coordinates(y, x, height, width)
    coords = torch.stack((y, x), -1).to(pred.dtype)
    delta = coords[:, None] - coords[None, :]
    weight = torch.exp(-delta.square().sum(-1)/(2*float(sigma)**2))
    weight = weight * (delta.abs().amax(-1) <= 3*float(sigma))
    weight = (weight / weight.sum(-1, keepdim=True)).detach()
    mp, mt = weight @ pred, weight @ gt
    dp, dt = pred[None] - mp[:, None], gt[None] - mt[:, None]
    w = weight[..., None]
    vp, vt = (w*dp.square()).sum(1), (w*dt.square()).sum(1)
    ve = (w*(dp-dt).square()).sum(1)
    a = (mp-mt).square() / (mp.square()+mt.square()+.0001)
    b = ve / (vp+vt+.0009)
    return (a+b-a*b).mean()


def _kernel(name):
    if name not in _kernels:
        import cupy as cp
        _kernels[name] = cp.RawKernel((HERE/'coordinate_ssim_points.cu').read_text(),
                                      name, options=('--std=c++14', '--fmad=false'))
    return _kernels[name]


def _launch(name, tensors, scalars, count):
    import cupy as cp
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        _kernel(name)(((count+127)//128,), (128,),
                      tuple(cp.from_dlpack(t.detach()) for t in tensors)+tuple(scalars))


def _spatial_index(y, x, sigma, height, width):
    # The image bounds contract is checked by the sampler, outside this path.
    pixel = y.to(torch.int64)*width+x.to(torch.int64)
    ordered_pixel, sample_order = torch.sort(pixel, stable=True)
    unique_pixel, counts = torch.unique_consecutive(ordered_pixel, return_counts=True)
    starts = torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(torch.int32)
    uy = (unique_pixel // width).to(torch.int32)
    ux = (unique_pixel % width).to(torch.int32)
    radius = math.floor(3*float(sigma))
    cell_size = max(1, radius)
    bh, bw = (height+cell_size-1)//cell_size, (width+cell_size-1)//cell_size
    cells = (uy.to(torch.int64)//cell_size)*bw+(ux.to(torch.int64)//cell_size)
    group_order = torch.argsort(cells, stable=True).to(torch.int32)
    bin_counts = torch.bincount(cells, minlength=bh*bw)
    bin_offsets = torch.cat((bin_counts.new_zeros(1), bin_counts.cumsum(0))).to(torch.int32)
    return (sample_order.to(torch.int32), starts, uy, ux, group_order, bin_offsets,
            cell_size, bh, bw, radius)


class _PointLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, gt, y, x, sigma, height, width):
        pred, gt = pred.contiguous(), gt.contiguous()
        y, x = y.to(torch.int32).contiguous(), x.to(torch.int32).contiguous()
        order, starts, uy, ux, group_order, offsets, cell_size, bh, bw, radius = (
            _spatial_index(y, x, sigma, height, width))
        n, groups = len(pred), len(uy)
        moments = pred.new_empty((groups, 3, 5))
        stats = pred.new_empty((groups, 3, 5))
        inv_norm = pred.new_empty(groups)
        values = pred.new_empty((groups, 3))
        _launch('aggregate_pixels', [pred, gt, order, starts, moments], [groups], 3*groups)
        _launch('point_ssim_forward',
                [moments, starts, uy, ux, group_order, offsets, values, stats, inv_norm],
                [groups, cell_size, bh, bw, radius, np.float32(sigma)], 3*groups)
        ctx.save_for_backward(pred, gt, y, x, starts, uy, ux, group_order, offsets, stats, inv_norm)
        ctx.params = (n, cell_size, bh, bw, radius, np.float32(sigma))
        return values.sum() / (3*n)

    @staticmethod
    def backward(ctx, upstream):
        pred, gt, y, x, starts, uy, ux, group_order, offsets, stats, inv_norm = ctx.saved_tensors
        grad = torch.empty_like(pred)
        _launch('point_ssim_backward',
                [pred, gt, y, x, starts, uy, ux, group_order, offsets, stats, inv_norm,
                 upstream.contiguous(), grad], list(ctx.params), pred.numel())
        return grad, None, None, None, None, None, None


def loss(pred, gt, y, x, sigma, height, width):
    """Mean SSIM loss over all observed centres/channels, repeated rays included.

Coordinates must lie inside [0,height) x [0,width). For CUDA this is a caller
contract; ``validate_coordinates`` provides an explicit preflight check. CPU
dispatch exists for small numerical tests only and uses quadratic storage.
"""
    _validate(pred, gt, y, x, sigma, height, width)
    if not pred.is_cuda:
        return reference_loss(pred, gt, y, x, sigma, height, width)
    if pred.dtype != torch.float32:
        raise TypeError('CUDA point SSIM requires float32 RGB')
    return _PointLoss.apply(pred, gt, y, x, float(sigma), height, width)
