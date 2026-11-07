"""Valid-window SSIM for the explicit 4x4 sampled-grid control.

Window weights use positions i-(window-1)/2. The even window is centred between
samples: [-1.5, -.5, .5, 1.5] for size four. Sigma remains 1.5 grid cells,
population moments and C1=.01**2/C2=.03**2 match the established SSIM formula.
No padding, variance clipping, Bessel correction, or cross-view windows.
The existing 11x11 training arm continues to use fused_ssim in controls.py.
"""
import math
import torch
import torch.nn.functional as F


def gaussian_weights(window, sigma, *, dtype, device):
    if type(window) is not int or window < 2 or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError('Expected an integer window >=2 and positive finite sigma')
    coords = torch.arange(window, dtype=dtype, device=device)-(window-1)/2
    weights = torch.exp(-coords.square()/(2*sigma*sigma))
    return weights/weights.sum()


def valid_ssim(prediction, target, *, window=4, sigma=1.5):
    if (prediction.ndim != 4 or prediction.shape != target.shape
            or prediction.dtype != target.dtype or prediction.device != target.device
            or not prediction.is_floating_point() or min(prediction.shape[-2:]) < window):
        raise ValueError('Expected equally shaped BCHW floating images with a complete window')
    channels = prediction.shape[1]
    weights = gaussian_weights(window, sigma, dtype=prediction.dtype, device=prediction.device)
    moments = torch.cat((prediction, target, prediction.square(), target.square(),
                         prediction*target), dim=1)
    horizontal = weights.reshape(1, 1, 1, window).expand(5*channels, 1, 1, window)
    vertical = weights.reshape(1, 1, window, 1).expand(5*channels, 1, window, 1)
    moments = F.conv2d(F.conv2d(moments, horizontal, groups=5*channels),
                       vertical, groups=5*channels)
    mx, my, xx, yy, xy = moments.chunk(5, dim=1)
    vx, vy, cov = xx-mx.square(), yy-my.square(), xy-mx*my
    return (((2*mx*my+.01**2)*(2*cov+.03**2))/
            ((mx.square()+my.square()+.01**2)*(vx+vy+.03**2))).mean()
