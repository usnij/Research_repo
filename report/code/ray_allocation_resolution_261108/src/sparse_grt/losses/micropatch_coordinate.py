"""Complete candidate support for micropatches; reuse the frozen SSIM kernels.

Only the virtual-grid candidate envelope differs from point sampling. Actual
coordinate Gaussian weights, physical square cutoff, normalization, all-centre
population SSIM and its complete overlapping backward remain the same.
"""
import torch
import torch.nn.functional as F
from . import coordinate_ssim_all as original


loss = original.loss
reference_loss = original.reference_loss
support_summary = original.support_summary


def radius_for(block, sigma, candidate_radius=None):
    if block != 3:
        raise ValueError('Micropatch coordinates use block3 virtual slots')
    required = 2*original.radius_for(6, sigma)+1
    if candidate_radius is None:
        return required
    if type(candidate_radius) is not int or candidate_radius < required:
        raise ValueError('Candidate envelope would omit valid micropatch neighbors')
    return candidate_radius


def reference_weights(y, x, block=3, sigma=2., nominal=False, *, candidate_radius=None):
    if nominal:
        raise ValueError('Micropatch loss uses actual observed coordinates')
    r = radius_for(block, sigma, candidate_radius); k = 2*r+1
    coords = torch.stack((y,x)).to(torch.float64 if y.dtype == torch.float64 else torch.float32)[None]
    neighbors = F.unfold(coords,k,padding=r).reshape(1,2,k*k,-1)
    delta = neighbors-coords.reshape(1,2,1,-1)
    valid = F.unfold(torch.ones_like(coords[:,:1]),k,padding=r) > .5
    valid &= (delta.abs() <= 3*sigma).all(1)
    weight = torch.exp(-delta.square().sum(1)/(2*sigma*sigma))*valid
    return (weight/weight.sum(1,keepdim=True)).detach(), k


def weights(y, x, block=3, sigma=2., nominal=False, *, candidate_radius=None):
    import numpy as np
    if nominal:
        raise ValueError('Micropatch loss uses actual observed coordinates')
    assert y.ndim==x.ndim==2 and y.shape==x.shape and y.is_cuda and x.device==y.device
    h,w=y.shape;r=radius_for(block,sigma,candidate_radius);k=2*r+1
    assert h>0 and w>0
    y=y.to(torch.int32).contiguous();x=x.to(torch.int32).contiguous()
    out=torch.empty((k*k,h*w),device=y.device)
    original.launch('make_weights',[y,x,out],[h,w,r,np.float32(sigma),0,block],h*w)
    return out[None],k
