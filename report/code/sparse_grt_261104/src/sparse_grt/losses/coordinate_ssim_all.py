"""All-centre observed-coordinate SSIM. Explicitly changes boundary objective.

No extra RGB samples, GT moments, or parameter-dependent neighborhood weights.
The CUDA implementation differentiates the complete overlapping neighborhood
objective; it does not restrict derivatives to window centres.
"""
from pathlib import Path
import math
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
_kernels = {}


def radius_for(block, sigma):
    # Samples from strata d cells apart can be (d-1)*block+1 pixels apart.
    # Integer pixel coordinates make this the complete candidate envelope.
    assert isinstance(block,int) and block>0 and math.isfinite(sigma) and sigma>0
    return math.floor((3*sigma+block-1)/block)


def reference_weights(y, x, block=8, sigma=6., nominal=False):
    r = radius_for(block, sigma); k = 2*r+1
    if nominal:
        yy, xx = torch.meshgrid(torch.arange(y.shape[0], device=y.device),
                                torch.arange(y.shape[1], device=y.device), indexing='ij')
        y, x = yy*block, xx*block
    coords = torch.stack((y, x)).to(torch.float64 if y.dtype == torch.float64 else torch.float32)[None]
    neighbors = F.unfold(coords, k, padding=r).reshape(1, 2, k*k, -1)
    delta = neighbors-coords.reshape(1,2,1,-1)
    valid = F.unfold(torch.ones_like(coords[:,:1]), k, padding=r) > .5
    valid &= (delta.abs() <= 3*sigma).all(1)
    w = torch.exp(-delta.square().sum(1)/(2*sigma*sigma))*valid
    return (w/w.sum(1,keepdim=True)).detach(), k


def reference_loss(pred, gt, weights, k, residual=True):
    """HWC reference. Empirical variances, no Bessel/HT correction or clipping."""
    h,w,c=pred.shape; r=k//2
    xp=F.unfold(pred.permute(2,0,1)[None],k,padding=r).reshape(c,k*k,h*w)
    yp=F.unfold(gt.permute(2,0,1)[None],k,padding=r).reshape(c,k*k,h*w)
    wx=weights[0];mx=(xp*wx).sum(1);my=(yp*wx).sum(1)
    if residual:
        dx=xp-mx[:,None];dy=yp-my[:,None]
        vx=(wx*dx.square()).sum(1);vy=(wx*dy.square()).sum(1)
        ve=(wx*(dx-dy).square()).sum(1)
        a=(mx-my).square()/(mx.square()+my.square()+.0001)
        b=ve/(vx+vy+.0009)
        values=a+b-a*b
    else:
        vx=(wx*xp.square()).sum(1)-mx.square()
        vy=(wx*yp.square()).sum(1)-my.square()
        cov=(wx*xp*yp).sum(1)-mx*my
        values=1-(2*mx*my+.0001)*(2*cov+.0009)/((mx.square()+my.square()+.0001)*(vx+vy+.0009))
    return values.mean()


def kernel(name):
    if name not in _kernels:
        import cupy as cp
        _kernels[name]=cp.RawKernel((HERE/'coordinate_ssim_all.cu').read_text(),name,
                                    options=('--std=c++14','--fmad=false'))
    return _kernels[name]


def launch(name, tensors, scalars, n):
    import cupy as cp
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        kernel(name)(((n+127)//128,),(128,),
                     tuple(cp.from_dlpack(t.detach()) for t in tensors)+tuple(scalars))


def weights(y,x,block=8,sigma=6.,nominal=False):
    import numpy as np
    assert y.ndim==x.ndim==2 and y.shape==x.shape and y.is_cuda and x.device==y.device
    h,w=y.shape;r=radius_for(block,sigma);k=2*r+1
    assert h>0 and w>0
    y=y.to(torch.int32).contiguous();x=x.to(torch.int32).contiguous()
    out=torch.empty((k*k,h*w),device=y.device)
    launch('make_weights',[y,x,out],[h,w,r,np.float32(sigma),int(nominal),block],h*w)
    return out[None],k


class _Loss(torch.autograd.Function):
    @staticmethod
    def forward(ctx,pred,gt,weight,k):
        assert pred.dtype==gt.dtype==weight.dtype==torch.float32
        assert pred.is_cuda and pred.shape==gt.shape and pred.ndim==3 and pred.shape[-1]==3
        assert not gt.requires_grad and not weight.requires_grad
        assert gt.device==weight.device==pred.device and k>0 and k%2==1
        assert weight.shape==(1,k*k,pred.shape[0]*pred.shape[1])
        assert min(pred.shape[:2])>0
        pred=pred.contiguous();gt=gt.contiguous();weight=weight.contiguous()
        h,w,c=pred.shape;values=torch.empty_like(pred);stats=torch.empty((h*w*c,5),device=pred.device)
        launch('ssim_forward',[pred,gt,weight,values,stats],[h,w,k//2],h*w*c)
        ctx.save_for_backward(pred,gt,weight,stats);ctx.k=k
        return values.mean()

    @staticmethod
    def backward(ctx,grad):
        pred,gt,weight,stats=ctx.saved_tensors;h,w,c=pred.shape
        result=torch.empty_like(pred)
        launch('ssim_backward',[pred,gt,weight,stats,grad.contiguous(),result],[h,w,ctx.k//2],h*w*c)
        return result,None,None,None


def loss(pred,gt,weight,k):
    return _Loss.apply(pred,gt,weight,k)


def support_summary(weight,k,shape):
    h,w=shape;r=k//2
    ww=weight[0].reshape(k*k,-1)
    n_eff=ww.square().sum(0).reciprocal();self_weight=ww[k*k//2]
    def stats(v):
        vf=v.float()
        q=torch.quantile(vf,vf.new_tensor([0.,.1,.5,.9,1.]))
        return dict(mean=float(vf.mean()),quantiles=dict(zip(['min','p10','median','p90','max'],q.tolist())))
    return dict(centres=ww.shape[1],effective_samples=stats(n_eff),centre_mass=stats(self_weight),
                valid_neighbors=stats((ww>0).sum(0)),mass_error=float((ww.sum(0)-1).abs().max()))
