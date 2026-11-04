"""Opt-in CUDA candidate; no measured speed or training-quality claim yet.

Exactly zero weights already occur outside the existing spatial cutoff.
Skipping them is valid for finite RGB values, without shortening the support.
"""
from pathlib import Path
import torch
from .coordinate_ssim import weights

_kernels = {}


def launch(name, tensors, scalars, n):
    import cupy as cp
    if name not in _kernels:
        source = Path(__file__).with_name("coordinate_ssim_skip_zero.cu").read_text()
        _kernels[name] = cp.RawKernel(source, name, options=("--std=c++14","--fmad=false"))
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        _kernels[name](((n+127)//128,),(128,),
                      tuple(cp.from_dlpack(t.detach()) for t in tensors)+tuple(scalars))


class _SkipZero(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, gt, weight, k):
        if not (pred.is_cuda and pred.dtype == gt.dtype == weight.dtype == torch.float32
                and pred.shape == gt.shape and pred.ndim == 3 and pred.shape[-1] == 3
                and gt.device == weight.device == pred.device and k > 0 and k%2 == 1
                and min(pred.shape[:2]) >= k
                and weight.shape == (1,k*k,pred.shape[0]*pred.shape[1])
                and not gt.requires_grad and not weight.requires_grad):
            raise ValueError("Expected finite CUDA float32 HWC RGB and detached weights/GT")
        pred,gt,weight = pred.contiguous(),gt.contiguous(),weight.contiguous()
        h,w,c = pred.shape
        values = torch.empty_like(pred)
        stats = torch.empty((h*w*c,5),device=pred.device)
        launch("ssim_forward",[pred,gt,weight,values,stats],[h,w,k//2],h*w*c)
        ctx.save_for_backward(pred,gt,weight,stats);ctx.k=k
        return values.sum()/((h-k+1)*(w-k+1)*c)

    @staticmethod
    def backward(ctx, grad):
        pred,gt,weight,stats=ctx.saved_tensors;h,w,c=pred.shape
        result=torch.empty_like(pred)
        launch("ssim_backward",[pred,gt,weight,stats,grad.contiguous(),result],
               [h,w,ctx.k//2],h*w*c)
        return result,None,None,None


loss = _SkipZero.apply
