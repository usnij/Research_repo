"""Check fused SSIM against an explicit 11x11 Gaussian-window reference."""
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
from fused_ssim import fused_ssim

torch.manual_seed(261102)
x=torch.rand(1,3,35,43,device='cuda',requires_grad=True)
y=torch.rand_like(x)
g=torch.arange(-5,6,device='cuda',dtype=x.dtype)
g=torch.exp(-g.square()/(2*1.5**2));g=g/g.sum()
kernel=(g[:,None]*g[None,:]).expand(3,1,11,11).contiguous()
conv=lambda z:F.conv2d(z,kernel,groups=3)
mx,my=conv(x),conv(y)
vx,vy=conv(x*x)-mx*mx,conv(y*y)-my*my
cov=conv(x*y)-mx*my
ref=(((2*mx*my+0.01**2)*(2*cov+0.03**2))/
     ((mx*mx+my*my+0.01**2)*(vx+vy+0.03**2))).mean()
value=fused_ssim(x,y,padding='valid')
grad_ref=torch.autograd.grad(ref,x,retain_graph=True)[0]
grad=torch.autograd.grad(value,x)[0]
result=dict(value_error=abs(float(ref-value)),gradient_max_error=float((grad-grad_ref).abs().max()))
result['passed']=result['value_error']<2e-5 and result['gradient_max_error']<2e-5
Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
print(result)
assert result['passed']
