"""Check four independent compact SSIM windows and their RGB gradients."""
from pathlib import Path
import json, sys
import torch
import torch.nn.functional as F
import fused_ssim
from sparse_grt.config import MethodConfig
from sparse_grt.sampling import StratifiedGrid
from compact_ssim_adapter_261103 import compact_objective

torch.set_num_threads(2)
torch.manual_seed(261103)
block = int(sys.argv[1])
cfg = MethodConfig(views=4, block=block, l1_weight=.8, structure_weight=.2, boundary='all')
grid = StratifiedGrid(211, 269, block, 'cuda')
h,w = grid.shape
p = torch.rand(4,3,h,w,device='cuda',requires_grad=True)
y = torch.rand_like(p)
flat = p.permute(0,2,3,1).reshape(4,1,h*w,3).contiguous()
gt = y.permute(0,2,3,1).reshape(4,1,h*w,3).contiguous()
plans = [dict(shape=(h,w),start=i*h*w,end=(i+1)*h*w) for i in range(4)]
d = torch.tensor([.1,.3],device='cuda',requires_grad=True)
s = torch.tensor([.2,.4],device='cuda',requires_grad=True)
loss, parts = compact_objective(flat,gt,plans,d,s,cfg)
g=torch.arange(-5,6,device='cuda',dtype=p.dtype)
g=torch.exp(-g.square()/(2*1.5**2));g/=g.sum()
kernel=(g[:,None]*g[None,:]).expand(3,1,11,11).contiguous()
conv=lambda x:F.conv2d(x,kernel,groups=3)
mx,my=conv(p),conv(y)
vx,vy=conv(p*p)-mx*mx,conv(y*y)-my*my
cov=conv(p*y)-mx*my
ref_ssim=(((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))).mean()
ref_loss=.8*(p-y).abs().mean()+.2*(1-ref_ssim)+.01*d.abs().mean()+.01*s.abs().mean()
actual_grads=torch.autograd.grad(loss,(p,d,s),retain_graph=True)
reference_grads=torch.autograd.grad(ref_loss,(p,d,s),retain_graph=True)
per_view=torch.stack([fused_ssim.fused_ssim(p[i:i+1].contiguous(),y[i:i+1].contiguous(),padding='valid') for i in range(4)]).mean()
result=dict(block=block,window=11,sigma_compact_grid=1.5,padding='valid',views=4,
            loss_error=abs(float(loss-ref_loss)),
            gradient_max_errors=[float((a-b).abs().max()) for a,b in zip(actual_grads,reference_grads)],
            per_view_average_error=abs(float(parts['compact_SSIM']-(1-per_view))),
            positive_gradient_count_by_view=[int((x.abs()>0).sum()) for x in actual_grads[0]],
            fused_ssim_source=str(Path(fused_ssim.__file__).resolve()))
result['passed']=(result['loss_error']<2e-5 and max(result['gradient_max_errors'])<2e-5
                  and result['per_view_average_error']<2e-6
                  and min(result['positive_gradient_count_by_view'])>0)
Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
print(result)
assert result['passed']
