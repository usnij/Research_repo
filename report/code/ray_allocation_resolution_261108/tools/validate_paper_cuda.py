"""Check the FINAL all-centre objective, without timing or training claims."""
import argparse
import json
from pathlib import Path
# Use the installed package. Set PYTHONPATH=src explicitly for a working-tree check.
import torch
from sparse_grt.sampling import StratifiedGrid
from sparse_grt.losses import coordinate_ssim_all as C


def main(output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.manual_seed(20261102)
    rows = []
    for height, width, block in [(1,1,8),(11,17,8),(31,29,6),(80,96,8)]:
        grid = StratifiedGrid(height, width, block, 'cuda')
        sample = grid.sample(torch.Generator(device='cuda').manual_seed(777))
        shape = grid.shape
        y,x=sample.y.reshape(shape),sample.x.reshape(shape)
        weights,k=C.weights(y,x,block,6.)
        w64,k64=C.reference_weights(y.double(),x.double(),block,6.)
        assert k==k64
        torch.testing.assert_close(weights.double(),w64,atol=2e-7,rtol=2e-6)
        for case in ('random','identical','flat','near_flat','constant_residual'):
            gt=torch.rand(*shape,3,device='cuda');pred=torch.rand_like(gt)
            if case=='identical':pred=gt.clone()
            elif case=='flat':pred.fill_(.4);gt.fill_(.5)
            elif case=='near_flat':pred=.5+pred*1e-7;gt=.5+gt*1e-7
            elif case=='constant_residual':pred=gt+.1
            pred.requires_grad_()
            loss=C.loss(pred,gt,weights,k)
            grad,=torch.autograd.grad(loss,pred)
            p64=pred.detach().double().requires_grad_()
            ref=C.reference_loss(p64,gt.double(),weights.double(),k)
            rg,=torch.autograd.grad(ref,p64)
            torch.testing.assert_close(loss.double(),ref,atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(grad.double(),rg,atol=2e-5,rtol=2e-4)
            rows.append(dict(image=[height,width],block=block,case=case,
                             loss_abs_error=float(abs(loss-ref)),
                             gradient_max_abs_error=float((grad-rg).abs().max()),
                             weight_max_abs_error=float((weights-w64).abs().max())))
    # One centre's statistical objective must reach its neighboring samples.
    y,x=torch.meshgrid(torch.arange(3,device='cuda')*8,torch.arange(3,device='cuda')*8,indexing='ij')
    w,k=C.weights(y,x,8,6.)
    gt=torch.rand(3,3,3,device='cuda');pred=gt.clone();pred[1,1]+=.07
    pred.requires_grad_();value=C.loss(pred,gt,w,k);g,=torch.autograd.grad(value,pred)
    assert torch.count_nonzero(g[0,0])>0
    result=dict(passed=True,device=torch.cuda.get_device_name(),cases=rows,
                cases_passed=len(rows),neighbor_gradient_verified=True,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                scope='Final boundary=all CUDA objective and RGB gradients; no training speed claim.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
