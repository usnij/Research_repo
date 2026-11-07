"""Run on an IDLE remote GPU, never alongside a timed training run.

Verifies default versus zero-weight-skipping kernels and sampled RGB caching.
This is a loss/input microbenchmark, not a Gaussian-gradient or training test.
"""
from pathlib import Path
import argparse
import json
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import torch
from sparse_grt.losses import coordinate_ssim as C
from sparse_grt.losses import experimental as E
from sparse_grt.sampling import StratifiedGrid


def main(output):
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(2);torch.manual_seed(261030)
    torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    for block,sigma in ((4,3.),(8,3.),(8,6.),(8,9.),(16,12.)):
        h,w=17,19
        grid=StratifiedGrid(h*block,w*block,block,"cuda")
        s=grid.sample(torch.Generator(device="cuda").manual_seed(261030))
        weights,k=C.weights(s.y.reshape(h,w),s.x.reshape(h,w),block,sigma)
        for case in ("random","flat","near_flat","identical","constant_residual"):
            gt=torch.rand(h,w,3,device="cuda");pred=torch.rand_like(gt)
            if case=="flat":pred.fill_(.4);gt.fill_(.5)
            elif case=="near_flat":pred=.5+1e-7*pred;gt=.5+1e-7*gt
            elif case=="identical":pred=gt.clone()
            elif case=="constant_residual":pred=gt+.1
            pred.requires_grad_()
            baseline=C.loss(pred,gt,weights,k);refgrad,=torch.autograd.grad(baseline,pred)
            new=E.loss(pred,gt,weights,k);grad,=torch.autograd.grad(new,pred)
            p64=pred.detach().double().requires_grad_()
            reference=C.reference_loss(p64,gt.double(),weights.double(),k)
            g64,=torch.autograd.grad(reference,p64)
            torch.testing.assert_close(new,baseline,atol=2e-7,rtol=2e-6)
            torch.testing.assert_close(grad,refgrad,atol=2e-7,rtol=2e-5)
            torch.testing.assert_close(grad.double(),g64,atol=2e-5,rtol=2e-4)
            rows.append(dict(block=block,sigma=sigma,case=case,
                        loss_delta=float(abs(new-baseline)),gradient_max_abs=float((grad-refgrad).abs().max()),
                        gradient_max_abs_vs_fp64=float((grad-g64).abs().max())))
    image=torch.randint(256,(1038,1558,3),dtype=torch.uint8,device="cuda")
    grid=StratifiedGrid(1038,1558,8,"cuda")
    sample=grid.sample(torch.Generator(device="cuda").manual_seed(261030))
    assert torch.equal((image/255.)[sample.y,sample.x],image[sample.y,sample.x]/255.)
    shape=grid.shape
    coords=[grid.sample(torch.Generator(device="cuda").manual_seed(261030+i)) for i in range(4)]
    rgb=[torch.rand(*shape,3,device="cuda",requires_grad=True) for _ in range(4)]
    gt=[torch.rand_like(v) for v in rgb]
    def run(backend):
        values=[]
        for p,g,s in zip(rgb,gt,coords):
            p.grad=None
            w,k=C.weights(s.y.reshape(shape),s.x.reshape(shape))
            values.append(backend.loss(p,g,w,k))
        torch.stack(values).mean().backward()
    for module in (C,E):
        for _ in range(5):run(module)
    measurements={"default":[],"skip_zero":[]}
    for rep in range(40):
        order=(("default",C),("skip_zero",E))
        if rep%2:order=order[::-1]
        for name,module in order:
            torch.cuda.synchronize();tick=time.perf_counter();run(module);torch.cuda.synchronize()
            measurements[name].append(1000*(time.perf_counter()-tick))
    result=dict(passed=True,device=torch.cuda.get_device_name(),cases=rows,
                selected_rgb_bit_equal=True,
                timing={k:dict(median_ms=statistics.median(v),samples_ms=v) for k,v in measurements.items()},
                scope="Weights+MV4 loss+RGB backward only. Not full training or Gaussian update parity.")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("cases","timing")}))
    print({k:v["median_ms"] for k,v in result["timing"].items()})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args().output)
