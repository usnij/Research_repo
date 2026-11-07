"""CPU sampling-geometry diagnostics, without optimization or quality claims."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import torch
import torch.nn.functional as F
from sparse_grt.sampling import StratifiedGrid
from sparse_grt.losses import coordinate_ssim as C


def main(output):
    torch.set_num_threads(2)
    rows=[]
    for block,sigma in ((4,6.),(8,1.5),(8,4.),(8,6.),(8,8.),(8,12.),(16,6.),(16,12.)):
        h,w=32,40;grid=StratifiedGrid(h*block,w*block,block)
        values=[]
        for seed in range(16):
            s=grid.sample(torch.Generator().manual_seed(261030+seed))
            weight,k=C.reference_weights(s.y.reshape(h,w).double(),s.x.reshape(h,w).double(),block,sigma)
            r=k//2;core=weight.reshape(k*k,h,w)[:,r:h-r,r:w-r]
            effective=core.square().sum(0).reciprocal()
            # Sum all valid-centre weights that touch each observed ray.
            active=weight.clone().reshape(k*k,h,w)
            active[:,:r]=0;active[:,h-r:]=0;active[:,:,:r]=0;active[:,:,w-r:]=0
            mass=F.fold(active.reshape(1,k*k,h*w),(h,w),k,padding=r)[0,0]
            interior=mass[r:h-r,r:w-r]
            values.append(dict(ess=float(effective.mean()),self_mass=float(core[k*k//2].mean()),
                positive_neighbors=float((core>0).double().sum(0).mean()),
                neighbor_pair_mass=float((1-core.square().sum(0)).mean()),
                interior_incoming_weight_cv=float(interior.std()/interior.mean()),
                minimum_interior_incoming_weight=float(interior.min()),
                maximum_interior_incoming_weight=float(interior.max())))
        rows.append(dict(block=block,sigma=sigma,sigma_over_block=sigma/block,
                         candidates=k*k,**{key:sum(v[key] for v in values)/len(values) for key in values[0]}))
    h,w=1038,1558;b=8;n=((h+b-1)//b)*((w+b-1)//b)
    result=dict(scope="16 synthetic independent masks per condition, CPU; weights only, not quality/independent observations",
                rows=rows,counter=dict(full_pixels=h*w,rays_per_view=n,
                image_transfer_bytes_per_step=4*h*w*3,
                resident_uint8_cache_bytes=210*h*w*3,
                old_float_conversion_pixels=4*h*w,new_float_conversion_pixels=4*n,
                valid_centres=(130-6)*(195-6),centres=n,
                omitted_centre_fraction=1-(130-6)*(195-6)/n))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    main(p.parse_args().output)
