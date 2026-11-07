"""Explicit idle-GPU loss check only: no dataset, renderer, model or training.

Run manually with --output. This file is outside unittest's test_* discovery.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import torch
from sparse_grt.config import MethodConfig
from sparse_grt.contracts import candidate_radius, ray_geometry
from sparse_grt.sampling import MicropatchGrid, StratifiedGrid
from sparse_grt.losses import micropatch_coordinate as C
from sparse_grt.losses.patch_ssim import loss as patch_loss


def coordinate_reference(pred, gt, y, x, sigma):
    coords=torch.stack((y.reshape(-1),x.reshape(-1)),-1).double()
    delta=coords[:,None]-coords[None,:]
    weight=torch.exp(-delta.square().sum(-1)/(2*sigma*sigma))
    weight*=(delta.abs()<=3*sigma).all(-1)
    weight/=weight.sum(1,keepdim=True)
    p,g=pred.reshape(-1,3),gt.reshape(-1,3)
    mp,mg=weight@p,weight@g
    dp,dg=p[None]-mp[:,None],g[None]-mg[:,None]
    vp=(weight[:,:,None]*dp.square()).sum(1)
    vg=(weight[:,:,None]*dg.square()).sum(1)
    covariance=(weight[:,:,None]*dp*dg).sum(1)
    return (1-(2*mp*mg+.0001)*(2*covariance+.0009)/
            ((mp.square()+mg.square()+.0001)*(vp+vg+.0009))).mean()


def patch_reference(pred,gt,indices):
    x,y=pred.reshape(-1,3)[indices],gt.reshape(-1,3)[indices]
    mx,my=x.mean(1),y.mean(1);dx,dy=x-mx[:,None],y-my[:,None]
    vx,vy=dx.square().mean(1),dy.square().mean(1)
    covariance=(dx*dy).mean(1)
    return (1-(2*mx*my+.0001)*(2*covariance+.0009)/
            ((mx.square()+my.square()+.0001)*(vx+vy+.0009))).mean()


def check(actual,pred,reference,p64):
    gradient,=torch.autograd.grad(actual,pred)
    expected,=torch.autograd.grad(reference,p64)
    assert torch.isfinite(actual) and torch.isfinite(gradient).all()
    torch.testing.assert_close(actual.detach().cpu().double(),reference.detach(),atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(gradient.cpu().double(),expected,atol=2e-5,rtol=2e-4)
    return dict(loss=float(actual.detach()),loss_abs_error=float(abs(actual.detach().cpu()-reference.detach())),
                gradient_max_abs_error=float((gradient.cpu().double()-expected).abs().max()),
                finite=True),gradient


def main(output):
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    generator=torch.Generator(device='cuda').manual_seed(261105)
    grid=MicropatchGrid(19,25,device='cuda')
    sample=grid.sample(torch.Generator(device='cuda').manual_seed(777))
    old=StratifiedGrid(19,25,3,'cuda').sample(torch.Generator(device='cuda').manual_seed(777))
    assert sample.y.numel()==7*9 and torch.unique(sample.y*25+sample.x).numel()==7*9
    assert ((sample.y>=0)&(sample.y<19)&(sample.x>=0)&(sample.x<25)).all()
    for key in ('y','x'):
        assert torch.equal(getattr(sample,key)[grid.boundary_mask],getattr(old,key)[grid.boundary_mask])
    py,px=sample.y[grid.patch_indices],sample.x[grid.patch_indices]
    assert torch.equal(py-py[:,:1],grid.local_y.expand_as(py))
    assert torch.equal(px-px[:,:1],grid.local_x.expand_as(px))
    y,x=sample.y.reshape(grid.shape),sample.x.reshape(grid.shape)
    rows=[]
    for sigma in (1.5,2.):
        method=MethodConfig(views=1,block=3,boundary='all',sampler='micropatch',sigma=sigma)
        radius=candidate_radius(method)
        weight,k=C.weights(y,x,3,sigma,candidate_radius=radius)
        w64,refk=C.reference_weights(y.cpu().double(),x.cpu().double(),3,sigma,candidate_radius=radius)
        assert k==refk==2*radius+1
        torch.testing.assert_close(weight.cpu().double(),w64,atol=2e-7,rtol=2e-5)
        for case in ('random','identical','flat','constant_residual','near_flat'):
            gt=torch.rand(*grid.shape,3,device='cuda',generator=generator)
            pred=torch.rand(*grid.shape,3,device='cuda',generator=generator)
            if case=='identical':pred=gt.clone()
            elif case=='flat':pred.fill_(.4);gt.fill_(.5)
            elif case=='constant_residual':pred=gt+.1
            elif case=='near_flat':pred=.5+pred*1e-7;gt=.5+gt*1e-7
            pred.requires_grad_();p64=pred.detach().cpu().double().requires_grad_()
            actual=C.loss(pred,gt,weight,k)
            reference=coordinate_reference(p64,gt.cpu().double(),y.cpu(),x.cpu(),sigma)
            result,_=check(actual,pred,reference,p64)
            rows.append(dict(kind='coordinate',sigma=sigma,candidate_radius=radius,case=case,**result))
    # Fractional physical cutoff 4.5: include actual (4,4), exclude either axis5.
    axis=torch.tensor([2,3,6,7],device='cuda')
    ay,ax=torch.meshgrid(axis,axis,indexing='ij')
    aw,ak=C.weights(ay,ax,3,1.5)
    assert ak==7 and aw[0,(2+3)*7+2+3,0]>0
    assert aw[0,3*7+3+3,0]==0 and aw[0,(3+3)*7+3,0]==0
    # R3 is required: virtual delta (3,3) can be physical delta (3,3).
    axis=torch.tensor([4,5,6,7],device='cuda')
    ay,ax=torch.meshgrid(axis,axis,indexing='ij')
    aw,ak=C.weights(ay,ax,3,1.5)
    assert ak==7 and aw[0,(3+3)*7+3+3,0]>0
    for case in ('random','identical','flat','constant_residual','near_flat'):
        gt=torch.rand(*grid.shape,3,device='cuda',generator=generator)
        pred=torch.rand(*grid.shape,3,device='cuda',generator=generator)
        if case=='identical':pred=gt.clone()
        elif case=='flat':pred.fill_(.4);gt.fill_(.5)
        elif case=='constant_residual':pred=gt+.1
        elif case=='near_flat':pred=.5+pred*1e-7;gt=.5+gt*1e-7
        pred.requires_grad_();p64=pred.detach().cpu().double().requires_grad_()
        actual=patch_loss(pred,gt,grid.patch_indices)
        reference=patch_reference(p64,gt.cpu().double(),grid.patch_indices.cpu())
        result,gradient=check(actual,pred,reference,p64)
        assert torch.count_nonzero(gradient.reshape(-1,3)[grid.boundary_mask])==0
        rows.append(dict(kind='actual_2x2_patch',case=case,**result))
    scenes=[]
    method=MethodConfig(views=1,block=3,boundary='all',sampler='micropatch',sigma=2.)
    for name,h,w in (('garden',840,1297),('bicycle',822,1237),('bonsai',1039,1559),('kitchen',1039,1558)):
        geometry=ray_geometry(h,w,method)
        scenes.append(dict(scene=name,rays=geometry['rays_per_view'],
                           fallback=geometry['boundary_fallback_rays'],patches=geometry['patch_count']))
    result=dict(passed=True,device=torch.cuda.get_device_name(),torch_version=torch.__version__,
                scope='Small synthetic loss/RGB backward and sampler check only; no training or renderer.',
                small_sampler_unique_in_bounds=True,legacy_edge_pairing=True,patch_contiguity=True,
                sigma1p5_fractional_cutoff=True,sigma1p5_required_radius3_neighbor=True,
                scene_ray_geometry=scenes,cases=rows)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','scene_ray_geometry')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
