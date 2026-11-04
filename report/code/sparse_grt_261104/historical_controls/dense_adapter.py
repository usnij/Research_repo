"""Matched-cap Dense MCMC control, using the frozen trainer/renderer unchanged.

Only the input adapter and image objective are replaced in this process. The
installed sparse package and active sparse jobs are never modified.
"""
from dataclasses import replace
import torch
from fused_ssim import fused_ssim
from sparse_grt.inputs import SelectedInputs
import sparse_grt.training as training
import sparse_grt.engine as engine


class DenseInputs(SelectedInputs):
    def __init__(self,dataset,batch_factory,config,**kwargs):
        if config.views!=1 or config.block!=1:
            raise ValueError('Dense control requires one view and every pixel')
        super().__init__(dataset,batch_factory,replace(config,structure_weight=0.),**kwargs)

    def make(self,views,seed,auxiliary='point_ssim'):
        if len(views)!=1: raise ValueError('Expected one dense view')
        vid=int(views[0]); _,ro,rd,_=self.cameras[self.intrinsics[vid]]
        h,w=rd.shape[1:3]; transform=self.poses[vid][0]
        origins=(ro @ transform[:3,:3].T+transform[:3,3]).reshape(1,1,-1,3).contiguous()
        directions=(rd @ transform[:3,:3].T).reshape(1,1,-1,3).contiguous()
        image=self.images[vid] if self.resident else self.items[vid]['data'][0].to(self.device,non_blocking=True)
        gt=(image/255.).reshape(1,1,-1,3)
        batch=self.batch_factory(rays_ori=origins,rays_dir=directions,rgb_gt=gt,
                                T_to_world=self.identity,intrinsics=None)
        plan=dict(shape=(h,w),gt=gt.reshape(1,h,w,3),weights=None,k=11,start=0,end=h*w)
        return batch,None,[plan]


def dense_objective(pred,gt,plans,density,scale,config,backend=None):
    h,w=plans[0]['shape']
    if len(plans)!=1 or pred.numel()!=h*w*3: raise ValueError('Incomplete dense image')
    prediction=pred.reshape(1,h,w,3).permute(0,3,1,2).contiguous()
    target=gt.reshape(1,h,w,3).permute(0,3,1,2).contiguous()
    l1=(pred-gt).abs().mean()
    structure=1-fused_ssim(prediction,target,padding='valid')
    regularization=config.opacity_weight*density.abs().mean()+config.scale_weight*scale.abs().mean()
    return config.l1_weight*l1+config.structure_weight*structure+regularization,dict(
        L1=l1,dense_SSIM=structure,regularization=regularization)


if __name__=='__main__':
    training.SelectedInputs=DenseInputs
    engine.objective=dense_objective
    training.cli()
