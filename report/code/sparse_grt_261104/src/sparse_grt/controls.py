"""Historical compact-grid and dense controls, exposed without monkey patches.

The formulas and input operations are retained from compact_ssim_adapter_261103.py
and dense_adapter.py. Fused SSIM uses its established 11x11, sigma=1.5 valid window.
"""
from dataclasses import replace
from .inputs import SelectedInputs


class GridInputs(SelectedInputs):
    def __init__(self, dataset, batch_factory, config, **kwargs):
        kwargs['weight_fn'] = lambda *args: (None, 11)
        super().__init__(dataset, batch_factory, config, **kwargs)
        for geometry in self.geometry.values():
            h, w = geometry['grid']
            if min(h, w) < 11:
                raise ValueError('Compact SSIM needs an 11x11 valid window')
            geometry.update(structural_loss='compact_ssim', candidate_grid=11,
                            valid_centres=(h-10)*(w-10), centre_domain='valid',
                            excluded_centre_fraction=1-(h-10)*(w-10)/(h*w),
                            sigma_pixels=None, sigma_over_block=None, sigma_compact_grid=1.5,
                            coordinate_sigma_config_unused=config.sigma)


class DenseInputs(SelectedInputs):
    def __init__(self, dataset, batch_factory, config, **kwargs):
        if config.views != 1 or config.block != 1:
            raise ValueError('Dense control requires one view and every pixel')
        super().__init__(dataset, batch_factory, replace(config, structure_weight=0.), **kwargs)
        for geometry in self.geometry.values():
            h, w = geometry['grid']
            if min(h, w) < 11:
                raise ValueError('Dense SSIM needs an 11x11 valid window')
            geometry.update(structural_loss='dense_ssim', candidate_grid=11,
                            valid_centres=(h-10)*(w-10), centre_domain='valid',
                            excluded_centre_fraction=1-(h-10)*(w-10)/(h*w),
                            sigma_pixels=1.5, sigma_over_block=1.5,
                            coordinate_sigma_config_unused=config.sigma)

    def make(self, views, seed, auxiliary='point_ssim'):
        if len(views) != 1:
            raise ValueError('Expected one dense view')
        vid = int(views[0])
        _, ro, rd, _ = self.cameras[self.intrinsics[vid]]
        h, w = rd.shape[1:3]
        transform = self.poses[vid][0]
        origins = (ro @ transform[:3, :3].T + transform[:3, 3]).reshape(1, 1, -1, 3).contiguous()
        directions = (rd @ transform[:3, :3].T).reshape(1, 1, -1, 3).contiguous()
        image = self.images[vid] if self.resident else self.items[vid]['data'][0].to(
            self.device, non_blocking=True)
        gt = (image/255.).reshape(1, 1, -1, 3)
        batch = self.batch_factory(rays_ori=origins, rays_dir=directions, rgb_gt=gt,
                                   T_to_world=self.identity, intrinsics=None)
        plan = dict(shape=(h, w), gt=gt.reshape(1, h, w, 3), weights=None,
                    k=11, start=0, end=h*w)
        return batch, None, [plan]


def grid_objective(pred, gt, plans, density, scale, config, backend=None):
    from fused_ssim import fused_ssim
    shapes = {tuple(p['shape']) for p in plans}
    if len(plans) != config.views or len(shapes) != 1:
        raise ValueError('Expected independent equally shaped compact grids')
    h, w = next(iter(shapes))
    if pred.numel() != len(plans)*h*w*3 or gt.shape != pred.shape:
        raise ValueError('RGB batch does not match compact grids')
    prediction = pred.reshape(len(plans), h, w, 3).permute(0, 3, 1, 2).contiguous()
    target = gt.reshape(len(plans), h, w, 3).permute(0, 3, 1, 2).contiguous()
    l1 = (pred-gt).abs().mean()
    structure = 1-fused_ssim(prediction, target, padding='valid')
    regularization = config.opacity_weight*density.abs().mean()+config.scale_weight*scale.abs().mean()
    return config.l1_weight*l1+config.structure_weight*structure+regularization, dict(
        L1=l1, compact_SSIM=structure, regularization=regularization)


def dense_objective(pred, gt, plans, density, scale, config, backend=None):
    from fused_ssim import fused_ssim
    h, w = plans[0]['shape']
    if len(plans) != 1 or pred.numel() != h*w*3:
        raise ValueError('Incomplete dense image')
    prediction = pred.reshape(1, h, w, 3).permute(0, 3, 1, 2).contiguous()
    target = gt.reshape(1, h, w, 3).permute(0, 3, 1, 2).contiguous()
    l1 = (pred-gt).abs().mean()
    structure = 1-fused_ssim(prediction, target, padding='valid')
    regularization = config.opacity_weight*density.abs().mean()+config.scale_weight*scale.abs().mean()
    return config.l1_weight*l1+config.structure_weight*structure+regularization, dict(
        L1=l1, dense_SSIM=structure, regularization=regularization)
