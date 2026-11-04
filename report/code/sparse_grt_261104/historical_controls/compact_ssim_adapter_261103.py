"""Compact-grid SSIM control; sampling, renderer and MCMC remain frozen."""
import torch
from fused_ssim import fused_ssim
from sparse_grt.inputs import SelectedInputs
import sparse_grt.training as training
import sparse_grt.engine as engine


class CompactInputs(SelectedInputs):
    def __init__(self, dataset, batch_factory, config, **kwargs):
        # Compact SSIM uses grid adjacency, not distances between sampled pixels.
        # Keep the identical sampler and RNG calls; omit unused coordinate weights.
        kwargs['weight_fn'] = lambda *args: (None, 11)
        super().__init__(dataset, batch_factory, config, **kwargs)
        for g in self.geometry.values():
            h, w = g['grid']
            if min(h, w) < 11:
                raise ValueError('Compact SSIM needs an 11x11 valid window')
            g.update(structural_loss='compact_ssim', candidate_grid=11,
                     valid_centres=(h-10)*(w-10), centre_domain='valid',
                     excluded_centre_fraction=1-(h-10)*(w-10)/(h*w),
                     sigma_pixels=None, sigma_over_block=None, sigma_compact_grid=1.5,
                     coordinate_sigma_config_unused=config.sigma)


def compact_objective(pred, gt, plans, density, scale, config, backend=None):
    shapes = {tuple(p['shape']) for p in plans}
    if len(plans) != config.views or len(shapes) != 1:
        raise ValueError('Expected independent equally shaped compact grids')
    h, w = next(iter(shapes))
    if pred.numel() != len(plans)*h*w*3 or gt.shape != pred.shape:
        raise ValueError('RGB batch does not match compact grids')
    prediction = pred.reshape(len(plans), h, w, 3).permute(0,3,1,2).contiguous()
    target = gt.reshape(len(plans), h, w, 3).permute(0,3,1,2).contiguous()
    l1 = (pred-gt).abs().mean()
    structure = 1-fused_ssim(prediction, target, padding='valid')
    regularization = config.opacity_weight*density.abs().mean()+config.scale_weight*scale.abs().mean()
    return config.l1_weight*l1+config.structure_weight*structure+regularization, dict(
        L1=l1, compact_SSIM=structure, regularization=regularization)


if __name__ == '__main__':
    training.SelectedInputs = CompactInputs
    engine.objective = compact_objective
    training.cli()
