"""Input/loss adapter for complete observed 2x2 patches only."""
import torch
from .inputs import SelectedInputs
from .losses.patch_ssim import loss


class PatchInputs(SelectedInputs):
    def __init__(self, dataset, batch_factory, config, **kwargs):
        if config.sampler != 'micropatch':
            raise ValueError('Actual patch SSIM requires the micropatch sampler')
        # No coordinate weights are allocated for this arm; the sampled rays and
        # the generator calls stay identical to the two coordinate-SSIM arms.
        kwargs['weight_fn'] = lambda *args, **kw: (None, 1)
        super().__init__(dataset, batch_factory, config, **kwargs)
        for geometry in self.geometry.values():
            if geometry['patch_count'] == 0:
                raise ValueError('Patch SSIM requires a complete 6x6 sampling tile')
            geometry.update(structural_loss='actual_2x2_patch_ssim', candidate_grid=2,
                            valid_centres=geometry['patch_count'],
                            centre_domain='complete_2x2_patch_centres',
                            excluded_centre_fraction=geometry['boundary_fallback_fraction'],
                            sigma_pixels=None, sigma_over_block=None,
                            physical_support_radius_pixels=None,
                            coordinate_support=None,
                            coordinate_sigma_config_unused=config.sigma,
                            structural_samples_per_patch=4, structural_pixel_weight=.25,
                            structural_boundary_policy='exclude_all_fallback_rays')


def patch_objective(pred, gt, plans, density, scale, config, backend=None):
    if config.sampler != 'micropatch' or len(plans) != config.views:
        raise ValueError('Expected independent micropatch view plans')
    flat = pred.reshape(-1, 3)
    l1 = (pred-gt).abs().mean()
    structure = torch.stack([
        loss(flat[p['start']:p['end']], p['gt'].reshape(-1, 3), p['patch_indices'])
        for p in plans
    ]).mean() if config.structure_weight else pred.new_zeros(())
    regularization = config.opacity_weight*density.abs().mean()+config.scale_weight*scale.abs().mean()
    total = config.l1_weight*l1+config.structure_weight*structure+regularization
    return total, dict(L1=l1, patch_SSIM=structure, regularization=regularization)
