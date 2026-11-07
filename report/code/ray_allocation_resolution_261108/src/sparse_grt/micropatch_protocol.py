"""The authorized 12-run single-view micropatch experiment, 2026-11-05."""
from pathlib import Path
from .protocol import CHECKPOINTS, StepLog, validate_gpu, verify_checkpoint
from .native5_protocol import configure_native_add
from .contracts import candidate_radius


BASE_CAPS = dict(garden=2044672, bicycle=3021939, bonsai=1209886, kitchen=1590685)
DOWNSAMPLE = dict(garden=4, bicycle=4, bonsai=2, kitchen=2)


def validate(method, args):
    if (method.steps != 30000 or method.initialization_seed != 777 or method.views != 1
            or method.view_seed != 2610297100 or method.ray_seed != 910000000):
        raise ValueError('Micropatch comparison requires 30k updates and the paired native5 seeds')
    if method.sampler != 'micropatch' or method.block != 3 or method.boundary != 'all':
        raise ValueError('Require micropatches, the exact block3 ray budget and all coordinate centres')
    if args.growth != 'native_5percent' or args.growth_end is not None or args.budget != 'fixed':
        raise ValueError('Keep native 5% growth and the previously registered fixed cap')
    scene = Path(args.scene).name
    if scene not in BASE_CAPS or method.cap != BASE_CAPS[scene] or args.downsample != DOWNSAMPLE[scene]:
        raise ValueError('Scene, cap or downsample differs from the registered four-scene comparison')
    for key, value in dict(l1_weight=.8, structure_weight=.2,
                           opacity_weight=.01, scale_weight=.01).items():
        if getattr(method, key) != value:
            raise ValueError(f'Micropatch comparison requires {key}={value}')
    if method.grid_window != 11 or method.grid_sigma != 1.5:
        raise ValueError('Keep unused legacy grid settings fixed')
    if args.structural_loss == 'coordinate':
        if method.sigma not in (1.5, 2.):
            raise ValueError('Coordinate micropatch arms require sigma1.5 or sigma2')
    elif args.structural_loss == 'patch':
        if method.sigma != 2.:
            raise ValueError('Patch SSIM uses sigma2 only as an unused configuration placeholder')
    else:
        raise ValueError('Only coordinate sigma1.5/sigma2 and actual 2x2 patch SSIM are authorized')


def record(method, structural_loss):
    patch = structural_loss == 'patch'
    return dict(sampler='micropatch', sampling_macro_tile=[6, 6], observed_patch=[2, 2],
                anchor_offsets=[0, 2, 4], anchor_probability=1/9,
                interior_pixel_inclusion_probability=1/9,
                edge_policy='legacy_3x3_points_in_partial_6x6_tiles',
                exact_ray_count='ceil(H/3)*ceil(W/3)',
                boundary_rng='legacy_point_draws_preserved_before_private_anchor_draws',
                coordinate_candidate_radius=None if patch else candidate_radius(method),
                coordinate_physical_support=None if patch else 3*method.sigma,
                coordinate_support_shape=None if patch else 'axis_aligned_square',
                coordinate_sigma_config_unused=method.sigma if patch else None,
                patch_loss=(dict(support='four_actual_observed_contiguous_pixels', weights=[.25]*4,
                    moments='population', C1=.0001, C2=.0009, scale='1-SSIM',
                    reduction='mean_over_complete_patches_and_RGB_channels',
                    boundary='fallback_rays_L1_only') if patch else None),
                ssim_backend='torch_observed_patch_population_moments' if patch else 'coordinate_ssim_all',
                observed_only=False, observed_RGB_only=True, full_model_regularization=True,
                native_5percent_growth_preserved=True,
                forced_cap_arrival=False)
