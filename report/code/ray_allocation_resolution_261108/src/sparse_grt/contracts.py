"""CPU preflight contracts for ray accounting and reproducible view schedules.

These checks describe the established objective; they do not change sampling,
loss weights, renderer mathematics, or optimizer updates.
"""
import hashlib
import json
import math
from pathlib import Path

from .sampling import view_sequence
from .losses.coordinate_ssim import radius_for


def candidate_radius(config):
    if getattr(config, 'sampler', 'point') == 'micropatch':
        # A physical support radius can cross D macrotiles. Their two virtual
        # rows/columns can differ by at most 2*D+1, including partial edge tiles.
        return 2*radius_for(6, config.sigma)+1
    return radius_for(config.block, config.sigma)


def ray_geometry(height, width, config):
    if type(height) is not int or type(width) is not int or min(height, width) <= 0:
        raise ValueError("Image dimensions must be positive integers")
    block = config.block
    rows, cols = math.ceil(height/block), math.ceil(width/block)
    rays = rows*cols
    radius = candidate_radius(config)
    boundary = getattr(config, 'boundary', 'valid')
    centres = rays if boundary == 'all' else max(0, rows-2*radius)*max(0, cols-2*radius)
    if config.structure_weight and not centres:
        raise ValueError("No valid structural-loss centres at this resolution/bandwidth")
    # Exact total-variation distance between equal-stratum weighting and
    # pixel-area weighting. It is a measure discrepancy, not a quality estimate.
    hh = [(block, rows-1), (height-block*(rows-1), 1)]
    ww = [(block, cols-1), (width-block*(cols-1), 1)]
    tv = .5*sum(nh*nw*abs(1/rays-h*w/(height*width))
                for h,nh in hh for w,nw in ww)
    result = dict(height=height, width=width, block=block, grid=[rows, cols],
                full_pixels=height*width, rays_per_view=rays,
                total_rays_if_equal_views=config.views*rays,
                actual_fraction_of_one_view=config.views*rays/(height*width),
                nominal_fraction_of_one_view=config.nominal_total_ray_fraction(),
                candidate_grid=2*radius+1, valid_centres=centres, centre_domain=boundary,
                excluded_centre_fraction=1-centres/rays,
                sigma_pixels=config.sigma, sigma_over_block=config.sigma/block,
                equal_stratum_vs_pixel_total_variation=tv)
    if getattr(config, 'sampler', 'point') == 'micropatch':
        patches = (height//6)*(width//6)
        result.update(sampler='micropatch', virtual_grid=True, macro_tile=[6, 6],
                      patch_shape=[2, 2], patch_count=patches, patch_rays=4*patches,
                      boundary_fallback_rays=rays-4*patches,
                      boundary_fallback_fraction=(rays-4*patches)/rays,
                      anchor_offsets=[0, 2, 4], anchor_probability=1/9,
                      interior_pixel_inclusion_probability=1/9,
                      edge_policy='legacy_3x3_points_in_partial_6x6_tiles',
                      coordinate_support='abs(delta_y),abs(delta_x)<=3*sigma',
                      physical_support_radius_pixels=3*config.sigma)
    return result


def validate_plans(plans, total_rays, view_count, boundary='valid'):
    """Check partition/index semantics without copying any tensor off GPU."""
    if len(plans) != view_count:
        raise ValueError("One independent structural-loss plan is required per view")
    offset = 0
    counts = []
    for p in plans:
        h,w = p['shape']; n = h*w
        if p['start'] != offset or p['end'] != offset+n:
            raise ValueError("View plans must partition the flattened ray batch exactly")
        if tuple(p['gt'].shape) != (1,h,w,3):
            raise ValueError("GT shape does not match the view plan")
        if p['weights'] is not None:
            if min(h,w) < (p['k'] if boundary == 'valid' else 1) or tuple(p['weights'].shape) != (1,p['k']**2,n):
                raise ValueError("Invalid structural support or weight shape")
        offset += n; counts.append(n)
    if offset != total_rays or len(set(counts)) != 1:
        raise ValueError("Current adapter requires equal ray counts and complete view coverage")


def _signature(config, view_count):
    return dict(view_count=view_count, views=config.views, planned_steps=config.steps,
                view_seed=config.view_seed, ray_seed=config.ray_seed)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def make_manifest(config, view_count):
    rows = [dict(step=t, views=v, ray_seed=config.ray_seed+t)
            for t,v in enumerate(view_sequence(view_count,config.steps,config.view_seed,config.views))]
    body = dict(schema=1, sequence='legacy_primary_shuffle_distinct_secondary',
                signature=_signature(config,view_count), rows=rows)
    return dict(**body, sha256=_digest(body))


def validate_manifest(manifest, config, view_count):
    body = {k:v for k,v in manifest.items() if k != 'sha256'}
    if manifest.get('sha256') != _digest(body):
        raise ValueError("Training manifest checksum mismatch")
    if manifest.get('schema') != 1 or manifest.get('signature') != _signature(config,view_count):
        raise ValueError("View/ray seeds or full schedule horizon differ from the saved manifest")
    rows = manifest['rows']
    if len(rows) != config.steps:
        raise ValueError("The saved manifest must cover the full planned training horizon")
    for t,row in enumerate(rows):
        views = row['views']
        if (row['step'] != t or row['ray_seed'] != config.ray_seed+t
                or len(views) != config.views or len(set(views)) != len(views)
                or any(type(v) is not int or not 0 <= v < view_count for v in views)):
            raise ValueError(f"Invalid view/ray assignment at step {t}")
    return rows


def read_manifest(path, config, view_count):
    manifest = json.loads(Path(path).read_text())
    validate_manifest(manifest,config,view_count)
    return manifest


def manifest_slice(manifest, start, stop):
    """A continuation slices a saved schedule, never regenerates a shorter one."""
    if not 0 <= start <= stop <= len(manifest['rows']):
        raise ValueError("Continuation is outside the original training horizon")
    return manifest['rows'][start:stop]
