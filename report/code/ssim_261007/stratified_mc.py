"""Stratified Monte Carlo sampling over a discrete full-resolution pixel grid.

This module selects one random pixel per non-overlapping rectangular stratum.
It does not resize the camera or jitter rays inside pixels.  The returned area
weights yield an unbiased estimator of a dense per-pixel mean, including when
the image dimensions are not divisible by the stratum size.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StratifiedSamples:
    """Pixel indices and full-image mean weights for one MC draw."""

    y: torch.Tensor
    x: torch.Tensor
    weights: torch.Tensor
    block_y: torch.Tensor
    block_x: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.y.numel())


def sample_stratified_pixels(
    height: int,
    width: int,
    block_height: int = 8,
    block_width: int = 8,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> StratifiedSamples:
    """Draw one uniformly random discrete pixel from each rectangular block."""

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if block_height <= 0 or block_width <= 0:
        raise ValueError("block dimensions must be positive")

    device = torch.device(device)
    y0 = torch.arange(0, height, block_height, device=device, dtype=torch.long)
    x0 = torch.arange(0, width, block_width, device=device, dtype=torch.long)
    block_y, block_x = torch.meshgrid(y0, x0, indexing="ij")
    block_y = block_y.reshape(-1)
    block_x = block_x.reshape(-1)

    valid_h = torch.clamp(height - block_y, max=block_height)
    valid_w = torch.clamp(width - block_x, max=block_width)

    random_y = torch.rand(block_y.shape, generator=generator, device=device)
    random_x = torch.rand(block_x.shape, generator=generator, device=device)
    offset_y = torch.floor(random_y * valid_h).to(torch.long)
    offset_x = torch.floor(random_x * valid_w).to(torch.long)

    y = block_y + offset_y
    x = block_x + offset_x
    block_area = valid_h * valid_w
    weights = block_area.to(torch.float64) / float(height * width)

    return StratifiedSamples(
        y=y,
        x=x,
        weights=weights,
        block_y=block_y,
        block_x=block_x,
    )


def stratified_mc_mean(values: torch.Tensor, samples: StratifiedSamples) -> torch.Tensor:
    """Estimate the dense mean of a scalar `[H, W]` tensor."""

    if values.ndim != 2:
        raise ValueError("values must be a scalar image with shape [H, W]")
    sampled_values = values[samples.y, samples.x]
    weights = samples.weights.to(device=values.device, dtype=values.dtype)
    return torch.sum(sampled_values * weights)


def samples_to_mask(
    height: int,
    width: int,
    samples: StratifiedSamples,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Scatter the samples to a boolean `[H, W]` mask."""

    target_device = torch.device(device) if device is not None else samples.y.device
    mask = torch.zeros((height, width), dtype=torch.bool, device=target_device)
    mask[samples.y.to(target_device), samples.x.to(target_device)] = True
    return mask


def sample_tile_coherent_pixels(
    height: int,
    width: int,
    micro_height: int = 8,
    micro_width: int = 8,
    macro_micro_height: int = 8,
    macro_micro_width: int = 8,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> StratifiedSamples:
    """Select one random micro-tile per macroblock and keep all its pixels.

    With the default 8x8 micro-tile and 8x8 micro-tiles per macroblock, this
    selects one 8x8 patch from every 64x64 macroblock: nominally 1/64 rays.
    Every pixel in a macroblock has the same inclusion probability when all
    micro-tiles are complete. Partial border micro-tiles use the exact
    Horvitz--Thompson weight for uniform micro-tile selection.
    """
    if min(height, width, micro_height, micro_width, macro_micro_height, macro_micro_width) <= 0:
        raise ValueError("image and tile dimensions must be positive")
    device = torch.device(device)
    micro_rows = (height + micro_height - 1) // micro_height
    micro_cols = (width + micro_width - 1) // micro_width
    group_y0 = torch.arange(0, micro_rows, macro_micro_height, device=device)
    group_x0 = torch.arange(0, micro_cols, macro_micro_width, device=device)
    group_y0, group_x0 = torch.meshgrid(group_y0, group_x0, indexing="ij")
    group_y0 = group_y0.reshape(-1)
    group_x0 = group_x0.reshape(-1)
    valid_rows = torch.clamp(micro_rows - group_y0, max=macro_micro_height)
    valid_cols = torch.clamp(micro_cols - group_x0, max=macro_micro_width)
    choices = torch.floor(
        torch.rand(group_y0.shape, generator=generator, device=device) * (valid_rows * valid_cols)
    ).long()
    selected_micro_y = group_y0 + choices // valid_cols
    selected_micro_x = group_x0 + choices % valid_cols
    selected_y0 = selected_micro_y * micro_height
    selected_x0 = selected_micro_x * micro_width

    oy, ox = torch.meshgrid(
        torch.arange(micro_height, device=device),
        torch.arange(micro_width, device=device),
        indexing="ij",
    )
    y = selected_y0[:, None] + oy.reshape(1, -1)
    x = selected_x0[:, None] + ox.reshape(1, -1)
    valid = (y < height) & (x < width)
    group_count = (valid_rows * valid_cols)[:, None].expand_as(y)
    group_origin_y = (group_y0 * micro_height)[:, None].expand_as(y)
    group_origin_x = (group_x0 * micro_width)[:, None].expand_as(x)
    return StratifiedSamples(
        y=y[valid],
        x=x[valid],
        weights=group_count[valid].to(torch.float64) / float(height * width),
        block_y=group_origin_y[valid],
        block_x=group_origin_x[valid],
    )


def sample_random_pixels_in_active_tiles(
    height: int,
    width: int,
    tile_height: int = 16,
    tile_width: int = 16,
    macro_tiles_y: int = 2,
    macro_tiles_x: int = 2,
    samples_per_active_tile: int = 16,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> StratifiedSamples:
    """Choose one renderer tile per macroblock, then random pixels in that tile.

    Defaults give inclusion probability (1/4)*(16/256)=1/64 for interior
    pixels while activating about one quarter of the 16x16 renderer tiles.
    """
    if min(height, width, tile_height, tile_width, macro_tiles_y, macro_tiles_x, samples_per_active_tile) <= 0:
        raise ValueError("image, tile, and sample dimensions must be positive")
    if samples_per_active_tile > tile_height * tile_width:
        raise ValueError("samples_per_active_tile cannot exceed the full tile area")
    device = torch.device(device)
    tile_rows = (height + tile_height - 1) // tile_height
    tile_cols = (width + tile_width - 1) // tile_width
    group_y0 = torch.arange(0, tile_rows, macro_tiles_y, device=device)
    group_x0 = torch.arange(0, tile_cols, macro_tiles_x, device=device)
    group_y0, group_x0 = torch.meshgrid(group_y0, group_x0, indexing="ij")
    group_y0 = group_y0.reshape(-1)
    group_x0 = group_x0.reshape(-1)
    valid_tile_rows = torch.clamp(tile_rows - group_y0, max=macro_tiles_y)
    valid_tile_cols = torch.clamp(tile_cols - group_x0, max=macro_tiles_x)
    tile_count = valid_tile_rows * valid_tile_cols
    tile_choice = torch.floor(
        torch.rand(group_y0.shape, generator=generator, device=device) * tile_count
    ).long()
    tile_y = group_y0 + tile_choice // valid_tile_cols
    tile_x = group_x0 + tile_choice % valid_tile_cols
    y0 = tile_y * tile_height
    x0 = tile_x * tile_width

    oy, ox = torch.meshgrid(
        torch.arange(tile_height, device=device),
        torch.arange(tile_width, device=device),
        indexing="ij",
    )
    y_candidates = y0[:, None] + oy.reshape(1, -1)
    x_candidates = x0[:, None] + ox.reshape(1, -1)
    valid = (y_candidates < height) & (x_candidates < width)
    random_keys = torch.rand(valid.shape, generator=generator, device=device)
    random_keys[~valid] = 2.0
    chosen = torch.topk(random_keys, samples_per_active_tile, dim=1, largest=False).indices
    y = torch.gather(y_candidates.expand_as(random_keys), 1, chosen)
    x = torch.gather(x_candidates.expand_as(random_keys), 1, chosen)
    chosen_valid = torch.gather(valid, 1, chosen)
    area = valid.sum(dim=1, keepdim=True)
    selected_count = torch.clamp(area, max=samples_per_active_tile)
    weight = (tile_count[:, None] * area / selected_count).to(torch.float64) / float(height * width)
    weight = weight.expand_as(y)
    group_origin_y = (group_y0 * tile_height)[:, None].expand_as(y)
    group_origin_x = (group_x0 * tile_width)[:, None].expand_as(x)
    return StratifiedSamples(
        y=y[chosen_valid],
        x=x[chosen_valid],
        weights=weight[chosen_valid],
        block_y=group_origin_y[chosen_valid],
        block_x=group_origin_x[chosen_valid],
    )


def _keyed_permutation(index, key, bits=3, rounds=4):
    """A bijection on [0, 2**(2*bits)) selected by `key`, computed per element.

    A Feistel network gives a different permutation for every key without storing
    any of them, which matters because the alternative -- one stored permutation
    per (image, block) -- would be 255 x 101,400 x 64 entries.
    """
    mask = (1 << bits) - 1
    left = (index >> bits) & mask
    right = index & mask
    for r in range(rounds):
        h = right * 2654435761 + key * 40503 + r * 2246822519
        h = (h ^ (h >> 13)) * 1274126177
        left, right = right, left ^ ((h >> 7) & mask)
    return (left << bits) | right


def sample_permuted_pixels(
    height: int,
    width: int,
    block_height: int,
    block_width: int,
    visit: int,
    state: dict,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> StratifiedSamples:
    """Sampling without replacement: a block never repeats a position until it has
    used them all.

    `sample_stratified_pixels` draws each block independently every step, so over
    the `steps / images` visits an image gets, the per-pixel count is Poisson. At
    30k steps over 255 images that is 117.6 visits and a mean of 1.84, which leaves
    **15.6% of pixels never sampled** -- and the loss says nothing about a pixel it
    never looks at. Drawing without replacement makes the count deterministic and
    drops its variance from 1.81 to 0.13.

    The order has to stay random. An arithmetic walk `offset + visit * stride`
    also covers every position, but blocks sharing a stride then hold the same
    relative offset for the whole run, which is a lattice baked into the sampling.
    Here each block draws a fresh keyed permutation per cycle instead, so relative
    positions are re-randomised every `block_height * block_width` visits and
    nothing is fixed across blocks.

    Border blocks are partial; their index is taken modulo the true area, which
    costs the exact-cycle guarantee for the 0.6% of blocks on two edges.
    """
    device = torch.device(device)
    y0 = torch.arange(0, height, block_height, device=device, dtype=torch.long)
    x0 = torch.arange(0, width, block_width, device=device, dtype=torch.long)
    block_y, block_x = torch.meshgrid(y0, x0, indexing="ij")
    block_y = block_y.reshape(-1)
    block_x = block_x.reshape(-1)

    valid_h = torch.clamp(height - block_y, max=block_height)
    valid_w = torch.clamp(width - block_x, max=block_width)
    area = valid_h * valid_w
    full = block_height * block_width
    bits = max(1, (full.bit_length() - 1) // 2)

    shape_key = (height, width, block_height, block_width)
    if shape_key not in state:
        state[shape_key] = torch.randint(
            0, 2 ** 30, (block_y.numel(),), generator=generator, device=device)
    block_seed = state[shape_key]

    cycle, position = divmod(visit, full)
    key = block_seed + cycle * 2654435761
    index = _keyed_permutation(
        torch.full_like(block_seed, position), key, bits=bits) % area

    return StratifiedSamples(
        y=block_y + index // valid_w,
        x=block_x + index % valid_w,
        weights=area.to(torch.float64) / float(height * width),
        block_y=block_y,
        block_x=block_x,
    )
