"""One observed pixel per stratum; unchanged random-number consumption.

The cached geometry removes repeated arange/meshgrid/clamp/area construction.
It does not change the sample count, generator, draw order, or boundary rule.
"""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class Samples:
    y: torch.Tensor
    x: torch.Tensor
    weights: torch.Tensor
    block_y: torch.Tensor
    block_x: torch.Tensor


class StratifiedGrid:
    def __init__(self, height, width, block=8, device="cpu"):
        if min(height, width, block) <= 0:
            raise ValueError("Positive image and stratum dimensions are required")
        self.height, self.width, self.block = height, width, block
        self.device = torch.device(device)
        by, bx = torch.meshgrid(
            torch.arange(0, height, block, device=self.device, dtype=torch.long),
            torch.arange(0, width, block, device=self.device, dtype=torch.long), indexing="ij")
        self.shape = by.shape
        self.block_y, self.block_x = by.reshape(-1), bx.reshape(-1)
        self.valid_h = torch.clamp(height-self.block_y, max=block)
        self.valid_w = torch.clamp(width-self.block_x, max=block)
        self.weights = (self.valid_h*self.valid_w).to(torch.float64)/float(height*width)

    def sample(self, generator):
        random_y = torch.rand(self.block_y.shape, generator=generator, device=self.device)
        random_x = torch.rand(self.block_x.shape, generator=generator, device=self.device)
        y = self.block_y + torch.floor(random_y*self.valid_h).to(torch.long)
        x = self.block_x + torch.floor(random_x*self.valid_w).to(torch.long)
        return Samples(y, x, self.weights, self.block_y, self.block_x)


def view_sequence(view_count, steps, seed, views=4):
    """Paired experimental sequence, generalized without changing the MV4 case."""
    if not 1 <= views <= view_count or steps < 0:
        raise ValueError("Invalid view count or number of steps")
    rng = np.random.default_rng(seed)
    primary = []
    while len(primary) < steps:
        primary.extend(rng.permutation(view_count).tolist())
    return [[first] + rng.choice([v for v in range(view_count) if v != first],
                                  views-1, replace=False).tolist()
            for first in primary[:steps]]
