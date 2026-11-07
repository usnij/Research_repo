"""Method settings, separated from machine paths and renderer installation."""
from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class MethodConfig:
    views: int = 4
    block: int = 8
    sigma: float = 6.0
    l1_weight: float = 1.0
    structure_weight: float = 0.2
    opacity_weight: float = 0.01
    scale_weight: float = 0.01
    cap: int = 1131968
    steps: int = 30000
    initialization_seed: int = 777
    view_seed: int = 2610297100
    ray_seed: int = 910000000
    boundary: str = "valid"
    grid_window: int = 11
    grid_sigma: float = 1.5
    sampler: str = "point"

    def __post_init__(self):
        if self.sampler not in ("point", "micropatch"):
            raise ValueError("Sampler must be point or micropatch")
        if self.sampler == "micropatch" and (self.block != 3 or self.boundary != "all"):
            raise ValueError("Micropatches require block=3 virtual slots and all coordinate centres")
        if self.boundary not in ("valid", "all"):
            raise ValueError("Boundary objective must be explicit: valid or all")
        for key in ("views", "block", "cap", "steps"):
            value = getattr(self, key)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("sigma", "l1_weight", "structure_weight", "opacity_weight", "scale_weight"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.sigma == 0 or self.l1_weight == self.structure_weight == 0:
            raise ValueError("A positive bandwidth and an image loss are required")
        if type(self.grid_window) is not int or self.grid_window not in (4, 11):
            raise ValueError("Grid SSIM window must be 4 or 11")
        if self.grid_sigma != 1.5:
            raise ValueError("Grid SSIM keeps sigma=1.5; only window support varies")

    @classmethod
    def read(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def nominal_total_ray_fraction(self):
        """Relative to ONE full view; edge strata produce a small difference."""
        return self.views / self.block**2
