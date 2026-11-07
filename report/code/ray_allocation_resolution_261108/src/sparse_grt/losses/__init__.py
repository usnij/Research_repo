"""The default coordinate SSIM is the unchanged validated implementation."""
from .coordinate_ssim import loss, weights, reference_loss, reference_weights

__all__ = ["loss", "weights", "reference_loss", "reference_weights"]
