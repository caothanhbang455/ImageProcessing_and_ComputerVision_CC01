# geometry/__init__.py
"""Shared geometric operations in explicit output-to-source coordinates."""

from .coordinates import identity_backward_map, normalized_to_pixel, pixel_to_normalized
from .grids import sampling_valid_mask
from .homography import homography_backward_map
