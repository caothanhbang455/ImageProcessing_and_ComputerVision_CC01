# geometry/composition.py
"""Compose two backward maps without repeatedly resampling RGB images."""

import cv2
import numpy as np

from .coordinates import MAP_SPACES, normalized_to_pixel, pixel_to_normalized
from .grids import sampling_valid_mask


def compose_backward_maps(map_a : np.ndarray, map_b : np.ndarray, source_size : tuple[int, int],
                          a_space : str = "source_pixels", b_space : str = "source_pixels",
                          output_space : str = "source_pixels", align_corners : bool = True,
                          a_valid_mask : np.ndarray | None = None,
                          b_valid_mask : np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray] :
    """Compose B_A: frame1 -> source0 and B_B: frame2 -> frame1.

    map_a shape is H1 x W1 x 2, map_b shape is H2 x W2 x 2. source_size
    describes source0 in (width, height) order. Returns (composed_map, valid_mask).
    Invalid map positions are set to (0,0) and MUST be interpreted via the mask.
    Bilinear map interpolation can differ from repeated image interpolation.
    """
    a, b = np.asarray(map_a), np.asarray(map_b)
    if (a.ndim != 3 or b.ndim != 3 or a.shape[-1] != 2 or b.shape[-1] != 2) :
        raise ValueError("Both maps must be H x W x 2")
    if (min(*a.shape[:2], *b.shape[:2], *source_size) < 2) :
        raise ValueError("Dimensions must be at least two")
    if (any(space not in MAP_SPACES for space in (a_space, b_space, output_space))) :
        raise ValueError("Unsupported map coordinate space")
    if (not np.isfinite(a).all() or not np.isfinite(b).all()) :
        raise ValueError("Cannot compose maps with non-finite coordinates")

    width1, height1 = a.shape[1], a.shape[0]
    source_width, source_height = source_size
    a_pixels = normalized_to_pixel(a, source_width, source_height, align_corners) if a_space != "source_pixels" else a
    b_pixels = normalized_to_pixel(b, width1, height1, align_corners) if b_space != "source_pixels" else b
    x, y = np.ascontiguousarray(b_pixels[..., 0], dtype=np.float32), np.ascontiguousarray(b_pixels[..., 1], dtype=np.float32)

    # Strict validity requires all contributing neighboring map cells to be valid.
    valid_a = sampling_valid_mask(a, a_space, source_width, source_height, align_corners)
    if (a_valid_mask is not None) :
        if (a_valid_mask.shape != valid_a.shape or a_valid_mask.dtype != np.bool_) :
            raise ValueError("a_valid_mask shape/dtype mismatch")
        valid_a &= a_valid_mask
    valid_b = sampling_valid_mask(b, b_space, width1, height1, align_corners)
    if (b_valid_mask is not None) :
        if (b_valid_mask.shape != valid_b.shape or b_valid_mask.dtype != np.bool_) :
            raise ValueError("b_valid_mask shape/dtype mismatch")
        valid_b &= b_valid_mask

    # INTER_LINEAR is sufficient for the geometric composition primitive; a later
    # research method can introduce a separately identified sampling variant.
    sampled_x = cv2.remap(np.ascontiguousarray(a_pixels[..., 0], dtype=np.float32), x, y,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sampled_y = cv2.remap(np.ascontiguousarray(a_pixels[..., 1], dtype=np.float32), x, y,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sampled_valid = cv2.remap(valid_a.astype(np.float32), x, y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    composed_pixels = np.stack([sampled_x, sampled_y], axis=-1)
    valid = valid_b & (sampled_valid >= 1.0 - 1e-6)
    valid &= sampling_valid_mask(composed_pixels, "source_pixels", source_width, source_height)
    if (output_space == "normalized_minus1_plus1") :
        result = pixel_to_normalized(composed_pixels, source_width, source_height, align_corners).astype(np.float32)
    else :
        result = composed_pixels.astype(np.float32)
    result[~valid] = 0
    return result, valid
