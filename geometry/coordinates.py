# geometry/coordinates.py
"""Convert (x, y) pixel-center coordinates to/from PyTorch normalized space.

Images use (height, width), while a point's last axis is always (x, y).
Functions preserve the leading point/map dimensions and return floating arrays.
"""

import numpy as np


MAP_SPACES = ("source_pixels", "normalized_minus1_plus1")


def _size(width : int, height : int, align_corners : bool) -> tuple[int, int] :
    if (not isinstance(width, (int, np.integer)) or not isinstance(height, (int, np.integer))) :
        raise TypeError("Image dimensions must be integers")
    minimum = 2 if align_corners else 1
    if (width < minimum or height < minimum) :
        raise ValueError(f"Dimensions must both be >= {minimum} for align_corners={align_corners}")
    return int(width), int(height)


def _points(points : np.ndarray) -> np.ndarray :
    values = np.asarray(points)
    if (values.ndim < 1 or values.shape[-1] != 2 or not np.issubdtype(values.dtype, np.number)) :
        raise ValueError("Points/maps must have a last axis of (x, y)")
    if (not np.isfinite(values).all()) :
        raise ValueError("Points must be finite")
    return values.astype(np.float64, copy=False)


def pixel_to_normalized(points : np.ndarray, width : int, height : int, align_corners : bool = True) -> np.ndarray :
    width, height = _size(width, height, align_corners)
    values = _points(points)
    scale = np.array([width - 1, height - 1] if align_corners else [width, height], dtype=np.float64)
    offset = 0.0 if align_corners else 0.5
    return (values + offset) * 2.0 / scale - 1.0


def normalized_to_pixel(points : np.ndarray, width : int, height : int, align_corners : bool = True) -> np.ndarray :
    width, height = _size(width, height, align_corners)
    values = _points(points)
    scale = np.array([width - 1, height - 1] if align_corners else [width, height], dtype=np.float64)
    offset = 0.0 if align_corners else 0.5
    return (values + 1.0) * scale / 2.0 - offset


def identity_backward_map(width : int, height : int, map_space : str = "normalized_minus1_plus1",
                          align_corners : bool = True) -> np.ndarray :
    """Return a dense H x W x 2 output-to-source identity map."""
    _size(width, height, align_corners)
    if (map_space not in MAP_SPACES) :
        raise ValueError(f"Unsupported map space: {map_space}")
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float64), np.arange(width, dtype=np.float64), indexing="ij")
    result = np.stack([xx, yy], axis=-1)
    if (map_space == "normalized_minus1_plus1") :
        result = pixel_to_normalized(result, width, height, align_corners)
    return result.astype(np.float32)
