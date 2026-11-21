# geometry/grids.py
"""Coarse-grid conversion, upsampling, valid sampling locations, and diagnostics."""

import numpy as np

from .coordinates import MAP_SPACES, normalized_to_pixel


def to_hwc_map(grid_chw : np.ndarray) -> np.ndarray :
    grid = np.asarray(grid_chw)
    if (grid.ndim != 3 or grid.shape[0] != 2) :
        raise ValueError("Expected a 2 x H x W grid")
    return np.moveaxis(grid, 0, -1)


def to_chw_grid(map_hwc : np.ndarray) -> np.ndarray :
    grid = np.asarray(map_hwc)
    if (grid.ndim != 3 or grid.shape[-1] != 2) :
        raise ValueError("Expected an H x W x 2 map")
    return np.moveaxis(grid, -1, 0)


def upsample_coarse_grid(grid_bchw, output_height : int, output_width : int, align_corners : bool = True) :
    """Upsample a torch B x 2 x H x W grid, matching the official UVDoc operation."""
    import torch
    import torch.nn.functional as F

    if (not isinstance(grid_bchw, torch.Tensor) or grid_bchw.ndim != 4 or grid_bchw.shape[1] != 2) :
        raise ValueError("Expected a torch B x 2 x H x W grid")
    if (not grid_bchw.is_floating_point()) :
        raise TypeError("Grid must be floating-point")
    if (output_height < 2 or output_width < 2) :
        raise ValueError("Output dimensions must be at least two")
    return F.interpolate(grid_bchw, size=(output_height, output_width), mode="bilinear", align_corners=align_corners)


def sampling_valid_mask(backward_map : np.ndarray, map_space : str,
                        source_width : int, source_height : int, align_corners : bool = True) -> np.ndarray :
    """True means source coordinates are finite and within the source image.

    This is not a document foreground mask and says nothing about visual quality.
    """
    values = np.asarray(backward_map)
    if (values.ndim != 3 or values.shape[-1] != 2) :
        raise ValueError("Expected an H x W x 2 backward map")
    if (map_space not in MAP_SPACES) :
        raise ValueError(f"Unknown map space: {map_space}")
    if (source_width < 2 or source_height < 2) :
        raise ValueError("Source dimensions must be at least two")
    finite = np.isfinite(values).all(axis=-1)
    if (map_space == "source_pixels") :
        x, y = values[..., 0], values[..., 1]
        return finite & (x >= 0) & (x <= source_width - 1) & (y >= 0) & (y <= source_height - 1)
    if (align_corners) :
        return finite & (np.abs(values[..., 0]) <= 1) & (np.abs(values[..., 1]) <= 1)
    # When align_corners=False the corner pixel centers lie inside [-1, 1].
    # Values at exactly +/-1 sample half from the padding region.
    x_limit = 1.0 - 1.0 / source_width
    y_limit = 1.0 - 1.0 / source_height
    return finite & (np.abs(values[..., 0]) <= x_limit) & (np.abs(values[..., 1]) <= y_limit)


def grid_statistics(grid_chw : np.ndarray) -> dict[str, float | int] :
    grid = np.asarray(grid_chw)
    if (grid.ndim != 3 or grid.shape[0] != 2) :
        raise ValueError("Expected a 2 x H x W coarse grid")
    finite = np.isfinite(grid)
    stats : dict[str, float | int] = {"total_values" : int(grid.size), "finite_values" : int(finite.sum())}
    for axis, name in enumerate(("x", "y")) :
        channel = grid[axis]
        valid = channel[np.isfinite(channel)]
        stats[f"{name}_min"] = float(valid.min()) if valid.size else float("nan")
        stats[f"{name}_max"] = float(valid.max()) if valid.size else float("nan")
    stats["outside_normalized_fraction"] = float(np.mean(np.any((np.abs(grid) > 1) | ~finite, axis=0)))
    stats["horizontal_step_mean"] = float(np.mean(np.linalg.norm(np.diff(grid, axis=2), axis=0)))
    stats["vertical_step_mean"] = float(np.mean(np.linalg.norm(np.diff(grid, axis=1), axis=0)))
    return stats
