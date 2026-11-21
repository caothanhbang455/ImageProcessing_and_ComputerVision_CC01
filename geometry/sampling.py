# geometry/sampling.py
"""Backward image sampling using PyTorch with explicit grid conventions."""

import numpy as np

from .coordinates import MAP_SPACES, pixel_to_normalized


def sample_backward(image : np.ndarray, backward_map : np.ndarray, map_space : str,
                    interpolation : str = "bilinear", padding : str = "zeros",
                    align_corners : bool = True, device : str = "cpu") -> np.ndarray :
    """Return a float32 H_out x W_out x C image without rounding or clipping.

    The source may be H x W or H x W x C. Input numerical scale is preserved.
    Map units must be declared explicitly. GPU and CPU outputs may differ slightly.
    """
    import torch
    import torch.nn.functional as F

    source = np.asarray(image)
    grid = np.asarray(backward_map)
    if (source.ndim not in (2, 3) or min(source.shape[:2]) < 2) :
        raise ValueError("Source image must be H x W [x C], at least 2 x 2")
    if (grid.ndim != 3 or grid.shape[-1] != 2 or min(grid.shape[:2]) < 2) :
        raise ValueError("Backward map must be H x W x 2, at least 2 x 2")
    if (not np.isfinite(source).all() or not np.isfinite(grid).all()) :
        raise ValueError("Source image and map must be finite")
    if (map_space not in MAP_SPACES) :
        raise ValueError(f"Invalid map_space: {map_space}")
    if (interpolation not in ("bilinear", "nearest") or padding not in ("zeros", "border", "reflection")) :
        raise ValueError("Unsupported sampling mode")
    if (map_space == "source_pixels") :
        grid = pixel_to_normalized(grid, source.shape[1], source.shape[0], align_corners)

    rgb = source[..., None] if source.ndim == 2 else source
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.astype(np.float32).transpose(2, 0, 1)))[None].to(device)
    grid_tensor = torch.from_numpy(np.ascontiguousarray(grid.astype(np.float32)))[None].to(device)
    with torch.inference_mode() :
        output = F.grid_sample(tensor, grid_tensor, mode=interpolation, padding_mode=padding,
                               align_corners=align_corners)
    result = output[0].cpu().numpy().transpose(1, 2, 0)
    return result[..., 0] if source.ndim == 2 else result
