# tests/test_geometry_grids.py
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from geometry.grids import (grid_statistics, sampling_valid_mask, to_chw_grid,
                            to_hwc_map, upsample_coarse_grid)
from geometry.coordinates import identity_backward_map


def test_layout_roundtrip() :
    grid = np.random.default_rng(0).random((2, 3, 5)).astype(np.float32)
    np.testing.assert_array_equal(to_chw_grid(to_hwc_map(grid)), grid)


def test_layout_invalid() :
    with pytest.raises(ValueError) :
        to_hwc_map(np.zeros((3, 2, 2)))
    with pytest.raises(ValueError) :
        to_chw_grid(np.zeros((2, 3, 3)))


def test_upsample_matches_reference_torch() :
    coarse = torch.randn((1, 2, 45, 31), generator=torch.Generator().manual_seed(21))
    result = upsample_coarse_grid(coarse, 83, 57)
    expected = F.interpolate(coarse, size=(83, 57), mode="bilinear", align_corners=True)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_upsample_refuses_wrong_shape() :
    with pytest.raises(ValueError) :
        upsample_coarse_grid(torch.zeros((1, 3, 5, 4)), 8, 9)


def test_sampling_mask_source_pixels() :
    grid = identity_backward_map(6, 8, "source_pixels")
    grid[0, 0] = [-1, 0]
    grid[1, 0] = [float("nan"), 0]
    valid = sampling_valid_mask(grid, "source_pixels", 6, 8)
    assert not valid[0, 0] and not valid[1, 0]
    assert valid[2, 2]


def test_sampling_mask_normalized() :
    grid = identity_backward_map(6, 8)
    grid[0, 0, 0] = 1.05
    valid = sampling_valid_mask(grid, "normalized_minus1_plus1", 6, 8)
    assert valid.sum() == 47


def test_grid_statistics() :
    grid = np.zeros((2, 45, 31), dtype=np.float32)
    stats = grid_statistics(grid)
    assert stats["finite_values"] == 2 * 45 * 31
    assert stats["outside_normalized_fraction"] == 0
    assert stats["x_min"] == 0


def test_half_pixel_valid_domain() :
    grid = np.zeros((2, 3, 2), dtype=np.float32)
    grid[0, 0] = [-1.0, 1.0]
    mask = sampling_valid_mask(grid, "normalized_minus1_plus1", 10, 8, align_corners=False)
    assert not mask[0, 0]
    assert mask[0, 1:].all()
    assert mask[1].all()
