# tests/test_geometry_sampling.py
import numpy as np
import pytest

from geometry.coordinates import identity_backward_map
from geometry.sampling import sample_backward


def test_bilinear_identity_float() :
    image = np.random.default_rng(4).random((32, 49, 3), dtype=np.float32)
    grid = identity_backward_map(49, 32)
    result = sample_backward(image, grid, "normalized_minus1_plus1")
    np.testing.assert_allclose(result, image, rtol=0, atol=3e-6)


def test_nearest_integer_translation() :
    image = np.arange(20 * 30, dtype=np.float32).reshape(20, 30)
    grid = identity_backward_map(30, 20, "source_pixels")
    grid[..., 0] -= 8
    result = sample_backward(image, grid, "source_pixels", interpolation="nearest")
    np.testing.assert_array_equal(result[:, 8:], image[:, :-8])
    assert (result[:, :8] == 0).all()


def test_default_zero_padding() :
    image = np.ones((9, 13, 3), dtype=np.float32)
    grid = identity_backward_map(13, 9, "source_pixels")
    grid[..., :] = [999, 999]
    result = sample_backward(image, grid, "source_pixels")
    assert result.sum() == 0


def test_invalid_map_fails() :
    with pytest.raises(ValueError) :
        sample_backward(np.ones((8, 8, 3)), np.full((8, 8, 2), float("nan")), "source_pixels")
    with pytest.raises(ValueError) :
        sample_backward(np.ones((8, 8, 3)), np.zeros((8, 8, 2)), "invented")
