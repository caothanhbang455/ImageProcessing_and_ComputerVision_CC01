# tests/test_geometry_coordinates.py
import numpy as np
import pytest

from geometry.coordinates import identity_backward_map, normalized_to_pixel, pixel_to_normalized


@pytest.mark.parametrize("align", [True, False])
def test_roundtrip_non_square(align) :
    points = np.array([[[0.0, 0.0], [6.5, 13.2]], [[13, 3], [4, 20]]])
    norm = pixel_to_normalized(points, 17, 31, align)
    np.testing.assert_allclose(normalized_to_pixel(norm, 17, 31, align), points, atol=1e-12)


def test_align_corners_endpoints() :
    points = np.array([[0., 0.], [16., 30.]])
    np.testing.assert_allclose(pixel_to_normalized(points, 17, 31), [[-1, -1], [1, 1]])


def test_half_pixel_coordinates() :
    points = np.array([[0., 0.], [15., 15.]])
    result = pixel_to_normalized(points, 16, 16, align_corners=False)
    np.testing.assert_allclose(result, [[-0.9375, -0.9375], [0.9375, 0.9375]])


@pytest.mark.parametrize("space", ["source_pixels", "normalized_minus1_plus1"])
def test_identity_map(space) :
    result = identity_backward_map(17, 9, space)
    assert result.shape == (9, 17, 2)
    expected = [16, 8] if space == "source_pixels" else [1, 1]
    np.testing.assert_allclose(result[-1, -1], expected)


@pytest.mark.parametrize("w,h", [(1, 9), (9, 1), (0, 2)])
def test_no_singular_align_corners_dimensions(w, h) :
    with pytest.raises(ValueError) :
        identity_backward_map(w, h)


def test_reject_nan_and_inconsistent_axes() :
    with pytest.raises(ValueError) :
        pixel_to_normalized(np.array([float('nan'), 0]), 10, 10)
    with pytest.raises(ValueError) :
        normalized_to_pixel(np.zeros((4, 3)), 10, 10)
    with pytest.raises(ValueError) :
        identity_backward_map(10, 10, "unknown")
