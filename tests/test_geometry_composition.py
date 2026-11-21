# tests/test_geometry_composition.py
import numpy as np
import pytest

from geometry.composition import compose_backward_maps
from geometry.coordinates import identity_backward_map, pixel_to_normalized


def test_identity_composition() :
    a = identity_backward_map(30, 20, "source_pixels")
    b = identity_backward_map(30, 20, "source_pixels")
    composed, valid = compose_backward_maps(a, b, (30, 20))
    np.testing.assert_allclose(composed, a, atol=1e-5)
    assert valid.all()


def test_unequal_translations_and_mask() :
    a = identity_backward_map(30, 20, "source_pixels")
    b = identity_backward_map(30, 20, "source_pixels")
    a[..., 0] += 2
    b[..., 1] += 3
    composed, valid = compose_backward_maps(a, b, (30, 20))
    np.testing.assert_allclose(composed[5, 5], [7, 8], atol=1e-5)
    assert valid[5, 5]
    assert not valid[-1, 5]
    assert not valid[5, -1]
    assert (composed[~valid] == 0).all()


def test_normalized_composition() :
    pixel_map = identity_backward_map(20, 12, "source_pixels")
    normalized = pixel_to_normalized(pixel_map, 20, 12)
    result, mask = compose_backward_maps(normalized, normalized, (20, 12),
                                          a_space="normalized_minus1_plus1", b_space="normalized_minus1_plus1",
                                          output_space="normalized_minus1_plus1")
    np.testing.assert_allclose(result, normalized, atol=5e-5)
    assert mask.all()


def test_invalid_a_mask() :
    a = identity_backward_map(20, 12, "source_pixels")
    b = a.copy()
    valid = np.ones((12, 20), dtype=bool)
    valid[5, 5] = False
    _, mask = compose_backward_maps(a, b, (20, 12), a_valid_mask=valid)
    assert not mask[5, 5]


def test_bad_coordinate_space() :
    a = identity_backward_map(20, 12)
    with pytest.raises(ValueError) :
        compose_backward_maps(a, a, (20, 12), a_space="magic")
