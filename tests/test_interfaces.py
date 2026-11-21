# tests/test_interfaces.py
"""Common result shape and failure-state validation."""

import numpy as np
import pytest

from models.interfaces import MethodResult


def test_success_with_geometry() :
    image = np.zeros((12, 18, 3), dtype=np.uint8)
    grid = np.zeros((12, 18, 2), dtype=np.float32)
    mask = np.ones((12, 18), dtype=bool)
    result = MethodResult("OK", image, backward_map=grid, map_space="source_pixels", valid_mask=mask)
    assert result.image.shape == (12, 18, 3)


def test_explicit_failure_without_image() :
    result = MethodResult("FAILED", None, failure_reason="No document boundary candidate")
    assert result.status == "FAILED" and result.image is None


@pytest.mark.parametrize("fields", [
    {"status": "OK", "image": None},
    {"status": "FAILED", "image": None},
    {"status": "OK", "image": np.zeros((3, 3, 3)), "map_space": "source_pixels"},
    {"status": "OK", "image": np.zeros((3, 3, 3)), "backward_map": np.zeros((3, 3, 2))},
    {"status": "OK", "image": np.zeros((3, 3, 3)), "valid_mask": np.zeros((3, 3), dtype=np.uint8)},
    {"status": "UNKNOWN", "image": None},
])
def test_invalid_results_fail_explicitly(fields) :
    with pytest.raises(ValueError) :
        MethodResult(**fields)
