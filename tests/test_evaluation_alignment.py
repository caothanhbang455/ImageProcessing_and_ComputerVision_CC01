# tests/test_evaluation_alignment.py
import numpy as np
import pytest

from evaluation.alignment import orientation_policy, preview_reference_pair


def test_preview_documents_nonreference_resize() :
    x = np.zeros((12, 8, 3), dtype=np.uint8)
    y = np.ones((20, 14, 3), dtype=np.uint8)
    a, b, details = preview_reference_pair(x, y, target_area=280)
    assert a.shape == b.shape
    assert details["preview_only"] and not details["reference_compatible"]
    with pytest.raises(ValueError) :
        preview_reference_pair(x, np.zeros((10, 10), dtype=float))


def test_docunet_orientation_policies_stay_separate() :
    assert not orientation_policy("64_1", "docunet_matlab_legacy_v1")["requires_corrected_inference"]
    assert orientation_policy("64_2", "docunet_matlab_corrected64_v1")["requires_corrected_inference"]
    assert not orientation_policy("63_1", "docunet_matlab_corrected64_v1")["requires_corrected_inference"]
    with pytest.raises(ValueError) :
        orientation_policy("64_1", "unknown")
