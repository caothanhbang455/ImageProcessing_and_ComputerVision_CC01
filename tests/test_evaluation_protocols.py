# tests/test_evaluation_protocols.py
import numpy as np
import pytest

from evaluation.protocols import dir300, docunet, uvdoc


def test_pairing_names() :
    assert docunet.reference_prediction_name("64_2") == "64_2.png"
    assert docunet.reference_name("64_2") == "64.png"
    assert dir300.reference_prediction_name("008") == "8_rec.png"
    assert dir300.reference_name("300") == "300.png"
    with pytest.raises(ValueError) :
        docunet.reference_prediction_name("66_1")
    with pytest.raises(ValueError) :
        dir300.reference_prediction_name("301")


def test_line_positions_and_input_guards() :
    assert len(uvdoc.LINE_POSITIONS) == 25
    assert list(uvdoc.LINE_POSITIONS[[0, -1]]) == [50, 950]
    assert np.all(np.diff(uvdoc.LINE_POSITIONS) > 0)
    with pytest.raises(ValueError) :
        uvdoc.warp_synthetic_line(np.zeros((4, 4, 2)), 1200, "horizontal")
    with pytest.raises(ValueError) :
        uvdoc.unwarp_synthetic_line(np.zeros((10, 10, 3)), np.ones((3, 3, 2)))
