# tests/test_geometry_homography.py
import cv2
import numpy as np
import pytest

from geometry.homography import (estimate_homography, homography_backward_map,
                                 invert_homography, transform_points)


def test_translation_forward_inverse() :
    h = np.array([[1, 0, 8], [0, 1, -3], [0, 0, 1]], dtype=np.float64)
    points = np.array([[2, 6], [11, 15], [20, 30], [33, 55]], dtype=np.float64)
    np.testing.assert_allclose(transform_points(points, h), points + [8, -3])
    np.testing.assert_allclose(transform_points(transform_points(points, h), invert_homography(h)), points)


def test_backwards_direction() :
    h = np.array([[1, 0, 8], [0, 1, 3], [0, 0, 1]], dtype=np.float64)
    backward = homography_backward_map(h, (50, 40), (50, 40))
    np.testing.assert_allclose(backward[10, 18], [10, 7])


def test_estimation_and_cv2_agree() :
    src = np.float32([[1, 2], [70, 4], [65, 55], [5, 50]])
    dst = np.float32([[10, 20], [82, 15], [74, 90], [12, 86]])
    estimated = estimate_homography(src, dst)
    np.testing.assert_allclose(transform_points(src, estimated), dst, atol=2e-4)
    cv = cv2.getPerspectiveTransform(src, dst)
    np.testing.assert_allclose(estimated / estimated[2, 2], cv / cv[2, 2], atol=1e-6)


def test_degenerate_rejection() :
    with pytest.raises(ValueError) :
        estimate_homography(np.array([[0,0], [1,1], [2,2], [3,3]]), np.array([[0,0], [1,0], [1,1], [0,1]]))
    with pytest.raises(ValueError) :
        invert_homography(np.zeros((3,3)))
    with pytest.raises(ValueError) :
        transform_points(np.array([[1.,0.]]), np.array([[1,0,0],[0,1,0],[0,0,0]]))
