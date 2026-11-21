# geometry/homography.py
"""Projective point geometry and explicit pixel-space backward maps."""

import cv2
import numpy as np

from .coordinates import identity_backward_map


def _matrix(matrix : np.ndarray) -> np.ndarray :
    h = np.asarray(matrix, dtype=np.float64)
    if (h.shape != (3, 3) or not np.isfinite(h).all()) :
        raise ValueError("Homography must be a finite 3 x 3 matrix")
    if (np.linalg.matrix_rank(h) != 3 or np.linalg.cond(h) > 1e12) :
        raise ValueError("Homography is singular or ill-conditioned")
    return h


def _landmarks(points : np.ndarray) -> np.ndarray :
    array = np.asarray(points, dtype=np.float64)
    if (array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 4 or not np.isfinite(array).all()) :
        raise ValueError("At least four finite (x, y) point pairs are required")
    if (np.unique(array, axis=0).shape[0] != array.shape[0] or np.linalg.matrix_rank(array - array.mean(axis=0)) < 2) :
        raise ValueError("Duplicate or collinear landmarks")
    return array


def estimate_homography(source_points : np.ndarray, target_points : np.ndarray) -> np.ndarray :
    src, dst = _landmarks(source_points), _landmarks(target_points)
    if (src.shape != dst.shape) :
        raise ValueError("Landmark arrays must have matching dimensions")
    if (len(src) == 4) :
        h = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    else :
        h, _ = cv2.findHomography(src, dst, method=0)
        if (h is None) :
            raise ValueError("Cannot determine homography from given landmarks")
    return _matrix(h)


def invert_homography(matrix : np.ndarray) -> np.ndarray :
    return _matrix(np.linalg.inv(_matrix(matrix)))


def transform_points(points : np.ndarray, matrix : np.ndarray) -> np.ndarray :
    pts = np.asarray(points, dtype=np.float64)
    if (pts.ndim < 1 or pts.shape[-1] != 2 or not np.isfinite(pts).all()) :
        raise ValueError("Points must end in a finite (x, y) axis")
    flat = pts.reshape(-1, 2)
    homogeneous = np.column_stack([flat, np.ones(len(flat))]) @ _matrix(matrix).T
    weights = homogeneous[:, 2]
    if (np.any(np.abs(weights) < 1e-12)) :
        raise ValueError("A transformed point lies at infinity")
    return (homogeneous[:, :2] / weights[:, None]).reshape(pts.shape)


def homography_backward_map(source_to_output_matrix : np.ndarray, source_size : tuple[int, int],
                            output_size : tuple[int, int]) -> np.ndarray :
    """Source/target dimensions are (width, height); returned map is output H x W x 2.

    The matrix maps source to output. Its inverse maps output pixels back to source.
    """
    sw, sh = source_size
    ow, oh = output_size
    if (min(sw, sh, ow, oh) < 2) :
        raise ValueError("Image dimensions must be at least two")
    target_pixels = identity_backward_map(ow, oh, "source_pixels")
    return transform_points(target_pixels, invert_homography(source_to_output_matrix)).astype(np.float32)
