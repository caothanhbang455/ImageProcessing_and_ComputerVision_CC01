# evaluation/alignment.py
"""Explicit reference preparation. OpenCV previews are not MATLAB-equivalent scores."""

import cv2
import numpy as np


TARGET_REFERENCE_AREA = 598400


def preview_reference_pair(predicted : np.ndarray, reference : np.ndarray,
                           target_area : int = TARGET_REFERENCE_AREA) -> tuple[np.ndarray, np.ndarray, dict] :
    """Preview the benchmark's resizing sequence without claiming MATLAB pixel parity.

    Use the actual MATLAB implementation for benchmark numbers; OpenCV and
    MATLAB may differ in grayscale conversion, resizing, and antialiasing.
    """
    if (target_area <= 0) :
        raise ValueError("Reference target area must be positive")
    for name, image in (("predicted", predicted), ("reference", reference)) :
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
            image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2) :
            raise ValueError(f"{name} needs a nonempty uint8 RGB image")
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    predicted_gray = cv2.cvtColor(predicted, cv2.COLOR_RGB2GRAY)
    h, w = reference_gray.shape
    scale = float(np.sqrt(target_area / (h * w)))
    ref_resized = cv2.resize(reference_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    pred_resized = cv2.resize(predicted_gray, (ref_resized.shape[1], ref_resized.shape[0]),
                              interpolation=cv2.INTER_CUBIC)
    return pred_resized, ref_resized, {
        "preview_only" : True, "reference_scale" : scale,
        "reference_wh" : (w, h), "prepared_wh" : (ref_resized.shape[1], ref_resized.shape[0]),
        "interpolation" : "opencv_inter_cubic", "target_area" : target_area,
        "reference_compatible" : False,
    }


def orientation_policy(sample_id : str, protocol_id : str) -> dict :
    if (protocol_id not in ("docunet_matlab_legacy_v1", "docunet_matlab_corrected64_v1")) :
        raise ValueError(f"Unsupported DocUNet orientation protocol: {protocol_id}")
    correction = protocol_id == "docunet_matlab_corrected64_v1" and sample_id in ("64_1", "64_2")
    return {"sample_id" : sample_id, "policy" : protocol_id,
            "requires_corrected_inference" : correction,
            "note" : "Corrected inference must be produced separately" if correction else "No input correction requested"}
