# models/interfaces.py
"""Small, model-independent output contract for future experiments."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


VALID_STATUSES = {"OK", "FAILED", "UNAVAILABLE", "REVIEW_REQUIRED"}
MAP_SPACES = {"source_pixels", "normalized_minus1_plus1"}


@dataclass
class MethodResult :
    """A rectified image and optional dense output-to-source sampling geometry.

    A backward_map, when returned, must be H x W x 2. A coarse learned grid
    belongs in metadata/artifacts until converted to a dense sampling map.
    """

    status : str
    image : np.ndarray | None
    backward_map : np.ndarray | None = None
    map_space : str | None = None
    valid_mask : np.ndarray | None = None
    failure_reason : str | None = None
    metadata : dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) :
        if (self.status not in VALID_STATUSES) :
            raise ValueError(f"Unknown method status: {self.status}")
        if (self.status == "OK") :
            if (not isinstance(self.image, np.ndarray) or self.image.ndim not in (2, 3)) :
                raise ValueError("OK results require a decoded image array")
            if (self.image.ndim == 3 and self.image.shape[2] not in (1, 3, 4)) :
                raise ValueError("Unsupported output channel count")
            if (not self.image.size or not all(self.image.shape[:2])) :
                raise ValueError("Output image must have positive dimensions")
            if (self.failure_reason) :
                raise ValueError("Successful results must not declare a failure_reason")
        elif (not self.failure_reason) :
            raise ValueError("Non-success results require a failure_reason")

        if (self.backward_map is not None) :
            if (self.map_space not in MAP_SPACES) :
                raise ValueError("backward_map requires explicit map_space")
            if (not isinstance(self.backward_map, np.ndarray) or self.backward_map.ndim != 3 or
                self.backward_map.shape[2] != 2) :
                raise ValueError("backward_map must be an H x W x 2 array")
            if (self.image is None or self.backward_map.shape[:2] != self.image.shape[:2]) :
                raise ValueError("Dense backward_map dimensions must match output image")
            if (not np.isfinite(self.backward_map).all()) :
                raise ValueError("backward_map must contain only finite values")
        elif (self.map_space is not None) :
            raise ValueError("map_space requires backward_map")

        if (self.valid_mask is not None) :
            if (self.image is None or self.valid_mask.shape != self.image.shape[:2]) :
                raise ValueError("valid_mask dimensions must match output image")
            if (self.valid_mask.dtype != np.bool_) :
                raise ValueError("valid_mask must be boolean")
