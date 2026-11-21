# data/images.py
"""Shared image decoding helpers. HDF5 is deliberately not required."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def read_image(path : Path, apply_exif : bool = False, mode : str = "RGB") -> np.ndarray :
    """Read source pixels without modifying the file; EXIF correction is opt-in."""
    with Image.open(path) as image :
        selected = ImageOps.exif_transpose(image) if apply_exif else image
        return np.asarray(selected.convert(mode)).copy()


def image_header(path : Path) -> dict[str, object] :
    with Image.open(path) as image :
        return {"raw_width": image.width, "raw_height": image.height,
                "mode": image.mode, "exif_orientation": int(image.getexif().get(274, 1))}


