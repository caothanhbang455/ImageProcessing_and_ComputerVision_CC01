# data/uvdoc.py
"""UVDoc image, HDF5 geometry, JSON, and auxiliary annotation readers."""

import json
from pathlib import Path

import numpy as np

from .images import image_header, read_image
from .paths import PathResolver, relative_manifest_path
from .records import SampleRecord


class HDF5EnvironmentError(RuntimeError):
    """HDF5 access is unavailable because the Python environment is incompatible."""


def _require_h5py():
    """Import h5py only for HDF5 operations, with an actionable ABI error."""
    try:
        import h5py
    except (ImportError, OSError, ValueError) as exc:
        raise HDF5EnvironmentError(
            "Unable to import h5py for UVDoc HDF5 annotations. "
            "This is an environment/dependency error, not evidence of corrupt data. "
            "If NumPy and h5py binaries are incompatible, create a clean Conda "
            "environment and install both from the same channel (for example, conda-forge). "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return h5py


def load_grid2d(path : Path) -> np.ndarray :
    """Match the EDA HDF5 convention: raw (2,61,89) -> (2,89,61)."""
    h5py = _require_h5py()
    with h5py.File(path, "r") as handle :
        if ("grid2d" not in handle) :
            raise KeyError(f"Missing grid2d in {path}")
        raw = np.asarray(handle["grid2d"])
    if (raw.shape != (2, 61, 89)) :
        raise ValueError(f"Unexpected UVDoc grid2d shape {raw.shape}: {path}")
    grid = np.asarray(raw.T.transpose(2, 0, 1), dtype=np.float64)
    if (not np.isfinite(grid).all()) :
        raise ValueError(f"Non-finite UVDoc grid2d: {path}")
    return grid


def load_grid3d(path : Path) -> np.ndarray :
    """Match the EDA HDF5 convention: raw (3,61,89) -> (89,61,3)."""
    h5py = _require_h5py()
    with h5py.File(path, "r") as handle :
        if ("grid3d" not in handle) :
            raise KeyError(f"Missing grid3d in {path}")
        raw = np.asarray(handle["grid3d"])
    if (raw.shape != (3, 61, 89)) :
        raise ValueError(f"Unexpected UVDoc grid3d shape {raw.shape}: {path}")
    grid = np.asarray(raw.T, dtype=np.float64)
    if (not np.isfinite(grid).all()) :
        raise ValueError(f"Non-finite UVDoc grid3d: {path}")
    return grid


def read_metadata(path : Path) -> dict :
    with path.open(encoding="utf-8") as handle :
        value = json.load(handle)
    if (not isinstance(value, dict)) :
        raise ValueError(f"Expected JSON object: {path}")
    return value


def auxiliary_path(record : SampleRecord, kind : str) -> str :
    """Construct only a path, not an assertion that an auxiliary file exists."""
    if (record.dataset not in ("UVDoc Final", "UVDoc Benchmark")) :
        raise ValueError("Auxiliary grids are specific to UVDoc records")
    if (kind not in ("uvmap", "seg")) :
        raise ValueError(f"Unsupported UVDoc auxiliary kind: {kind}")
    image_relative = relative_manifest_path(record.input_path)
    prefix = image_relative.split("/img/", 1)
    if (len(prefix) != 2) :
        raise ValueError(f"Unexpected UVDoc image path: {record.input_path}")
    geom = record.metadata["geom_name"]
    return f"{prefix[0]}/{kind}/{geom}.mat"


def read_auxiliary(path : Path) -> dict[str, np.ndarray] :
    """Return raw HDF5 arrays, without assuming UV-map axis semantics."""
    h5py = _require_h5py()
    with h5py.File(path, "r") as handle :
        arrays = {name: np.asarray(obj) for name, obj in handle.items()
                  if isinstance(obj, h5py.Dataset)}
    if (not arrays) :
        raise ValueError(f"No HDF5 datasets in auxiliary file: {path}")
    return arrays


def load_sample(record : SampleRecord, resolver : PathResolver, include_geometry : bool = False,
                apply_exif : bool = False) -> dict[str, object] :
    if (record.dataset not in ("UVDoc Final", "UVDoc Benchmark")) :
        raise ValueError("Expected UVDoc record")
    sample = {"record": record, "image": read_image(resolver.resolve(record.input_path), apply_exif=apply_exif)}
    if (include_geometry) :
        sample["grid2d"] = load_grid2d(resolver.resolve(record.annotation_paths["grid2d"]))
        sample["grid3d"] = load_grid3d(resolver.resolve(record.annotation_paths["grid3d"]))
    return sample
