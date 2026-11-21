# tests/test_optional_dependencies.py
"""Stage 1 smoke checks must isolate broken optional HDF5 dependencies."""

import subprocess
import sys
from pathlib import Path

from PIL import Image

from data.manifests import load_records
from data.uvdoc import HDF5EnvironmentError
from scripts.validate_datasets import smoke_read


def test_image_readers_do_not_require_h5py_at_import_time() -> None:
    # A fresh process makes this independent of pytest's own h5py imports.
    script = """
import builtins
real_import = builtins.__import__
def block_h5py(name, *args, **kwargs):
    if name == 'h5py':
        raise ValueError('numpy.dtype size changed, may indicate binary incompatibility')
    return real_import(name, *args, **kwargs)
builtins.__import__ = block_h5py
from data import uvdoc, docunet, dir300
from data.images import read_image, image_header
from pathlib import Path
try:
    uvdoc.load_grid2d(Path('unused.mat'))
except uvdoc.HDF5EnvironmentError as error:
    assert 'environment/dependency error' in str(error)
else:
    raise AssertionError('Broken h5py import should become HDF5EnvironmentError')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_smoke_classifies_hdf5_import_failure_as_environment_error(
    small_project: Path, monkeypatch
) -> None:
    from data import uvdoc

    record = load_records("uvdoc_final", small_project)[0]
    image = small_project / record.input_path
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 40)).save(image)

    def broken_hdf5(*args, **kwargs):
        raise HDF5EnvironmentError("Test-simulated NumPy/h5py ABI mismatch")

    monkeypatch.setattr(uvdoc, "load_sample", broken_hdf5)
    rows = smoke_read(small_project, 1)
    uvdoc_row = next(row for row in rows if row["dataset"] == "UVDoc Final")
    assert uvdoc_row["status"] == "ENVIRONMENT_ERROR"
    assert "ABI mismatch" in uvdoc_row["reason"]
    assert all(row["status"] != "FAIL" for row in rows)
