# Stage 1 environment compatibility hotfix

This overlay addresses an import problem found on Windows Anaconda `base` on 21 November 2026.
It does not change the dataset manifests, the existing UVDoc split, model implementations,
or the research protocol.

## Root cause

`ValueError: numpy.dtype size changed, may indicate binary incompatibility` during
`import h5py` means the local NumPy and `h5py` compiled extensions are incompatible.
The traceback is evidence of an environment problem, **not** a malformed image or annotation.
Installing packages into a busy `base` environment can produce mixed binary dependency versions.

## Recommended isolated Windows environment

Open Anaconda PowerShell Prompt:

```powershell
conda create -n docunwarp -c conda-forge python=3.11 numpy h5py pillow pyyaml pytest -y
conda activate docunwarp
cd "G:\My Drive\Document Unwarping"
python -c "import numpy, h5py; print('NumPy:', numpy.__version__, 'h5py:', h5py.__version__)"
python -m pytest -q tests
python scripts/validate_datasets.py --root "G:\My Drive\Document Unwarping" --check-files --max-files-per-dataset 3 --smoke-read 2
```

This environment is intended for Stage 1; later GPU training/inference may need a
separate dependency setup. Avoid changing the Anaconda `base` environment to fix this issue.

## Code changes

- `data/images.py` provides shared image decoding independently of HDF5.
- `data/docunet.py` and `data/dir300.py` import image helpers from `data.images`.
- `data/uvdoc.py` loads `h5py` only when accessing `.mat` HDF5 annotations; image readers
  remain available, and the old `data.uvdoc.read_image` path remains available.
- `scripts/validate_datasets.py` records `ENVIRONMENT_ERROR` rather than aborting or
  mislabeling a broken HDF5 import as a corrupt dataset. Exit code `3` indicates this issue.
- `tests/test_optional_dependencies.py` verifies that image reader imports succeed without
  HDF5 and that broken HDF5 produces an explicit environmental failure.

The overlay contains only a small set of changed source, test, and documentation files.
It must be extracted into the **existing project root**, not a nested directory.

## Validation outcome interpretation

- `PASS`: requested check succeeds.
- `UNAVAILABLE`: required image, reference, or annotation was not found in the configured roots.
- `ENVIRONMENT_ERROR`: a dependency such as `h5py` could not import; fix the environment.
- `FAIL`: a record, annotation or decoded image did not pass the implemented check.

An `ENVIRONMENT_ERROR` is still a failed smoke test, but it is distinct from dataset corruption.
