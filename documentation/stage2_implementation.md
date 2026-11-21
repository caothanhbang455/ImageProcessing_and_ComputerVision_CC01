# Stage 2 implementation: Shared geometry and official UVDoc inference

**Date:** 21 November 2026  
**Status at package handoff:** Source implemented and synthetic/CPU integration tested. Official pinned-checkpoint parity and actual-image inference require a Colab run with the user's external assets and GPU.

## Scope and files

Stage 2 adds `geometry/{coordinates,grids,homography,sampling,composition}.py`, `models/deep/uvdoc.py`, `configs/models/uvdoc_official.yaml`, `scripts/run_inference.py`, and `scripts/verify_uvdoc_reference.py`, together with corresponding tests and Colab notebook. It does not edit the Stage 1 readers or method contract, change datasets or training splits, implement new models, tune hyperparameters, or score the published benchmarks.

The official source is an external checkout of `https://github.com/tanguymagne/UVDoc.git` at **`4c9b82b537057aff2526e6dd118a847cdd072e82`**. The required checkpoint is `model/best_model.pkl`, SHA-256 **`7e90861b8a516eb4bc51f84bd889cb77275743d2d1d3ca8091951ec9f2b7da23`**. No original weights or third-party model source are included in the team patch. `verify_source` checks the Git revision and that `demo.py`, `utils.py`, and `model.py` are unchanged. The adapter refuses different source commits, checkpoint hashes, model settings, or architecture parameter counts.

## Minimal API

```python
from pathlib import Path
from data.manifests import load_records
from data.paths import PathResolver
from models.deep.uvdoc import load_uvdoc_model, predict_uvdoc

root = Path('/content/drive/MyDrive/Document Unwarping')
record = next(r for r in load_records('uvdoc_benchmark', root) if r.sample_id == '00010')
input_file = PathResolver(root).resolve(record.input_path)
loaded = load_uvdoc_model(source_dir='/content/uvdoc_reference_stage2', device='cuda')
prediction = predict_uvdoc(loaded, input_file)
assert prediction.result.status == 'OK'
print(prediction.result.image.shape, prediction.grid2d.shape, prediction.grid3d.shape)
```

The default `source_dir` is `/content/uvdoc_reference_source`, which matches the old pilot. Supplying an explicit source path as above is preferred for the new notebook. `predict_uvdoc_from_rgb` accepts a predecoded uint8 RGB image for unit testing or controlled use, but the official file-oriented adapter uses OpenCV decoding and should be used for reference comparison.

## Run structure and execution

Use the supplied `[21.11] Stage_2_Geometry_and_UVDoc_Inference.ipynb` in Colab. It applies the source patch with backups, checks the existing Stage 1 tests, runs the new tests, stages only missing UVDoc Final representative members where necessary, obtains and verifies the pinned upstream source, runs model preflight, performs a separate-process official demo comparison, and runs small multi-dataset diagnostics.

For a short Colab command invoked **inside a notebook Python cell**:

```python
import subprocess, sys
subprocess.run([sys.executable, 'scripts/run_inference.py',
                '--root', str(ROOT), '--dataset', 'uvdoc_benchmark',
                '--sample-id', '00010', '--source-dir', str(OFFICIAL_SOURCE),
                '--device', 'cuda'], cwd=str(ROOT), check=True)
```

The runner requires `--sample-id` or `--limit`; it will not silently run all 20,000 UVDoc Final samples. Supported datasets are `uvdoc_final`, `uvdoc_benchmark`, `docunet` (`crop` and `original` variants), and `dir300`. Outputs live under `runs/inference/<run_id>/` by default and include:

- `config.json`, `environment.json`, `provenance.json` with source, checkpoint and manifest identities.
- `samples.csv` with a record for every attempted sample, including missing inputs or failures.
- `predictions/uvdoc_official/images/<sample_key>.png` and `coarse_grids/<sample_key>.npz` for successful predictions.
- Optional `dense_maps/*.npz` and UVDoc Benchmark `benchmark_maps/*.mat` (MATLAB-compatible `bm` array, stored as 45×31×2).
- `diagnostics/timings.csv`, `grid_statistics.csv`, `failures.csv`, `artifacts.csv`, `summary.json` and `report.md`.

A run refuses to overwrite an existing directory. All timing sections are synchronized at CUDA boundaries. Peak-memory comparisons and benchmark accuracy scoring are intentionally deferred.

## Independent official comparison

`scripts/verify_uvdoc_reference.py` uses an independently executed upstream `demo.py` in a subprocess on a *temporary byte-for-byte copy* of the selected image. A separate upstream network invocation also writes the preprocessed input and coarse grids. The verifier compares model input, 2D/3D grids, and final uint8 output. The PNG gate demands exact pixel equality in the matching Colab environment; numeric coarse grids must agree within `1e-5`. It writes a CSV and JSON audit even when parity fails. This audit is not a benchmark score.

The preferred first comparison sample is the previously inspected UVDoc Benchmark `00010`, for which the old pilot reported exact official-demo equality. That past result is not proof of parity for this new implementation. The new Colab notebook must produce its own actual parity report.

## Verification status at package creation

The current work performed automated unit and fixture integration tests in an isolated Stage 1 checkout. The new tests use a deterministic synthetic grid model; they do not load the actual UVDoc checkpoint, and they must not be described as pretrained inference. The package's verification summary records actual pytest results and any discovered limitations.

## Environment and storage

- The geometry package can be tested on CPU.
- Official model parity is intended for float32 with a CUDA-enabled PyTorch build, ideally Colab L4 as in the pilot. Do not automatically replace Colab's CUDA PyTorch installation.
- The Stage 2 requirements list only non-Torch dependencies. Install anything absent from Colab without forcing a NumPy upgrade into an incompatible environment.
- Clone the upstream reference to `/content/uvdoc_reference_stage2`; keep it outside the team Git repository.
- Keep checkpoints, staged samples, and artifacts out of Git. The official checkpoint is verified against SHA-256 before any model inference.
- The Stage 1 validation report from the user's earlier run reported two UVDoc Final samples unavailable, and the user subsequently staged eight missing files. A successful second Stage 1 physical check was not yet supplied at the time of this package's creation.

## Research boundaries

Stage 2 does not select a classical nonrigid method, additional pretrained network, new training policy, experiment matrix, hybrid strategy, or published metric set. Stage 3 should validate reference metrics and image alignment using saved inference artifacts, not rerun models merely because an evaluation implementation changes.
