# Project Structure and Implementation Guide

This guide explains how to use the existing repository to implement and evaluate document image unwarping methods. It covers the main directories, the shared data and method interfaces, inference outputs, and the available evaluation procedures. The examples use the current Python modules and can be run from Google Colab after mounting the project folder.

**Quick start:** Select records with `data.manifests.load_records()`, write a function that receives an RGB image and returns `MethodResult`, run it with `scripts.run_method.run_method()`, and evaluate the saved predictions with a compatible protocol. The official UVDoc baseline has its own inference runner and remains unchanged.

## 1. Project overview

Document unwarping corrects geometric distortion in photographed pages. Perspective changes the apparent page shape, while folds and curvature can distort text and illustrations within the page. A rectification method produces an image intended to represent a flatter document.

The repository supports classical computer vision methods and deep-learning methods. Classical approaches may use page boundaries, homographies, or other explicit geometric rules. Deep-learning approaches may predict a corrected image or a sampling map. Both can use the same dataset identities and output conventions.

Three operations form the common workflow: load a sample, run a method, and evaluate the saved result. Evaluation is protocol-specific. Producing a visually plausible image does not automatically establish a better geometric correction or a valid published benchmark score.

## 2. Repository structure

The project folder is also the Git repository root. Python modules are imported from this root. The following tree shows the main existing locations and the intended location for classical implementations:

```text
Document Unwarping/
├── README.md
├── configs/
│   ├── datasets.yaml
│   ├── evaluation.yaml
│   └── models/uvdoc_official.yaml
├── data/
│   ├── records.py
│   ├── manifests.py
│   ├── paths.py
│   ├── images.py
│   ├── validation.py
│   ├── uvdoc.py
│   ├── docunet.py
│   └── dir300.py
├── geometry/
│   ├── coordinates.py
│   ├── grids.py
│   ├── homography.py
│   ├── sampling.py
│   └── composition.py
├── models/
│   ├── interfaces.py
│   ├── deep/uvdoc.py
│   └── classical/                  # Intended location for new methods
├── evaluation/
│   ├── metrics.py
│   ├── alignment.py
│   ├── aggregation.py
│   ├── reference_metrics.py
│   ├── runner.py
│   └── protocols/
│       ├── uvdoc.py
│       ├── docunet.py
│       └── dir300.py
├── scripts/
│   ├── run_method.py
│   ├── run_inference.py
│   ├── evaluate.py
│   ├── validate_datasets.py
│   └── verify_uvdoc_reference.py
├── tests/
├── notebooks/
├── documentation/
├── dataset/                     # Local source data, not tracked
├── checkpoints/                 # Local model weights, not tracked
├── runs/                        # Generated experiment outputs, not tracked
└── paper/                       # Local research papers, not tracked
```

| Directory | Purpose and typical use |
|---|---|
| `configs/` | Dataset locations, evaluation protocols, and model settings. A new model-specific configuration belongs under `configs/models/` when needed. |
| `data/` | Stable sample records, manifest readers, path resolution, image decoding, and dataset-specific annotation access. |
| `dataset/` | Original images, annotations, and generated local metadata. Do not modify original dataset files. |
| `geometry/` | Reusable coordinate conversion, homography, backward sampling, and map composition. |
| `models/` | The shared output contract and method implementations. Place new classical code in `models/classical/` and neural adapters in `models/deep/`. |
| `evaluation/` | Benchmark preparation, metric calculations, external reference integrations, and result summaries. |
| `scripts/` | Reusable entry points for inference, dataset validation, and evaluation. |
| `tests/` | Unit and integration tests for data, geometry, methods, and evaluation. |
| `notebooks/` | Colab setup, exploration, visual inspection, and execution examples. Reusable algorithms should remain in Python modules. |
| `documentation/` | Dataset rules, geometry conventions, implementation explanations, and research findings. |
| `runs/` | New inference and evaluation outputs, including images, CSV results, and reports. |

The distinction between `data/` and `dataset/` is intentional. The first contains Git-tracked Python source; the second contains local research data. Similarly, `models/` contains code, while `checkpoints/` stores model weights.

For a new transformation used by several methods, add a focused function in `geometry/` rather than copying it into multiple models. For a new benchmark metric, use `evaluation/` and document its reference protocol. Avoid creating empty packages or moving existing files to match a hypothetical architecture.

## 3. Working with datasets

`data.manifests.load_records()` reads the established dataset manifests and returns `SampleRecord` objects. It does not download data or assign a new training split. Records use stable sample identities, independent of where the files are stored.

Supported dataset keys and input variants are:

| Dataset key | Variant | Main use |
|---|---|---|
| `uvdoc_final` | `rendered` | Annotated rendered document data |
| `uvdoc_benchmark` | `rendered` | UVDoc benchmark inference and geometry evaluation |
| `docunet` | `crop` or `original` | Distorted crops or original photographs |
| `dir300` | `dist` | Distorted images paired with flat references |

The following example loads two DIR300 records and resolves the first image path:

```python
from pathlib import Path
from data.manifests import load_records
from data.paths import PathResolver
from data.images import read_image

ROOT = Path("/content/drive/MyDrive/Document Unwarping")
records = load_records("dir300", ROOT, variant="dist")[:2]
resolver = PathResolver(ROOT)

sample = records[0]
image_path = resolver.resolve(sample.input_path)
image = read_image(image_path, mode="RGB")
print(sample.sample_key, image.shape)
```

`SampleRecord` includes `sample_key`, `dataset`, `sample_id`, `variant`, `input_path`, optional `reference_path`, `group_id`, `annotation_paths`, and `metadata`. Use `sample_key` as the unique identity within run artifacts. `group_id` identifies a related document or geometry group when available. Paths in the manifest are relative to the repository root; `PathResolver` locates the physical file, including supported Colab runtime staging locations.

Do not pair input and reference images by directory order. DocUNet contains multiple views of some documents, and DIR300 pairs are identified by numbered IDs. The UVDoc datasets provide separate geometry annotations, whose representation must be interpreted through their dataset-specific readers. See [Dataset protocols](reference/dataset_protocols.md) for the full pairing and split rules.

## 4. Implementing methods

A general method is a Python callable that receives a `uint8` RGB image and returns `models.interfaces.MethodResult`. The method can be a function or a callable object. The runner does not assume that the method uses PyTorch.

The smallest valid image-only example is:

```python
from models.interfaces import MethodResult

def identity_method(image) :
    return MethodResult(status="OK", image=image.copy())
```

This identity function is useful for checking the interface. It does not correct distortion and should not be used to claim benchmark improvement.

`MethodResult` has the following fields:

| Field | Meaning |
|---|---|
| `status` | `OK`, `FAILED`, `UNAVAILABLE`, or `REVIEW_REQUIRED` |
| `image` | Corrected image array for an `OK` result; the saved output must be `uint8` |
| `backward_map` | Optional dense output-to-source sampling map |
| `map_space` | Required with a map: `source_pixels` or `normalized_minus1_plus1` |
| `valid_mask` | Optional boolean mask matching the output image size |
| `failure_reason` | Required when the status is not `OK` |
| `metadata` | Optional JSON-compatible method details |

For example, a classical page detector that finds no usable page boundary should return a failure rather than an apparently successful copy of the input:

```python
return MethodResult(
    status="FAILED",
    image=None,
    failure_reason="No valid page quadrilateral found",
)
```

A method that produces geometry can return a dense backward map. This example uses the identity mapping to demonstrate the required coordinate format:

```python
import numpy as np

def identity_with_map(image) :
    height, width = image.shape[:2]
    y, x = np.indices((height, width), dtype=np.float32)
    backward_map = np.stack((x, y), axis=-1)
    return MethodResult(status="OK", image=image.copy(),
                        backward_map=backward_map, map_space="source_pixels")
```

The map must match the output dimensions and contain finite coordinates. A validity mask is optional.

Classical methods generally belong in `models/classical/`; neural model adapters belong in `models/deep/`. Model loading and checkpoint initialization should happen once, before processing the sample list. The callable can then capture the initialized model. Preserve the original UVDoc adapter as a reproducible reference rather than changing its preprocessing to fit an experimental variant.

## 5. Geometry conventions

Image arrays use `(height, width, channels)` order. Coordinates use `(x, y)`, with the origin at the center of the upper-left pixel, `x` increasing rightward, and `y` downward.

A **forward** transform maps points from the source to the target. A **backward** map instead tells the image sampler which source pixel supplies each output pixel:

```text
backward_map[y_output, x_output] = (x_source, y_source)
```

A dense backward map has shape `(H_output, W_output, 2)`. It may contain source pixel coordinates or normalized coordinates, but the choice must be declared with `map_space`. Interpolation mode, padding behavior, and PyTorch's `align_corners` setting affect the result and should never be implicit in a new method's geometric interpretation.

The official UVDoc network predicts coarse 2D and 3D grids. These are different from the dense map accepted by `MethodResult`. In particular, adding a dense map to a new method does not make it compatible with the official UVDoc line metric, which expects UVDoc's coarse backward grid representation.

Use the tested utilities in `geometry/` for repeated operations. See [Geometry conventions](reference/geometry_conventions.md) for homography direction, map composition, coordinate conversion, and sampling formulas.

## 6. Running inference and saving predictions

Two inference paths are available:

- `scripts/run_method.py` runs independently implemented image methods through the shared `MethodResult` interface.
- `scripts/run_inference.py` executes the pinned official UVDoc baseline and saves its model-specific coarse grids.

The general runner accepts records from one dataset and input variant per run. It validates their identities against the manifest, reads images, calls the method, and saves outputs and failures.

This complete example uses the same records from Section 3:

```python
from datetime import datetime, timezone
from uuid import uuid4
from scripts.run_method import run_method

run_id = "identity_dir300_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
run_id += "_" + uuid4().hex[:6]
inference_dir = ROOT / "runs/inference" / run_id

summary = run_method(
    records=records,
    method=identity_method,
    root=ROOT,
    output_dir=inference_dir,
    method_id="identity_example",
    config={"purpose": "interface verification"},
)
print(summary)
```

Run the code after defining `identity_method` in Section 4. The selected source images must exist in the current environment. Each run uses a fresh output directory; the runner refuses to overwrite an existing one.

A general inference run includes:

```text
runs/inference/<run_id>/
├── config.json
├── environment.json
├── provenance.json
├── samples.csv
├── artifacts.csv
├── summary.json
└── predictions/<method_id>/
    ├── images/              # Successful images
    ├── dense_maps/          # Created only when geometry is returned
    └── metadata/            # Created only when metadata is returned
```

`samples.csv` keeps every attempted sample and its status. `artifacts.csv` records saved file paths, hashes, and sizes. `config.json` identifies the method and supplied settings; `provenance.json` records the method, manifest fingerprint, and repository revision when available. Include model/checkpoint identity in the supplied configuration when applicable, because a general callable does not automatically expose its weight file.

The runner converts output RGB images to the channel order expected by PNG encoding. Inputs and existing dataset files are not modified. Experiment outputs belong under `runs/`, not in source directories or Git.

## 7. Evaluating results

`evaluation.runner.evaluate_inference_run()` reads a saved inference run. It checks manifest identities and the recorded artifact hashes before using any prediction. Evaluation creates a separate output directory and does not rerun the method.

The available protocol IDs are defined in `configs/evaluation.yaml`:

| Protocol | Required output and result |
|---|---|
| `uvdoc_lines_official_n25_v1` | UVDoc-compatible coarse backward grid and benchmark UV map; horizontal and vertical line deviation in pixels |
| `uvdoc_texture_prepare_v1` | UVDoc coarse grid; prepares unwarped texture images |
| `uvdoc_visual_matlab_v1` | UVDoc texture preparation and reference MATLAB dependencies for MS-SSIM/AD |
| `docunet_matlab_legacy_v1` | DocUNet `crop` predictions; prepares the historical reference procedure for MS-SSIM/LD/AD |
| `docunet_matlab_corrected64_v1` | Separate policy for document 64; corrected-input inference may be required |
| `dir300_matlab_v1` | DIR300 `dist` predictions; prepares the selected MS-SSIM/LD reference procedure |

For the identity DIR300 run above, evaluate the saved images using:

```python
from evaluation.runner import evaluate_inference_run

eval_id = "identity_dir300_eval_" + uuid4().hex[:8]
evaluation_dir = ROOT / "runs/evaluation" / eval_id

result = evaluate_inference_run(
    root=ROOT,
    inference_run=inference_dir,
    protocol_id="dir300_matlab_v1",
    output_dir=evaluation_dir,
)
print(result["status_counts"])
print(evaluation_dir / "metrics/per_sample.csv")
```

This call can successfully **prepare** the reference inputs while MS-SSIM and LD remain `UNAVAILABLE_DEPENDENCY` if the MATLAB reference environment is unavailable. A prepared image is not a computed benchmark score. To attempt supported external evaluation, the runner also accepts `run_matlab=True` with the relevant reference dependencies available. Its CLI is `scripts/evaluate.py`, which takes `--root`, `--inference-run`, and `--protocol` (and optional reference arguments).

A general image-only method cannot receive an official UVDoc line score. It needs a verified coarse backward grid in the expected representation. The evaluator records unsupported cases as `NOT_APPLICABLE` rather than inventing a map or metric. A failure during inference remains visible as `UNAVAILABLE_PREDICTION` during evaluation.

Evaluation outputs are stored under `runs/evaluation/<evaluation_id>/`. Inspect `samples.csv` for evaluation status, `metrics/per_sample.csv` for values and reasons, `metrics/summary.csv` for coverage and statistics, and `report.md` for the run summary. UVDoc line runs also save `metrics/per_line.csv` and reference parity evidence when requested.

The recorded Colab validation completed the UVDoc 50-image line evaluation and obtained exact independent line-metric parity on samples `00010` and `00011`. MATLAB reference metrics were not computed in that environment because MATLAB Engine was unavailable. These are separate verification statements; neither implies image similarity or OCR performance. For protocol definitions and limitations, see [Evaluation protocols](reference/evaluation_protocols.md).

## 8. Testing and maintaining the repository

New methods should have focused tests for valid output images, required failure information, input/output sizes, and any optional backward-map convention. When saving runs, check that the sample identities, output paths, artifact hashes, and reported failures are correct. Test an applicable evaluation protocol on a small fixed sample set before attempting full-benchmark comparison.

In Google Colab, after mounting the project and setting its root as the working directory:

```python
%cd "/content/drive/MyDrive/Document Unwarping"
!python -m pytest -q tests/
```

The last reported complete suite with the general method integration passed **95 tests** in Colab. This is an observed test result, not a guarantee that later modifications will pass.

Use separate Git branches for independent work. Stage only the intended source or documentation files, inspect the staged diff, and record actual verification in the commit message. Keep `dataset/`, `checkpoints/`, `runs/`, temporary files, and the locally ignored `CLAUDE.md` outside Git. Preserve completed notebooks and previous experiment outputs.

| Problem | First place to check |
|---|---|
| Dataset file cannot be found | `configs/datasets.yaml`, manifest path, `data/paths.py`, and local dataset staging |
| A sample is paired with the wrong reference | `SampleRecord.sample_key`, `reference_path`, and [Dataset protocols](reference/dataset_protocols.md) |
| A method produces `FAILED` | Its `failure_reason`, `samples.csv`, and method-specific tests |
| Backward map has wrong shape or direction | `models/interfaces.py` and [Geometry conventions](reference/geometry_conventions.md) |
| Evaluation reports `NOT_APPLICABLE` | Selected protocol and required grid representation |
| Evaluation reports `UNAVAILABLE_DEPENDENCY` | MATLAB/SIFT-flow availability and `evaluation/reference_metrics.py` |
| A saved result is rejected | `artifacts.csv`, recorded hashes, and manifest fingerprint |
| A Colab-only command fails in PowerShell | Run `%cd` and `!python` in a Colab cell, not in Windows PowerShell |

Older implementation notes are kept in the [documentation archive](archive/implementation/) for historical reference. For current benchmark requirements, consult [Evaluation protocols](reference/evaluation_protocols.md). The shared interfaces described here should remain small; a new method normally needs a callable, its configuration, and focused tests rather than changes throughout the repository.
