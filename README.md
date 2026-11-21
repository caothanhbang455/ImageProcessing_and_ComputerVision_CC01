# Document Image Unwarping

This project studies ways to correct geometric distortion in photographed documents. It covers classical image processing and deep-learning methods, with the official pretrained [UVDoc](https://github.com/tanguymagne/UVDoc) model as the initial reference. The goal is to compare methods on consistent datasets and understand where they work, where they fail, and why.

A flat page photographed at an angle may need only perspective correction. A curved or folded page needs a more flexible transformation. These are geometric problems, distinct from removing shadows, sharpening text, or improving OCR results.

## Documentation

The [documentation index](documentation/README.md) is the best place to start. The main guides are:

| English | Tiếng Việt | Topic |
|---|---|---|
| [Implementation guide](documentation/implementation_guide.md) | [Hướng dẫn sử dụng repository](documentation/implementation_guide-vi.md) | Dataset access, writing methods, inference, saved outputs |
| [Understanding methods](documentation/methods_overview.md) | [Tìm hiểu các phương pháp](documentation/methods_overview-vi.md) | Perspective correction, nonrigid geometry, neural and hybrid methods |
| [Experiments and evaluation](documentation/experiments_and_evaluation.md) | [Thực nghiệm và đánh giá](documentation/experiments_and_evaluation-vi.md) | Fair comparisons, metrics, failures, reporting |

Precise [dataset](documentation/reference/dataset_protocols.md), [geometry](documentation/reference/geometry_conventions.md), and [evaluation](documentation/reference/evaluation_protocols.md) conventions are in `documentation/reference/`. Earlier implementation notes and research reports are retained in `documentation/archive/`; they are historical records, not instructions for the current setup.

## Datasets

| Dataset | Manifest records | Main use |
|---|---:|---|
| [UVDoc Final](https://github.com/tanguymagne/UVDoc-Dataset) | 20,000 | Geometric annotations for method development and possible training |
| UVDoc Benchmark | 50 | Official UVDoc line evaluation and texture preparation |
| [DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html) | 130 | Real document images with scanned references |
| [DIR300](https://github.com/fh2019ustc/DocScanner) | 300 | An additional real-image evaluation set |

These are manifest counts, not a promise that every dataset is present on a particular machine. Raw files and metadata live under `dataset/` and are not tracked in Git. DocUNet and DIR300 are evaluation datasets, not default training sets.

## Project layout

```text
Document Unwarping/
├── configs/            # Dataset, model and evaluation settings
├── data/               # Sample records, dataset readers and image loading
├── geometry/           # Coordinates, grids, homography and sampling
├── models/             # MethodResult interface and model implementations
├── evaluation/         # Evaluation protocols, metrics and run reader
├── scripts/            # Method inference, UVDoc inference and evaluation
├── tests/              # Automated tests
├── notebooks/          # Colab experiments and investigation
├── requirements.txt    # Shared Python dependencies
├── documentation/      # Guides, references and archived research notes
├── dataset/            # Local datasets (ignored by Git)
├── checkpoints/        # Local model weights (ignored by Git)
├── runs/               # Inference and evaluation outputs (ignored by Git)
└── paper/              # Local papers (ignored by Git)
```

`data/` contains Python dataset utilities; `dataset/` contains the actual images and annotations. New classical methods can be placed in `models/classical/` when needed, while neural implementations belong in `models/deep/`. There is no need to create unused directories in advance.

## Getting started

Google Colab is the environment used for the verified UVDoc and evaluation runs. In a Colab cell, mount Drive and work from the repository root:

```python
from google.colab import drive

drive.mount("/content/drive")
%cd "/content/drive/MyDrive/Document Unwarping"
```

Install the shared Python dependencies and run the tests:

```python
%pip install -r requirements.txt
!python -m pytest -q tests/
```

The latest reported complete suite passed **95 tests** in Colab on 21 November 2026. The single requirements file uses the CUDA-enabled PyTorch already provided by Colab; it does not install a replacement PyTorch build or the MATLAB reference evaluation environment. On Windows, use normal PowerShell commands without the Colab `%` and `!` prefixes.

To select DIR300 samples:

```python
from pathlib import Path
from data.manifests import load_records

root = Path("/content/drive/MyDrive/Document Unwarping")
records = load_records("dir300", root, variant="dist")
print(records[0].sample_key)
```

For a new method, `scripts.run_method.run_method()` accepts a function that receives an RGB image and returns `MethodResult`. The official UVDoc path uses `scripts/run_inference.py` and retains its model-specific coarse grids. Prediction runs are saved in `runs/inference/`; compatible evaluation results go in `runs/evaluation/`. See the implementation guide for a complete example.

## Evaluation and verified status

Evaluation is dataset-specific. UVDoc Benchmark has an official line metric that requires a compatible coarse grid. DocUNet and DIR300 support reference-data preparation; published image metrics require the appropriate MATLAB and SIFT-flow dependencies. Successfully preparing reference images is not the same as computing benchmark scores.

In the recorded Colab work, official UVDoc inference and line evaluation completed for all 50 benchmark samples. The independent line-metric and texture comparisons passed for two selected samples. MATLAB-based benchmark scores were not calculated in that environment. These checks establish reproducibility of the tested paths, not the performance of any future method.

The [evaluation guide](documentation/experiments_and_evaluation.md) explains how to interpret comparisons; the exact scoring requirements are in the [evaluation protocols](documentation/reference/evaluation_protocols.md).

## References

- [UVDoc model](https://github.com/tanguymagne/UVDoc) and [UVDoc dataset](https://github.com/tanguymagne/UVDoc-Dataset)
- [DocUNet benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html)
- [DocScanner / DIR300](https://github.com/fh2019ustc/DocScanner)

Do not commit `dataset/`, `runs/`, `checkpoints/`, local research PDFs, caches, or local `CLAUDE.md` files. Earlier reports are preserved under `documentation/archive/` so the active documentation can stay focused on current use.
