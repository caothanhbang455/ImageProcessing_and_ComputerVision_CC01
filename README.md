# Document Image Unwarping

Research and implementation work on correcting geometric distortions in photographed paper documents. The project studies a pretrained neural unwarping baseline, evaluates it on real documents, and develops classical image processing methods for comparison and possible integration.

This is an active course project. The repository is being set up; **the presence of a planned file or experiment in this README does not mean it has been implemented or run**.

## Project scope

A photographed document may contain perspective distortion, page curvature, folds, occlusion, shadows, and complex backgrounds. Geometric unwarping aims to produce an image in which document content is arranged as it would be on a flat page. Appearance correction and text readability are related, but they are evaluated separately from geometric alignment.

The initial deep learning baseline is **UVDoc: Neural Grid-based Document Unwarping**. The project will also investigate classical computer vision methods, but the first implementation milestone is to establish reliable data loading, baseline inference, evaluation, and error analysis.

## Datasets

| Dataset | Role | Current status |
|---|---|---|
| [DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html) | Primary real-world evaluation benchmark | Original photographs, crops, reference scans, and evaluation files are present locally; full audit pending |
| [DIR300](https://github.com/fh2019ustc/DocScanner) | Independent external real-world test benchmark | Planned acquisition and audit |
| [UVDoc Final](https://github.com/tanguymagne/UVDoc-Dataset) | Supervised training data with 2D/3D geometric annotations | Selected; download and annotation audit pending |
| Generated synthetic samples | Controlled geometry experiments and diagnostic tests | Planned, not yet generated |

DocUNet and DIR300 are evaluation datasets, not default training sets. UVDoc Final is the primary candidate for training the original UVDoc-style geometry objective. The full Doc3D download is outside the current storage budget and is not a prerequisite for baseline inference.

See [`context.md`](context.md) for the complete dataset decisions, evaluation plan, and current blockers.

## Repository layout

The local project folder **is the Git repository root**. Do not create a nested clone of the shared team repository.

```text
Document Unwarping/
├── README.md
├── CLAUDE.md                  # Agent working conventions, when installed
├── context.md                 # Project state, decisions, and next work
├── .gitignore
├── requirements-eda.txt      # Existing team EDA dependencies
├── paper/                    # Local papers, ignored by Git
├── dataset/                  # Local datasets, ignored by Git
│   └── DocUNet/
│       ├── original/
│       ├── crop/
│       ├── scan/
│       └── eval/
├── documentation/            # Tracked notes and evaluation decisions
├── notebooks/                # Existing team notebooks and future notebooks
├── notebook/                 # Existing local notebook, preserved and ignored
├── data/                     # Planned tracked dataset loaders and manifests
├── models/                   # Planned model-specific code
│   ├── deep/
│   └── classical/
├── evaluation/               # Planned metrics, protocols, and error analysis
├── training/                 # Planned training code, if needed
├── tests/                    # Planned correctness tests
├── checkpoints/              # Local weights, ignored by Git
└── runs/                     # Local experiment outputs, ignored by Git
```

The directories marked *planned* should be created when implementation needs them. Existing research material and notebooks must not be moved merely to match this layout.

## Getting started

### Local environment

The project is developed in Windows PowerShell and Google Colab. The working directory on Windows is:

```text
G:\My Drive\Document Unwarping
```

The corresponding Google Drive path after mounting in Colab is:

```text
/content/drive/MyDrive/Document Unwarping
```

For the existing dataset-exploration notebooks, the team provides a small CPU-oriented environment:

```powershell
cd "G:\My Drive\Document Unwarping"
python -m pip install -r requirements-eda.txt
python -m jupyter lab
```

These commands install only the existing EDA requirements. **UVDoc inference and training dependencies are not finalized yet.** Do not assume that this environment can load the pretrained checkpoint or train a model.

### Data locations

The existing local DocUNet data is expected at:

```text
dataset/DocUNet/original/
dataset/DocUNet/crop/
dataset/DocUNet/scan/
dataset/DocUNet/eval/
```

Do not rename source files to simplify pairing. The future dataset manifest should handle filenames, extensions, document IDs, and multiple views explicitly. All image and annotation files remain excluded from Git.

## Evaluation approach

We will use the same verified sample manifest and documented preprocessing protocol for all competing methods.

- **Geometric quality:** Local Distortion (LD) and Aligned Distortion (AD), using the appropriate reference evaluation protocol.
- **Image similarity:** Multi-Scale Structural Similarity (MS-SSIM), with the implementation and alignment rules recorded.
- **Text readability:** Character Error Rate (CER) and Edit Distance when a defensible text-reference protocol is available.
- **Efficiency:** Inference runtime measured on a specified device and input size.
- **Synthetic geometry:** 2D/3D grid error when corresponding ground-truth annotations are available.

Evaluation will retain per-image results, metric distributions, and visual failure cases rather than report only average scores. A Python metric implementation will not be described as benchmark-equivalent until its preprocessing and numerical results are validated against the relevant reference implementation.

## Current status and next work

The local branch `research/classical-document-unwarping` was created from the shared repository's `main` branch. Its first push failed with HTTP 403 because the GitHub account lacked permission to write to the team's repository. **The branch exists locally but has not been confirmed on the remote.**

The immediate work is to finalize the dataset manifests and EDA, prepare official UVDoc pretrained inference, implement and validate the evaluation metrics, and establish image-level error analysis. The UVDoc Final archive is reported to be about 27 GB; download and extraction will be planned for Google Colab and Google Drive, with storage and integrity checks. No baseline scores or training results are claimed here.

## References

- [UVDoc model and evaluation code](https://github.com/tanguymagne/UVDoc)
- [UVDoc dataset release and annotation layout](https://github.com/tanguymagne/UVDoc-Dataset)
- [DocUNet benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html)
- [DocScanner and DIR300](https://github.com/fh2019ustc/DocScanner)

For agent behavior, follow `CLAUDE.md`. For decisions, dataset status, evaluation conventions, and unresolved questions, read [`context.md`](context.md). For implementation details and actual results, use the relevant files in `documentation/` and recorded run outputs.
