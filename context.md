# Project Context

**Last updated:** November 18, 2026  
**Status:** Initial repository, data, baseline, and evaluation setup. No new baseline results or completed training runs have been verified.

This file records the project state and agreed technical direction so later conversations and agents can continue without reconstructing the discussion. It is not the agent behavior guide (`CLAUDE.md`), a complete implementation specification, or an experiment log. Update it when a decision changes or a proposed component becomes verified.

## 1. Goal and scope

The project studies geometric document image unwarping, with classical image processing as the main area for new implementation and a pretrained neural method as the starting point for comparison. A successful experiment should produce geometrically rectified document images, evaluate them under a consistent protocol, and explain where the methods succeed or fail.

The first baseline is the **official pretrained UVDoc model**. Classical methods are intentionally deferred during the initial setup phase, but future work may include document localization, boundary handling, geometric transformations, interpolation, and classical preprocessing or postprocessing. The specific improvement will be selected after inspecting baseline errors.

Do not present illumination correction, improved OCR, and improved geometric alignment as the same outcome. They may correlate, but they answer different experimental questions.

## 2. Local repository and collaboration

The local Git repository root is the user's existing folder:

```text
Windows: G:\My Drive\Document Unwarping\
Colab:   /content/drive/MyDrive/Document Unwarping/
```

There must not be a nested folder named after the shared GitHub repository.

Shared remote:

```text
origin: https://github.com/caothanhbang455/ImageProcessing_and_ComputerVision_CC01.git
```

Long-lived local development branch:

```text
research/classical-document-unwarping
```

The branch was created locally from `origin/main` at commit `c243dc7` on November 18, 2026. `git fetch origin` succeeded, but the first attempt to push the branch returned HTTP 403 for GitHub user `phatpinkkk`. Remote publication is therefore **pending repository write access**. The user will resolve access with a teammate. Do not claim the branch is already on GitHub.

The team may use several independent long-lived branches, without merging every experiment into `main`. Respect the user's active branch and do not perform Git operations on the user's behalf.

## 3. Existing local content

The user's local folder already contains the following material:

```text
paper/                       # PDFs of relevant research papers
dataset/DocUNet/_archives/   # Original downloaded ZIP files
dataset/DocUNet/original/    # Photographs including surrounding background
dataset/DocUNet/crop/        # Document-centered crop photographs
dataset/DocUNet/scan/        # Flatbed-scanned references
dataset/DocUNet/eval/        # MATLAB evaluation and SIFT-flow code
documentation/              # Existing DocUNet inspection and mapping reports
notebook/Untitled0.ipynb     # Existing local exploratory notebook
```

The original directory listing includes `documentation/docunet_inspection.md`, `documentation/docunet_manifest.json`, and `documentation/docunet_document_groups.csv`. They are earlier inspection artifacts, not evidence that the new canonical dataset loader and evaluation protocol have been implemented.

After switching to the team branch, the root also contains tracked team files, including `README.md`, `.gitignore`, `requirements-eda.txt`, and `notebooks/`. The team's existing EDA notebooks are useful references but are not the required implementation design. Keep them intact unless a change is specifically requested.

The supplied DocUNet directory listing contains the original photos, crops, scans, and evaluation support files. A previous teammate EDA reported 130 readable crops, 65 readable reference scans, and 130 matched pairs. The new project should verify those counts against the **user's own current files**, especially when adding the 130 original photographs to the manifest.

## 4. Dataset decisions

| Dataset | Intended use | Acquisition / implementation state |
|---|---|---|
| **DocUNet** | Primary real-world benchmark; separate original and cropped input variants | Present locally; new manifest and audit pending |
| **DIR300** | External real-world evaluation | Selected; complete local acquisition and audit not yet verified |
| **UVDoc Final** | Main candidate for supervised dual-grid training and geometry EDA | Selected; ZIP size reported by user as approximately 27 GB; download pending |
| **Generated synthetic samples** | Small controlled geometry checks or possible lightweight 2D training | Proposed, not generated |
| **Doc3D** | Historical UVDoc training source | Full dataset excluded from immediate project because of its large storage footprint |

### DocUNet

The benchmark has 65 physical documents with two distorted photographs per document and a flatbed reference scan per document. The local `original/` and `crop/` directories contain different input representations. Pair each input with its reference using the document ID and view ID; do not match by filename order. Source extensions vary in case, and cropped filenames may have an additional ` copy` suffix.

Run the initial baseline on `crop/` first, and report results from `original/` separately. The input variant must appear in result metadata. The reference images do not provide a ground-truth dense unwarping field or a validated character transcript.

The benchmark's orientation and image-alignment conventions require explicit verification. In particular, previous work documents an orientation mismatch for samples associated with document 64. Do not silently correct this or any other sample without recording the protocol.

### DIR300

DIR300 contains 300 distorted/reference image pairs, typically organized as `dist/` and `gt/` with corresponding numeric IDs. The teammate's shared EDA was performed on an incomplete download (148 distorted images, 150 references, 83 valid pairs). Those figures describe that incomplete run, **not** the final dataset or the user's local folder.

Audit its complete image set before using it for an external test. Keep DIR300 out of training and development-time checkpoint selection.

### UVDoc Final

Official dataset repository: https://github.com/tanguymagne/UVDoc-Dataset

The published final dataset contains, among other items:

```text
UVDoc_final/
├── img/
├── grid2d/
├── grid3d/
├── img_geom/
├── metadata_geom/
├── metadata_sample/
├── seg/
├── textures/
├── uvmap/
├── warped_textures/
├── wc/
└── split.json
```

Use it to verify the official UVDoc training input, geometry targets, annotation shapes, coordinate conventions, and split metadata. The user reports an archive size of about 27 GB. Plan a Colab notebook to download it directly from the official URL to Google Drive, with transfer progress, safe retries, archive integrity verification, free-space checks, and extraction. Verify the uncompressed size before extraction and avoid downloading the archive multiple times.

A 2D synthetic warp does not automatically provide the physical 3D supervision expected by the original UVDoc model. Do not assume DocUNet or DIR300's flat scan pairs can directly replace UVDoc's grid annotations.

## 5. Baseline and training plan

**Baseline model:** the official pretrained UVDoc implementation and checkpoint from https://github.com/tanguymagne/UVDoc.

The initial run should preserve the authors' inference conventions, including model input preparation, predicted grid interpretation, geometric image sampling, and output framing. Record any differences rather than introducing them silently. The official implementation must be runnable on a small representative sample before batch evaluation.

A complete baseline run should save predictions, a per-sample result table, run metadata, and examples for manual review. Preserve outputs so metric changes do not require unnecessary inference reruns.

**Training:** full training is not required to establish the pretrained baseline. After acquiring compatible UVDoc Final annotations, validate the data loader and run a small forward/backward/checkpoint smoke test before deciding whether any training or fine-tuning is needed. Do not claim training readiness until the checks have run successfully.

**Classical research:** deferred until the baseline and evaluation infrastructure are stable. Future classical methods should run independently before integration with UVDoc, so experimental improvements can be attributed to a specific change.

## 6. EDA design

Separate real-image benchmark EDA from annotated training-data EDA.

**Real-image EDA** should cover file integrity, valid pairs, image sizes and aspect ratios, EXIF orientation, channel/mode differences, transparency, brightness and contrast distributions, representative samples, background content, reference alignment, and visually reviewed distortion/failure characteristics.

**Training-data EDA** should cover the image-to-annotation mapping, 2D/3D grid shapes and ranges, coordinate orientation and normalization, invalid samples, mask coverage, repeated geometries across splits, and overlays of grids on source images. Verify whether released splits separate underlying geometries, not just differently rendered examples.

The outputs should be reusable manifests, audit summaries, compact statistics, and representative galleries. Original dataset files must remain unchanged. EDA brightness or contrast statistics are descriptive, not scores for unwarping quality.

## 7. Evaluation protocol and metrics

The evaluation implementation is **planned, not yet completed**. Keep its mathematical functions separate from dataset-specific image preparation and alignment.

| Metric | Role | Main condition |
|---|---|---|
| MS-SSIM | Structural image similarity | Record resizing, alignment, image mode, and implementation |
| LD | Residual geometric distortion | Use benchmark-compatible SIFT-flow computation where comparing with papers |
| AD | Aligned geometric distortion | Match the selected benchmark's definition and implementation |
| Edit Distance / CER | Text recognition quality | Requires a defined OCR engine and defensible reference text |
| Runtime | Performance and practicality | Report hardware, input size, measured stages, and repetitions |
| 2D / 3D grid errors | Synthetic geometric supervision | Only when ground-truth annotated grids exist |

The real-world benchmark protocols must distinguish Python diagnostic metrics from verified reproduction of historical MATLAB-based metrics. Do not silently substitute OpenCV optical flow for SIFT flow or resize mismatched images to force a score.

Evaluation results should contain sample ID, document ID, input variant, dataset, method, checkpoint, protocol, prediction and reference paths, metric values, runtime, and failure status. Preserve sample-level results and aggregate with explicit denominators. For DocUNet, remember that 130 images correspond to 65 underlying documents.

OCR from a reference scan is not automatically ground-truth transcription. If using OCR-derived pseudo-reference text, label the resulting metric as OCR agreement rather than CER against verified text.

## 8. Error analysis

The evaluation should support more than a single benchmark average. Identify the best and worst samples, per-image differences between runs, metric disagreements, outliers, and method-specific failures.

Inspect suspected issues using the original input, corrected output, reference, and relevant geometric grids. Candidate visual categories include residual curvature, incomplete page boundaries, clipping, local stretching, perspective errors, blur, and shading artifacts. Confirm categories by visual inspection rather than inferring them solely from brightness, contrast, or one low metric value.

When comparing methods, use the same validated sample manifest and evaluation protocol. Keep all failures visible, with exclusions and unavailable metrics reported explicitly.

## 9. Repository organization

The agreed directory roles are:

```text
models/deep/          # UVDoc-specific implementation and later neural methods
models/classical/     # Future classical algorithms
data/                 # Tracked dataset readers, audits, and manifests
evaluation/           # Metrics, benchmark protocols, run evaluation, error analysis
training/             # Training logic only when needed
notebooks/            # Colab and EDA notebooks; reuse Python modules
tests/                # Metric, data, and integration checks
documentation/        # Project decisions, inspection reports, validated results
dataset/              # Local raw and extracted data, never committed
checkpoints/          # Local model weights, never committed
runs/                 # Saved model predictions and run outputs, never committed
paper/                # Local research reference files, not committed
```

The actual folder tree may contain fewer directories until code is written. Do not add placeholder implementations merely to fill a planned structure. Preserve the legacy local `notebook/` directory until its contents have been reviewed.

The repository's original `.gitignore` ignored `/data/` because the team used it for raw images. This branch's planned layout uses `/data/` for Python source, and `/dataset/` for raw files. Ensure ignore rules reflect that distinction before committing new code.

## 10. Immediate next work and open issues

1. Place and review the branch-specific `.gitignore`, `README.md`, and `context.md`. Keep existing data and notebooks intact.
2. Resolve GitHub write access before attempting to publish the local branch again.
3. Build and verify the canonical DocUNet manifest, including original, crop, and reference pairing.
4. Obtain and audit DIR300 as a separate held-out real-image test set.
5. Prepare the pretrained UVDoc environment and run a small inference smoke test.
6. Implement and validate metrics and benchmark-specific protocols, then establish image-level error analysis.
7. Prepare a resumable Colab download and integrity-check workflow for UVDoc Final, then inspect its geometric annotations before considering training.

**Open issues:** Remote write permission; exact validated UVDoc archive contents and extracted storage requirement; completion of DIR300 acquisition; MATLAB/SIFT-flow evaluation compatibility; OCR text-reference protocol; actual UVDoc inference and training compatibility in the chosen Colab runtime.

Progress should be recorded through actual file changes and verified outputs. This context document is a statement of decisions and current status, not proof that planned steps have been executed.
