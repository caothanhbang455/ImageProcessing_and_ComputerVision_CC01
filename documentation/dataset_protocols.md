# Dataset contracts and file access

**Status:** Stage 1 implementation, 21 November 2026. This document describes verified manifest formats and the implemented readers. It does not define the team's future experiments, training datasets, or evaluation metrics.

## 1. One source of truth

Original CSV/JSON files are stored under `dataset/_metadata/` and are read in place. The Python `data/` package does not rewrite them and does not alter original images or annotations. `configs/datasets.yaml` identifies their filenames, default input variants, and release-count expectations. Local machine paths are supplied at runtime, not embedded in the manifests.

| Dataset key | Human-readable label | Manifest | Default input | Inventory records | Group identity |
|---|---|---|---|---:|---|
| `uvdoc_final` | UVDoc Final | `uvdoc_final_manifest.csv` | `rendered` | 20,000 | Base geometry |
| `uvdoc_benchmark` | UVDoc Benchmark | `uvdoc_benchmark_manifest.csv` | `rendered` | 50 | Base geometry |
| `docunet` | DocUNet | `docunet_manifest.csv` | `crop` | 130 | Document ID |
| `dir300` | DIR300 | `dir300_manifest.csv` | `dist` | 300 | Document ID |

The inventory counts above are verified using the files supplied on 21 November 2026, but are not a guarantee of physical file availability on another machine.

## 2. Stable identities

`load_records(dataset, root, variant=None)` returns `SampleRecord` values with these fields:

- `sample_key`, `dataset`, `sample_id`, `variant`: canonical identity, independent of the filesystem.
- `input_path`: original manifest-relative input path, retained verbatim.
- `reference_path`: optional flat reference; absent for UVDoc records.
- `group_id`: underlying page for DocUNet, base deformation geometry for UVDoc, document ID for DIR300.
- `annotation_paths`: available annotated file locations; UVDoc has `grid2d`, `grid3d`, and `metadata`.
- `metadata`: all original manifest columns, so original names, statuses, textures, and background labels are not lost.

Examples: `UVDoc_Final_rendered_00010`, `UVDoc_Benchmark_rendered_00010`, `DocUNet_crop_38_1`, `DocUNet_original_38_1`, and `DIR300_dist_8`. CSV IDs are read as strings; leading zeros are never coerced to integers.

A given DocUNet row describes the `crop` input by default. Calling `load_records('docunet', root, variant='original')` returns a *different input variant* with the same sample ID and scan reference. This is not a second physical document or a new training sample.

## 3. UVDoc format

The raw UVDoc 2D HDF5 dataset `grid2d` has shape `(2, 61, 89)` and is exposed as `(2, 89, 61)`, using the exact transpose convention confirmed in the November 20 EDA. The raw HDF5 dataset `grid3d` has shape `(3, 61, 89)` and is exposed as `(89, 61, 3)`. These are **dataset annotations**, not the pretrained UVDoc network's predicted coarse `(2, 45, 31)` and `(3, 45, 31)` grids.

Grid readers validate the expected shape, key, and finite values. Neither loader silently interprets a coordinate as normalized, resizes an annotation, or substitutes a simulated 3D target. UV and segmentation auxiliary HDF5 files can be addressed through `auxiliary_path(record, 'uvmap' | 'seg')`; `read_auxiliary` returns *raw* arrays without claiming a particular axis convention.

UVDoc Final has 1,008 base-geometry groups and 4,032 geometry variants. All base groups have four variant names (`0_0`, `0_1`, `1_0`, `1_1`), but their number of rendered samples is not uniform. In the supplied manifest, 968 base groups have 20 renderings and 40 have 16. Validation checks the variant set, not an invalid assumption of 20 renderings per group.

The previously saved EDA group-disjoint split remains accessible through `load_existing_split(root)` and `validate_existing_split(root)`. It contains 17,992 training images in 907 geometry groups and 2,008 validation images in 101 groups, with no group overlap. The code does **not** create a new split or decide that future experiments must adopt this particular split. Textures/backgrounds can overlap across the saved split, and the released pretrained UVDoc weights may have seen some of UVDoc Final during their original training.

## 4. DocUNet format

The 130 manifest records correspond to 65 physical documents with two views each. Their default crop filenames include a **literal space and `copy` suffix**, for example `dataset/DocUNet/crop/38_1 copy.png`. Original photographs retain the exact `.jpg`/`.JPG` extensions stored in the manifest, and both views of a document share one flat scan reference `dataset/DocUNet/scan/<document_id>.png`.

The general image loader defaults to **no EXIF rotation** and records the original EXIF orientation. `apply_exif=True` is explicitly available when a documented experiment requires it. The specific DocUNet `64_1`/`64_2` scan-reference rotation issue is **not corrected in Stage 1**; benchmark-specific reference alignment belongs to the evaluation protocol.

The older `documentation/docunet_manifest.json` reports a `crop_to_original` matching count of zero from an early audit. The November 20 EDA CSV is the current source of truth for actual two-view crop/original/scan pairing; do not use the historical count as a reason to discard otherwise valid samples.

## 5. DIR300 format

The 300 default `dist` inputs and their `gt` reference images are paired by explicit numeric IDs, preserving the original paths under `dataset/DIR300/_source/DIR300/`. A flat reference image is not a 2D/3D UVDoc geometry annotation.

## 6. Path resolution

The supported project root can be passed directly or through `DOCUMENT_UNWARPING_ROOT`.

- Windows example: `G:\My Drive\Document Unwarping`
- Colab example: `/content/drive/MyDrive/Document Unwarping`

Only safe project-relative paths are accepted from the manifests. The resolver does not choose a file merely because its basename matches.

UVDoc Final is special: the dataset archive need not be fully extracted. When resolving `dataset/UVDoc/UVDoc_final/<suffix>`, `PathResolver` checks an explicit `DOCUMENT_UNWARPING_UVDOC_FINAL_ROOT` override (or constructor argument), the logical project path, the existing `dataset/UVDoc/_staging_UVDoc_final/UVDoc_final/` path, and the active Colab `/content/uvdoc_runtime/UVDoc_final/` location, if that directory exists.

`find(relative_path)` returns `None` if a file has not been staged. `resolve(relative_path)` raises `FileNotFoundError` with candidate paths rather than silently returning a missing file. `resolve(..., require_exists=False)` can be used to inspect a candidate path, but is not proof of existence.

Stage 1 does not download, extract, or duplicate any dataset. Selective archive extraction can be extracted into a dedicated utility later if the team actually needs it.

## 7. Image decoding and official model fidelity

`data.uvdoc.read_image` is a general Pillow-based reader that returns an RGB array by default. It does not automatically apply EXIF rotation. The completed November 21 pretrained UVDoc pilot instead used `cv2.imread(..., cv2.IMREAD_COLOR)` and BGR-to-RGB conversion, matching the original model demo. **These decoding procedures are not interchangeable without verification.** The official UVDoc adapter in Stage 2 must preserve the reference demo behavior and associated regression test.

## 8. Validation and limitations

```powershell
python -m pip install -r requirements-stage1.txt
python -m pytest -q tests
python scripts/validate_datasets.py --root "G:\My Drive\Document Unwarping"
python scripts/validate_datasets.py --root "G:\My Drive\Document Unwarping" --check-files --max-files-per-dataset 3 --smoke-read 2
```

In Colab, use the equivalent mounted root. The first validator command reads only manifests and split metadata. The second optionally checks physical file availability and decodes a few representative images/grids. Missing unstaged files are `UNAVAILABLE`, not silently counted as geometry failures.

The default report path is `runs/data_validation/<timestamp>_<id>/`. It contains `report.md`, `summary.json`, `issues.csv`, `missing_files.csv`, and, for a smoke read, `smoke.csv`. File checks can be slow over Google Drive; use a sample limit first.

Stage 1 does not establish correct reference registration, ground-truth OCR, published MS-SSIM/LD/AD compatibility, or model improvement. Those remain future shared evaluation and research decisions.
