# Stage 1 implementation and integration handoff

**Date:** 21 November 2026

**Scope:** Shared data records, manifest readers, safe path resolution, existing UVDoc split verification, representative image/grid loaders, model result contract, and a CPU-oriented validation report. No new method or experiment matrix was selected.

## Integration into the existing repository

The deliverable is an **overlay**, not a second Git repository. Extract its contents directly into the existing project root `G:\My Drive\Document Unwarping\`, preserving the folder structure. It contains only newly implemented files. It does not contain `.git`, raw datasets, checkpoints, notebooks, source-paper PDFs, or previously generated outputs.

Before extraction, review the list of files and any local/uncommitted changes in the shared repository. The supplied Git status shows existing modified and deleted tracked files, which this overlay intentionally does not overwrite. No Git operations are performed automatically.

Install and validate from PowerShell:

```powershell
cd "G:\My Drive\Document Unwarping"
python -m pip install -r requirements-stage1.txt
python -m pytest -q tests
python scripts/validate_datasets.py --root "G:\My Drive\Document Unwarping"
python scripts/validate_datasets.py --root "G:\My Drive\Document Unwarping" --check-files --max-files-per-dataset 3 --smoke-read 2
```

In Colab:

```python
%cd "/content/drive/MyDrive/Document Unwarping"
%pip install -r requirements-stage1.txt
!python -m pytest -q tests
!python scripts/validate_datasets.py --root "/content/drive/MyDrive/Document Unwarping" --check-files --max-files-per-dataset 3 --smoke-read 2
```

For UVDoc Final members staged elsewhere, set `DOCUMENT_UNWARPING_UVDOC_FINAL_ROOT` to the directory that directly contains `img/`, `grid2d/`, `grid3d/`, and `metadata_sample/`.

## Example use

```python
from data import PathResolver, load_records
from data.uvdoc import load_sample as load_uvdoc_sample

root = r"G:\My Drive\Document Unwarping"
resolver = PathResolver(root)
uvdoc = load_records("uvdoc_benchmark", root)
sample = load_uvdoc_sample(uvdoc[0], resolver, include_geometry=True)
print(sample["record"].sample_key, sample["image"].shape, sample["grid2d"].shape)

# DocUNet crops and original photographs are distinct input variants.
from data.docunet import load_sample as load_docunet_sample
crop = load_records("docunet", root)[0]
original = load_records("docunet", root, variant="original")[0]
print(crop.sample_key, original.sample_key, crop.reference_path)
```

The example requires the actual dataset assets in the user's environment. It is not executed by the local metadata-only check.

## Checks performed before handoff

The Stage 1 package was tested on small synthetic fixtures, including mixed-extension DocUNet names, explicit EXIF orientation, missing assets, HDF5 grid shapes/transposition, existing split logic, and successful/failed method result contracts.

The supplied original CSV/JSON metadata files were copied into an **isolated temporary test root** for inventory validation. The actual rows were read without changes, and the saved UVDoc split was checked for coverage and group disjointness. See the separately attached validation report for observed counts and statuses.

The ZIP did not include actual document images or UVDoc `.mat` annotations, so complete integration smoke reads against real assets remain `UNAVAILABLE` until run in the user's repository/Colab environment. Likewise, the completed UVDoc notebook's official pixel-equivalence check belongs to Stage 2 and was not rerun as part of this data-only stage.

## Cleanup / reorganizing existing files

No existing file was deleted, moved, or renamed. The provided working tree contains uncommitted changes; removing files now could interfere with teammates' work. Candidates for a later, **separate reviewed cleanup** include a zero-byte root `files.txt`, obsolete `__MACOSX` extraction metadata, or notebook outputs that can be cleared after preserving an unmodified archive. Original EDA outputs, manifests, dataset archives, source evaluation scripts, and baseline pilot results should be retained.

`README.md` and `context.md` were intentionally left untouched because the supplied Git status shows a modified tracked README and an untracked context file. After merging Stage 1, update their status sections to accurately link to `documentation/dataset_protocols.md` and this implementation note, without rewriting prior experimental evidence.

## Upcoming engineering work (not a fixed experimental schedule)

Stage 2: shared coordinate/mapping primitives and faithful pretrained inference extracted from the verified November 21 pilot.

Stage 3: dataset-specific metric protocols, reference alignment, failure accounting, error analysis, and reproducible reporting.

Stage 4: independently selected classical and learned method implementations using the same interfaces.

Stage 5: research decisions, experiment selection, matched comparisons, and possible integration, contingent on baseline analysis and team agreement.

Stage 6: complete evaluation, sensitivity/error analysis, research report, and reproducible demonstration.
