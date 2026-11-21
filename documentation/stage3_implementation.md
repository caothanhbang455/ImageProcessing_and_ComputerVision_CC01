# Stage 3 implementation guide

## Purpose

Stage 3 scores already saved Stage 2 predictions. It does not train models or change inference preprocessing. The evaluation code records sample identities, missing references, metric values, numerical parity checks, and the environment used for each run.

## Files

| File | Responsibility |
|---|---|
| `configs/evaluation.yaml` | Named metric protocols and expected dataset variants |
| `evaluation/metrics.py` | Per-metric result contract and simple pixel diagnostics |
| `evaluation/protocols/uvdoc.py` | Official-style UVDoc line measurement, ground-truth UV-map loading, and texture preparation |
| `evaluation/protocols/docunet.py` | DocUNet naming and orientation policy |
| `evaluation/protocols/dir300.py` | DIR300 naming and selected metric definitions |
| `evaluation/alignment.py` | Explicit, non-reference OpenCV alignment previews |
| `evaluation/reference_metrics.py` | Conditional external MATLAB and SIFT-flow calls |
| `evaluation/runner.py` | Stage 2 artifact checks, per-sample scoring, and result export |
| `evaluation/aggregation.py` | Metric summaries with missing-value counts |
| `scripts/evaluate.py` | Command-line evaluation entry point |
| `notebooks/[21.11] Stage_3_Evaluation_and_Metric_Validation.ipynb` | Colab execution workflow |

Existing Stage 1 and Stage 2 source files are not replaced.

## Requirements

Install Stage 1 and Stage 2 dependencies first. Stage 3 also needs `scikit-image` for the original UVDoc line-mask erosion. Keep the installed CUDA-enabled PyTorch build in Colab; do not replace it with a CPU wheel when installing other packages.

`hdf5storage` reads the UVDoc benchmark ground-truth UV maps. The MATLAB Engine and SIFT flow are optional external dependencies. They are needed to run the reference MS-SSIM, LD, and AD calculations. No Python optical-flow approximation is installed as a fallback.

## Colab workflow

1. Extract the Stage 3 source patch into the existing repository. It adds source files but does not replace any Stage 1/2 file.
2. Open the Stage 3 Colab notebook and mount Google Drive.
3. Verify the imported Stage 3 modules and run the complete automated test suite.
4. Check the UVDoc Benchmark ground-truth UV-map and texture assets. If files are missing, use the selective archive staging cell instead of extracting full training data.
5. Evaluate the two saved Stage 2 UVDoc Benchmark predictions, `00010` and `00011`. The notebook compares the new line evaluator with the separate pinned official code.
6. Generate a new, complete 50-image Stage 2 UVDoc Benchmark inference run. This uses the same verified checkpoint and saves the original predictions separately.
7. Run all 50 UVDoc line evaluations, then prepare the unwarped texture images and verify selected outputs.
8. Prepare DocUNet and DIR300 predictions for their reference evaluation conventions. Run MATLAB reference scoring only in an environment where the required MATLAB and SIFT-flow code is available.
9. Inspect the generated CSV, JSON, and Markdown reports before using scores in a team comparison.

The notebook selects new output directories. It does not replace existing runs or downloaded datasets.

## Command-line examples

From the repository root, for two saved UVDoc samples:

```bash
python scripts/evaluate.py \
  --root "/content/drive/MyDrive/Document Unwarping" \
  --inference-run "/content/drive/MyDrive/Document Unwarping/runs/inference/stage2_integration_20261121T104333Z_cca74c/uvdoc_benchmark" \
  --protocol uvdoc_lines_official_n25_v1 \
  --device cuda \
  --official-source /content/uvdoc_reference_stage2 \
  --verify-parity-id 00010 \
  --verify-parity-id 00011
```

Check UVDoc Benchmark files without scoring:

```bash
python scripts/evaluate.py \
  --root "/content/drive/MyDrive/Document Unwarping" \
  --protocol uvdoc_lines_official_n25_v1 \
  --preflight-assets
```

The evaluator reads the Stage 2 `samples.csv` and its artifact registry. It verifies the Stage 1 manifest hash, checks the identity of every sample, and validates the hash and size of each prediction it uses.

## Output files

Each evaluation run saves:

```text
runs/evaluation/<run-id>/
  config.json
  environment.json
  provenance.json
  samples.csv
  prepared_predictions.csv
  metrics/per_line.csv
  metrics/per_sample.csv
  metrics/summary.csv
  reference_validation/numerical_parity.csv
  reference_validation/matlab_result.json
  prepared_predictions/...
  artifacts.csv
  summary.json
  report.md
```

The evaluator writes partial CSV records after each processed sample. If the process is interrupted, the directory retains completed records for inspection. To rerun, select a new directory. Existing run directories are never overwritten automatically.

## Result interpretation

A successful inference is not necessarily a successful evaluation. A method can produce an image but have no usable ground-truth UV map, an invalid benchmark line, or an unavailable MATLAB dependency. Stage 3 records these cases separately.

The UVDoc line evaluation measures geometric straightness on synthetic test lines. It does not measure readability, appearance, OCR accuracy, or geometric alignment to a scanned page.

DocUNet and DIR300 MATLAB results should only be described as published-reference compatible when the required external tools have actually run and their output procedure has been checked. The Python resizing preview is intended for inspection, not benchmark scoring.

## Verification evidence and remaining checks

The source package is tested in an isolated environment containing the Stage 1/2 files. Its local test report is provided in the handoff. The user previously confirmed that all 68 Stage 1/2 tests passed in Colab, that the representative physical dataset smoke gate passed, and that sample `00010` matched the pinned official UVDoc demo exactly.

The full 50-image line benchmark, two-sample independent line parity, texture-prediction pixel parity, and optional MATLAB evaluation must be executed in the user's Colab environment because the model checkpoint and benchmark images are not present in this isolated build.
