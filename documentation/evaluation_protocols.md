# Evaluation protocols

This document defines how Stage 3 scores saved document-unwarping predictions. It is meant for anyone on the team running an experiment or reading a result table.

## General rules

Each image has a stable `sample_key` from the Stage 1 manifest. Stage 2 saves an inference run with `samples.csv`, predicted images, coarse grids, `artifacts.csv`, and provenance files. Stage 3 reads those files without running the neural model again.

A metric is identified by its protocol, not just its name. In particular, a Python implementation of SSIM or optical flow cannot be reported as the published DocUNet metric unless its numerical behavior has been verified against the benchmark code. Missing predictions and invalid references are recorded as missing measurements, not as zero scores.

Results are stored in a new directory under `runs/evaluation/`. The evaluator refuses to overwrite an existing directory. The input inference run remains unchanged.

## UVDoc Benchmark: line deviation

Protocol: `uvdoc_lines_official_n25_v1`.

The UVDoc authors evaluate geometry by drawing synthetic straight lines on a 1000 by 1000 pixel texture. They deform each line using the ground-truth UV map, then use the model's predicted backward mapping to straighten it. Horizontal and vertical deviations are measured separately. Lower values indicate straighter lines.

The reference uses 25 lines in each direction, at integer positions given by `np.linspace(50, 950, 25, dtype=int)`. A line with no black pixels produces an undefined value. Stage 3 records every line and reports the number of valid lines instead of filling undefined measurements with zero.

The reference performs the initial UV deformation with PyTorch `grid_sample(..., align_corners=False)`. The prediction's coarse backward map is upsampled with bilinear interpolation and `align_corners=True`, and the final image sampling also uses `align_corners=True`. The different settings are part of the published algorithm. The reference also applies a binary mask erosion before measuring line deviations.

Input files:

- UVDoc Benchmark metadata: `metadata_sample/<sample_id>.json`
- Ground-truth UV map: `uvmap/<geom_name>.mat`, where `geom_name` comes from the metadata
- Stage 2 saved coarse 2D grid: `predictions/uvdoc_official/coarse_grids/<sample_key>.npz`

The coarse predicted grid is already in normalized coordinates. It must not be normalized a second time. When a Stage 2 benchmark `bm` MAT file exists, Stage 3 checks that its values agree with the NPZ grid.

For independent parity, Stage 3 invokes `uvdocBenchmark_metric.py` from the verified pinned UVDoc Git checkout in a separate Python process. It compares individual line measurements, including the placement of undefined values. A result is not marked reference verified until this check passes.

Reference: https://github.com/tanguymagne/UVDoc/blob/main/uvdocBenchmark_metric.py

## UVDoc Benchmark: visual metrics

Protocols: `uvdoc_texture_prepare_v1` and `uvdoc_visual_matlab_v1`.

The official benchmark also evaluates unwarped textures. These are different from the rectified photographic images saved by the Stage 2 demo. Stage 3 can apply the saved coarse 2D backward map to each `warped_textures/<sample_id>.png`, using the official unwarping convention. It saves the result as `prepared_predictions/uwp_texture/<sample_id>.png`.

A separate parity function compares these generated texture images with the pinned official `utils.py` implementation. The comparison checks decoded uint8 pixels without rerunning the network.

The published visual evaluation uses MATLAB and the author's `evalScriptUVDoc.m`. It compares the unwarped textures with `texture_sample` and relies on the SIFT-flow evaluation dependencies. The MATLAB backend is optional. When those dependencies are absent, Stage 3 reports unavailable visual metrics, not approximate substitutes.

References: https://github.com/tanguymagne/UVDoc/blob/main/uvdocBenchmark_pred.py and https://github.com/tanguymagne/UVDoc/blob/main/eval/eval_code/evalScriptUVDoc.m

## DocUNet

Protocols: `docunet_matlab_legacy_v1` and `docunet_matlab_corrected64_v1`.

Use the 130 cropped inputs and their 65 scanned references. The stable sample ID retains the document number and view, such as `38_1`. The MATLAB evaluator expects predicted images named `38_1.png` and reference scans named `38.png`.

The published reference uses grayscale images and resizes the scan toward a target area of 598,400 pixels. It then resizes the prediction to the resulting scan dimensions with bicubic interpolation. SIFT flow provides alignment information for LD and AD. These operations must be performed by the selected MATLAB reference when producing published benchmark scores.

The documented document-64 orientation problem is not corrected by the general Stage 1 reader. The corrected protocol explicitly requires separate inference on the corrected input for `64_1` and `64_2`. It records `REVIEW_REQUIRED` if these predictions are not available, rather than pretending a post-inference rotation is equivalent.

The `alignment.py` OpenCV function is for previewing dimensions and transformations. It is **not** a numerical substitute for MATLAB's grayscale conversion and bicubic resize.

References: https://github.com/tanguymagne/UVDoc/blob/main/eval/eval_code/evalScript.m and https://github.com/fh2019ustc/DocScanner

## DIR300

Protocol: `dir300_matlab_v1`.

DIR300 has 300 distorted/reference pairs identified by their numeric IDs. The DocGeoNet MATLAB wrapper expects predictions named `<id>_rec.png`, such as `8_rec.png`, and reference scans named `8.png`.

The released wrapper computes MS-SSIM and LD after MATLAB grayscale conversion and reference-area normalization. It uses the released `evalUnwarp.m` and its SIFT-flow dependencies. Stage 3 can prepare the prediction filenames and has a conditional MATLAB bridge for the complete set. The derived driver must be checked numerically against the original wrapper before claiming full reference compatibility. DIR300 AD is not defined by this selected wrapper.

References: https://github.com/fh2019ustc/DocGeoNet/blob/main/ssim_ld_eval_DIR300.m and https://github.com/fh2019ustc/DocGeoNet/blob/main/evalUnwarp.m

## Metric status and aggregation

The per-sample metric table includes `sample_key`, `method_id`, `protocol_id`, `metric_name`, `value`, `unit`, `higher_is_better`, `status`, `reason`, and `reference_verified`.

A successful metric uses `OK` and a finite value. Other statuses include `UNAVAILABLE_PREDICTION`, `UNAVAILABLE_REFERENCE`, `UNAVAILABLE_DEPENDENCY`, `INVALID_INPUT`, `METRIC_FAILED`, `NOT_APPLICABLE`, and `REVIEW_REQUIRED`. Any non-OK metric has an empty value and a reason.

The summary reports attempted, successful, and missing metric counts. Means and percentiles are calculated over finite measurements only. The coverage fraction is reported alongside the mean so that a method cannot appear better simply because difficult samples were omitted.

A score computed on a diagnostic subset must not be described as a full benchmark result. The initial Stage 2 integration run has only two UVDoc Benchmark samples, two DocUNet crops, and two DIR300 samples. A new Stage 2 run is required for complete benchmark coverage.

## Verification status

As of Stage 3 source delivery, the Stage 1/2 Colab implementation has been verified by the user with 68 passing tests, a representative dataset smoke check, and exact official-demo pixel parity for sample `00010`. The Stage 3 evaluator and the external reference scripts have **not yet** been run together on the full real benchmark in Colab. Such scores remain pending until the notebook records per-sample results and the parity reports.
