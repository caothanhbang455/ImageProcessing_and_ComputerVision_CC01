# evaluation/runner.py
"""Evaluate saved inference runs, retaining attempted samples and failures."""

import csv
import json
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from data.manifests import load_records, manifest_path
from data.paths import PathResolver, project_root
from evaluation.aggregation import aggregate_rows
from evaluation.metrics import MetricResult
from evaluation.protocols import dir300, docunet, uvdoc
from evaluation.reference_metrics import matlab_availability, run_uvdoc_or_docunet_matlab, run_dir300_matlab
from models.deep.uvdoc import hash_file


METRIC_FIELDS = ("sample_key", "method_id", "protocol_id", "metric_name", "value", "unit",
                 "higher_is_better", "status", "reason", "reference_verified")
LINE_FIELDS = ("sample_key", "sample_id", "protocol_id", "direction", "position", "value", "status", "reason")
SAMPLE_FIELDS = ("sample_key", "sample_id", "dataset", "variant", "group_id", "method_id", "prediction_status",
                 "evaluation_status", "reason", "n_valid_lines", "n_missing_lines", "artifact_path")


SAFE_METHOD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


class IncompatiblePrediction(ValueError) :
    """The selected metric requires an output representation this method lacks."""


def load_evaluation_config(root : Path) -> dict :
    cfg_path = root / "configs/evaluation.yaml"
    if (not cfg_path.is_file()) :
        raise FileNotFoundError(f"Stage 3 configuration missing: {cfg_path}")
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if (not isinstance(config, dict) or config.get("version") != 1 or not isinstance(config.get("protocols"), dict)) :
        raise ValueError("Invalid Stage 3 evaluation configuration")
    return config


def read_csv(path : Path) -> list[dict[str, str]] :
    if (not path.is_file()) :
        raise FileNotFoundError(f"Required Stage 2 CSV unavailable: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle :
        reader = csv.DictReader(handle)
        if (not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames))) :
            raise ValueError(f"Invalid CSV columns: {path}")
        return list(reader)


def save_csv(path : Path, rows : list[dict], fields : list[str] | tuple[str, ...]) -> None :
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle :
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_json(path : Path, data : dict | list) -> None :
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def safe_relative_file(run : Path, path : str) -> Path :
    """Reject absolute, escaping, missing, and symlink-escaping artifact paths."""
    relative = Path(path)
    if (not path or relative.is_absolute() or any(x == ".." for x in relative.parts)) :
        raise ValueError(f"Unsafe inference artifact path: {path!r}")
    full = (run / relative).resolve()
    if (not full.is_relative_to(run.resolve()) or not full.is_file()) :
        raise FileNotFoundError(f"Inference artifact missing or outside run: {path}")
    return full


def verify_registered_artifact(run : Path, sample_key : str, filename : str, artifact_type : str,
                               index : dict[tuple[str, str], dict]) -> Path :
    entry = index.get((sample_key, artifact_type))
    if (entry is None or entry["relative_path"] != filename) :
        raise ValueError(f"Artifact registration mismatch for {sample_key}/{artifact_type}")
    full = safe_relative_file(run, filename)
    if (full.stat().st_size != int(entry["bytes"]) or hash_file(full) != entry["sha256"]) :
        raise ValueError(f"Inference artifact SHA-256/size mismatch: {filename}")
    return full


def reference_source_identity(source_dir : str | Path | None) -> dict :
    if (source_dir is None) :
        return {"path" : None, "git_revision" : None, "tracked_changes" : None}
    directory = Path(source_dir).expanduser().resolve()
    if (not directory.is_dir()) :
        return {"path" : str(directory), "git_revision" : None, "tracked_changes" : None}
    revision = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True)
    known = ["uvdocBenchmark_metric.py", "utils.py", "eval/eval_code/evalScript.m",
             "eval/eval_code/evalScriptUVDoc.m", "evalUnwarp.m", "ssim_ld_eval_DIR300.m"]
    return {"path" : str(directory),
            "git_revision" : revision.stdout.strip() if revision.returncode == 0 else None,
            "tracked_changes" : bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
            "source_sha256" : {f : hash_file(directory / f) for f in known if (directory / f).is_file()}}


def inspect_inference_run(root : Path, inference_run : Path, protocol : dict) -> tuple[list[dict], dict, dict] :
    """Bind saved predictions to Stage 1 identities before writing results."""
    dataset = protocol["dataset"]
    variant = protocol["variant"]
    records = {r.sample_key : r for r in load_records(dataset, root, variant=variant)}
    source_rows = read_csv(inference_run / "samples.csv")
    if (not source_rows) :
        raise ValueError("Inference run has no sample records")
    if (len(source_rows) != len({r["sample_key"] for r in source_rows})) :
        raise ValueError("Duplicate sample_key in Stage 2 samples.csv")
    bound = []
    for row in source_rows :
        key = row["sample_key"]
        if (key not in records) :
            raise ValueError(f"Inference identity absent from {dataset} manifest: {key}")
        rec = records[key]
        if (row["dataset"] != rec.dataset or row["sample_id"] != rec.sample_id or
            row["variant"] != rec.variant or row["input_path"] != rec.input_path or
            row["group_id"] != (rec.group_id or "")) :
            raise ValueError(f"Stage 2 manifest identity mismatch: {key}")
        bound.append({**row, "_record" : rec})
    provenance_path = inference_run / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    method_id = provenance.get("method_id")
    if (not isinstance(method_id, str) or not SAFE_METHOD_ID.fullmatch(method_id)) :
        raise ValueError("Inference provenance requires a safe nonempty method_id")
    actual_sha = hash_file(manifest_path(dataset, root))
    if (provenance.get("manifest_sha256") != actual_sha) :
        raise ValueError("Stage 1 manifest SHA-256 differs from recorded inference")
    artifacts = read_csv(inference_run / "artifacts.csv")
    index = {}
    valid_keys = {r["sample_key"] for r in source_rows}
    for row in artifacts :
        key = (row["sample_key"], row["artifact_type"])
        if (key in index or row["sample_key"] not in valid_keys) :
            raise ValueError(f"Duplicate or unexpected inference artifact identity: {key}")
        relative = row.get("relative_path", "").replace("\\", "/")
        parts = relative.split("/")
        if (len(parts) < 4 or parts[:2] != ["predictions", method_id]) :
            raise ValueError(f"Artifact path does not match inference method {method_id}: {relative}")
        index[key] = row
    for row in bound :
        row["_method_id"] = method_id
    return bound, index, provenance


def _measure(sample : dict, protocol_id : str, metric : str, status : str,
             value : float | None = None, reason : str = "", verified : bool = False) -> dict :
    return MetricResult(sample_key=sample["sample_key"], method_id=sample["_method_id"],
                        protocol_id=protocol_id, metric_name=metric,
                        value=value, unit="px" if metric.endswith("_px") else "score",
                        higher_is_better=(metric == "ms_ssim"), status=status,
                        reason=reason, reference_verified=verified).as_row()


def _save_png(image_rgb : np.ndarray, path : Path) -> None :
    path.parent.mkdir(parents=True, exist_ok=True)
    if (not cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))) :
        raise OSError(f"Could not save output: {path}")


def _copy_prediction(image_path : Path, destination : Path) -> None :
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (destination.exists()) :
        raise FileExistsError(f"Duplicate prepared image: {destination}")
    shutil.copyfile(image_path, destination)


def evaluate_inference_run(root : str | Path, inference_run : str | Path, protocol_id : str,
                           output_dir : str | Path, device : str = "cpu",
                           official_source : str | Path | None = None,
                           verify_parity_ids : list[str] | None = None,
                           run_matlab : bool = False) -> dict[str, Any] :
    """Evaluate saved method predictions without invoking inference.

    Creates a fresh run directory. Results are written after each sample so
    interrupted runs preserve completed sample records for inspection.
    """
    root, inference_run, output_dir = project_root(root), Path(inference_run).resolve(), Path(output_dir).resolve()
    config = load_evaluation_config(root)
    if (protocol_id not in config["protocols"]) :
        raise ValueError(f"Unknown Stage 3 protocol ID: {protocol_id}")
    protocol = config["protocols"][protocol_id]
    if (not inference_run.is_dir()) :
        raise FileNotFoundError(f"Inference run missing: {inference_run}")
    if (output_dir.exists() or output_dir == inference_run or output_dir.is_relative_to(inference_run)) :
        raise FileExistsError(f"Refusing to reuse/overwrite an evaluation directory: {output_dir}")
    bound, registry, provenance = inspect_inference_run(root, inference_run, protocol)
    if (device.startswith("cuda") and not torch.cuda.is_available()) :
        raise RuntimeError(f"Requested CUDA device is unavailable: {device}")
    if (verify_parity_ids and (protocol["kind"] not in ("uvdoc_lines", "uvdoc_texture", "uvdoc_matlab") or official_source is None)) :
        raise ValueError("Official parity requires a UVDoc protocol and official_source")
    if (set(verify_parity_ids or []) - {x["sample_id"] for x in bound}) :
        raise ValueError("Requested parity sample is absent from Stage 2 inference")
    if (run_matlab and protocol["kind"] not in ("uvdoc_matlab", "docunet_matlab", "dir300_matlab")) :
        raise ValueError("MATLAB execution is only available for UVDoc visual, DocUNet, and DIR300 protocols")
    resolver = PathResolver(root)
    output_dir.mkdir(parents=True, exist_ok=False)
    metric_rows : list[dict] = []
    line_rows : list[dict] = []
    sample_rows : list[dict] = []
    parity_rows : list[dict] = []
    prepared_rows : list[dict] = []
    parity_set = set(verify_parity_ids or [])
    if (parity_set - {x["sample_id"] for x in bound}) :
        raise ValueError(f"Requested parity IDs absent from run: {sorted(parity_set - {x['sample_id'] for x in bound})}")
    protocol_kind = protocol["kind"]

    save_json(output_dir / "config.json", {"protocol_id" : protocol_id, "protocol" : protocol,
                                              "device" : device, "verify_parity_ids" : sorted(parity_set),
                                              "run_matlab" : run_matlab})
    save_json(output_dir / "provenance.json", {
        "stage2_run" : str(inference_run), "inference_run" : str(inference_run),
        "method_id" : provenance["method_id"],
        "stage2_manifest_sha256" : provenance["manifest_sha256"],
        "stage2_checkpoint_sha256" : provenance.get("checkpoint_sha256"),
        "stage2_source_commit" : provenance.get("official_source_commit"),
        "external_reference_source" : reference_source_identity(official_source),
        "evaluation_config_sha256" : hash_file(root / "configs/evaluation.yaml"),
        "generated_utc" : datetime.now(timezone.utc).isoformat(),
        "note" : "Saved-prediction evaluation; no model inference or benchmark tuning",
    })
    save_json(output_dir / "environment.json", {"python" : sys.version.split()[0],
                                                  "platform" : platform.platform(),
                                                  "numpy" : np.__version__, "opencv" : cv2.__version__,
                                                  "torch" : torch.__version__, "device" : device,
                                                  "matlab" : matlab_availability(official_source, protocol_id)})

    for index, row in enumerate(bound, start=1) :
        sample = {k : row.get(k, "") for k in ("sample_key", "sample_id", "dataset", "variant", "group_id")}
        sample["prediction_status"] = row["status"]
        sample["method_id"] = row["_method_id"]
        sample.update(evaluation_status="", reason="", n_valid_lines=0, n_missing_lines=0, artifact_path="")
        verified = False
        phase = "prediction"
        try :
            if (row["status"] != "OK") :
                raise FileNotFoundError(f"Stage 2 inference status is {row['status']}: {row.get('failure_reason')}")
            image = verify_registered_artifact(inference_run, row["sample_key"], row["image_artifact"],
                                                "image", registry)
            sample["artifact_path"] = row["image_artifact"]
            decoded_image = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if (decoded_image is None) :
                raise ValueError(f"Stage 2 prediction PNG cannot be decoded: {image}")
            if (row.get("output_width") and row.get("output_height") and
                (decoded_image.shape[1], decoded_image.shape[0]) !=
                (int(row["output_width"]), int(row["output_height"]))) :
                raise ValueError(f"Stage 2 PNG dimensions disagree with samples.csv for {row['sample_key']}")
            grid_file = None
            if (protocol_kind in ("uvdoc_lines", "uvdoc_texture", "uvdoc_matlab")) :
                if (not row.get("grid_artifact")) :
                    raise IncompatiblePrediction("UVDoc protocol requires a compatible coarse backward grid; "
                                                 "an image or dense map alone is insufficient")
                grid_file = verify_registered_artifact(inference_run, row["sample_key"], row["grid_artifact"],
                                                       "coarse_grids", registry)
            record = row["_record"]
            phase = "reference"
            if (protocol_kind == "uvdoc_lines") :
                uv = uvdoc.load_uvmap(record, resolver)
                bm = uvdoc.load_predicted_grid(grid_file, row["sample_key"])
                if (row.get("bm_artifact")) :
                    mat_file = verify_registered_artifact(inference_run, row["sample_key"], row["bm_artifact"],
                                                          "benchmark_map", registry)
                    if (uvdoc.compare_npz_mat(grid_file, mat_file, row["sample_key"]) != 0) :
                        raise ValueError("Stage 2 NPZ and MAT grids disagree")
                phase = "metric"
                lines, summary = uvdoc.compute_sample_lines(uv, bm, device=device)
                for line in lines :
                    line_rows.append({"sample_key" : sample["sample_key"], "sample_id" : sample["sample_id"],
                                      "protocol_id" : protocol_id, **line})
                sample["n_valid_lines"] = summary["horizontal_valid_lines"] + summary["vertical_valid_lines"]
                sample["n_missing_lines"] = summary["horizontal_missing_lines"] + summary["vertical_missing_lines"]
                if (sample["sample_id"] in parity_set) :
                    bench_root = root / "dataset/UVDoc/UVDoc_benchmark"
                    parity = uvdoc.verify_official_line_parity(Path(official_source), bench_root,
                                                               sample["sample_id"], bm, lines)
                    parity_rows.append({"sample_key" : sample["sample_key"], **parity})
                    verified = parity["status"] == "PASS"
                    if (not verified) :
                        sample["evaluation_status"] = "REVIEW_REQUIRED"
                        sample["reason"] = "Independent official line parity failed"
                for direction in ("horizontal", "vertical") :
                    val = summary[direction]
                    key = f"{direction}_line_std_px"
                    metric_rows.append(_measure(row, protocol_id, key, "OK" if val is not None else "METRIC_FAILED",
                                                 val, "" if val is not None else "No valid synthetic lines", verified))
                if (not sample["evaluation_status"]) :
                    sample["evaluation_status"] = "OK" if sample["n_valid_lines"] else "METRIC_FAILED"
                    sample["reason"] = "" if sample["n_valid_lines"] else "All synthetic lines were invalid"
            elif (protocol_kind in ("uvdoc_texture", "uvdoc_matlab")) :
                bm = uvdoc.load_predicted_grid(grid_file, row["sample_key"])
                source = resolver.resolve(f"dataset/UVDoc/UVDoc_benchmark/warped_textures/{sample['sample_id']}.png")
                phase = "metric"
                corrected = uvdoc.unwarp_texture_from_grid(source, bm, device=device)
                target = output_dir / "prepared_predictions/uwp_texture" / f"{sample['sample_id']}.png"
                _save_png(corrected, target)
                sample.update(evaluation_status="OK", artifact_path=str(target.relative_to(output_dir)))
                prepared_rows.append({"sample_key" : sample["sample_key"], "relative_path" : sample["artifact_path"],
                                      "sha256" : hash_file(target), "reference_variant" : "texture_sample"})
                if (sample["sample_id"] in parity_set) :
                    parity = uvdoc.verify_official_texture_parity(Path(official_source), source, bm, corrected)
                    parity_rows.append({"sample_key" : sample["sample_key"], **parity})
                    if (parity["status"] != "PASS") :
                        sample.update(evaluation_status="REVIEW_REQUIRED", reason="Official texture pixels differ")
                if (protocol_kind == "uvdoc_matlab") :
                    for metric in protocol["metric_names"] :
                        metric_rows.append(_measure(row, protocol_id, metric, "UNAVAILABLE_DEPENDENCY",
                                                     reason="MATLAB reference results not computed in per-sample runner"))
            elif (protocol_kind in ("docunet_matlab", "dir300_matlab")) :
                if (protocol_kind == "docunet_matlab") :
                    if (docunet.corrected_input_required(sample["sample_id"], protocol_id)) :
                        sample.update(evaluation_status="REVIEW_REQUIRED",
                                      reason="Corrected-64 policy requires independently generated rotated-input predictions")
                    name = docunet.reference_prediction_name(sample["sample_id"])
                else :
                    name = dir300.reference_prediction_name(sample["sample_id"])
                if (not sample["evaluation_status"]) :
                    if (record.reference_path is None or resolver.find(record.reference_path) is None) :
                        raise FileNotFoundError(f"Missing reference scan for {sample['sample_id']}")
                    target = output_dir / "prepared_predictions/reference_input" / name
                    _copy_prediction(image, target)
                    sample.update(evaluation_status="OK", artifact_path=str(target.relative_to(output_dir)))
                    prepared_rows.append({"sample_key" : sample["sample_key"], "relative_path" : sample["artifact_path"],
                                          "sha256" : hash_file(target), "reference_path" : record.reference_path})
                for metric in protocol["metric_names"] :
                    metric_rows.append(_measure(row, protocol_id, metric,
                                                 "REVIEW_REQUIRED" if sample["evaluation_status"] == "REVIEW_REQUIRED"
                                                 else "UNAVAILABLE_DEPENDENCY",
                                                 reason=sample["reason"] or "External MATLAB/SIFT-flow reference scoring pending"))
            else :
                raise ValueError(f"Unknown evaluation protocol kind: {protocol_kind}")
        except Exception as error :
            status = "UNAVAILABLE_PREDICTION" if (row["status"] != "OK") else (
                "NOT_APPLICABLE" if isinstance(error, IncompatiblePrediction) else
                ("UNAVAILABLE_PREDICTION" if phase == "prediction" else "UNAVAILABLE_REFERENCE")
                if isinstance(error, FileNotFoundError) else
                "UNAVAILABLE_DEPENDENCY" if isinstance(error, (ImportError, ModuleNotFoundError)) or
                (isinstance(error, RuntimeError) and "requires hdf5storage" in str(error)) else
                "INVALID_INPUT" if isinstance(error, ValueError) and phase != "metric" else "METRIC_FAILED")
            sample.update(evaluation_status=status, reason=f"{type(error).__name__}: {error}")
            for metric in protocol["metric_names"] :
                if (not any(r["sample_key"] == row["sample_key"] and r["metric_name"] == metric for r in metric_rows)) :
                    metric_rows.append(_measure(row, protocol_id, metric, status, reason=sample["reason"]))
        sample_rows.append(sample)
        save_csv(output_dir / "metrics/per_sample.csv", metric_rows, METRIC_FIELDS)
        save_csv(output_dir / "metrics/per_line.csv", line_rows, LINE_FIELDS)
        save_csv(output_dir / "samples.csv", sample_rows, SAMPLE_FIELDS)
        save_csv(output_dir / "reference_validation/numerical_parity.csv", parity_rows,
                 ("sample_key", "status", "reason", "max_abs", "atol", "lines", "reference_commit"))
        save_csv(output_dir / "prepared_predictions.csv", prepared_rows,
                 ("sample_key", "relative_path", "sha256", "reference_variant", "reference_path"))
        print(f"[{index}/{len(bound)}] {sample['sample_key']}: {sample['evaluation_status']}", flush=True)

    matlab_record = {"status" : "NOT_REQUESTED", "reason" : "External MATLAB not requested"}
    if (run_matlab) :
        expected = {"uvdoc_matlab" : 50, "docunet_matlab" : 130, "dir300_matlab" : 300}[protocol_kind]
        pred_dir = output_dir / ("prepared_predictions/uwp_texture" if protocol_kind == "uvdoc_matlab"
                                 else "prepared_predictions/reference_input")
        reference_dir = root / ("dataset/UVDoc/UVDoc_benchmark/texture_sample" if protocol_kind == "uvdoc_matlab"
                                else "dataset/DocUNet/scan" if protocol_kind == "docunet_matlab"
                                else "dataset/DIR300/_source/DIR300/gt")
        try :
            if (protocol_kind == "dir300_matlab") :
                matlab_record = run_dir300_matlab(Path(official_source) if official_source else Path("."),
                                                   reference_dir, pred_dir)
            else :
                matlab_record = run_uvdoc_or_docunet_matlab(protocol_id, Path(official_source) if official_source else Path("."),
                                                            reference_dir, pred_dir, expected)
        except Exception as error :
            matlab_record = {"status" : "METRIC_FAILED", "reason" : f"{type(error).__name__}: {error}"}
        if (matlab_record["status"] == "OK") :
            try :
                if (protocol_kind == "dir300_matlab") :
                    scores = {str(int(a[0])) : {"ms_ssim" : float(a[1]), "ld" : float(a[2])}
                              for a in matlab_record["rows"]}
                else :
                    values = np.loadtxt(matlab_record["reference_output"], ndmin=2)
                    if (len(values) < expected + 1) :
                        raise ValueError("Official MATLAB individual_res.txt is incomplete")
                    scores = {}
                    for item in values[:expected] :
                        if (protocol_kind == "uvdoc_matlab") :
                            key = f"{int(item[0]):05d}"
                            scores[key] = {"ad" : float(item[1]), "ms_ssim" : float(item[2])}
                        else :
                            key = f"{int(item[0])}_{int(item[1])}"
                            scores[key] = {"ad" : float(item[2]), "ms_ssim" : float(item[3]), "ld" : float(item[4])}
                if (len(scores) != expected) :
                    raise ValueError("MATLAB results contain duplicate or missing sample IDs")
                for metric in metric_rows :
                    sample = next((r for r in sample_rows if r["sample_key"] == metric["sample_key"]), None)
                    if (sample is None or sample["evaluation_status"] != "OK") :
                        continue
                    value = scores.get(sample["sample_id"], {}).get(metric["metric_name"])
                    if (value is not None and np.isfinite(value) and value >= 0) :
                        metric.update(value=value, status="OK", reason="", reference_verified=False)
                save_csv(output_dir / "metrics/per_sample.csv", metric_rows, METRIC_FIELDS)
                matlab_record["parsed_samples"] = len(scores)
            except Exception as error :
                matlab_record = {"status" : "METRIC_FAILED", "reason" : f"MATLAB result parse: {type(error).__name__}: {error}"}
    save_json(output_dir / "reference_validation/matlab_result.json", matlab_record)
    summaries = aggregate_rows(metric_rows)
    save_csv(output_dir / "metrics/summary.csv", summaries,
             ["protocol_id", "metric_name", "unit", "higher_is_better", "n_attempted",
              "n_metric_ok", "n_metric_missing", "coverage_fraction", "reference_verified",
              "n_expected", "n_valid", "n_missing", "mean", "median", "std", "min", "max", "p05", "p95"])
    counts = Counter(r["evaluation_status"] for r in sample_rows)
    result = {"protocol_id" : protocol_id, "method_id" : provenance["method_id"],
              "inference_run" : str(inference_run), "evaluation_run" : str(output_dir),
              "n_attempted" : len(sample_rows), "status_counts" : dict(counts),
              "metrics" : summaries, "parity" : parity_rows,
              "matlab" : matlab_record, "reference_verified" : bool(parity_rows) and
              all(p["status"] == "PASS" for p in parity_rows)}
    save_json(output_dir / "summary.json", result)
    inventory = []
    for file in sorted(output_dir.rglob("*")) :
        if (file.is_file() and file.name != "artifacts.csv") :
            inventory.append({"relative_path" : file.relative_to(output_dir).as_posix(),
                              "bytes" : file.stat().st_size, "sha256" : hash_file(file)})
    save_csv(output_dir / "artifacts.csv", inventory, ["relative_path", "bytes", "sha256"])
    report = ["# Stage 3 evaluation report", "", f"Protocol: `{protocol_id}`",
              f"Method: `{provenance['method_id']}`", f"Inference run: `{inference_run}`",
              f"Attempted samples: **{len(sample_rows)}**", "",
              "## Evaluation status", ""]
    report.extend(f"- {key}: {counts[key]}" for key in sorted(counts))
    report.extend(["", "## Metric availability", ""])
    report.extend(f"- {r['metric_name']}: {r['n_metric_ok']}/{r['n_attempted']} valid; "
                  f"mean = {r['mean'] if r['mean'] is not None else 'unavailable'} {r['unit']}"
                  for r in summaries)
    report.extend(["", "## Reference validation", "",
                   f"Independent UVDoc parity runs: {len(parity_rows)}; "
                   f"all passed: {result['reference_verified'] if parity_rows else 'not checked'}.",
                   f"MATLAB reference status: {matlab_record['status']}.", "",
                   "Only metrics with independent reference validation may be reported as "
                   "published-benchmark equivalent. Unavailable measurements are not zero scores."])
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    # Refresh inventory after the final report is written.
    inventory.append({"relative_path" : "report.md", "bytes" : (output_dir / "report.md").stat().st_size,
                      "sha256" : hash_file(output_dir / "report.md")})
    save_csv(output_dir / "artifacts.csv", inventory, ["relative_path", "bytes", "sha256"])
    return result
