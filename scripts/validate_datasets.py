# scripts/validate_datasets.py
"""CPU-only dataset readiness report. No inference, training, or dataset mutation."""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.manifests import DATASET_KEYS, load_records
from data.paths import PathResolver, project_root
from data.validation import validate_inventory


def smoke_read(root : Path, count : int) -> list[dict] :
    from data import dir300, docunet, uvdoc
    from data.uvdoc import HDF5EnvironmentError

    resolver = PathResolver(root)
    rows = []
    for dataset in DATASET_KEYS :
        for record in load_records(dataset, root)[:count] :
            row = {"dataset": record.dataset, "sample_key": record.sample_key, "status": "", "reason": ""}
            if (resolver.find(record.input_path) is None) :
                row.update(status="UNAVAILABLE", reason="Input image not present in this environment")
            else :
                try :
                    if (dataset.startswith("uvdoc")) :
                        sample = uvdoc.load_sample(record, resolver, include_geometry=True)
                        row["shape"] = list(sample["image"].shape)
                        row["grid2d_shape"] = list(sample["grid2d"].shape)
                        row["grid3d_shape"] = list(sample["grid3d"].shape)
                    elif (dataset == "docunet") :
                        sample = docunet.load_sample(record, resolver, include_reference=True)
                        row["shape"] = list(sample["image"].shape)
                    else :
                        sample = dir300.load_sample(record, resolver, include_reference=True)
                        row["shape"] = list(sample["image"].shape)
                    row["status"] = "PASS"
                except HDF5EnvironmentError as error :
                    row.update(status="ENVIRONMENT_ERROR", reason=str(error))
                except FileNotFoundError as error :
                    row.update(status="UNAVAILABLE", reason=str(error))
                except (OSError, ValueError, KeyError) as error :
                    row.update(status="FAIL", reason=f"{type(error).__name__}: {error}")
            rows.append(row)
    return rows


def write_csv(path : Path, rows : list[dict], columns : list[str]) :
    with path.open("w", newline="", encoding="utf-8") as handle :
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(output : Path, result : dict, smoke : list[dict] | None, root : Path) :
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps({"inventory": result, "smoke": smoke}, indent=2), encoding="utf-8")
    issue_rows = []
    missing_rows = []
    lines = ["# Stage 1 dataset validation", "", "This report checks the existing EDA manifests without modifying source data.",
             "No neural inference, benchmark scores, or geometric performance results are implied.", "",
             f"Project root: `{root}`", "", "## Inventory", "",
             "| Dataset | Records | Groups | Integrity | Files checked | Files unavailable |",
             "|---|---:|---:|---|---:|---:|"]
    for key in DATASET_KEYS :
        row = result[key]
        files_checked = str(row["files_checked"]) if row["file_check_status"] != "NOT_CHECKED" else "Not checked"
        missing = str(row["missing_file_count"]) if row["file_check_status"] != "NOT_CHECKED" else "Not checked"
        lines.append(f"| {row['dataset']} | {row['records']:,} | {row['groups']:,} | {row['status']} | {files_checked} | {missing} |")
        issue_rows += [{"dataset": key, **issue} for issue in row["issues"]]
        missing_rows += [{"dataset": key, **missing} for missing in row["missing_files"]]
    split = result["split"]
    lines.extend(["", "## Existing UVDoc split", "",
                  f"Status: **{split['status']}**. Training: {split['training_samples']:,}; validation: {split['validation_samples']:,}; "
                  f"base geometry groups: {split['base_groups']:,}; overlapping groups: {split['overlapping_groups']}.",
                  "These are previously saved EDA memberships, not a newly approved training policy.", ""])
    issue_rows += [{"dataset": "split", **issue} for issue in split["issues"]]
    if (issue_rows) :
        lines.extend(["## Integrity issues", "", f"{len(issue_rows)} issue(s) saved to `issues.csv`.", ""])
    else :
        lines.extend(["## Integrity issues", "", "No manifest or split integrity issues detected.", ""])
    if (any(result[key]["files_checked"] for key in DATASET_KEYS)) :
        lines.extend(["## File availability", "",
                      "File checks are environment-specific. Missing staged UVDoc Final members do not imply corrupt manifests.",
                      "The JSON retains the total unavailable count and the CSV lists up to 50 examples per dataset.", ""])
    else :
        lines.extend(["## File availability", "", "Not checked. This was a manifest-only validation.", ""])
    if (smoke is not None) :
        counts = {status: sum(x["status"] == status for x in smoke) for status in ("PASS", "UNAVAILABLE", "FAIL", "ENVIRONMENT_ERROR")}
        lines.extend(["## Representative image/geometry decoding", "",
                      f"PASS: {counts['PASS']}; UNAVAILABLE: {counts['UNAVAILABLE']}; "
                      f"FAIL: {counts['FAIL']}; ENVIRONMENT_ERROR: {counts['ENVIRONMENT_ERROR']}.",
                      "Per-sample records are in `smoke.csv`.", ""])
        write_csv(output / "smoke.csv", smoke, ["dataset", "sample_key", "status", "reason", "shape", "grid2d_shape", "grid3d_shape"])
    write_csv(output / "issues.csv", issue_rows, ["dataset", "code", "sample_key", "detail"])
    write_csv(output / "missing_files.csv", missing_rows, ["dataset", "sample_key", "role", "path"])
    lines.extend(["## Status interpretation", "",
                  "- PASS: The requested check completed and satisfied its structural criteria.",
                  "- FAIL: A manifest, split, or decoded content violation was found.",
                  "- UNAVAILABLE: Required physical files are not present in this environment.",
                  "- ENVIRONMENT_ERROR: A required Python dependency could not load; this is not dataset corruption.",
                  "- REVIEW_REQUIRED: A question requiring human review, such as benchmark orientation, is not resolved by Stage 1.", "",
                  "The DocUNet document-64 reference orientation is not silently corrected by these readers.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int :
    parser = argparse.ArgumentParser(description="Validate the existing four dataset inventories and optional file availability")
    parser.add_argument("--root", default=None, help="Project root (defaults to DOCUMENT_UNWARPING_ROOT or current directory)")
    parser.add_argument("--check-files", action="store_true", help="Check physical input/reference/annotation file presence")
    parser.add_argument("--max-files-per-dataset", type=int, default=None, help="Limit the records used for physical file checks")
    parser.add_argument("--smoke-read", type=int, default=0, help="Decode the first N records and relevant annotations per dataset")
    parser.add_argument("--output", default=None, help="Report directory; default is runs/data_validation/<unique run id>")
    args = parser.parse_args()
    if ((args.max_files_per_dataset is not None and args.max_files_per_dataset < 1) or args.smoke_read < 0) :
        parser.error("File-check limit must be positive and smoke-read count must be nonnegative")
    root = project_root(args.root)
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output = Path(args.output) if args.output else root / "runs/data_validation" / run_name
    result = validate_inventory(root, args.check_files, args.max_files_per_dataset)
    integrity_failed = any(result[key]["issues"] for key in DATASET_KEYS) or bool(result["split"]["issues"])
    smoke = smoke_read(root, args.smoke_read) if (args.smoke_read and not integrity_failed) else None
    if (args.smoke_read and integrity_failed) :
        print("Smoke reading skipped because manifest/split integrity checks failed")
    write_report(output, result, smoke, root)
    print("Dataset validation report:", output / "report.md")
    for key in DATASET_KEYS :
        availability = (f"{result[key]['missing_file_count']} unavailable physical files" if args.check_files
                        else "physical files not checked")
        print(f"{key}: {result[key]['records']} rows, {result[key]['status']}, {availability}")
    print("saved UVDoc split:", result["split"]["status"])
    if (smoke is not None) :
        print("smoke read:", {status: sum(x["status"] == status for x in smoke) for status in ("PASS", "UNAVAILABLE", "FAIL", "ENVIRONMENT_ERROR")})
    physical_unavailable = args.check_files and any(result[key]["missing_file_count"] for key in DATASET_KEYS)
    content_failed = smoke is not None and any(x["status"] == "FAIL" for x in smoke)
    environment_failed = smoke is not None and any(x["status"] == "ENVIRONMENT_ERROR" for x in smoke)
    if integrity_failed or content_failed:
        return 1
    if environment_failed:
        return 3
    return 2 if physical_unavailable else 0


if (__name__ == "__main__") :
    raise SystemExit(main())
