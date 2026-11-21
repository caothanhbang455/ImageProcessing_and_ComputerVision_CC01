# scripts/run_inference.py
"""Reproducible, selection-limited, pretrained UVDoc inference on Stage 1 records.

This runner does not compute benchmark metrics, alter source images, or tune UVDoc.
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.manifests import DATASET_KEYS, load_records, manifest_path
from data.paths import PathResolver, project_root
from models.deep.uvdoc import (LoadedUVDoc, hash_file, load_uvdoc_config,
                               load_uvdoc_model, predict_uvdoc, verify_source)


COLUMNS = ("dataset", "sample_id", "variant", "sample_key", "group_id", "input_path", "resolved_path",
           "status", "failure_reason", "width", "height", "output_width", "output_height",
           "decode_ms", "preprocess_ms", "forward_ms", "warp_ms", "conversion_ms",
           "save_ms", "end_to_end_ms", "image_artifact", "grid_artifact", "dense_artifact", "bm_artifact")


def save_rgb_png(image : np.ndarray, destination : Path) -> None :
    if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3) :
        raise ValueError("Only uint8 RGB H x W x 3 output can be saved")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.stem + ".partial.png")
    try :
        if (not cv2.imwrite(str(temp), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))) :
            raise OSError(f"OpenCV could not write {destination}")
        os.replace(temp, destination)
    finally :
        if (temp.exists()) :
            temp.unlink()


def save_npz_atomic(destination : Path, **arrays) -> None :
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".partial.npz")
    try :
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
    finally :
        if (temporary.exists()) :
            temporary.unlink()


def write_csv(destination : Path, rows : list[dict], fields : tuple[str, ...] | list[str]) -> None :
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle :
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(destination : Path, value : dict) -> None :
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _team_git_info(root : Path) -> dict[str, object] :
    def git(*args) :
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)

    rev = git("rev-parse", "HEAD")
    if (rev.returncode != 0) :
        return {"revision" : None, "working_tree_modified" : None}
    dirty = git("status", "--porcelain", "--untracked-files=no")
    return {"revision" : rev.stdout.strip(),
            "working_tree_modified" : bool(dirty.stdout.strip()) if dirty.returncode == 0 else None}


def environment_record(device : torch.device) -> dict[str, object] :
    return {"python" : sys.version.split()[0], "platform" : platform.platform(),
            "numpy" : np.__version__, "opencv" : cv2.__version__, "torch" : torch.__version__,
            "cuda_version" : torch.version.cuda, "device" : str(device),
            "gpu" : torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "timing" : "CUDA synchronized before/after each timed inference segment",
            "precision" : "float32", "batch_size" : 1}


def selected_records(dataset : str, root : Path, sample_ids : list[str] | None = None,
                     limit : int | None = None, variant : str | None = None) -> list :
    """Never infer 20k samples accidentally; require explicit selection."""
    if (not sample_ids and limit is None) :
        raise ValueError("Specify --sample-id and/or --limit to select an intentional diagnostic subset")
    if (limit is not None and limit < 1) :
        raise ValueError("--limit must be positive")
    records = load_records(dataset, root, variant=variant)
    if (sample_ids) :
        requested = set(sample_ids)
        found = {row.sample_id for row in records}
        missing = requested - found
        if (missing) :
            raise ValueError(f"Requested IDs not in {dataset} manifest: {sorted(missing)}")
        records = [row for row in records if row.sample_id in requested]
    if (limit is not None) :
        records = records[:limit]
    return records


def _safe_key(value : str) -> str :
    if (not value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in value)) :
        raise ValueError(f"Unsafe sample key for artifact path: {value!r}")
    return value


def _write_bm_mat(path : Path, grid2d : np.ndarray) -> None :
    try :
        import hdf5storage
    except ImportError as error :
        raise RuntimeError("Optional benchmark MAT export requires hdf5storage>=0.2,<0.3") from error
    if (grid2d.shape != (2, 45, 31)) :
        raise ValueError("Expected UVDoc coarse grid (2,45,31)")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.mat")
    try :
        hdf5storage.savemat(str(temporary), {"bm" : grid2d.transpose(1, 2, 0)})
        os.replace(temporary, path)
    finally :
        if (temporary.exists()) :
            temporary.unlink()


def run_records(records : list, resolver : PathResolver, loaded : LoadedUVDoc, output : Path,
                manifest_sha256 : str, save_dense_maps : bool = False,
                export_benchmark_mat : bool = False) -> dict[str, object] :
    """Run selected records, preserve all per-sample outcomes and artifact identities."""
    if (not records) :
        raise ValueError("At least one manifest record must be selected")
    sample_keys = [row.sample_key for row in records]
    if (len(sample_keys) != len(set(sample_keys))) :
        raise ValueError("Repeated sample keys would overwrite inference artifacts")
    if (export_benchmark_mat and any(row.dataset == "UVDoc Benchmark" for row in records)) :
        try :
            import hdf5storage
        except ImportError as error :
            raise RuntimeError("Benchmark MAT export requires hdf5storage>=0.2,<0.3") from error
    if (output.exists()) :
        raise FileExistsError(f"Refusing to overwrite an existing inference run: {output}")
    output.mkdir(parents=True, exist_ok=False)
    method_dir = output / "predictions/uvdoc_official"
    diag_dir = output / "diagnostics"
    artifacts : list[dict] = []
    sample_rows : list[dict] = []
    grid_rows : list[dict] = []
    failures : list[dict] = []
    provenance = {
        "method_id" : "uvdoc_official", "official_source_commit" : loaded.source_commit,
        "official_source_url" : loaded.config["source"]["repository"],
        "checkpoint_path" : str(loaded.checkpoint_path), "checkpoint_sha256" : loaded.checkpoint_sha256,
        "official_source_file_sha256" : verify_source(loaded.source_dir, loaded.source_commit),
        "manifest_sha256" : manifest_sha256, "team_git" : _team_git_info(resolver.root),
        "note" : "No benchmark metrics computed; source references and training memberships were not used in inference",
    }
    write_json(output / "provenance.json", provenance)
    write_json(output / "config.json", {"model" : loaded.config, "sample_keys" : [r.sample_key for r in records],
                                         "save_dense_maps" : save_dense_maps,
                                         "export_benchmark_mat" : export_benchmark_mat})
    write_json(output / "environment.json", environment_record(loaded.device))

    for index, record in enumerate(records, start=1) :
        start = perf_counter()
        key = _safe_key(record.sample_key)
        row = {"dataset" : record.dataset, "sample_id" : record.sample_id, "variant" : record.variant,
               "sample_key" : key, "group_id" : record.group_id, "input_path" : record.input_path,
               "status" : "", "failure_reason" : ""}
        resolved = resolver.find(record.input_path)
        if (resolved is None) :
            row.update(status="UNAVAILABLE", failure_reason="Input image unavailable in current environment")
        else :
            row["resolved_path"] = str(resolved)
            produced_files = []
            try :
                prediction = predict_uvdoc(loaded, resolved, return_dense_map=save_dense_maps)
                result = prediction.result
                if (result.status != "OK" or result.image is None) :
                    raise ValueError("UVDoc inference did not return a valid successful MethodResult")
                image_file = method_dir / "images" / f"{key}.png"
                coarse_file = method_dir / "coarse_grids" / f"{key}.npz"
                save_start = perf_counter()
                save_rgb_png(result.image, image_file)
                produced_files.append(image_file)
                save_npz_atomic(coarse_file, grid2d=prediction.grid2d, grid3d=prediction.grid3d,
                                map_space=np.array("normalized_minus1_plus1"),
                                sample_key=np.array(key), original_size_wh=np.array(prediction.input_size))
                produced_files.append(coarse_file)
                paths = [("image", image_file), ("coarse_grids", coarse_file)]
                if (save_dense_maps) :
                    dense_file = method_dir / "dense_maps" / f"{key}.npz"
                    save_npz_atomic(dense_file, backward_map=result.backward_map, valid_mask=result.valid_mask,
                                    map_space=np.array("normalized_minus1_plus1"))
                    produced_files.append(dense_file)
                    paths.append(("dense_map", dense_file))
                    row["dense_artifact"] = str(dense_file.relative_to(output))
                if (export_benchmark_mat and record.dataset == "UVDoc Benchmark") :
                    mat_file = method_dir / "benchmark_maps" / f"{record.sample_id}.mat"
                    _write_bm_mat(mat_file, prediction.grid2d)
                    produced_files.append(mat_file)
                    paths.append(("benchmark_map", mat_file))
                    row["bm_artifact"] = str(mat_file.relative_to(output))
                save_ms = (perf_counter() - save_start) * 1000
                pending_artifacts = []
                for kind, path in paths :
                    pending_artifacts.append({"sample_key" : key, "artifact_type" : kind,
                                              "relative_path" : str(path.relative_to(output)),
                                              "sha256" : hash_file(path), "bytes" : path.stat().st_size})
                artifacts.extend(pending_artifacts)
                timings = prediction.timings_ms
                row.update(status="OK", width=prediction.input_size[0], height=prediction.input_size[1],
                           output_width=prediction.output_size[0], output_height=prediction.output_size[1],
                           image_artifact=str(image_file.relative_to(output)),
                           grid_artifact=str(coarse_file.relative_to(output)), save_ms=round(save_ms, 3))
                row.update({name : round(timings.get(name, 0.0), 3) for name in
                            ("decode_ms", "preprocess_ms", "forward_ms", "warp_ms", "conversion_ms")})
                grid_rows.append({"sample_key" : key, **result.metadata["grid_diagnostics"]})
            except Exception as error :
                # A per-image failure is a recoverable event. Preserve the failure
                # and continue with the rest of the explicitly selected samples.
                for partial_artifact in produced_files :
                    partial_artifact.unlink(missing_ok=True)
                row.update(status="FAILED", failure_reason=f"{type(error).__name__}: {error}")
        row["end_to_end_ms"] = round((perf_counter() - start) * 1000, 3)
        sample_rows.append(row)
        if (row["status"] != "OK") :
            failures.append({"sample_key" : key, "status" : row["status"],
                             "failure_reason" : row["failure_reason"], "input_path" : record.input_path})
        print(f"[{index}/{len(records)}] {key}: {row['status']}", flush=True)

    write_csv(output / "samples.csv", sample_rows, COLUMNS)
    write_csv(diag_dir / "timings.csv", sample_rows,
              ["sample_key", "dataset", "status", "decode_ms", "preprocess_ms", "forward_ms",
               "warp_ms", "conversion_ms", "save_ms", "end_to_end_ms"])
    write_csv(diag_dir / "failures.csv", failures, ["sample_key", "status", "failure_reason", "input_path"])
    write_csv(diag_dir / "grid_statistics.csv", grid_rows,
              ["sample_key", "total_values", "finite_values", "x_min", "x_max", "y_min", "y_max",
               "outside_normalized_fraction", "horizontal_step_mean", "vertical_step_mean"])
    write_csv(output / "artifacts.csv", artifacts, ["sample_key", "artifact_type", "relative_path", "sha256", "bytes"])
    counts = Counter(row["status"] for row in sample_rows)
    summary = {"attempted" : len(records), "success" : counts["OK"], "unavailable" : counts["UNAVAILABLE"],
               "failed" : counts["FAILED"], "run_dir" : str(output)}
    write_json(output / "summary.json", summary)
    report = ["# Stage 2: official UVDoc inference", "",
              "This run produces predictions and geometry diagnostics only. No benchmark accuracy is claimed.", "",
              f"- Attempted records: **{len(records)}**",
              f"- Successful predictions: **{counts['OK']}**",
              f"- Missing inputs: **{counts['UNAVAILABLE']}**",
              f"- Failed predictions: **{counts['FAILED']}**", "",
              f"- Official source: `{loaded.source_commit}`",
              f"- Checkpoint SHA-256: `{loaded.checkpoint_sha256}`", "",
              "## Artifacts", "",
              "`samples.csv` retains every attempted identity; `diagnostics/` includes grid statistics,",
              "timings, and failure reasons. `artifacts.csv` records every saved binary with SHA-256.", "",
              "Independent official-demo pixel parity is a separate verification gate. This run alone does not establish it.", ""]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def main() -> int :
    parser = argparse.ArgumentParser(description="Pinned official UVDoc inference; no evaluation or training")
    parser.add_argument("--root", default=None)
    parser.add_argument("--dataset", required=True, choices=DATASET_KEYS)
    parser.add_argument("--sample-id", action="append", default=[], help="Repeat for multiple original sample IDs")
    parser.add_argument("--limit", type=int, default=None, help="Process first N manifest records, intentionally")
    parser.add_argument("--variant", default=None, help="DocUNet: crop or original")
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    parser.add_argument("--save-dense-maps", action="store_true")
    parser.add_argument("--export-benchmark-mat", action="store_true")
    args = parser.parse_args()

    root = project_root(args.root)
    records = selected_records(args.dataset, root, args.sample_id, args.limit, args.variant)
    cfg = load_uvdoc_config()
    loaded = load_uvdoc_model(cfg, device=args.device, source_dir=args.source_dir, checkpoint_path=args.checkpoint)
    run_id = datetime.now(timezone.utc).strftime("stage2_%Y%m%dT%H%M%SZ_") + uuid4().hex[:8]
    run_path = Path(args.output) if args.output else root / "runs/inference" / run_id
    summary = run_records(records, PathResolver(root), loaded, run_path,
                          hash_file(manifest_path(args.dataset, root)),
                          save_dense_maps=args.save_dense_maps, export_benchmark_mat=args.export_benchmark_mat)
    print("Run directory:", run_path)
    print(json.dumps(summary, indent=2))
    return 0 if summary["success"] == summary["attempted"] else 2


if (__name__ == "__main__") :
    raise SystemExit(main())
