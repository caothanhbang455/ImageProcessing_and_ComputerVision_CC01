# scripts/run_method.py
"""Run a Python image method on Stage 1 records and save evaluable predictions.

The callable receives one uint8 RGB image and returns MethodResult. This
runner does not load checkpoints, compute metrics, or change source datasets.
"""

import csv
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.images import read_image
from data.manifests import load_records, manifest_path
from data.paths import PathResolver, project_root
from data.records import SampleRecord
from models.interfaces import MethodResult


SAMPLE_FIELDS = ("dataset", "sample_id", "variant", "sample_key", "group_id", "input_path",
                 "resolved_path", "status", "failure_reason", "width", "height",
                 "output_width", "output_height", "end_to_end_ms", "image_artifact",
                 "dense_artifact", "metadata_artifact")
ARTIFACT_FIELDS = ("sample_key", "artifact_type", "relative_path", "sha256", "bytes")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


def _hash_file(path : Path) -> str :
    digest = sha256()
    with path.open("rb") as handle :
        for chunk in iter(lambda : handle.read(1024 * 1024), b"") :
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path : Path, payload : dict) -> None :
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _write_csv(path : Path, rows : list[dict], columns : tuple[str, ...]) -> None :
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle :
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _save_image(image : np.ndarray, path : Path) -> None :
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim not in (2, 3)) :
        raise ValueError("Method image must be a uint8 grayscale, RGB, or RGBA array")
    if (image.ndim == 3) :
        if (image.shape[2] == 1) :
            stored = image[:, :, 0]
        elif (image.shape[2] == 3) :
            stored = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif (image.shape[2] == 4) :
            stored = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        else :
            raise ValueError("Method image channels must be 1, 3, or 4")
    else :
        stored = image
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.png")
    try :
        if (not cv2.imwrite(str(temporary), np.ascontiguousarray(stored))) :
            raise OSError(f"Unable to save prediction: {path}")
        os.replace(temporary, path)
    finally :
        temporary.unlink(missing_ok=True)


def _save_geometry(result : MethodResult, sample_key : str, path : Path) -> None :
    arrays = {"sample_key" : np.array(sample_key)}
    if (result.backward_map is not None) :
        arrays["backward_map"] = np.asarray(result.backward_map)
        arrays["map_space"] = np.array(result.map_space)
    if (result.valid_mask is not None) :
        arrays["valid_mask"] = np.asarray(result.valid_mask)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.npz")
    try :
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally :
        temporary.unlink(missing_ok=True)


def _git_revision(root : Path) -> str | None :
    try :
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True)
    except FileNotFoundError :
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def run_method(records : list[SampleRecord], method : Callable[[np.ndarray], MethodResult],
               root : str | Path, output_dir : str | Path, method_id : str,
               config : dict | None = None) -> dict :
    """Save a fresh method run compatible with the Stage 3 artifact reader.

    The records must belong to one dataset and input variant. Model loading
    should happen before this call, not once per image.
    """
    root = project_root(root)
    output = Path(output_dir).expanduser().resolve()
    records = list(records)
    if (not callable(method) or not SAFE_ID.fullmatch(method_id)) :
        raise ValueError("method must be callable and method_id must use letters, digits, _ or -")
    if (not records or not all(isinstance(r, SampleRecord) for r in records)) :
        raise ValueError("Select at least one Stage 1 SampleRecord")
    identities = {(r.dataset, r.variant) for r in records}
    keys = [r.sample_key for r in records]
    if (len(identities) != 1 or len(set(keys)) != len(keys) or
        any(not SAFE_ID.fullmatch(key) for key in keys)) :
        raise ValueError("Use one dataset/variant with unique safe sample keys per run")
    dataset_label, variant = next(iter(identities))
    dataset_names = {"UVDoc Final" : "uvdoc_final", "UVDoc Benchmark" : "uvdoc_benchmark",
                     "DocUNet" : "docunet", "DIR300" : "dir300"}
    if (dataset_label not in dataset_names) :
        raise ValueError(f"Unsupported Stage 1 dataset: {dataset_label}")
    dataset = dataset_names[dataset_label]
    canonical = {r.sample_key : r for r in load_records(dataset, root, variant=variant)}
    for rec in records :
        original = canonical.get(rec.sample_key)
        if (original is None or (original.dataset, original.sample_id, original.variant,
            original.group_id, original.input_path) != (rec.dataset, rec.sample_id, rec.variant,
            rec.group_id, rec.input_path)) :
            raise ValueError(f"Record does not match the current Stage 1 manifest: {rec.sample_key}")
    method_config = {} if config is None else config
    if (not isinstance(method_config, dict)) :
        raise TypeError("config must be a JSON-compatible dictionary")
    # Validate the configuration before creating a run directory.
    json.dumps(method_config, allow_nan=False)
    if (output.exists()) :
        raise FileExistsError(f"Refusing to overwrite an inference run: {output}")

    resolver = PathResolver(root)
    manifest_sha = _hash_file(manifest_path(dataset, root))
    output.mkdir(parents=True, exist_ok=False)
    method_dir = output / "predictions" / method_id
    _write_json(output / "config.json", {"method_id" : method_id, "method_config" : method_config,
                                          "sample_keys" : keys})
    _write_json(output / "provenance.json", {"method_id" : method_id, "manifest_sha256" : manifest_sha,
                                              "team_git_revision" : _git_revision(root),
                                              "note" : "Inference only; no benchmark metric computed"})
    _write_json(output / "environment.json", {"python" : sys.version.split()[0],
                                               "platform" : platform.platform(),
                                               "numpy" : np.__version__, "opencv" : cv2.__version__})
    rows, artifacts = [], []
    for position, rec in enumerate(records, start=1) :
        start = perf_counter()
        row = {"dataset" : rec.dataset, "sample_id" : rec.sample_id, "variant" : rec.variant,
               "sample_key" : rec.sample_key, "group_id" : rec.group_id or "",
               "input_path" : rec.input_path, "status" : "", "failure_reason" : ""}
        resolved = resolver.find(rec.input_path)
        if (resolved is None) :
            row.update(status="UNAVAILABLE", failure_reason="Source image not available")
        else :
            row["resolved_path"] = str(resolved)
            written : list[Path] = []
            pending : list[dict] = []
            try :
                image = read_image(resolved, mode="RGB")
                row.update(width=image.shape[1], height=image.shape[0])
                result = method(image)
                if (not isinstance(result, MethodResult)) :
                    raise TypeError("Method must return models.interfaces.MethodResult")
                if (result.status != "OK") :
                    row.update(status=result.status, failure_reason=result.failure_reason)
                else :
                    if (result.image is None or result.image.dtype != np.uint8) :
                        raise ValueError("Successful output must contain a uint8 image")
                    if (result.backward_map is not None and
                        (result.map_space not in ("source_pixels", "normalized_minus1_plus1") or
                         result.backward_map.shape != (*result.image.shape[:2], 2) or
                         not np.isfinite(result.backward_map).all())) :
                        raise ValueError("Invalid dense backward map or coordinate space")
                    if (result.valid_mask is not None and
                        (result.valid_mask.shape != result.image.shape[:2] or result.valid_mask.dtype != np.bool_)) :
                        raise ValueError("Invalid output validity mask")
                    image_file = method_dir / "images" / f"{rec.sample_key}.png"
                    _save_image(result.image, image_file)
                    written.append(image_file)
                    row.update(status="OK", image_artifact=image_file.relative_to(output).as_posix(),
                               output_width=result.image.shape[1], output_height=result.image.shape[0])
                    saved = [("image", image_file)]
                    if (result.backward_map is not None or result.valid_mask is not None) :
                        geometry_file = method_dir / "dense_maps" / f"{rec.sample_key}.npz"
                        _save_geometry(result, rec.sample_key, geometry_file)
                        written.append(geometry_file)
                        row["dense_artifact"] = geometry_file.relative_to(output).as_posix()
                        saved.append(("dense_map", geometry_file))
                    if (result.metadata) :
                        metadata_file = method_dir / "metadata" / f"{rec.sample_key}.json"
                        _write_json(metadata_file, result.metadata)
                        written.append(metadata_file)
                        row["metadata_artifact"] = metadata_file.relative_to(output).as_posix()
                        saved.append(("metadata", metadata_file))
                    pending = [{"sample_key" : rec.sample_key, "artifact_type" : kind,
                                "relative_path" : path.relative_to(output).as_posix(),
                                "sha256" : _hash_file(path), "bytes" : path.stat().st_size}
                               for kind, path in saved]
            except Exception as error :
                for file in written :
                    file.unlink(missing_ok=True)
                row.update(status="FAILED", failure_reason=f"{type(error).__name__}: {error}")
                row.pop("image_artifact", None)
                row.pop("dense_artifact", None)
                row.pop("metadata_artifact", None)
                pending.clear()
            artifacts.extend(pending)
        row["end_to_end_ms"] = round((perf_counter() - start) * 1000, 3)
        rows.append(row)
        _write_csv(output / "samples.csv", rows, SAMPLE_FIELDS)
        _write_csv(output / "artifacts.csv", artifacts, ARTIFACT_FIELDS)
        print(f"[{position}/{len(records)}] {rec.sample_key}: {row['status']}", flush=True)
    counts = Counter(row["status"] for row in rows)
    summary = {"attempted" : len(rows), "success" : counts["OK"],
               "unavailable" : counts["UNAVAILABLE"], "failed" : counts["FAILED"],
               "review_required" : counts["REVIEW_REQUIRED"], "run_dir" : str(output)}
    _write_json(output / "summary.json", summary)
    return summary
