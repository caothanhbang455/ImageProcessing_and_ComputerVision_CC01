# tests/test_evaluation_runner.py
"""Exercise Stage 2 to Stage 3 integration using artificial saved artifacts."""

import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from data.manifests import load_records, manifest_path
from evaluation import runner
from models.deep.uvdoc import hash_file


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def make_csv(path, rows, fields) :
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle :
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fake_run(tmp_path, prediction_status="OK") :
    root = tmp_path / "repository"
    (root / "configs").mkdir(parents=True)
    shutil.copyfile(SOURCE_ROOT / "configs/evaluation.yaml", root / "configs/evaluation.yaml")
    shutil.copyfile(SOURCE_ROOT / "configs/datasets.yaml", root / "configs/datasets.yaml")
    metadata = root / "dataset/_metadata"
    metadata.mkdir(parents=True)
    shutil.copyfile(SOURCE_ROOT / "dataset/_metadata/uvdoc_benchmark_manifest.csv",
                    metadata / "uvdoc_benchmark_manifest.csv")
    record = next(x for x in load_records("uvdoc_benchmark", root) if x.sample_id == "00010")
    run = tmp_path / "inference"
    run.mkdir()
    image = run / f"predictions/uvdoc_official/images/{record.sample_key}.png"
    grid = run / f"predictions/uvdoc_official/coarse_grids/{record.sample_key}.npz"
    artifacts = []
    if (prediction_status == "OK") :
        image.parent.mkdir(parents=True)
        grid.parent.mkdir(parents=True)
        assert cv2.imwrite(str(image), np.zeros((9, 9, 3), dtype=np.uint8))
        np.savez_compressed(grid, grid2d=np.zeros((2, 45, 31), dtype=np.float32),
                            map_space=np.array("normalized_minus1_plus1"), sample_key=np.array(record.sample_key))
        for kind, f in (("image", image), ("coarse_grids", grid)) :
            artifacts.append(dict(sample_key=record.sample_key, artifact_type=kind,
                                  relative_path=f.relative_to(run).as_posix(), sha256=hash_file(f), bytes=f.stat().st_size))
    make_csv(run / "samples.csv", [{"dataset":record.dataset, "sample_id":record.sample_id,
                                     "variant":record.variant, "sample_key":record.sample_key,
                                     "group_id":record.group_id, "input_path":record.input_path,
                                     "status":prediction_status, "failure_reason":"input unavailable" if prediction_status != "OK" else "",
                                     "image_artifact":image.relative_to(run).as_posix() if prediction_status == "OK" else "",
                                     "grid_artifact":grid.relative_to(run).as_posix() if prediction_status == "OK" else ""}],
             ["dataset", "sample_id", "variant", "sample_key", "group_id", "input_path", "status",
              "failure_reason", "image_artifact", "grid_artifact"])
    make_csv(run / "artifacts.csv", artifacts, ["sample_key", "artifact_type", "relative_path", "sha256", "bytes"])
    (run / "provenance.json").write_text(json.dumps({"manifest_sha256":hash_file(manifest_path("uvdoc_benchmark", root)),
                                                     "method_id":"uvdoc_official", "checkpoint_sha256":"dummy",
                                                     "official_source_commit":"dummy"}))
    return root, run, record


def test_stage2_run_scored_without_model_or_real_dataset(tmp_path, monkeypatch) :
    root, inference, record = fake_run(tmp_path)
    monkeypatch.setattr(runner.uvdoc, "load_uvmap", lambda r, resolver: np.zeros((8, 8, 2)))
    monkeypatch.setattr(runner.uvdoc, "compute_sample_lines", lambda uv, bm, device: (
        [{"direction":"horizontal", "position":50, "value":1.2,"status":"OK", "reason":""},
         {"direction":"vertical", "position":50, "value":2.0,"status":"OK", "reason":""}],
        {"horizontal":1.2,"vertical":2.0,"horizontal_valid_lines":1,"vertical_valid_lines":1,
         "horizontal_missing_lines":0,"vertical_missing_lines":0}))
    output = tmp_path / "scored"
    result = runner.evaluate_inference_run(root, inference, "uvdoc_lines_official_n25_v1", output)
    assert result["n_attempted"] == 1 and result["status_counts"] == {"OK":1}
    assert result["metrics"][0]["n_metric_ok"] == 1
    assert not result["reference_verified"]
    assert (output / "metrics/per_sample.csv").exists()
    assert (output / "metrics/per_line.csv").exists()
    assert (output / "artifacts.csv").exists()
    with pytest.raises(FileExistsError) :
        runner.evaluate_inference_run(root, inference, "uvdoc_lines_official_n25_v1", output)


def test_missing_prediction_is_counted_not_dropped(tmp_path) :
    root, inference, rec = fake_run(tmp_path, "UNAVAILABLE")
    result = runner.evaluate_inference_run(root, inference, "uvdoc_lines_official_n25_v1", tmp_path / "eval")
    assert result["status_counts"] == {"UNAVAILABLE_PREDICTION":1}
    assert all(m["n_metric_ok"] == 0 for m in result["metrics"])


def test_bad_hash_is_rejected_per_sample(tmp_path) :
    root, inference, rec = fake_run(tmp_path)
    path = inference / f"predictions/uvdoc_official/images/{rec.sample_key}.png"
    path.write_bytes(b"tampered")
    result = runner.evaluate_inference_run(root, inference, "uvdoc_lines_official_n25_v1", tmp_path / "eval")
    assert result["status_counts"] == {"INVALID_INPUT":1}
    assert all(m["n_metric_ok"] == 0 for m in result["metrics"])


def test_changed_manifest_is_rejected_before_output_creation(tmp_path) :
    root, inference, rec = fake_run(tmp_path)
    with (root / "dataset/_metadata/uvdoc_benchmark_manifest.csv").open("a") as stream :
        stream.write("\n")
    with pytest.raises(ValueError, match="SHA-256") :
        runner.evaluate_inference_run(root, inference, "uvdoc_lines_official_n25_v1", tmp_path / "eval")
    assert not (tmp_path / "eval").exists()


def test_traversal_rejected(tmp_path) :
    run = tmp_path / "inference"
    run.mkdir()
    with pytest.raises(ValueError) :
        runner.safe_relative_file(run, "../../secret")
    with pytest.raises(ValueError) :
        runner.safe_relative_file(run, "/tmp/secret")


def test_missing_registered_prediction_reports_prediction_unavailable(tmp_path) :
    root, inference, rec = fake_run(tmp_path)
    path = inference / f"predictions/uvdoc_official/images/{rec.sample_key}.png"
    path.unlink()
    result = runner.evaluate_inference_run(root, inference, "uvdoc_lines_official_n25_v1", tmp_path / "eval")
    assert result["status_counts"] == {"UNAVAILABLE_PREDICTION":1}


def test_uvdoc_texture_preparation_reads_saved_grid(tmp_path) :
    root, inference, rec = fake_run(tmp_path)
    texture = root / f"dataset/UVDoc/UVDoc_benchmark/warped_textures/{rec.sample_id}.png"
    texture.parent.mkdir(parents=True)
    assert cv2.imwrite(str(texture), np.ones((10, 11, 3), dtype=np.uint8) * 180)
    result = runner.evaluate_inference_run(root, inference, "uvdoc_texture_prepare_v1", tmp_path / "eval")
    assert result["status_counts"] == {"OK":1}
    prepared = tmp_path / "eval/prepared_predictions/uwp_texture" / f"{rec.sample_id}.png"
    assert prepared.is_file()
    assert cv2.imread(str(prepared)).shape == (10, 11, 3)
