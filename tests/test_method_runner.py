# tests/test_method_runner.py
"""Small method functions must produce auditable, evaluable saved runs."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data.manifests import load_records
from evaluation.runner import evaluate_inference_run
from models.interfaces import MethodResult
from scripts.run_method import run_method


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def make_test_root(tmp_path, dataset) :
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    (root / "dataset/_metadata").mkdir(parents=True)
    for filename in ("configs/datasets.yaml", "configs/evaluation.yaml",
                     f"dataset/_metadata/{dataset}_manifest.csv") :
        target = root / filename
        shutil.copyfile(SOURCE_ROOT / filename, target)
    return root


def add_image(root, logical, color=(24, 90, 167)) :
    path = root / logical
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (17, 11), color=color).save(path)
    return path


def csv_rows(path) :
    with path.open(newline="", encoding="utf-8") as handle :
        return list(csv.DictReader(handle))


def test_image_only_dir300_can_be_prepared_for_reference_metrics(tmp_path) :
    root = make_test_root(tmp_path, "dir300")
    rec = load_records("dir300", root)[0]
    add_image(root, rec.input_path)
    add_image(root, rec.reference_path)

    def identity(image) :
        return MethodResult("OK", image.copy(), metadata={"note": "identity diagnostic"})

    run = tmp_path / "inference"
    result = run_method([rec], identity, root, run, "identity_classical", {"version": 1})
    assert result["success"] == 1
    rows = csv_rows(run / "samples.csv")
    artifact = run / rows[0]["image_artifact"]
    assert artifact.is_file()
    assert rows[0]["dense_artifact"] == ""
    assert json.loads((run / "provenance.json").read_text())["method_id"] == "identity_classical"
    assert len(csv_rows(run / "artifacts.csv")) == 2

    evaluation = tmp_path / "evaluation"
    summary = evaluate_inference_run(root, run, "dir300_matlab_v1", evaluation)
    assert summary["status_counts"] == {"OK": 1}
    metrics = csv_rows(evaluation / "metrics/per_sample.csv")
    assert {m["method_id"] for m in metrics} == {"identity_classical"}
    assert {m["status"] for m in metrics} == {"UNAVAILABLE_DEPENDENCY"}
    assert (evaluation / "prepared_predictions/reference_input/1_rec.png").is_file()
    assert csv_rows(evaluation / "samples.csv")[0]["method_id"] == "identity_classical"


def test_image_only_docunet_uses_same_runner(tmp_path) :
    root = make_test_root(tmp_path, "docunet")
    rec = load_records("docunet", root, variant="crop")[0]
    add_image(root, rec.input_path)
    add_image(root, rec.reference_path)
    run = tmp_path / "inference"
    run_method([rec], lambda image: MethodResult("OK", image[:, :, 0]), root, run, "gray_classical")
    summary = evaluate_inference_run(root, run, "docunet_matlab_legacy_v1", tmp_path / "evaluation")
    assert summary["status_counts"] == {"OK": 1}
    assert (tmp_path / "evaluation/prepared_predictions/reference_input/1_1.png").is_file()


def test_dense_geometry_preserved_without_uvdoc_grid_assumption(tmp_path) :
    root = make_test_root(tmp_path, "uvdoc_benchmark")
    rec = load_records("uvdoc_benchmark", root)[0]
    add_image(root, rec.input_path)

    def with_grid(image) :
        h, w = image.shape[:2]
        grid = np.zeros((h, w, 2), dtype=np.float32)
        grid[:, :, 0] = np.arange(w)[None, :]
        grid[:, :, 1] = np.arange(h)[:, None]
        return MethodResult("OK", image, backward_map=grid, map_space="source_pixels",
                            valid_mask=np.ones((h, w), dtype=bool))

    run = tmp_path / "inference"
    run_method([rec], with_grid, root, run, "identity_geometry")
    row = csv_rows(run / "samples.csv")[0]
    with np.load(run / row["dense_artifact"], allow_pickle=False) as saved :
        assert saved["backward_map"].shape == (11, 17, 2)
        assert saved["map_space"].item() == "source_pixels"
        assert saved["valid_mask"].dtype == np.bool_
    # A dense backward map cannot be used as UVDoc's native coarse 45x31 grid.
    result = evaluate_inference_run(root, run, "uvdoc_lines_official_n25_v1", tmp_path / "evaluation")
    assert result["status_counts"] == {"NOT_APPLICABLE": 1}
    assert {x["status"] for x in csv_rows(tmp_path / "evaluation/metrics/per_sample.csv")} == {"NOT_APPLICABLE"}


def test_failures_and_missing_inputs_are_retained(tmp_path) :
    root = make_test_root(tmp_path, "dir300")
    records = load_records("dir300", root)[:3]
    add_image(root, records[0].input_path)
    add_image(root, records[1].input_path)
    calls = 0

    def method(image) :
        nonlocal calls
        calls += 1
        if (calls == 1) :
            return MethodResult("FAILED", None, failure_reason="No page boundary")
        raise RuntimeError("Synthetic model failure")

    run = tmp_path / "inference"
    summary = run_method(records, method, root, run, "failure_example")
    assert (summary["failed"], summary["unavailable"]) == (2, 1)
    assert [r["status"] for r in csv_rows(run / "samples.csv")] == ["FAILED", "FAILED", "UNAVAILABLE"]
    assert not csv_rows(run / "artifacts.csv")
    evaluation = evaluate_inference_run(root, run, "dir300_matlab_v1", tmp_path / "evaluation")
    assert evaluation["status_counts"] == {"UNAVAILABLE_PREDICTION": 3}


def test_bad_return_and_partial_artifacts_are_not_registered(tmp_path) :
    root = make_test_root(tmp_path, "dir300")
    rec = load_records("dir300", root)[0]
    add_image(root, rec.input_path)
    run = tmp_path / "inference"
    run_method([rec], lambda img: MethodResult("OK", img.astype(np.float32)), root, run, "bad_output")
    assert csv_rows(run / "samples.csv")[0]["status"] == "FAILED"
    assert csv_rows(run / "artifacts.csv") == []
    assert not list((run / "predictions").rglob("*.png"))


def test_invalid_selection_and_existing_run_rejected(tmp_path) :
    root = make_test_root(tmp_path, "dir300")
    rec = load_records("dir300", root)[0]
    method = lambda image: MethodResult("OK", image)
    run = tmp_path / "inference"
    with pytest.raises(ValueError, match="method_id") :
        run_method([rec], method, root, run, "../unsafe")
    with pytest.raises(ValueError, match="unique") :
        run_method([rec, rec], method, root, run, "simple")
    with pytest.raises(ValueError) :
        run_method([rec], method, root, run, "simple", {"invalid": float("nan")})
    assert not run.exists()
    run.mkdir()
    with pytest.raises(FileExistsError) :
        run_method([rec], method, root, run, "simple")


def test_eval_detects_modified_non_uvdoc_artifact(tmp_path) :
    root = make_test_root(tmp_path, "dir300")
    rec = load_records("dir300", root)[0]
    add_image(root, rec.input_path)
    add_image(root, rec.reference_path)
    run = tmp_path / "inference"
    run_method([rec], lambda img: MethodResult("OK", img), root, run, "neural_example")
    row = csv_rows(run / "samples.csv")[0]
    (run / row["image_artifact"]).write_bytes(b"tampered")
    summary = evaluate_inference_run(root, run, "dir300_matlab_v1", tmp_path / "evaluation")
    assert summary["status_counts"] == {"INVALID_INPUT": 1}
    assert {x["method_id"] for x in csv_rows(tmp_path / "evaluation/metrics/per_sample.csv")} == {"neural_example"}
