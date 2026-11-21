# tests/test_inference_runner.py
import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from data.manifests import load_records
from data.paths import PathResolver
from models.deep.uvdoc import LoadedUVDoc, load_uvdoc_config
from scripts.run_inference import run_records, selected_records


class DummyModel(torch.nn.Module) :
    def forward(self, x) :
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 45), torch.linspace(-1, 1, 31), indexing="ij")
        return torch.stack([xx, yy])[None], torch.zeros((1, 3, 45, 31))


def test_selection_requires_ids_or_limit(small_project) :
    with pytest.raises(ValueError) :
        selected_records("uvdoc_benchmark", small_project)
    selected = selected_records("uvdoc_benchmark", small_project, sample_ids=["00000"])
    assert selected[0].sample_id == "00000"
    with pytest.raises(ValueError) :
        selected_records("uvdoc_benchmark", small_project, sample_ids=["99999"])


def test_runner_artifact_roundtrip(small_project, monkeypatch, tmp_path) :
    from scripts import run_inference
    monkeypatch.setattr(run_inference, "verify_source", lambda *_: {"model.py": "fixture"})
    records = load_records("uvdoc_benchmark", small_project)
    path = small_project / records[0].input_path
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.random.default_rng(2).integers(0, 256, (33, 45, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    missing = load_records("dir300", small_project)[0]
    loaded = LoadedUVDoc(DummyModel().eval(), torch.device("cpu"), load_uvdoc_config(),
                         tmp_path, tmp_path / "none", "fixture", "fixture")
    output = tmp_path / "run"
    summary = run_records(records + [missing], PathResolver(small_project), loaded, output, "fixturehash")
    assert summary["attempted"] == 2
    assert summary["success"] == 1
    assert summary["unavailable"] == 1
    with (output / "samples.csv").open(newline="") as handle :
        rows = list(csv.DictReader(handle))
    assert [row["status"] for row in rows] == ["OK", "UNAVAILABLE"]
    assert (output / rows[0]["image_artifact"]).is_file()
    with np.load(output / rows[0]["grid_artifact"], allow_pickle=False) as grid :
        assert grid["grid2d"].shape == (2, 45, 31)
        assert grid["grid3d"].shape == (3, 45, 31)
    with pytest.raises(FileExistsError) :
        run_records(records, PathResolver(small_project), loaded, output, "fixturehash")


def test_runner_cleans_partially_written_artifacts(small_project, monkeypatch, tmp_path) :
    from scripts import run_inference
    monkeypatch.setattr(run_inference, "verify_source", lambda *_: {"model.py": "fixture"})
    records = load_records("uvdoc_benchmark", small_project)
    path = small_project / records[0].input_path
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.ones((19, 25, 3), dtype=np.uint8) * 80
    assert cv2.imwrite(str(path), image)
    loaded = LoadedUVDoc(DummyModel(), torch.device("cpu"), load_uvdoc_config(),
                         tmp_path, tmp_path / "none", "fixture", "fixture")
    def failed_npz(*args, **kwargs) :
        raise OSError("Simulated disk write failure")
    monkeypatch.setattr(run_inference, "save_npz_atomic", failed_npz)
    output = tmp_path / "failed_run"
    summary = run_records(records, PathResolver(small_project), loaded, output, "fixturehash")
    assert summary["failed"] == 1
    assert not list(output.rglob("*.png"))
    with (output / "samples.csv").open(newline="") as handle :
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == "FAILED"
    assert "Simulated disk write failure" in rows[0]["failure_reason"]


def test_repeated_identity_rejected_before_output(small_project, monkeypatch, tmp_path) :
    records = load_records("uvdoc_benchmark", small_project)
    loaded = LoadedUVDoc(DummyModel(), torch.device("cpu"), load_uvdoc_config(),
                         tmp_path, tmp_path / "none", "fixture", "fixture")
    output = tmp_path / "repeated"
    with pytest.raises(ValueError, match="Repeated sample keys") :
        run_records(records * 2, PathResolver(small_project), loaded, output, "fixturehash")
    assert not output.exists()
