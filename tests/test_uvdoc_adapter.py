# tests/test_uvdoc_adapter.py
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from models.deep.uvdoc import (LoadedUVDoc, hash_file, load_uvdoc_config,
                               predict_uvdoc, predict_uvdoc_from_rgb, preprocess_official_rgb)


class DeterministicGridModel(torch.nn.Module) :
    def forward(self, image) :
        x = torch.linspace(-1, 1, 31, device=image.device)
        y = torch.linspace(-1, 1, 45, device=image.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([xx, yy])[None].to(torch.float32)
        return grid, torch.zeros((1, 3, 45, 31), dtype=torch.float32, device=image.device)


@pytest.fixture
def fake_loaded(tmp_path) :
    return LoadedUVDoc(DeterministicGridModel().eval(), torch.device("cpu"), load_uvdoc_config(),
                       tmp_path, tmp_path / "checkpoint.pkl", "synthetic-sha", "synthetic-commit")


def test_official_preprocess_exact(fake_loaded) :
    img = np.random.default_rng(5).integers(0, 256, (38, 49, 3), dtype=np.uint8)
    prepared, source = preprocess_official_rgb(img, torch.device("cpu"))
    expected = cv2.resize(img.astype(np.float32) / 255, (488, 712), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(source, img.astype(np.float32) / 255)
    np.testing.assert_array_equal(prepared.numpy(), expected.transpose(2, 0, 1)[None])


def test_adapter_matches_formula_on_synthetic_model(fake_loaded) :
    img = np.random.default_rng(1).integers(0, 256, (41, 53, 3), dtype=np.uint8)
    prediction = predict_uvdoc_from_rgb(fake_loaded, img, return_dense_map=True)
    assert prediction.result.status == "OK"
    assert prediction.result.image.shape == img.shape
    assert prediction.result.image.dtype == np.uint8
    assert prediction.result.backward_map.shape == (41, 53, 2)
    assert prediction.result.valid_mask.shape == (41, 53)
    inp, float_img = preprocess_official_rgb(img, "cpu")
    g2, _ = fake_loaded.model(inp)
    full = F.interpolate(g2, size=(41, 53), mode="bilinear", align_corners=True)
    direct = F.grid_sample(torch.from_numpy(float_img.transpose(2, 0, 1).copy())[None],
                           full.permute(0, 2, 3, 1), align_corners=True)
    expected = (direct[0].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    np.testing.assert_array_equal(prediction.result.image, expected)


def test_predict_image_path_and_missing(fake_loaded, tmp_path) :
    source = tmp_path / "source.png"
    rgb = np.zeros((30, 40, 3), dtype=np.uint8)
    rgb[..., 0] = 120
    assert cv2.imwrite(str(source), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    p = predict_uvdoc(fake_loaded, source)
    assert p.result.backward_map is None and p.result.map_space is None
    assert p.timings_ms["decode_ms"] >= 0
    with pytest.raises(FileNotFoundError) :
        predict_uvdoc(fake_loaded, tmp_path / "missing.png")


def test_wrong_image_dtype() :
    with pytest.raises(ValueError) :
        preprocess_official_rgb(np.zeros((20, 20, 3), dtype=np.float32), "cpu")


def test_load_config_rejects_mutation(tmp_path) :
    import yaml
    config = load_uvdoc_config()
    config["inference"]["align_corners"] = False
    file = tmp_path / "modified.yaml"
    file.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError) :
        load_uvdoc_config(file)


def test_hash_file(tmp_path) :
    file = tmp_path / "a"
    file.write_bytes(b"stage2")
    assert hash_file(file) == hashlib.sha256(b"stage2").hexdigest()


def test_checkpoint_hash_mismatch_prevents_deserialization(tmp_path, monkeypatch) :
    from models.deep import uvdoc
    monkeypatch.setattr(uvdoc, "verify_source", lambda *_: {"model.py" : "test"})
    checkpoint = tmp_path / "bad_weights.pkl"
    checkpoint.write_bytes(b"not the approved pretrained checkpoint" * 40000)
    with pytest.raises(RuntimeError, match="Checkpoint hash mismatch") :
        uvdoc.load_uvdoc_model(device="cpu", source_dir=tmp_path, checkpoint_path=checkpoint)
