# tests/test_evaluation_uvdoc.py
import numpy as np
import cv2
import pytest

from evaluation.protocols import uvdoc


def identity_coarse() :
    ys, xs = np.meshgrid(np.linspace(-1, 1, 45), np.linspace(-1, 1, 31), indexing="ij")
    return np.stack((xs, ys), axis=-1).astype(np.float32)


def test_stage2_npz_contract(tmp_path) :
    target = tmp_path / "grid.npz"
    coarse = identity_coarse()
    np.savez_compressed(target, grid2d=coarse.transpose(2, 0, 1),
                        map_space=np.array("normalized_minus1_plus1"), sample_key=np.array("id"))
    assert np.array_equal(uvdoc.load_predicted_grid(target, "id"), coarse)
    with pytest.raises(ValueError, match="metadata mismatch") :
        uvdoc.load_predicted_grid(target, "wrong")


def test_uvdoc_texture_warp_has_image_contract(tmp_path) :
    test_img = np.arange(24 * 18 * 3, dtype=np.uint8).reshape(24, 18, 3)
    path = tmp_path / "texture.png"
    assert cv2.imwrite(str(path), cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
    unwarped = uvdoc.unwarp_texture_from_grid(path, identity_coarse())
    assert unwarped.shape == test_img.shape and unwarped.dtype == np.uint8
    assert float(np.abs(unwarped.astype(int) - test_img.astype(int)).mean()) <= 1.0


def test_line_metric_returns_valid_or_explicit_missing() :
    # A synthetic 1000-pixel identity UV map keeps a 1-pixel stripe measurable.
    ys, xs = np.meshgrid(np.linspace(0, 1, uvdoc.CANVAS),
                         np.linspace(0, 1, uvdoc.CANVAS), indexing="ij")
    uv = np.stack((xs, ys), axis=-1).astype(np.float32)
    bm = identity_coarse()
    result = uvdoc.single_line_deviation(uv, bm, 500, "horizontal")
    assert result is None or (np.isfinite(result) and result >= 0.0)
    with pytest.raises(ValueError) :
        uvdoc.single_line_deviation(uv, bm, 500, "sideways")


def test_uvdoc_uvmap_allows_nan_outside_document(monkeypatch, tmp_path) :
    import sys
    from types import SimpleNamespace

    uv = np.zeros((12, 10, 2), dtype=np.float32)
    uv[:, :, 0] = np.linspace(0, 1, 10)[None, :]
    uv[:, :, 1] = np.linspace(0, 1, 12)[:, None]
    uv[0, :, :] = np.nan
    uv[-1, :, :] = np.nan
    monkeypatch.setitem(sys.modules, "hdf5storage", SimpleNamespace(loadmat=lambda _: {"uv" : uv}))
    record = SimpleNamespace(metadata={"geom_name" : "example"})
    resolver = SimpleNamespace(resolve=lambda _: tmp_path / "example.mat")
    loaded = uvdoc.load_uvmap(record, resolver)
    assert loaded.shape == uv.shape and np.isnan(loaded[0]).all()
    result = uvdoc.warp_synthetic_line(loaded, 500, "horizontal")
    assert result.shape == (12, 10, 3) and result.dtype == np.uint8


def test_uvdoc_uvmap_rejects_infinity_and_fully_invalid_map() :
    uv = np.full((12, 10, 2), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="usable coordinates") :
        uvdoc.warp_synthetic_line(uv, 500, "horizontal")
    uv[5, 5] = [0.5, np.inf]
    with pytest.raises(ValueError, match="usable coordinates") :
        uvdoc.warp_synthetic_line(uv, 500, "horizontal")
