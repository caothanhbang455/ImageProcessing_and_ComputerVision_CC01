# models/deep/uvdoc.py
"""Reference-faithful inference for the pinned, unmodified UVDoc implementation.

The official model.py is loaded from a separately verified Git checkout. No
architecture or pretrained weights are distributed in this project's sources.
This module does not score a benchmark or apply image enhancements.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any
import importlib.util
import os
import subprocess

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from geometry.grids import grid_statistics, sampling_valid_mask, upsample_coarse_grid
from models.interfaces import MethodResult


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/models/uvdoc_official.yaml"
DEFAULT_SOURCE = Path("/content/uvdoc_reference_source")


@dataclass
class LoadedUVDoc :
    model : torch.nn.Module
    device : torch.device
    config : dict[str, Any]
    source_dir : Path
    checkpoint_path : Path
    checkpoint_sha256 : str
    source_commit : str


@dataclass
class UVDocPrediction :
    result : MethodResult
    grid2d : np.ndarray
    grid3d : np.ndarray
    timings_ms : dict[str, float]
    input_size : tuple[int, int]
    output_size : tuple[int, int]


def load_uvdoc_config(path : str | Path = DEFAULT_CONFIG) -> dict[str, Any] :
    file = Path(path)
    config = yaml.safe_load(file.read_text(encoding="utf-8"))
    if (not isinstance(config, dict) or config.get("model_id") != "uvdoc_official") :
        raise ValueError(f"Unsupported UVDoc config: {file}")
    required = {"source", "checkpoint", "architecture", "inference"}
    if (not required.issubset(config)) :
        raise ValueError(f"UVDoc config missing sections: {sorted(required - set(config))}")
    inference = config["inference"]
    expected = {"input_width" : 488, "input_height" : 712, "grid_height" : 45,
                "grid_width" : 31, "dtype" : "float32", "batch_size" : 1,
                "interpolation" : "bilinear", "padding_mode" : "zeros", "align_corners" : True}
    for key, value in expected.items() :
        if (inference.get(key) != value or type(inference.get(key)) is not type(value)) :
            raise ValueError(f"Official UVDoc setting {key!r} must be {value!r}")
    if (config["architecture"] != {"name" : "UVDocnet", "num_filter" : 32,
                                  "kernel_size" : 5, "parameter_count" : 8003463}) :
        raise ValueError("Architecture must match the approved pretrained UVDoc network")
    if (config["source"].get("commit") != "4c9b82b537057aff2526e6dd118a847cdd072e82") :
        raise ValueError("UVDoc source commit does not match the pinned release")
    digest = config["checkpoint"].get("sha256", "")
    if (digest != "7e90861b8a516eb4bc51f84bd889cb77275743d2d1d3ca8091951ec9f2b7da23") :
        raise ValueError("UVDoc checkpoint SHA-256 does not match the approved weights")
    if (config["checkpoint"].get("relative_path") != "model/best_model.pkl") :
        raise ValueError("Unexpected official checkpoint path")
    return config


def hash_file(path : Path) -> str :
    digest = sha256()
    with path.open("rb") as handle :
        for chunk in iter(lambda : handle.read(8 * 1024 * 1024), b"") :
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(source_dir : str | Path, expected_commit : str) -> dict[str, str] :
    """Require exact source commit and unchanged official inference files."""
    source = Path(source_dir).expanduser().resolve()
    for name in ("model.py", "utils.py", "demo.py") :
        if (not (source / name).is_file()) :
            raise FileNotFoundError(f"Official UVDoc source is missing {name}: {source}")
    try :
        head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.PIPE).strip()
        edited = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--",
                                          "model.py", "utils.py", "demo.py"],
                                         text=True, stderr=subprocess.PIPE).strip()
    except (OSError, subprocess.CalledProcessError) as error :
        raise RuntimeError("Official UVDoc source must be an accessible Git checkout") from error
    if (head != expected_commit) :
        raise RuntimeError(f"Official UVDoc commit mismatch: expected {expected_commit}, found {head}")
    if (edited) :
        raise RuntimeError(f"Official UVDoc inference files have local changes: {edited}")
    return {name : hash_file(source / name) for name in ("model.py", "utils.py", "demo.py")}


def resolve_reference_paths(config : dict[str, Any], source_dir : str | Path | None = None,
                            checkpoint_path : str | Path | None = None) -> tuple[Path, Path] :
    selected = source_dir or os.environ.get("DOCUMENT_UNWARPING_UVDOC_SOURCE_DIR") or DEFAULT_SOURCE
    source = Path(selected).expanduser().resolve()
    selected_checkpoint = checkpoint_path or os.environ.get("DOCUMENT_UNWARPING_UVDOC_CHECKPOINT")
    checkpoint = Path(selected_checkpoint).expanduser().resolve() if selected_checkpoint else source / config["checkpoint"]["relative_path"]
    return source, checkpoint


def load_uvdoc_model(config : dict[str, Any] | str | Path = DEFAULT_CONFIG, device : str = "cuda",
                     source_dir : str | Path | None = None,
                     checkpoint_path : str | Path | None = None) -> LoadedUVDoc :
    """Preflight source/weights first, then strictly load trusted model_state weights."""
    cfg = load_uvdoc_config(config) if not isinstance(config, dict) else config
    if (cfg.get("model_id") != "uvdoc_official") :
        raise ValueError("Only the unmodified official UVDoc model is supported")
    # Reapply pin checks even when the caller supplies a dictionary.
    pinned = load_uvdoc_config()
    if (cfg != pinned) :
        raise ValueError("Official UVDoc configuration must match pinned release settings")
    source, checkpoint = resolve_reference_paths(cfg, source_dir, checkpoint_path)
    verify_source(source, cfg["source"]["commit"])
    if (not checkpoint.is_file()) :
        raise FileNotFoundError(f"Official pretrained checkpoint missing: {checkpoint}")
    if (checkpoint.stat().st_size < 1024 * 1024) :
        raise ValueError(f"Checkpoint is too small (possibly a Git LFS pointer): {checkpoint}")
    actual_digest = hash_file(checkpoint)
    if (actual_digest != cfg["checkpoint"]["sha256"]) :
        raise RuntimeError(f"Checkpoint hash mismatch: expected {cfg['checkpoint']['sha256']}, got {actual_digest}")
    device_obj = torch.device(device)
    if (device_obj.type == "cuda" and not torch.cuda.is_available()) :
        raise RuntimeError("CUDA is unavailable; choose a GPU runtime or explicitly use device='cpu'")

    module_name = "_verified_uvdoc_official_model"
    spec = importlib.util.spec_from_file_location(module_name, source / "model.py")
    if (spec is None or spec.loader is None) :
        raise ImportError(f"Cannot load the pinned UVDoc architecture from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.UVDocnet(num_filter=32, kernel_size=5)
    count = sum(parameter.numel() for parameter in net.parameters())
    if (count != cfg["architecture"]["parameter_count"]) :
        raise ValueError(f"Official UVDoc architecture has {count} parameters, expected 8003463")
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint_data, dict) or "model_state" not in checkpoint_data) :
        raise ValueError("Checkpoint must be a dictionary containing model_state")
    net.load_state_dict(checkpoint_data["model_state"], strict=True)
    net = net.to(device_obj).float().eval()
    return LoadedUVDoc(net, device_obj, cfg, source, checkpoint, actual_digest, cfg["source"]["commit"])


def decode_official_rgb(image_path : str | Path) -> np.ndarray :
    """Match cv2.imread default orientation/color behavior used in demo.py."""
    path = Path(image_path)
    if (not path.is_file()) :
        raise FileNotFoundError(f"Input image unavailable: {path}")
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if (bgr is None) :
        raise ValueError(f"OpenCV could not decode input image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess_official_rgb(image_rgb : np.ndarray, device : torch.device | str) -> tuple[torch.Tensor, np.ndarray] :
    """Source float image and 488x712 model tensor exactly follow demo.py."""
    rgb = np.asarray(image_rgb)
    if (rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 2) :
        raise ValueError("Expected an H x W x 3 uint8 RGB image")
    source_float = rgb.astype(np.float32) / 255.0
    resized = cv2.resize(source_float, (488, 712), interpolation=cv2.INTER_LINEAR)
    inp = torch.from_numpy(resized.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    return inp, source_float


def _sync(device : torch.device) -> None :
    if (device.type == "cuda") :
        torch.cuda.synchronize(device)


def _timestamp(device : torch.device) -> float :
    _sync(device)
    return perf_counter()


def predict_uvdoc_from_rgb(loaded : LoadedUVDoc, image_rgb : np.ndarray,
                            return_dense_map : bool = False) -> UVDocPrediction :
    """Run unchanged float32 official sampling at original image dimensions.

    The coarse grid is already normalized; do NOT renormalize it. The final
    byte cast intentionally truncates, like the official demo.
    """
    device = loaded.device
    start = _timestamp(device)
    model_tensor, source_float = preprocess_official_rgb(image_rgb, device)
    preprocess_ms = (_timestamp(device) - start) * 1000
    h, w = source_float.shape[:2]
    t0 = _timestamp(device)
    with torch.inference_mode() :
        grid2, grid3 = loaded.model(model_tensor)
    forward_ms = (_timestamp(device) - t0) * 1000
    if (tuple(grid2.shape) != (1, 2, 45, 31) or tuple(grid3.shape) != (1, 3, 45, 31)) :
        raise ValueError(f"Official UVDoc unexpected grid shapes: {tuple(grid2.shape)}, {tuple(grid3.shape)}")
    if (not bool(torch.isfinite(grid2).all()) or not bool(torch.isfinite(grid3).all())) :
        raise ValueError("UVDoc model produced a non-finite grid")
    if (grid2.dtype != torch.float32 or grid3.dtype != torch.float32) :
        raise ValueError("Official float32 inference required; autocast or incompatible weights detected")

    t1 = _timestamp(device)
    with torch.inference_mode() :
        full_grid = upsample_coarse_grid(grid2, h, w, align_corners=True)
        source_tensor = torch.from_numpy(source_float.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
        warped = F.grid_sample(source_tensor, full_grid.permute(0, 2, 3, 1),
                               mode="bilinear", padding_mode="zeros", align_corners=True)
    warp_ms = (_timestamp(device) - t1) * 1000

    t2 = _timestamp(device)
    output = (warped[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    coarse_2d = grid2[0].detach().cpu().numpy().copy()
    coarse_3d = grid3[0].detach().cpu().numpy().copy()
    dense = full_grid[0].permute(1, 2, 0).detach().cpu().numpy().copy() if return_dense_map else None
    mask = sampling_valid_mask(dense, "normalized_minus1_plus1", w, h) if return_dense_map else None
    conversion_ms = (_timestamp(device) - t2) * 1000
    diagnostics = grid_statistics(coarse_2d)
    timing = {"preprocess_ms" : preprocess_ms, "forward_ms" : forward_ms,
              "warp_ms" : warp_ms, "conversion_ms" : conversion_ms}
    result = MethodResult(status="OK", image=output, backward_map=dense,
                          map_space="normalized_minus1_plus1" if dense is not None else None,
                          valid_mask=mask,
                          metadata={"method_id" : "uvdoc_official", "source_size" : [w, h],
                                    "coordinate_space" : "normalized_minus1_plus1",
                                    "grid_diagnostics" : diagnostics, "timings_ms" : timing})
    return UVDocPrediction(result, coarse_2d, coarse_3d, timing, (w, h), (w, h))


def predict_uvdoc(loaded : LoadedUVDoc, image_path : str | Path,
                  return_dense_map : bool = False) -> UVDocPrediction :
    start = perf_counter()
    rgb = decode_official_rgb(image_path)
    decode_ms = (perf_counter() - start) * 1000
    prediction = predict_uvdoc_from_rgb(loaded, rgb, return_dense_map=return_dense_map)
    prediction.timings_ms["decode_ms"] = decode_ms
    prediction.result.metadata["timings_ms"]["decode_ms"] = decode_ms
    return prediction
