# evaluation/protocols/uvdoc.py
"""UVDoc line-deviation evaluation following the pinned authors' algorithm.

The ground-truth UV deformation uses align_corners=False. The predicted
backward-grid operation uses align_corners=True. These distinct settings are
intentional and match uvdocBenchmark_metric.py and utils.py respectively.

Numerical equivalence must be checked independently against the official
pinned checkout before calling real-data scores reference verified.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import binary_erosion

from data.paths import PathResolver


CANVAS = 1000
LINE_POSITIONS = np.linspace(50, 950, 25, dtype=int)
OFFICIAL_COMMIT = "4c9b82b537057aff2526e6dd118a847cdd072e82"


def load_uvmap(record, resolver : PathResolver) -> np.ndarray :
    """Read the UV ground truth named by the sample's physical geometry."""
    try :
        import hdf5storage
    except ImportError as error :
        raise RuntimeError("UVDoc line scoring requires hdf5storage>=0.2,<0.3") from error
    path = resolver.resolve(f"dataset/UVDoc/UVDoc_benchmark/uvmap/{record.metadata['geom_name']}.mat")
    data = hdf5storage.loadmat(str(path))
    if ("uv" not in data) :
        raise ValueError(f"UVDoc UV-map MATLAB file has no uv array: {path}")
    uv = np.asarray(data["uv"])
    # UVDoc uses NaN coordinates outside the segmented document region.
    # The official metric passes these values to grid_sample and handles the
    # resulting invalid pixels during stripe masking. Reject only infinities
    # and maps with no usable coordinate pairs.
    if (uv.ndim != 3 or uv.shape[2] != 2 or min(uv.shape[:2]) < 2 or
        np.isinf(uv).any() or not np.any(np.isfinite(uv).all(axis=-1))) :
        raise ValueError(f"Invalid UVDoc UV-map shape or values in {path}: {uv.shape}")
    return uv


def load_predicted_grid(npz_path : Path, sample_key : str) -> np.ndarray :
    """Read the normalized coarse Stage 2 backward map without renormalizing it."""
    with np.load(npz_path, allow_pickle=False) as data :
        if (not {"grid2d", "map_space", "sample_key"}.issubset(data.files)) :
            raise ValueError(f"Stage 2 grid artifact lacks required fields: {npz_path}")
        if (str(data["map_space"].item()) != "normalized_minus1_plus1" or
            str(data["sample_key"].item()) != sample_key) :
            raise ValueError(f"Stage 2 grid metadata mismatch: {npz_path}")
        grid = np.asarray(data["grid2d"], dtype=np.float32)
    if (grid.shape != (2, 45, 31) or not np.isfinite(grid).all()) :
        raise ValueError(f"UVDoc expected finite coarse 2D grid (2,45,31); got {grid.shape}")
    return grid.transpose(1, 2, 0).copy()


def load_predicted_mat(mat_file : Path) -> np.ndarray :
    import hdf5storage
    data = hdf5storage.loadmat(str(mat_file))
    if ("bm" not in data) :
        raise ValueError("UVDoc MAT prediction has no bm field")
    grid = np.asarray(data["bm"], dtype=np.float32)
    if (grid.shape != (45, 31, 2) or not np.isfinite(grid).all()) :
        raise ValueError(f"Unexpected UVDoc MAT bm shape {grid.shape}")
    return grid


def compare_npz_mat(npz_path : Path, mat_path : Path, sample_key : str) -> float :
    return float(np.max(np.abs(load_predicted_grid(npz_path, sample_key) - load_predicted_mat(mat_path))))


def warp_synthetic_line(uvmap : np.ndarray, position : int, direction : str,
                        device : str | torch.device = "cpu") -> np.ndarray :
    """Create and deform one synthetic stripe as the official line metric does."""
    if (direction not in ("horizontal", "vertical") or not 0 <= position < CANVAS) :
        raise ValueError("Invalid UVDoc line direction or position")
    if (uvmap.ndim != 3 or uvmap.shape[2] != 2 or np.isinf(uvmap).any() or
        not np.any(np.isfinite(uvmap).all(axis=-1))) :
        raise ValueError("UV map must be H x W x 2 with usable coordinates; NaN masks are permitted")
    stripe = np.full((CANVAS, CANVAS, 3), 255, dtype=np.uint8)
    if (direction == "horizontal") :
        stripe[position, :] = 0
    else :
        stripe[:, position] = 0
    source = torch.from_numpy(stripe.transpose(2, 0, 1).copy()).float()[None].to(device)
    uv_tensor = torch.from_numpy(np.ascontiguousarray(uvmap * 2 - 1)).float()[None].to(device)
    with torch.inference_mode() :
        warped = F.grid_sample(source, uv_tensor, align_corners=False)
    result = np.clip(warped[0].cpu().numpy().transpose(1, 2, 0), 0, 255) / 255.0
    grey = np.all(result == 0.5, axis=-1)
    result[grey] = np.nan
    valid = 1 - np.all(np.isnan(result), axis=-1).astype(int)
    eroded = binary_erosion(valid).astype(int)
    result[np.repeat(~eroded.astype(bool)[:, :, None], 3, axis=-1)] = 1.0
    return (result * 255).astype(np.uint8)


def unwarp_synthetic_line(warped : np.ndarray, predicted_bm : np.ndarray,
                          device : str | torch.device = "cpu") -> np.ndarray :
    """Apply official bilinear unwarping at the metric's fixed 1000-pixel canvas."""
    if (warped.ndim != 3 or warped.shape[2] != 3 or
        predicted_bm.shape != (45, 31, 2) or not np.isfinite(predicted_bm).all()) :
        raise ValueError("Invalid UVDoc warped stripe or predicted backward map")
    # Match official numpy-to-float32 conversion and bilinear_unwarping.
    img_float = warped.astype(np.float64) / 255.0
    img = torch.from_numpy(np.ascontiguousarray(img_float.transpose(2, 0, 1))).float()[None].to(device)
    coarse = torch.from_numpy(np.ascontiguousarray(predicted_bm.transpose(2, 0, 1))).float()[None].to(device)
    with torch.inference_mode() :
        grid = F.interpolate(coarse, size=(CANVAS, CANVAS), mode="bilinear", align_corners=True)
        output = F.grid_sample(img, grid.permute(0, 2, 3, 1), align_corners=True)
    return output[0].cpu().numpy().transpose(1, 2, 0)


def single_line_deviation(uvmap : np.ndarray, predicted_bm : np.ndarray, position : int,
                          direction : str, device : str | torch.device = "cpu") -> float | None :
    warped = warp_synthetic_line(uvmap, position, direction, device)
    corrected = unwarp_synthetic_line(warped, predicted_bm, device)[:, :, 0]
    if (direction == "horizontal") :
        coords = np.where(corrected < 0.5)[0]
    elif (direction == "vertical") :
        coords = np.where(corrected < 0.5)[1]
    else :
        raise ValueError("Invalid line direction")
    return float(np.std(coords)) if len(coords) else None


def compute_sample_lines(uvmap : np.ndarray, predicted_bm : np.ndarray,
                         device : str | torch.device = "cpu", positions : np.ndarray | None = None) -> tuple[list[dict], dict] :
    """Return each line, sample means, and explicit missing-line counts."""
    positions = LINE_POSITIONS if positions is None else positions
    results = []
    for direction in ("horizontal", "vertical") :
        for position in positions :
            deviation = single_line_deviation(uvmap, predicted_bm, int(position), direction, device)
            results.append({"direction" : direction, "position" : int(position),
                            "value" : deviation, "status" : "OK" if deviation is not None else "METRIC_FAILED",
                            "reason" : "" if deviation is not None else "No black pixels after rectification"})
    summary = {}
    for direction in ("horizontal", "vertical") :
        subset = [line["value"] for line in results if line["direction"] == direction]
        valid = [x for x in subset if x is not None]
        summary[direction] = float(np.mean(valid)) if valid else None
        summary[f"{direction}_valid_lines"] = len(valid)
        summary[f"{direction}_missing_lines"] = len(subset) - len(valid)
    return results, summary


def unwarp_texture_from_grid(texture_path : Path, bm : np.ndarray,
                              device : str | torch.device = "cpu") -> np.ndarray :
    """Reproduce uvdocBenchmark_pred.py texture warping without model inference."""
    bgr = cv2.imread(str(texture_path), cv2.IMREAD_COLOR)
    if (bgr is None) :
        raise ValueError(f"Cannot decode benchmark warped texture: {texture_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    source = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1) / 255.0)).float()[None].to(device)
    coarse = torch.from_numpy(np.ascontiguousarray(bm.transpose(2, 0, 1))).float()[None].to(device)
    with torch.inference_mode() :
        grid = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=True)
        out = F.grid_sample(source, grid.permute(0, 2, 3, 1), align_corners=True)
    return (out[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def inspect_uvdoc_assets(records : list, resolver : PathResolver) -> list[dict] :
    """Check geometry, metadata, warped textures, and reference textures per sample."""
    report = []
    for record in records :
        geom = record.metadata["geom_name"]
        required = {
            "metadata" : record.annotation_paths["metadata"],
            "uvmap" : f"dataset/UVDoc/UVDoc_benchmark/uvmap/{geom}.mat",
            "warped_texture" : f"dataset/UVDoc/UVDoc_benchmark/warped_textures/{record.sample_id}.png",
            "reference_texture" : f"dataset/UVDoc/UVDoc_benchmark/texture_sample/{record.sample_id}.png",
        }
        report.append({"sample_key" : record.sample_key, "sample_id" : record.sample_id,
                       **{f"{kind}_available" : resolver.find(path) is not None for kind, path in required.items()},
                       **{f"{kind}_logical_path" : path for kind, path in required.items()}})
    return report


def verify_official_line_parity(source_dir : Path, benchmark_root : Path, sample_id : str,
                                bm : np.ndarray, local_lines : list[dict],
                                atol : float = 1e-5) -> dict :
    """Compute per-line values in an independent process loading pinned official code."""
    try :
        import hdf5storage
    except ImportError as error :
        raise RuntimeError("Reference parity requires hdf5storage") from error
    source = Path(source_dir).resolve()
    if (not (source / "uvdocBenchmark_metric.py").is_file()) :
        raise FileNotFoundError("Pinned official UVDoc line evaluator is missing")
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--",
                                     "uvdocBenchmark_metric.py", "utils.py"], text=True).strip()
    if (head != OFFICIAL_COMMIT or dirty) :
        raise RuntimeError("Official UVDoc evaluator must be at the pinned clean commit")
    program = '''import json, sys, numpy as np, hdf5storage as h5
from pathlib import Path
from uvdocBenchmark_metric import compute_metric_single_line
root, sample, path = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
metadata = json.loads((root / 'metadata_sample' / (sample + '.json')).read_text())
uv = h5.loadmat(str(root / 'uvmap' / (metadata['geom_name'] + '.mat')))['uv']
bm = h5.loadmat(str(path))['bm']
rows = []
for direction in ('horizontal', 'vertical'):
 for position in np.linspace(50, 950, 25, dtype=int):
  v = float(compute_metric_single_line(uv, bm, int(position), direction))
  rows.append(dict(direction=direction, position=int(position), value=v if np.isfinite(v) else None))
print('REFERENCE_LINES_JSON=' + json.dumps(rows))
'''
    with tempfile.TemporaryDirectory(prefix="uvdoc_stage3_reference_") as temp :
        mat_file = Path(temp) / "bm.mat"
        hdf5storage.savemat(str(mat_file), {"bm" : bm})
        completed = subprocess.run([sys.executable, "-c", program, str(benchmark_root), sample_id,
                                    str(mat_file)], cwd=source, capture_output=True, text=True)
        if (completed.returncode != 0) :
            raise RuntimeError(f"Official line evaluator failed: {completed.stderr[-3000:]}")
        marker = "REFERENCE_LINES_JSON="
        line = next((x[len(marker):] for x in completed.stdout.splitlines() if x.startswith(marker)), None)
        if (line is None) :
            raise RuntimeError("Official line evaluator did not emit structured results")
        official = json.loads(line)
    if (len(official) != len(local_lines)) :
        raise ValueError("Reference and Stage 3 line counts differ")
    max_abs = 0.0
    for a, b in zip(official, local_lines) :
        if ((a["direction"], a["position"]) != (b["direction"], b["position"])) :
            raise ValueError("Reference line identities differ")
        if (a["value"] is None or b["value"] is None) :
            if (a["value"] is not None or b["value"] is not None) :
                return {"status" : "FAIL", "reason" : "Invalid-line placement differs", "max_abs" : None}
        else :
            max_abs = max(max_abs, abs(a["value"] - b["value"]))
    return {"status" : "PASS" if max_abs <= atol else "FAIL",
            "reason" : "" if max_abs <= atol else "Per-line deviations exceed tolerance",
            "max_abs" : max_abs, "atol" : atol, "lines" : len(official),
            "reference_commit" : head}


def verify_official_texture_parity(source_dir : Path, texture_path : Path, bm : np.ndarray,
                                   output_rgb : np.ndarray) -> dict :
    """Compare a prepared texture with independently executed official utils.py."""
    try :
        import hdf5storage
    except ImportError as error :
        raise RuntimeError("Texture parity requires hdf5storage") from error
    source = Path(source_dir).resolve()
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--",
                                     "utils.py", "model.py"], text=True).strip()
    if (head != OFFICIAL_COMMIT or dirty or not (source / "utils.py").is_file()) :
        raise RuntimeError("Texture reference source is missing or differs from pinned checkout")
    program = '''import cv2, numpy as np, hdf5storage as h5, sys
from utils import bilinear_unwarping_from_numpy
texture, bm_path, output = sys.argv[1:4]
rgb = cv2.cvtColor(cv2.imread(texture), cv2.COLOR_BGR2RGB)
if rgb is None: raise ValueError('Official texture missing')
bm = h5.loadmat(bm_path)['bm']
result = bilinear_unwarping_from_numpy(rgb.astype(np.float64)/255.0, bm, (rgb.shape[1],rgb.shape[0]))
result = (result * 255).astype(np.uint8)
if not cv2.imwrite(output, cv2.cvtColor(result, cv2.COLOR_RGB2BGR)):
 raise RuntimeError('Official texture output was not saved')
'''
    with tempfile.TemporaryDirectory(prefix="uvdoc_stage3_texture_") as temp :
        mat_file = Path(temp) / "bm.mat"
        png_file = Path(temp) / "official.png"
        hdf5storage.savemat(str(mat_file), {"bm" : bm})
        run = subprocess.run([sys.executable, "-c", program, str(texture_path), str(mat_file),
                              str(png_file)], cwd=source, capture_output=True, text=True)
        if (run.returncode != 0) :
            raise RuntimeError(f"Official texture parity failed: {run.stderr[-3000:]}")
        official = cv2.imread(str(png_file), cv2.IMREAD_COLOR)
        if (official is None) :
            raise RuntimeError("Official texture reference PNG missing")
        official = cv2.cvtColor(official, cv2.COLOR_BGR2RGB)
    if (official.shape != output_rgb.shape) :
        return {"status" : "FAIL", "reason" : "Output size differs", "max_abs" : None,
                "atol" : 0, "lines" : 0, "reference_commit" : head}
    difference = float(np.max(np.abs(official.astype(np.int16) - output_rgb.astype(np.int16))))
    return {"status" : "PASS" if difference == 0 else "FAIL",
            "reason" : "" if difference == 0 else "Official texture pixels differ",
            "max_abs" : difference, "atol" : 0, "lines" : 0, "reference_commit" : head}
