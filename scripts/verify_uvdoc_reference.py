# scripts/verify_uvdoc_reference.py
"""Verify Stage 2 against the independently executed pinned official UVDoc demo.

The demo is executed in a separate Python process on a temporary copy of the
original source image, so it cannot overwrite or modify the dataset file.
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.deep.uvdoc import (decode_official_rgb, hash_file, load_uvdoc_model,
                               predict_uvdoc, preprocess_official_rgb)


# Independent forward path imports the official model.py inside the verified
# reference checkout. It does not import the team's wrapper or geometry code.
OFFICIAL_FORWARD = r'''
import cv2
import numpy as np
import torch
from model import UVDocnet
from utils import IMG_SIZE
import sys

source, checkpoint, destination, device = sys.argv[1:]
image = cv2.imread(source)
if image is None:
    raise ValueError("OpenCV cannot read reference input")
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
inp = torch.from_numpy(cv2.resize(img, IMG_SIZE).transpose(2, 0, 1)).unsqueeze(0)
net = UVDocnet(num_filter=32, kernel_size=5)
ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
net.load_state_dict(ckpt["model_state"], strict=True)
net = net.to(device).float().eval()
with torch.inference_mode():
    grid2, grid3 = net(inp.to(device))
np.savez_compressed(destination, input_tensor=inp.cpu().numpy(),
                    grid2d=grid2.detach().cpu().numpy(),
                    grid3d=grid3.detach().cpu().numpy())
'''


def compare_with_official_demo(loaded, input_path : Path, output_dir : Path,
                               tolerance_grid : float = 1e-5) -> dict :
    """Return a numerical audit. Neither parity nor checkpoint success is assumed."""
    if (output_dir.exists()) :
        raise FileExistsError(f"Refusing to overwrite parity directory: {output_dir}")
    if (not input_path.is_file()) :
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    image_path = Path(input_path).resolve()
    result = {"source_image" : str(image_path), "source_image_sha256" : hash_file(image_path),
              "source_commit" : loaded.source_commit, "checkpoint_sha256" : loaded.checkpoint_sha256,
              "device" : str(loaded.device), "preprocess_max_abs" : None,
              "grid2d_max_abs" : None, "grid3d_max_abs" : None,
              "output_max_abs_uint8" : None, "output_mean_abs_uint8" : None,
              "status" : "FAIL", "reason" : "Not executed"}
    try :
        with TemporaryDirectory(prefix="stage2_uvdoc_official_") as folder :
            temp = Path(folder)
            source_copy = temp / ("reference_input" + image_path.suffix.lower())
            shutil.copyfile(image_path, source_copy)
            if (hash_file(source_copy) != result["source_image_sha256"]) :
                raise ValueError("Temporary source image bytes differ from original")
            prediction = predict_uvdoc(loaded, source_copy)
            reference_forward = temp / "reference_forward.npz"
            forward_command = [sys.executable, "-c", OFFICIAL_FORWARD, str(source_copy),
                               str(loaded.checkpoint_path), str(reference_forward), str(loaded.device)]
            subprocess.run(forward_command, cwd=loaded.source_dir, check=True,
                           capture_output=True, text=True)
            with np.load(reference_forward, allow_pickle=False) as arrays :
                our_tensor, _ = preprocess_official_rgb(decode_official_rgb(source_copy), loaded.device)
                our_tensor_np = our_tensor.detach().cpu().numpy()
                result["preprocess_max_abs"] = float(np.max(np.abs(our_tensor_np - arrays["input_tensor"])))
                result["grid2d_max_abs"] = float(np.max(np.abs(prediction.grid2d[None] - arrays["grid2d"])))
                result["grid3d_max_abs"] = float(np.max(np.abs(prediction.grid3d[None] - arrays["grid3d"])))

            # The actual authors' demo is the independent final-output reference.
            demo_command = [sys.executable, str(loaded.source_dir / "demo.py"),
                            "--ckpt-path", str(loaded.checkpoint_path), "--img-path", str(source_copy)]
            subprocess.run(demo_command, cwd=loaded.source_dir, check=True,
                           capture_output=True, text=True)
            reference_png = temp / "reference_input_unwarp.png"
            if (not reference_png.is_file()) :
                raise FileNotFoundError(f"Official demo did not write {reference_png}")
            reference_image = decode_official_rgb(reference_png)
            our_image = prediction.result.image
            if (our_image.shape != reference_image.shape) :
                raise ValueError(f"Output shape differs: {our_image.shape}, {reference_image.shape}")
            diff = np.abs(our_image.astype(np.int16) - reference_image.astype(np.int16))
            result["output_max_abs_uint8"] = int(diff.max())
            result["output_mean_abs_uint8"] = float(diff.mean())
            passed = (result["preprocess_max_abs"] == 0 and
                      result["grid2d_max_abs"] <= tolerance_grid and
                      result["grid3d_max_abs"] <= tolerance_grid and
                      result["output_max_abs_uint8"] == 0)
            result["status"] = "PASS" if passed else "FAIL"
            result["reason"] = "All reference equality gates passed" if passed else "Reference tensors or PNG differ"
            # Preserve the final comparison, but not an unnecessary duplicate of
            # the entire source image and reference architecture.
            if (passed) :
                shutil.copy2(reference_png, output_dir / "official_demo_output.png")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error :
        result["status"] = "FAIL"
        result["reason"] = f"{type(error).__name__}: {error}"
        if (isinstance(error, subprocess.CalledProcessError)) :
            result["reference_stderr_tail"] = (error.stderr or "")[-2000 :]

    (output_dir / "reference_equivalence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output_dir / "reference_equivalence.csv").open("w", newline="", encoding="utf-8") as handle :
        fields = ["source_image", "source_image_sha256", "source_commit", "checkpoint_sha256", "device",
                  "preprocess_max_abs", "grid2d_max_abs", "grid3d_max_abs",
                  "output_max_abs_uint8", "output_mean_abs_uint8", "status", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(result)
    return result


def main() -> int :
    parser = argparse.ArgumentParser(description="Run pinned original UVDoc demo against the team's adapter")
    parser.add_argument("--image", required=True, help="Resolved real image path, never overwritten")
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True, help="New dedicated comparison directory")
    args = parser.parse_args()
    loaded = load_uvdoc_model(device=args.device, source_dir=args.source_dir, checkpoint_path=args.checkpoint)
    row = compare_with_official_demo(loaded, Path(args.image), Path(args.output))
    print(json.dumps(row, indent=2))
    return 0 if row["status"] == "PASS" else 2


if (__name__ == "__main__") :
    raise SystemExit(main())
