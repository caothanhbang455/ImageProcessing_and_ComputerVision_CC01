# tests/test_uvdoc_reference.py
import subprocess

import pytest

from models.deep.uvdoc import verify_source


def test_verified_source_requires_real_checkout(tmp_path) :
    for name in ("model.py", "utils.py", "demo.py") :
        (tmp_path / name).write_text("# fixture", encoding="utf-8")
    with pytest.raises(RuntimeError) :
        verify_source(tmp_path, "incorrect")


def test_verified_source_rejects_commit_mismatch(tmp_path) :
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Tester"], check=True)
    for name in ("model.py", "utils.py", "demo.py") :
        (tmp_path / name).write_text("# fixture", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "model.py", "utils.py", "demo.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "fixture"], check=True)
    with pytest.raises(RuntimeError, match="commit mismatch") :
        verify_source(tmp_path, "1234567890")


def test_independent_reference_harness_with_synthetic_upstream(tmp_path) :
    """Exercise separate-process comparison without representing it as real UVDoc parity."""
    import cv2
    import numpy as np
    import torch
    from models.deep.uvdoc import LoadedUVDoc, load_uvdoc_config
    from scripts.verify_uvdoc_reference import compare_with_official_demo

    source = tmp_path / "fake_upstream"
    source.mkdir()
    model_text = '''import torch
class UVDocnet(torch.nn.Module):
    def __init__(self, num_filter=32, kernel_size=5):
        super().__init__()
    def forward(self, x):
        yy, xx = torch.meshgrid(torch.linspace(-1,1,45,device=x.device), torch.linspace(-1,1,31,device=x.device), indexing="ij")
        return torch.stack([xx, yy])[None], torch.zeros((1,3,45,31),device=x.device)
'''
    utils_text = '''import torch
import torch.nn.functional as F
from model import UVDocnet
IMG_SIZE = [488, 712]
def load_model(ckpt_path):
    model = UVDocnet(num_filter=32, kernel_size=5)
    ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model
def bilinear_unwarping(warped_img, point_positions, img_size):
    full = F.interpolate(point_positions, size=(img_size[1], img_size[0]), mode="bilinear", align_corners=True)
    return F.grid_sample(warped_img, full.permute(0,2,3,1), align_corners=True)
'''
    demo_text = '''import argparse
import cv2
import numpy as np
import torch
from utils import IMG_SIZE, load_model, bilinear_unwarping
def unwarp_img(ckpt_path, img_path, img_size):
    model = load_model(ckpt_path).eval()
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    inp = torch.from_numpy(cv2.resize(image, img_size).transpose(2,0,1)).unsqueeze(0)
    with torch.inference_mode():
        grid, _ = model(inp)
        out = bilinear_unwarping(torch.from_numpy(image.transpose(2,0,1)).unsqueeze(0), grid, (image.shape[1], image.shape[0]))
    png = (out[0].detach().numpy().transpose(1,2,0) * 255).astype(np.uint8)
    cv2.imwrite(img_path.rsplit(".", 1)[0] + "_unwarp.png", cv2.cvtColor(png, cv2.COLOR_RGB2BGR))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path")
    parser.add_argument("--img-path")
    a = parser.parse_args()
    unwarp_img(a.ckpt_path, a.img_path, IMG_SIZE)
'''
    (source / "model.py").write_text(model_text)
    (source / "utils.py").write_text(utils_text)
    (source / "demo.py").write_text(demo_text)
    # Import the synthetic architecture separately from the team's model adapter.
    import importlib.util
    spec = importlib.util.spec_from_file_location("fake_model", source / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.UVDocnet()
    checkpoint = source / "fake.pkl"
    torch.save({"model_state" : model.state_dict()}, checkpoint)
    loaded = LoadedUVDoc(model, torch.device("cpu"), load_uvdoc_config(), source, checkpoint,
                         "synthetic_only", "synthetic_only")
    image = np.random.default_rng(12).integers(0, 256, (25, 39, 3), dtype=np.uint8)
    image_file = tmp_path / "input.png"
    cv2.imwrite(str(image_file), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    audit = compare_with_official_demo(loaded, image_file, tmp_path / "comparison")
    assert audit["status"] == "PASS", audit["reason"]
    assert audit["output_max_abs_uint8"] == 0
    assert audit["grid2d_max_abs"] == 0
    assert audit["preprocess_max_abs"] == 0
