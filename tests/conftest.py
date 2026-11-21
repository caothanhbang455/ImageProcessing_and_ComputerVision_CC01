# tests/conftest.py
"""Small, independent metadata fixture. Tests never need the original datasets."""

import csv
import json
from pathlib import Path

import pytest
import yaml


def _csv(path : Path, rows : list[dict]) :
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle :
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def small_project(tmp_path : Path) -> Path :
    root = tmp_path / "Document Unwarping"
    config_folder = root / "configs"
    metadata = root / "dataset/_metadata"
    config_folder.mkdir(parents=True)
    metadata.mkdir(parents=True)

    source_config = Path(__file__).resolve().parents[1] / "configs/datasets.yaml"
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    config["datasets"]["uvdoc_final"]["expected_records"] = 8
    config["datasets"]["uvdoc_benchmark"]["expected_records"] = 1
    config["datasets"]["docunet"]["expected_records"] = 2
    config["datasets"]["dir300"]["expected_records"] = 1
    (config_folder / "datasets.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    uv_final = []
    split_rows = {"train": [], "val": []}
    for base_id, split in ((0, "train"), (1, "val")) :
        for i, variant in enumerate(("0_0", "0_1", "1_0", "1_1")) :
            sample_id = f"{base_id * 4 + i:05d}"
            geom = f"00_{base_id:05d}_1_{variant}"
            row = {"dataset": "UVDoc Final", "sample_id": sample_id, "geom_name": geom,
                   "image_path": f"dataset/UVDoc/UVDoc_final/img/{sample_id}.png",
                   "grid2d_path": f"dataset/UVDoc/UVDoc_final/grid2d/{geom}.mat",
                   "grid3d_path": f"dataset/UVDoc/UVDoc_final/grid3d/{geom}.mat",
                   "metadata_path": f"dataset/UVDoc/UVDoc_final/metadata_sample/{sample_id}.json",
                   "texture_name": "texture.png", "background_name": "background.png",
                   "annotation_status": "valid"}
            uv_final.append(row)
            split_rows[split].append({**row, "base_geometry": f"00_{base_id:05d}_1", "variant": variant, "split": split})
    _csv(metadata / "uvdoc_final_manifest.csv", uv_final)
    _csv(metadata / "uvdoc_train.csv", split_rows["train"])
    _csv(metadata / "uvdoc_val.csv", split_rows["val"])
    (metadata / "split_spec.json").write_text(json.dumps({"policy": "base_geometry_group_disjoint",
                                                            "training_samples": 4, "validation_samples": 4,
                                                            "training_groups": 1, "validation_groups": 1}), encoding="utf-8")
    benchmark = {**uv_final[0], "dataset": "UVDoc Benchmark", "geom_name": "09_00000_1_0_0"}
    benchmark["image_path"] = "dataset/UVDoc/UVDoc_benchmark/img/00000.png"
    benchmark["grid2d_path"] = "dataset/UVDoc/UVDoc_benchmark/grid2d/09_00000_1_0_0.mat"
    benchmark["grid3d_path"] = "dataset/UVDoc/UVDoc_benchmark/grid3d/09_00000_1_0_0.mat"
    benchmark["metadata_path"] = "dataset/UVDoc/UVDoc_benchmark/metadata_sample/00000.json"
    _csv(metadata / "uvdoc_benchmark_manifest.csv", [benchmark])

    docunet = []
    for view in ("1", "2") :
        docunet.append({"dataset": "DocUNet", "sample_id": f"38_{view}", "document_id": "38",
                        "view_id": view, "original_path": f"dataset/DocUNet/original/38_{view}.JPG",
                        "input_path": f"dataset/DocUNet/crop/38_{view} copy.png",
                        "reference_path": "dataset/DocUNet/scan/38.png", "input_variant": "crop", "status": "valid"})
    _csv(metadata / "docunet_manifest.csv", docunet)
    _csv(metadata / "dir300_manifest.csv", [{"dataset": "DIR300", "sample_id": "8", "document_id": "8",
                                             "view_id": "", "input_path": "dataset/DIR300/_source/DIR300/dist/8.png",
                                             "reference_path": "dataset/DIR300/_source/DIR300/gt/8.png",
                                             "input_variant": "dist", "status": "valid"}])
    return root
