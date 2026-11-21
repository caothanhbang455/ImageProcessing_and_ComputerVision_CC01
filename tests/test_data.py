# tests/test_data.py
"""Manifest, path, split, image, and geometry contracts."""

import csv
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image

from data.manifests import load_existing_split, load_records
from data.paths import PathResolver, relative_manifest_path
from data.validation import validate_existing_split, validate_inventory
from data import uvdoc, docunet, dir300


def test_all_dataset_manifests_and_existing_split(small_project : Path) :
    result = validate_inventory(small_project)
    assert all(result[key]["status"] == "PASS" for key in ("uvdoc_final", "uvdoc_benchmark", "docunet", "dir300"))
    assert result["split"]["status"] == "PASS"
    assert result["split"]["overlapping_groups"] == 0
    assert load_existing_split(small_project)[0]["00000"] == "train"
    assert load_existing_split(small_project)[0]["00004"] == "val"
    assert load_records("uvdoc_final", small_project)[0].sample_key == "UVDoc_Final_rendered_00000"
    assert load_records("uvdoc_benchmark", small_project)[0].sample_key == "UVDoc_Benchmark_rendered_00000"
    assert load_records("dir300", small_project)[0].sample_key == "DIR300_dist_8"


def test_docunet_crop_and_original_variants(small_project : Path) :
    crop = load_records("docunet", small_project)[0]
    original = load_records("docunet", small_project, variant="original")[0]
    assert crop.sample_key == "DocUNet_crop_38_1"
    assert original.sample_key == "DocUNet_original_38_1"
    assert crop.group_id == "38"
    assert crop.input_path.endswith("38_1 copy.png")
    assert original.input_path.endswith("38_1.JPG")
    assert original.reference_path == crop.reference_path
    with pytest.raises(ValueError) :
        load_records("dir300", small_project, variant="crop")


def test_path_resolution_and_staging(small_project : Path) :
    resolver = PathResolver(small_project)
    record = load_records("uvdoc_final", small_project)[0]
    expected = small_project / record.input_path
    assert resolver.find(record.input_path) is None
    assert resolver.resolve(record.input_path, require_exists=False) == expected
    with pytest.raises(FileNotFoundError) :
        resolver.resolve(record.input_path)

    staged = small_project / "dataset/UVDoc/_staging_UVDoc_final/UVDoc_final/img/00000.png"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"fixture")
    assert resolver.resolve(record.input_path) == staged
    override = small_project / "custom_stage/img/00000.png"
    override.parent.mkdir(parents=True)
    override.write_bytes(b"override")
    assert PathResolver(small_project, uvdoc_final_root=small_project / "custom_stage").resolve(record.input_path) == override
    assert relative_manifest_path(r"dataset\DocUNet\scan\38.png") == "dataset/DocUNet/scan/38.png"
    for invalid in ("../other/file.png", "C:/private/file.jpg", "/tmp/absolute.png", "") :
        with pytest.raises(ValueError) :
            relative_manifest_path(invalid)


def test_original_manifest_files_are_not_modified(small_project : Path) :
    file = small_project / "dataset/_metadata/uvdoc_final_manifest.csv"
    before = hashlib.sha256(file.read_bytes()).hexdigest()
    validate_inventory(small_project, check_files=True, max_files_per_dataset=1)
    assert hashlib.sha256(file.read_bytes()).hexdigest() == before


def test_split_overlap_is_detected(small_project : Path) :
    file = small_project / "dataset/_metadata/uvdoc_val.csv"
    with file.open(encoding="utf-8", newline="") as handle :
        rows = list(csv.DictReader(handle))
        columns = list(rows[0])
    rows[0]["base_geometry"] = "00_00000_1"
    # Changing metadata alone must be caught even when group memberships stay unchanged.
    with file.open("w", encoding="utf-8", newline="") as handle :
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(rows)
    check = validate_existing_split(small_project)
    assert check["status"] == "FAIL"
    assert any(issue["code"] == "SPLIT_METADATA" for issue in check["issues"])


def test_hdf5_geometry_and_auxiliary_readers(tmp_path : Path) :
    grid2 = np.arange(2 * 61 * 89, dtype=np.float32).reshape(2, 61, 89)
    grid3 = np.arange(3 * 61 * 89, dtype=np.float32).reshape(3, 61, 89)
    file2 = tmp_path / "grid2.mat"
    file3 = tmp_path / "grid3.mat"
    with h5py.File(file2, "w") as handle :
        handle.create_dataset("grid2d", data=grid2)
    with h5py.File(file3, "w") as handle :
        handle.create_dataset("grid3d", data=grid3)
    loaded2 = uvdoc.load_grid2d(file2)
    loaded3 = uvdoc.load_grid3d(file3)
    assert loaded2.shape == (2, 89, 61)
    assert loaded3.shape == (89, 61, 3)
    assert loaded2[0, 7, 11] == grid2[0, 11, 7]
    assert loaded3[7, 11, 1] == grid3[1, 11, 7]
    with h5py.File(tmp_path / "aux.mat", "w") as handle :
        handle.create_dataset("uv", data=np.array([[3, 4]], dtype=np.float32))
    assert np.array_equal(uvdoc.read_auxiliary(tmp_path / "aux.mat")["uv"], [[3, 4]])
    with pytest.raises(KeyError) :
        uvdoc.load_grid2d(file3)


def test_exif_is_explicit(tmp_path : Path) :
    im = Image.new("RGB", (40, 20), (130, 30, 20))
    exif = Image.Exif()
    exif[274] = 6
    photo = tmp_path / "photo.jpg"
    im.save(photo, exif=exif)
    assert uvdoc.image_header(photo)["exif_orientation"] == 6
    assert uvdoc.read_image(photo, apply_exif=False).shape[:2] == (20, 40)
    assert uvdoc.read_image(photo, apply_exif=True).shape[:2] == (40, 20)


def test_sample_images_and_geometry_are_read(small_project : Path) :
    r_uv = load_records("uvdoc_final", small_project)[0]
    r_doc = load_records("docunet", small_project)[0]
    r_dir = load_records("dir300", small_project)[0]
    images = [(r_uv.input_path, (20, 30)), (r_doc.input_path, (30, 40)),
              (r_doc.reference_path, (25, 20)), (r_dir.input_path, (10, 20)),
              (r_dir.reference_path, (10, 25))]
    for relative, size in images :
        file = small_project / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, "white").save(file)
    for name, shape, key in (("grid2d", (2, 61, 89), "grid2d"), ("grid3d", (3, 61, 89), "grid3d")) :
        file = small_project / r_uv.annotation_paths[name]
        file.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(file, "w") as handle :
            handle.create_dataset(key, data=np.zeros(shape, dtype=np.float32))
    resolver = PathResolver(small_project)
    assert uvdoc.load_sample(r_uv, resolver, include_geometry=True)["grid2d"].shape == (2, 89, 61)
    assert docunet.load_sample(r_doc, resolver, include_reference=True)["reference"].shape == (20, 25, 3)
    assert dir300.load_sample(r_dir, resolver, include_reference=True)["reference"].shape == (25, 10, 3)
