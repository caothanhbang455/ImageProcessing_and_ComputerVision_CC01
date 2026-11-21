# data/manifests.py
"""Read the original EDA CSV manifests and expose stable, normalized sample records."""

import csv
from pathlib import Path
from typing import Any

import yaml

from .paths import project_root, relative_manifest_path
from .records import SampleRecord


REQUIRED_COLUMNS = {
    "uvdoc_final": {"dataset", "sample_id", "geom_name", "image_path", "grid2d_path", "grid3d_path", "metadata_path", "annotation_status"},
    "uvdoc_benchmark": {"dataset", "sample_id", "geom_name", "image_path", "grid2d_path", "grid3d_path", "metadata_path", "annotation_status"},
    "docunet": {"dataset", "sample_id", "document_id", "view_id", "original_path", "input_path", "reference_path", "input_variant", "status"},
    "dir300": {"dataset", "sample_id", "document_id", "input_path", "reference_path", "input_variant", "status"},
}

DATASET_KEYS = tuple(REQUIRED_COLUMNS)


def load_config(root : str | Path | None = None) -> dict[str, Any] :
    path = project_root(root) / "configs/datasets.yaml"
    if (not path.is_file()) :
        raise FileNotFoundError(f"Stage 1 dataset configuration missing: {path}")
    with path.open(encoding="utf-8") as handle :
        config = yaml.safe_load(handle)
    if (not isinstance(config, dict) or config.get("version") != 1 or not isinstance(config.get("datasets"), dict)) :
        raise ValueError(f"Invalid datasets.yaml schema: {path}")
    for name in DATASET_KEYS :
        if (name not in config["datasets"]) :
            raise ValueError(f"Missing dataset configuration: {name}")
        entry = config["datasets"][name]
        if (not isinstance(entry, dict) or not all(entry.get(key) for key in ("label", "manifest", "default_variant"))) :
            raise ValueError(f"Incomplete configuration for {name}")
    relative_manifest_path(str(config.get("metadata_directory", "")))
    return config


def read_csv(path : Path) -> list[dict[str, str]] :
    if (not path.is_file()) :
        raise FileNotFoundError(f"Manifest not available: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle :
        reader = csv.DictReader(handle)
        if (not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames)) :
            raise ValueError(f"Invalid or duplicate CSV headers: {path}")
        rows = list(reader)
    if (any(None in row for row in rows)) :
        raise ValueError(f"Rows contain columns beyond the declared CSV header: {path}")
    return rows


def manifest_path(dataset : str, root : str | Path | None = None) -> Path :
    cfg = load_config(root)
    if (dataset not in DATASET_KEYS) :
        raise ValueError(f"Unknown dataset key: {dataset!r}. Choose from {DATASET_KEYS}")
    metadata_dir = relative_manifest_path(cfg["metadata_directory"])
    name = relative_manifest_path(cfg["datasets"][dataset]["manifest"])
    if ("/" in name) :
        raise ValueError(f"Manifest name must be a file within metadata directory: {name}")
    return project_root(root) / metadata_dir / name


def load_manifest_rows(dataset : str, root : str | Path | None = None) -> list[dict[str, str]] :
    path = manifest_path(dataset, root)
    rows = read_csv(path)
    missing = REQUIRED_COLUMNS[dataset] - set(rows[0] if rows else read_headers(path))
    if (missing) :
        raise ValueError(f"{dataset} manifest missing columns: {sorted(missing)}")
    return rows


def read_headers(path : Path) -> list[str] :
    with path.open(encoding="utf-8-sig", newline="") as handle :
        return next(csv.reader(handle), [])


def _key(dataset_label : str, variant : str, sample_id : str) -> str :
    return "_".join([dataset_label.replace(" ", "_"), variant, sample_id])


def _record(dataset : str, row : dict[str, str], variant : str) -> SampleRecord :
    label = {"uvdoc_final": "UVDoc Final", "uvdoc_benchmark": "UVDoc Benchmark", "docunet": "DocUNet", "dir300": "DIR300"}[dataset]
    if (row["dataset"] != label) :
        raise ValueError(f"Expected {label} row; found {row['dataset']!r}, sample {row.get('sample_id')}")

    sample_id = row["sample_id"]
    if (dataset.startswith("uvdoc")) :
        if (variant != "rendered") :
            raise ValueError(f"UVDoc variant is 'rendered', not {variant}")
        geom = row["geom_name"]
        if (len(geom.rsplit("_", 2)) != 3) :
            raise ValueError(f"Malformed UVDoc geometry name {geom!r}")
        input_path = row["image_path"]
        group_id = geom.rsplit("_", 2)[0]
        reference_path = None
        annotations = {"grid2d": row["grid2d_path"], "grid3d": row["grid3d_path"], "metadata": row["metadata_path"]}
    elif (dataset == "docunet") :
        if (variant not in ("crop", "original")) :
            raise ValueError("DocUNet supports 'crop' and 'original' input variants")
        input_path = row["original_path"] if variant == "original" else row["input_path"]
        reference_path = row["reference_path"]
        group_id = row["document_id"]
        annotations = {}
    else :
        if (variant != "dist") :
            raise ValueError("DIR300 supports only the 'dist' input variant")
        input_path = row["input_path"]
        reference_path = row["reference_path"]
        group_id = row["document_id"]
        annotations = {}

    for name, value in [("input_path", input_path), ("reference_path", reference_path), *annotations.items()] :
        if (value) :
            relative_manifest_path(value)
        elif (name != "reference_path") :
            raise ValueError(f"Missing {name} for {label}/{sample_id}")
    return SampleRecord(
        sample_key=_key(label, variant, sample_id), dataset=label,
        sample_id=sample_id, variant=variant, input_path=input_path,
        reference_path=reference_path, group_id=group_id,
        annotation_paths=annotations, metadata=dict(row),
    )


def load_records(dataset : str, root : str | Path | None = None, variant : str | None = None) -> list[SampleRecord] :
    cfg = load_config(root)
    if (dataset not in DATASET_KEYS) :
        raise ValueError(f"Unknown dataset: {dataset!r}")
    chosen_variant = variant or cfg["datasets"][dataset]["default_variant"]
    records = [_record(dataset, row, chosen_variant) for row in load_manifest_rows(dataset, root)]
    keys = [record.sample_key for record in records]
    if (len(set(keys)) != len(keys)) :
        raise ValueError(f"Duplicate sample keys in {dataset} manifest")
    return records


def load_existing_split(root : str | Path | None = None) -> tuple[dict[str, str], dict[str, Any]] :
    """Read existing EDA memberships. Do not generate or assign a new split."""
    import json

    directory = project_root(root) / "dataset/_metadata"
    spec_file = directory / "split_spec.json"
    if (not spec_file.is_file()) :
        raise FileNotFoundError(f"Split specification missing: {spec_file}")
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    membership = {}
    for filename, expected_label in (("uvdoc_train.csv", "train"), ("uvdoc_val.csv", "val")) :
        file = directory / filename
        for row in read_csv(file) :
            if (row.get("split") != expected_label or row.get("dataset") != "UVDoc Final") :
                raise ValueError(f"Unexpected split label/dataset in {file}, sample {row.get('sample_id')}")
            sample_id = row.get("sample_id", "")
            if (not sample_id or sample_id in membership) :
                raise ValueError(f"Empty or repeated sample ID in split manifests: {sample_id!r}")
            membership[sample_id] = expected_label
    return membership, spec
