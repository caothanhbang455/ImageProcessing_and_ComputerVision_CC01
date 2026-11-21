# data/validation.py
"""Dataset integrity checks on manifests, existing split assignments, and files."""

from collections import Counter, defaultdict
from pathlib import Path

from .manifests import DATASET_KEYS, load_config, load_existing_split, load_records, read_csv
from .paths import PathResolver, project_root


def _issue(issues : list[dict], code : str, detail : str, sample_key : str = "") :
    issues.append({"code": code, "sample_key": sample_key, "detail": detail})


def validate_inventory(root : str | Path | None = None, check_files : bool = False,
                       max_files_per_dataset : int | None = None) -> dict :
    """Never changes the original files. File checks are opt-in for large remote datasets."""
    root = project_root(root)
    cfg = load_config(root)
    resolver = PathResolver(root)
    results = {}
    for key in DATASET_KEYS :
        issues = []
        try :
            records = load_records(key, root)
        except (FileNotFoundError, ValueError, KeyError) as error :
            results[key] = {"dataset": cfg["datasets"][key]["label"], "records": 0,
                            "groups": 0, "issues": [{"code": "MANIFEST_ERROR", "sample_key": "",
                                                   "detail": f"{type(error).__name__}: {error}"}],
                            "files_checked": 0, "missing_file_count": 0,
                            "file_check_status": "NOT_CHECKED", "missing_files": [], "status": "FAIL"}
            continue
        expected_count = cfg["datasets"][key].get("expected_records")
        if (expected_count is not None and len(records) != expected_count) :
            _issue(issues, "UNEXPECTED_COUNT", f"Expected {expected_count} records, found {len(records)}")
        status_column = "annotation_status" if key.startswith("uvdoc") else "status"
        for record in records :
            status = record.metadata.get(status_column, "")
            if (status != "valid") :
                _issue(issues, "INVALID_SOURCE_RECORD", f"Source status={status!r}", record.sample_key)

        if (key == "docunet") :
            grouped = defaultdict(list)
            for record in records :
                grouped[record.group_id].append(record)
                view_id = record.metadata["view_id"]
                expected = f"{record.group_id}_{view_id}"
                if (record.sample_id != expected) :
                    _issue(issues, "DOCUNET_ID", f"Expected {expected}", record.sample_key)
                if (Path(record.input_path).name != f"{record.sample_id} copy.png") :
                    _issue(issues, "DOCUNET_CROP", "Unexpected crop filename", record.sample_key)
                if (Path(record.reference_path).stem != record.group_id) :
                    _issue(issues, "DOCUNET_REFERENCE", "Scan ID differs from document ID", record.sample_key)
            for group, members in grouped.items() :
                if (len(members) != 2 or set(r.metadata["view_id"] for r in members) != {"1", "2"} or
                    len({r.reference_path for r in members}) != 1) :
                    _issue(issues, "DOCUNET_GROUP", f"Invalid two-view pairing for document {group}")
        if (key == "dir300") :
            for record in records :
                if (Path(record.input_path).stem != record.sample_id or
                    Path(record.reference_path).stem != record.sample_id) :
                    _issue(issues, "DIR300_PAIR", "Input/reference IDs do not match", record.sample_key)
        if (key.startswith("uvdoc")) :
            variants = defaultdict(set)
            for record in records :
                geom = record.metadata["geom_name"]
                variant = "_".join(geom.rsplit("_", 2)[-2:])
                variants[record.group_id].add(variant)
                if (Path(record.input_path).stem != record.sample_id or
                    Path(record.annotation_paths["grid2d"]).stem != geom or
                    Path(record.annotation_paths["grid3d"]).stem != geom) :
                    _issue(issues, "UVDOC_PAIR", "Image/grid geometry IDs do not match", record.sample_key)
            if (key == "uvdoc_final") :
                allowed = {"0_0", "0_1", "1_0", "1_1"}
                for group, values in variants.items() :
                    if (values != allowed) :
                        _issue(issues, "UVDOC_VARIANTS", f"Incomplete variants for {group}: {sorted(values)}")

        files_checked = 0
        missing_files = []
        if (check_files) :
            subset = records[:max_files_per_dataset] if max_files_per_dataset is not None else records
            for record in subset :
                paths = [("input", record.input_path)]
                if (record.reference_path) :
                    paths.append(("reference", record.reference_path))
                paths.extend(record.annotation_paths.items())
                for role, logical_path in paths :
                    files_checked += 1
                    if (resolver.find(logical_path) is None) :
                        missing_files.append({"sample_key": record.sample_key, "role": role, "path": logical_path})
        results[key] = {"dataset": cfg["datasets"][key]["label"], "records": len(records),
                        "groups": len({r.group_id for r in records}), "issues": issues,
                        "files_checked": files_checked, "missing_file_count": len(missing_files),
                        "file_check_status": ("UNAVAILABLE" if missing_files else "PASS") if check_files else "NOT_CHECKED",
                        "missing_files": missing_files[:50], "status": "FAIL" if issues else "PASS"}

    try :
        results["split"] = validate_existing_split(root)
    except (FileNotFoundError, ValueError, KeyError) as error :
        results["split"] = {"status": "FAIL", "training_samples": 0,
                            "validation_samples": 0, "base_groups": 0, "overlapping_groups": 0,
                            "issues": [{"code": "SPLIT_ERROR", "sample_key": "",
                                        "detail": f"{type(error).__name__}: {error}"}]}
    return results


def validate_existing_split(root : str | Path | None = None) -> dict :
    """Check the previously saved split; do not create one or assume it is obligatory."""
    root = project_root(root)
    membership, spec = load_existing_split(root)
    records = load_records("uvdoc_final", root)
    by_id = {record.sample_id: record for record in records}
    issues = []
    if (set(membership) != set(by_id)) :
        _issue(issues, "SPLIT_COVERAGE", f"Membership={len(membership)}, manifest={len(by_id)}, shared={len(set(membership) & set(by_id))}")
    grouped = defaultdict(set)
    counts = Counter(membership.values())
    for sample_id, split in membership.items() :
        if (sample_id in by_id) :
            grouped[by_id[sample_id].group_id].add(split)
    overlap = sorted(group for group, value in grouped.items() if len(value) > 1)
    if (overlap) :
        _issue(issues, "SPLIT_GROUP_OVERLAP", f"Shared base geometry groups: {overlap[:8]}")
    if (spec.get("policy") != "base_geometry_group_disjoint") :
        _issue(issues, "SPLIT_POLICY", "Unexpected stored split policy")
    if (counts.get("train", 0) != spec.get("training_samples") or counts.get("val", 0) != spec.get("validation_samples")) :
        _issue(issues, "SPLIT_COUNTS", f"Observed {dict(counts)}, stored spec {spec.get('training_samples')}/{spec.get('validation_samples')}")
    groups_per_split = Counter(next(iter(labels)) for labels in grouped.values() if len(labels) == 1)
    if (groups_per_split.get("train", 0) != spec.get("training_groups") or
        groups_per_split.get("val", 0) != spec.get("validation_groups")) :
        _issue(issues, "SPLIT_GROUP_COUNTS", f"Observed {dict(groups_per_split)}")

    # Verify the saved geom names, not just IDs and group counts.
    metadata = root / "dataset/_metadata"
    for filename in ("uvdoc_train.csv", "uvdoc_val.csv") :
        for row in read_csv(metadata / filename) :
            record = by_id.get(row["sample_id"])
            if (record is not None and (record.metadata["geom_name"] != row.get("geom_name") or
                                       record.group_id != row.get("base_geometry"))) :
                _issue(issues, "SPLIT_METADATA", "Geometry metadata inconsistent with original manifest", row["sample_id"])
                if (len(issues) >= 100) :
                    break
    return {"status": "FAIL" if issues else "PASS", "training_samples": counts.get("train", 0),
            "validation_samples": counts.get("val", 0), "base_groups": len(grouped),
            "overlapping_groups": len(overlap), "issues": issues}
