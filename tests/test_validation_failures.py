# tests/test_validation_failures.py
"""Failures must be visible rather than dropped from dataset audits."""

import csv
from pathlib import Path

import pytest

from data.manifests import load_records
from data.validation import validate_inventory


def _change_csv(path : Path, transform) :
    with path.open(encoding="utf-8", newline="") as handle :
        reader = csv.DictReader(handle)
        names = reader.fieldnames
        rows = list(reader)
    transform(rows)
    with path.open("w", encoding="utf-8", newline="") as handle :
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def test_duplicate_sample_is_rejected(small_project : Path) :
    path = small_project / "dataset/_metadata/dir300_manifest.csv"
    _change_csv(path, lambda rows : rows.append(dict(rows[0])))
    with pytest.raises(ValueError, match="Duplicate sample keys") :
        load_records("dir300", small_project)


def test_broken_docunet_reference_is_reported(small_project : Path) :
    path = small_project / "dataset/_metadata/docunet_manifest.csv"
    _change_csv(path, lambda rows : rows[0].update(reference_path="dataset/DocUNet/scan/999.png"))
    audit = validate_inventory(small_project)
    assert audit["docunet"]["status"] == "FAIL"
    assert any(issue["code"] == "DOCUNET_REFERENCE" for issue in audit["docunet"]["issues"])


def test_bad_status_is_retained_as_error(small_project : Path) :
    path = small_project / "dataset/_metadata/dir300_manifest.csv"
    _change_csv(path, lambda rows : rows[0].update(status="missing"))
    audit = validate_inventory(small_project)
    assert any(issue["code"] == "INVALID_SOURCE_RECORD" for issue in audit["dir300"]["issues"])


def test_file_absence_is_separate_from_manifest_integrity(small_project : Path) :
    audit = validate_inventory(small_project, check_files=True, max_files_per_dataset=1)
    assert audit["uvdoc_final"]["status"] == "PASS"
    assert audit["uvdoc_final"]["missing_file_count"] > 0


def test_validator_reports_duplicate_manifest_failure(small_project : Path) :
    path = small_project / "dataset/_metadata/dir300_manifest.csv"
    _change_csv(path, lambda rows : rows.append(dict(rows[0])))
    audit = validate_inventory(small_project)
    assert audit["dir300"]["status"] == "FAIL"
    assert audit["dir300"]["issues"][0]["code"] == "MANIFEST_ERROR"
    assert audit["split"]["status"] == "PASS"
