# tests/test_evaluation_reference.py
from evaluation.reference_metrics import matlab_availability, run_uvdoc_or_docunet_matlab


def test_matlab_missing_script_does_not_fake_scores(tmp_path) :
    check = matlab_availability(tmp_path, "docunet_matlab_legacy_v1")
    assert not check["ready_to_attempt"] and not check["reference_script_present"]
    result = run_uvdoc_or_docunet_matlab("docunet_matlab_legacy_v1", tmp_path, tmp_path, tmp_path, 130)
    assert result["status"] == "UNAVAILABLE_DEPENDENCY"
