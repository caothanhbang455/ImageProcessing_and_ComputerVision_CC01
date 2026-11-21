# evaluation/reference_metrics.py
"""External MATLAB reference bridge; never replace SIFT flow with optical flow."""

import importlib.util
from pathlib import Path


REQUIRED_MATLAB_SCRIPTS = {
    "uvdoc_visual_matlab_v1" : "evalScriptUVDoc.m",
    "docunet_matlab_legacy_v1" : "evalScript.m",
    "docunet_matlab_corrected64_v1" : "evalScript.m",
}


def matlab_availability(source_dir : str | Path | None = None, protocol_id : str | None = None) -> dict :
    """Report prerequisites without changing the Python environment."""
    installed = importlib.util.find_spec("matlab.engine") is not None if importlib.util.find_spec("matlab") else False
    script = REQUIRED_MATLAB_SCRIPTS.get(protocol_id or "")
    repo = Path(source_dir) if source_dir else None
    code = repo / "eval/eval_code" if repo else None
    file_ok = bool(code and (code / script).is_file()) if script else False
    sift_ok = bool(code and (code / "SIFTflow").is_dir())
    return {"matlab_engine_installed" : installed, "reference_script_present" : file_ok,
            "siftflow_directory_present" : sift_ok, "protocol_id" : protocol_id,
            "ready_to_attempt" : bool(installed and file_ok and sift_ok)}


def run_uvdoc_or_docunet_matlab(protocol_id : str, official_source : Path, reference_dir : Path,
                                 predictions_dir : Path, expected_predictions : int) -> dict :
    """Run the external authors' MATLAB script on a complete separately staged dataset.

    MATLAB writes individual_res.txt under predictions_dir. No file within
    the original inference run is ever passed as the writable destination.
    """
    if (protocol_id not in REQUIRED_MATLAB_SCRIPTS) :
        raise ValueError(f"Unknown external MATLAB protocol: {protocol_id}")
    availability = matlab_availability(official_source, protocol_id)
    if (not availability["ready_to_attempt"]) :
        return {"status" : "UNAVAILABLE_DEPENDENCY", "reason" : str(availability)}
    if (not reference_dir.is_dir() or not predictions_dir.is_dir()) :
        return {"status" : "UNAVAILABLE_REFERENCE", "reason" : "MATLAB input directories are absent"}
    if (len(list(predictions_dir.glob("*.png"))) != expected_predictions) :
        return {"status" : "UNAVAILABLE_PREDICTION", "reason" : "MATLAB requires a complete benchmark prediction set"}
    if ((predictions_dir / "individual_res.txt").exists()) :
        raise FileExistsError("Refusing to overwrite the external MATLAB result")
    import matlab.engine
    engine = matlab.engine.start_matlab()
    try :
        engine.cd(str(official_source / "eval/eval_code"), nargout=0)
        if (protocol_id == "uvdoc_visual_matlab_v1") :
            ms, ad = engine.evalScriptUVDoc(str(reference_dir), str(predictions_dir), False, nargout=2)
            metrics = {"ms_ssim" : float(ms), "ad" : float(ad)}
        else :
            ms, ld, ad = engine.evalScript(str(reference_dir), str(predictions_dir), False, nargout=3)
            metrics = {"ms_ssim" : float(ms), "ld" : float(ld), "ad" : float(ad)}
    finally :
        engine.quit()
    return {"status" : "OK", "reference_output" : str(predictions_dir / "individual_res.txt"),
            "metrics" : metrics, "verification" : "reference_execution_only; numerical_parity_pending"}


def run_dir300_matlab(source_dir : Path, reference_dir : Path, predictions_dir : Path) -> dict :
    """Execute DocGeoNet's evalUnwarp with its DIR300 preparation convention.

    This uses the released MATLAB evalUnwarp implementation, with an explicit
    Python-constructed driver instead of editing the original hardcoded paths.
    Report 'reference_execution_only' until matched to the original wrapper.
    """
    source = Path(source_dir)
    required = [source / "evalUnwarp.m", source / "ssim_ld_eval_DIR300.m"]
    installed = importlib.util.find_spec("matlab.engine") is not None if importlib.util.find_spec("matlab") else False
    if (not installed or not all(p.is_file() for p in required) or not (source / "SIFTflow").is_dir()) :
        return {"status" : "UNAVAILABLE_DEPENDENCY",
                "reason" : "DocGeoNet MATLAB Engine, released evaluation files, or SIFTflow is unavailable"}
    if (not reference_dir.is_dir() or len(list(predictions_dir.glob("*_rec.png"))) != 300 or
        not all((reference_dir / f"{i}.png").is_file() for i in range(1, 301))) :
        return {"status" : "UNAVAILABLE_REFERENCE", "reason" : "The complete DIR300 pairs are required"}
    import matlab.engine
    engine = matlab.engine.start_matlab()
    try :
        engine.cd(str(source), nargout=0)
        engine.addpath(engine.genpath(str(source / "SIFTflow")), nargout=0)
        engine.workspace["stage3_pred_root"] = str(predictions_dir)
        engine.workspace["stage3_ref_root"] = str(reference_dir)
        matlab_script = '''
stage3_rows = zeros(300, 3);
for stage3_i = 1:300
    stage3_pred = imread(fullfile(stage3_pred_root, sprintf('%d_rec.png', stage3_i)));
    stage3_ref = imread(fullfile(stage3_ref_root, sprintf('%d.png', stage3_i)));
    stage3_pred = rgb2gray(stage3_pred);
    stage3_ref = rgb2gray(stage3_ref);
    stage3_scale = sqrt(598400 / size(stage3_ref, 1) / size(stage3_ref, 2));
    stage3_ref = imresize(stage3_ref, stage3_scale);
    stage3_pred = imresize(stage3_pred, [size(stage3_ref,1), size(stage3_ref,2)]);
    [stage3_ms, stage3_ld] = evalUnwarp(stage3_pred, stage3_ref);
    stage3_rows(stage3_i,:) = [stage3_i, stage3_ms, stage3_ld];
end
'''
        engine.eval(matlab_script, nargout=0)
        rows = [list(row) for row in engine.workspace["stage3_rows"]]
    finally :
        engine.quit()
    return {"status" : "OK", "rows" : rows, "verification" : "reference_execution_only; parity_with_released_wrapper_pending"}
