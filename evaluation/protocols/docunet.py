# evaluation/protocols/docunet.py
"""DocUNet image pairing and published MATLAB scoring prerequisites."""

from evaluation.alignment import orientation_policy


METRICS = ("ms_ssim", "ld", "ad")


def reference_prediction_name(sample_id : str) -> str :
    parts = sample_id.split("_")
    if (len(parts) != 2 or not all(x.isdigit() for x in parts)) :
        raise ValueError(f"DocUNet sample ID must be document_view: {sample_id!r}")
    doc, view = (int(x) for x in parts)
    if (not 1 <= doc <= 65 or view not in (1, 2)) :
        raise ValueError(f"Invalid DocUNet document/view: {sample_id}")
    return f"{doc}_{view}.png"


def reference_name(sample_id : str) -> str :
    return f"{int(sample_id.split('_')[0])}.png" if reference_prediction_name(sample_id) else ""


def corrected_input_required(sample_id : str, protocol_id : str) -> bool :
    return bool(orientation_policy(sample_id, protocol_id)["requires_corrected_inference"])
