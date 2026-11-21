# evaluation/protocols/dir300.py
"""DIR300 numerical pairing and reference metric availability."""


METRICS = ("ms_ssim", "ld")


def reference_prediction_name(sample_id : str) -> str :
    if (not sample_id.isdigit() or not 1 <= int(sample_id) <= 300) :
        raise ValueError(f"Invalid DIR300 sample ID: {sample_id!r}")
    return f"{int(sample_id)}_rec.png"


def reference_name(sample_id : str) -> str :
    reference_prediction_name(sample_id)
    return f"{int(sample_id)}.png"
