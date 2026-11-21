# data/dir300.py
"""DIR300 input and corresponding flat reference image loading."""

from .paths import PathResolver
from .records import SampleRecord
from .images import image_header, read_image


def load_sample(record : SampleRecord, resolver : PathResolver, include_reference : bool = False) -> dict[str, object] :
    if (record.dataset != "DIR300") :
        raise ValueError("Expected DIR300 record")
    input_path = resolver.resolve(record.input_path)
    sample = {"record": record, "image": read_image(input_path),
              "image_header": image_header(input_path)}
    if (include_reference) :
        if (not record.reference_path) :
            raise ValueError(f"Reference missing from manifest: {record.sample_key}")
        sample["reference"] = read_image(resolver.resolve(record.reference_path))
    return sample
