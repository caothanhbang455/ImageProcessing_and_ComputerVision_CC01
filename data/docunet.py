# data/docunet.py
"""DocUNet input variants and scan pairing, with EXIF handling made explicit."""

from .paths import PathResolver
from .records import SampleRecord
from .images import image_header, read_image


def load_sample(record : SampleRecord, resolver : PathResolver, include_reference : bool = False,
                apply_exif : bool = False) -> dict[str, object] :
    if (record.dataset != "DocUNet") :
        raise ValueError("Expected DocUNet record")
    input_path = resolver.resolve(record.input_path)
    sample = {"record": record, "image": read_image(input_path, apply_exif=apply_exif),
              "image_header": image_header(input_path)}
    if (include_reference) :
        if (not record.reference_path) :
            raise ValueError(f"Reference missing from manifest: {record.sample_key}")
        sample["reference"] = read_image(resolver.resolve(record.reference_path), apply_exif=apply_exif)
    return sample
