# data/records.py
"""Dataset-neutral identities without imposing a training or evaluation policy."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SampleRecord :
    """One input variant. Paths are relative to the project root, not the machine."""

    sample_key : str
    dataset : str
    sample_id : str
    variant : str
    input_path : str
    reference_path : str | None = None
    group_id : str | None = None
    annotation_paths : dict[str, str] = field(default_factory=dict)
    metadata : dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) :
        if (not all([self.sample_key, self.dataset, self.sample_id, self.variant, self.input_path])) :
            raise ValueError("SampleRecord identity, variant, and input_path must be nonempty")
        if (not isinstance(self.annotation_paths, dict) or not isinstance(self.metadata, dict)) :
            raise TypeError("SampleRecord annotation_paths and metadata must be dictionaries")
