# data/__init__.py
"""Shared document dataset records, loaders, and validation."""

from .manifests import load_records
from .paths import PathResolver
from .records import SampleRecord

__all__ = ["SampleRecord", "PathResolver", "load_records"]
