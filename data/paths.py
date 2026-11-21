# data/paths.py
"""Resolve manifest-relative paths, including optional UVDoc Final staging."""

import os
from pathlib import Path, PurePosixPath


FINAL_PREFIX = "dataset/UVDoc/UVDoc_final/"


def project_root(root : str | Path | None = None) -> Path :
    """Use an explicit root, environment setting, or current working directory."""
    value = root if root is not None else os.environ.get("DOCUMENT_UNWARPING_ROOT")
    resolved = Path(value).expanduser() if value else Path.cwd()
    if (not resolved.is_dir()) :
        raise FileNotFoundError(f"Project root is not accessible: {resolved}")
    return resolved.resolve()


def relative_manifest_path(value : str) -> str :
    """Normalize only relative manifest paths, never arbitrary absolute paths."""
    if (not isinstance(value, str) or not value.strip()) :
        raise ValueError("Manifest path is empty")
    normalized = value.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if (normalized.startswith("/") or any(":" in p or p in (".", "..") for p in parts)) :
        raise ValueError(f"Unsafe manifest path: {value!r}")
    return "/".join(parts)


class PathResolver :
    """Find real files without changing canonical manifest paths."""

    def __init__(self, root : str | Path | None = None, uvdoc_final_root : str | Path | None = None) :
        self.root = project_root(root)
        override = uvdoc_final_root or os.environ.get("DOCUMENT_UNWARPING_UVDOC_FINAL_ROOT")
        self.uvdoc_final_root = Path(override).expanduser() if override else None

    def candidates(self, value : str) -> list[Path] :
        logical = relative_manifest_path(value)
        primary = self.root / logical
        if (not logical.startswith(FINAL_PREFIX)) :
            return [primary]

        suffix = logical[len(FINAL_PREFIX):]
        candidates = []
        if (self.uvdoc_final_root is not None) :
            candidates.append(self.uvdoc_final_root / suffix)
        candidates.extend([
            primary,
            self.root / "dataset/UVDoc/_staging_UVDoc_final/UVDoc_final" / suffix,
        ])
        if (Path("/content/uvdoc_runtime/UVDoc_final").is_dir()) :
            candidates.append(Path("/content/uvdoc_runtime/UVDoc_final") / suffix)

        return list(dict.fromkeys(candidates))

    def find(self, value : str) -> Path | None :
        return next((candidate for candidate in self.candidates(value) if candidate.is_file()), None)

    def resolve(self, value : str, require_exists : bool = True) -> Path :
        found = self.find(value)
        if (found is not None) :
            return found
        if (require_exists) :
            locations = "; ".join(str(x) for x in self.candidates(value))
            raise FileNotFoundError(f"File not available: {value}. Checked: {locations}")
        return self.candidates(value)[0]
