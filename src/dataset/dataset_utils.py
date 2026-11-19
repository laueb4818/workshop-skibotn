"""Fetch-if-missing dataset helper for Kaggle datasets."""

from __future__ import annotations

from pathlib import Path

import kagglehub
from kagglehub.config import get_cache_folder

# Short names -> Kaggle handles, so the rest of the code never hardcodes handles.
REGISTRY: dict[str, str] = {
    "kinetics-train-5per": "rohanmallick/kinetics-train-5per",
}


def _split(handle: str) -> tuple[str, str]:
    parts = handle.strip("/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"expected '<owner>/<dataset>', got {handle!r}")
    return parts[0], parts[1]


def _local_versions(handle: str) -> dict[int, Path]:
    """Versions of ``handle`` that are already in the kagglehub cache (non-empty dirs)."""
    owner, name = _split(handle)
    root = Path(get_cache_folder()) / "datasets" / owner / name / "versions"
    if not root.is_dir():
        return {}
    return {
        int(p.name): p
        for p in root.iterdir()
        if p.is_dir() and p.name.isdigit() and any(p.iterdir())
    }


def get_dataset(
    name_or_handle: str,
    *,
    version: int | None = None,
    force_download: bool = False,
    verbose: bool = True,
) -> Path:
    """Return the local path of a Kaggle dataset, downloading it only if it is missing.

    ``name_or_handle`` is a key of ``REGISTRY`` or a raw ``owner/dataset`` handle.
    With ``version=None`` and a cached copy, the newest cached version is used without
    touching the network; pin ``version`` for exact reproducibility.
    """
    handle = REGISTRY.get(name_or_handle, name_or_handle)
    _split(handle)  # validate

    if not force_download:
        local = _local_versions(handle)
        if version is not None and version in local:
            if verbose:
                print(f"[datasets] {handle} v{version} ready at {local[version]}")
            return local[version]
        if version is None and local:
            newest = max(local)
            if verbose:
                print(f"[datasets] {handle} v{newest} ready at {local[newest]}")
            return local[newest]

    if verbose:
        print(f"[datasets] downloading {handle}" + (f" v{version}" if version else "") + " ...")
    ref = handle if version is None else f"{handle}/versions/{version}"
    path = Path(kagglehub.dataset_download(ref, force_download=force_download))
    if verbose:
        print(f"[datasets] ready at {path}")
    return path