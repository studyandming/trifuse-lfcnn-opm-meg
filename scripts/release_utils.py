"""Shared path helpers for the anonymous TriFuse-LFCNN release."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = REPO_ROOT / "cache"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"


def default_zip_path() -> Path:
    """Return the dataset archive path, overridable with OPM_MEG_ZIP."""
    return Path(os.environ.get("OPM_MEG_ZIP", DATA_DIR / "dog_day_afternoon_OPM.zip"))


def default_cache_dir() -> Path:
    """Return the preprocessing cache directory, overridable with OPM_MEG_CACHE."""
    return Path(os.environ.get("OPM_MEG_CACHE", CACHE_DIR / "opm_movie_validation_cache"))


def default_result_path(filename: str) -> Path:
    """Return a result path under results/, overridable by per-script CLI args."""
    return RESULTS_DIR / filename

