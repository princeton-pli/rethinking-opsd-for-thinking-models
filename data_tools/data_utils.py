"""Shared path constants for the data pipeline."""

import os
import pathlib


# Repo root is the parent of data_tools/.
_WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_ROOT = pathlib.Path(os.environ.get("DATA_ROOT", str(_WORKSPACE_ROOT / "data")))
RAW_DATA_DIR = DATA_ROOT / "raw"
EVAL_DATA_DIR = DATA_ROOT / "eval"


def get_raw_path(dataset_name, sample_size=None):
    """Canonical path for a raw/sampled dataset parquet file."""
    suffix = f"_{sample_size}" if sample_size else ""
    return RAW_DATA_DIR / f"{dataset_name}{suffix}.parquet"
