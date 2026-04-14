"""
Data loading utilities for the behavior clustering pipeline.

Reads LISBET embedding CSV files from a directory tree where each subdirectory
corresponds to one video segment.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EMBEDDING_FILENAME = "features_lisbet_embedding.csv"


def load_embeddings(
    root: Path,
) -> tuple[list[np.ndarray], list[str]]:
    """
    Load embedding arrays from all segment subdirectories under *root*.

    Each subdirectory must contain ``features_lisbet_embedding.csv`` with an
    index column and numeric embedding columns.

    Returns
    -------
    segments : list[np.ndarray]
        One array per segment, shape (n_frames, n_dims).
    names : list[str]
        Corresponding subdirectory names (segment identifiers).

    Raises
    ------
    FileNotFoundError
        If *root* does not exist or contains no valid segments.
    ValueError
        If segments have inconsistent feature dimensions or contain NaN/Inf.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Embeddings directory not found: {root}")

    segments: list[np.ndarray] = []
    names: list[str] = []
    n_dims: int | None = None

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        csv_path = folder / EMBEDDING_FILENAME
        if not csv_path.exists():
            logger.debug("Skipping %s (no %s)", folder.name, EMBEDDING_FILENAME)
            continue

        df = pd.read_csv(csv_path, index_col=0)
        arr = df.to_numpy(dtype=np.float32)

        # Validate dimensions
        if n_dims is None:
            n_dims = arr.shape[1]
        elif arr.shape[1] != n_dims:
            raise ValueError(
                f"Dimension mismatch in {folder.name}: "
                f"expected {n_dims}, got {arr.shape[1]}"
            )

        # Validate values
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"NaN or Inf values in {folder.name}")

        segments.append(arr)
        names.append(folder.name)
        logger.debug("Loaded %s: %d frames × %d dims", folder.name, *arr.shape)

    if len(segments) < 2:
        raise FileNotFoundError(
            f"Need at least 2 segments under {root}, found {len(segments)}"
        )

    logger.info(
        "Loaded %d segments (%d total frames, %d dims) from %s",
        len(segments),
        sum(s.shape[0] for s in segments),
        n_dims,
        root,
    )
    return segments, names
