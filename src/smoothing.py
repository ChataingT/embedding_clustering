"""
Bidirectional exponential smoothing for multivariate time-series embeddings.

Reduces frame-level noise while preserving the temporal structure of
behaviour-relevant dynamics.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def exponential_smoothing(X: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """
    Bidirectional (forward–backward) exponential smoothing.

    Parameters
    ----------
    X : np.ndarray, shape (T, D)
        Multivariate time series.
    alpha : float in (0, 1]
        Smoothing factor.  Lower values produce heavier smoothing.

    Returns
    -------
    np.ndarray, shape (T, D)
        Smoothed series (average of forward and backward passes).
    """
    X = np.asarray(X, dtype=np.float32)
    T = len(X)

    fwd = np.empty_like(X)
    fwd[0] = X[0]
    for t in range(1, T):
        fwd[t] = alpha * X[t] + (1.0 - alpha) * fwd[t - 1]

    bwd = np.empty_like(X)
    bwd[-1] = X[-1]
    for t in range(T - 2, -1, -1):
        bwd[t] = alpha * X[t] + (1.0 - alpha) * bwd[t + 1]

    return 0.5 * (fwd + bwd)


def median_smoothing(X: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Median smoothing over time using an odd-sized centered window.

    Parameters
    ----------
    X : np.ndarray, shape (T, D)
        Multivariate time series.
    window : int
        Positive odd window size in frames.

    Returns
    -------
    np.ndarray, shape (T, D)
        Smoothed series.
    """
    X = np.asarray(X, dtype=np.float32)
    if window < 1 or window % 2 == 0:
        raise ValueError(f"median window must be a positive odd integer, got {window}")

    # pandas rolling median with centered window keeps temporal alignment and
    # uses local samples only; min_periods=1 preserves segment edges.
    return (
        pd.DataFrame(X)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=np.float32)
    )


def compute_temporal_energy(X: np.ndarray) -> float:
    """Mean L2-norm of consecutive-frame differences."""
    return float(np.mean(np.linalg.norm(np.diff(X, axis=0), axis=1)))


def smooth_segments(
    segments: Sequence[np.ndarray],
    names: Sequence[str],
    alpha: float,
    method: str = "ema",
    median_window: int = 5,
) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """
    Smooth each segment independently, then concatenate.

    Parameters
    ----------
    segments : list of np.ndarray
        Raw embedding arrays, one per video segment.
    names : list of str
        Segment identifiers (same length as *segments*).
    alpha : float
        Smoothing parameter forwarded to :func:`exponential_smoothing`.
    method : str
        One of ``'none'``, ``'ema'``, or ``'median'``.
    median_window : int
        Window size for median smoothing.

    Returns
    -------
    df : pd.DataFrame
        Combined smoothed embeddings with columns
        ``[0, 1, …, D-1, 'index', 'segment_name', 'segment_id']``.
    boundaries : list of (start, end) tuples
        Frame index ranges for each segment in the concatenated array.
    """
    smoothed_parts: list[pd.DataFrame] = []
    boundaries: list[tuple[int, int]] = []
    offset = 0

    for i, (seg, name) in enumerate(zip(segments, names)):
        raw_energy = compute_temporal_energy(seg)
        if method == "none":
            sm = np.asarray(seg, dtype=np.float32)
        elif method == "ema":
            sm = exponential_smoothing(seg, alpha=alpha)
        elif method == "median":
            sm = median_smoothing(seg, window=median_window)
        else:
            raise ValueError(f"Unsupported smoothing method: {method}")

        sm_energy = compute_temporal_energy(sm)
        ratio = sm_energy / raw_energy if raw_energy > 0 else 0.0
        logger.debug(
            "Segment %s: energy ratio %.3f (raw=%.3f, smoothed=%.3f)",
            name, ratio, raw_energy, sm_energy,
        )

        part = pd.DataFrame(sm)
        part["segment_name"] = name
        part.reset_index(inplace=True)
        smoothed_parts.append(part)

        end = offset + len(seg)
        boundaries.append((offset, end))
        offset = end

    df = pd.concat(smoothed_parts, axis=0, ignore_index=True)

    segment_ids: list[int] = []
    for i, part in enumerate(smoothed_parts):
        segment_ids.extend([i] * len(part))
    df["segment_id"] = segment_ids

    logger.info("Smoothed %d segments (method=%s) → %d total frames", len(segments), method, len(df))
    return df, boundaries
