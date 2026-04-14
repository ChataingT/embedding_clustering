"""
Clustering evaluation: quality metrics, temporal coherence, cross-video validation.

All three analyses are designed to operate on the compact subcluster centroids
or on frame-level labels + segment boundaries, avoiding O(N²) computations on
the full frame set.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

logger = logging.getLogger(__name__)


def summarize_labels(labels: np.ndarray) -> dict[str, Any]:
    """Summarize label counts including optional HDBSCAN noise label (-1)."""
    unique, counts = np.unique(labels, return_counts=True)
    mapping = {int(k): int(v) for k, v in zip(unique, counts)}
    n_total = int(len(labels))
    n_noise = int(mapping.get(-1, 0))
    n_non_noise = n_total - n_noise
    n_clusters = len([k for k in mapping if k >= 0])
    return {
        "n_total": n_total,
        "n_clusters": int(n_clusters),
        "n_noise": n_noise,
        "noise_ratio": float(n_noise / n_total) if n_total else 0.0,
        "counts": mapping,
        "n_non_noise": int(n_non_noise),
    }


def filter_noise_labels(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Filter out noise label (-1) samples for metrics requiring cluster IDs."""
    mask = labels >= 0
    return X[mask], labels[mask]


# ── 1. Clustering quality metrics ────────────────────────────

def sweep_distance_thresholds(
    subcluster_centers: np.ndarray,
    linkage_matrix: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    """
    Compute Silhouette, Davies-Bouldin, and Calinski-Harabasz scores for
    each distance threshold applied to the dendrogram.

    Metrics are evaluated on subcluster centroids (not raw frames).

    Returns a DataFrame with columns:
        distance_threshold, n_clusters, silhouette_score,
        davies_bouldin_index, calinski_harabasz_score
    """
    rows: list[dict[str, Any]] = []
    for t in thresholds:
        labs = fcluster(linkage_matrix, t=t, criterion="distance")
        n_clusters = len(np.unique(labs))
        if n_clusters < 2:
            logger.warning("Threshold %.1f → %d cluster (skipped)", t, n_clusters)
            continue
        rows.append({
            "distance_threshold": t,
            "n_clusters": n_clusters,
            "silhouette_score": silhouette_score(subcluster_centers, labs),
            "davies_bouldin_index": davies_bouldin_score(subcluster_centers, labs),
            "calinski_harabasz_score": calinski_harabasz_score(subcluster_centers, labs),
        })

    df = pd.DataFrame(rows)
    logger.info(
        "Quality metrics computed for %d thresholds [%.0f … %.0f]",
        len(df), min(thresholds), max(thresholds),
    )
    return df


def compute_birch_radii(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    subcluster_centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the RMS radius and size of each occupied BIRCH subcluster.

    Returns
    -------
    radii : np.ndarray, shape (n_occupied,)
    sizes : np.ndarray, shape (n_occupied,)
    """
    radii, sizes = [], []
    for i in range(len(subcluster_centers)):
        idx = np.where(labels == i)[0]
        if len(idx) == 0:
            continue
        diff = X_scaled[idx] - subcluster_centers[i]
        radius = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        radii.append(radius)
        sizes.append(len(idx))
    return np.array(radii), np.array(sizes)


# ── 2. Temporal coherence ────────────────────────────────────

def compute_temporal_coherence(
    labels: np.ndarray,
    segment_boundaries: list[tuple[int, int]],
) -> dict[str, Any]:
    """
    Measure how stable cluster assignments are over time.

    Returns
    -------
    dict with keys:
        mean_bout_length, median_bout_length, std_bout_length,
        switch_rate, stability_score, total_bouts,
        segment_metrics (pd.DataFrame per segment).
    """
    boundary_ends = {end for _, end in segment_boundaries}

    switches = 0
    bout_lengths: list[int] = []
    current_bout = 1

    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            if i not in boundary_ends:
                switches += 1
            bout_lengths.append(current_bout)
            current_bout = 1
        else:
            current_bout += 1
    bout_lengths.append(current_bout)

    # Per-segment metrics
    seg_rows: list[dict[str, Any]] = []
    for seg_idx, (start, end) in enumerate(segment_boundaries):
        seg_labs = labels[start:end]
        n = len(seg_labs)
        seg_sw = int(np.sum(seg_labs[1:] != seg_labs[:-1])) if n > 1 else 0
        rate = seg_sw / (n - 1) if n > 1 else 0.0
        seg_rows.append({
            "segment": seg_idx,
            "length": n,
            "switches": seg_sw,
            "switch_rate": rate,
            "stability": 1.0 - rate,
        })

    total_valid = len(labels) - len(segment_boundaries)
    switch_rate = switches / total_valid if total_valid > 0 else 0.0
    bout_arr = np.array(bout_lengths, dtype=float)

    return {
        "mean_bout_length": float(np.mean(bout_arr)),
        "median_bout_length": float(np.median(bout_arr)),
        "std_bout_length": float(np.std(bout_arr)),
        "switch_rate": switch_rate,
        "stability_score": 1.0 - switch_rate,
        "total_bouts": len(bout_lengths),
        "segment_metrics": pd.DataFrame(seg_rows),
    }


# ── 3. Cross-video cluster validation ────────────────────────

def analyze_cross_video_clusters(
    labels: np.ndarray,
    segment_boundaries: list[tuple[int, int]],
) -> dict[str, Any]:
    """
    Analyse how clusters distribute across video segments.

    Returns
    -------
    dict with keys:
        cluster_segment_matrix  (n_clusters × n_segments, binary),
        cluster_prevalence      (how many segments each cluster appears in),
        segment_diversity       (unique clusters per segment),
        unique_clusters,
        segment_info (pd.DataFrame).
    """
    n_segments = len(segment_boundaries)
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}

    matrix = np.zeros((n_clusters, n_segments), dtype=int)
    seg_rows: list[dict[str, Any]] = []

    for seg_idx, (start, end) in enumerate(segment_boundaries):
        seg_labs = labels[start:end]
        unique_in_seg = np.unique(seg_labs)
        for cid in unique_in_seg:
            matrix[cluster_to_idx[cid], seg_idx] = 1

        counts = pd.Series(seg_labs).value_counts()
        seg_rows.append({
            "segment": seg_idx,
            "n_frames": len(seg_labs),
            "n_unique_clusters": len(unique_in_seg),
            "most_common_cluster": int(counts.index[0]),
            "most_common_count": int(counts.iloc[0]),
            "most_common_pct": float(counts.iloc[0] / len(seg_labs) * 100),
        })

    prevalence = matrix.sum(axis=1)
    diversity = matrix.sum(axis=0)

    seg_specific = int((prevalence == 1).sum())
    widespread = int((prevalence >= n_segments * 0.5).sum())
    logger.info(
        "Cross-video: %d clusters, %d segment-specific (%.0f%%), "
        "%d widespread (%.0f%%)",
        n_clusters,
        seg_specific, seg_specific / n_clusters * 100,
        widespread, widespread / n_clusters * 100,
    )

    return {
        "cluster_segment_matrix": matrix,
        "cluster_prevalence": prevalence,
        "segment_diversity": diversity,
        "unique_clusters": unique_clusters,
        "segment_info": pd.DataFrame(seg_rows),
    }
