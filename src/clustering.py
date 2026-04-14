"""
Two-stage clustering: BIRCH subclustering -> agglomerative or HDBSCAN refinement.

Stage 1 — BIRCH
    Online, scalable subclustering via incremental ``partial_fit`` batches.
    Produces O(hundreds–thousands) subclusters from millions of frames.

Stage 2 — Refinement on subcluster centroids
    Agglomerative linkage (Ward/complete/average) or HDBSCAN density clustering
    on the compact centroid set.

Label assignment uses GPU-accelerated batched nearest-centroid (torch).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import Birch

logger = logging.getLogger(__name__)


# ── Stage 1: BIRCH ───────────────────────────────────────────

def fit_birch(
    X: np.ndarray,
    threshold: float = 9.0,
    branching_factor: int = 500,
    batch_size: int = 10_000,
) -> Birch:
    """
    Train a BIRCH model via incremental partial_fit.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Scaled embedding array.
    threshold : float
        CF-tree leaf radius — controls subcluster granularity.
    branching_factor : int
        Max children per CF-tree node.
    batch_size : int
        Frames per ``partial_fit`` call.

    Returns
    -------
    Birch
        Fitted BIRCH model (``subcluster_centers_`` populated).
    """
    brc = Birch(
        threshold=threshold,
        branching_factor=branching_factor,
        n_clusters=None,
        compute_labels=False,
    )

    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        brc.partial_fit(X[start:end])
        if start % (10 * batch_size) == 0:
            logger.debug(
                "BIRCH partial_fit %d/%d — %d subclusters so far",
                end, n, len(brc.subcluster_centers_),
            )

    logger.info(
        "BIRCH fitted: %d subclusters (threshold=%.1f, branching_factor=%d)",
        len(brc.subcluster_centers_), threshold, branching_factor,
    )
    return brc


def save_birch(model: Birch, path: Path) -> None:
    joblib.dump(model, path)
    logger.info("BIRCH model saved → %s", path)


def load_birch(path: Path) -> Birch:
    model = joblib.load(path)
    logger.info("BIRCH model loaded ← %s", path)
    return model


# ── GPU label assignment ─────────────────────────────────────

@torch.no_grad()
def assign_labels_gpu(
    X: np.ndarray,
    centroids: np.ndarray,
    batch_size: int = 256,
    device: str = "cuda",
) -> np.ndarray:
    """
    Assign each frame to its nearest centroid using batched squared-Euclidean
    distance on GPU.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Scaled data points.
    centroids : np.ndarray, shape (M, D)
        Subcluster centres.
    batch_size : int
        Number of frames per GPU batch.
    device : str
        PyTorch device identifier.

    Returns
    -------
    np.ndarray, shape (N,), dtype int32
        Nearest-centroid indices.
    """
    N = X.shape[0]
    C = torch.tensor(centroids.astype(np.float32), device=device)
    C_norm = (C ** 2).sum(dim=1).unsqueeze(0)  # (1, M)

    labels = torch.empty(N, dtype=torch.int32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        Xb = torch.tensor(X[start:end], dtype=torch.float32, device=device)

        X_norm = (Xb ** 2).sum(dim=1).unsqueeze(1)  # (B, 1)
        dists = X_norm + C_norm - 2.0 * (Xb @ C.T)  # (B, M)

        labels[start:end] = torch.argmin(dists, dim=1).cpu()

        del Xb, X_norm, dists

    torch.cuda.empty_cache()
    logger.info("GPU label assignment: %d frames → %d centroids", N, len(centroids))
    return labels.numpy()


# ── Stage 2: Ward hierarchical refinement ────────────────────

def hierarchical_refinement(
    subcluster_centers: np.ndarray,
    distance_threshold: float,
    linkage_method: str = "ward",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Agglomerative clustering on BIRCH subcluster centroids.

    Parameters
    ----------
    subcluster_centers : np.ndarray, shape (M, D)
    distance_threshold : float
        Dendrogram cut height.
    linkage_method : str
        ``'ward'``, ``'complete'``, or ``'average'``.

    Returns
    -------
    subcluster_labels : np.ndarray, shape (M,)
        Cluster ID for each subcluster (1-based from ``fcluster``).
    linkage_matrix : np.ndarray, shape (M-1, 4)
        Scipy linkage matrix.
    """
    n_sub = len(subcluster_centers)
    logger.info(
        "Computing %s linkage on %d subcluster centroids …",
        linkage_method, n_sub,
    )

    if n_sub < 2:
        logger.warning(
            "Only %d subcluster(s) — too few for hierarchical linkage. "
            "All frames will be assigned to a single cluster. "
            "Consider lowering the BIRCH threshold.", n_sub,
        )
        return np.ones(n_sub, dtype=np.intp), np.empty((0, 4))

    linkage_matrix = linkage(
        subcluster_centers,
        method=linkage_method,
        metric="euclidean",
    )

    subcluster_labels = fcluster(
        linkage_matrix, t=distance_threshold, criterion="distance"
    )

    n_clusters = len(np.unique(subcluster_labels))
    logger.info(
        "Hierarchical cut at distance=%.1f → %d clusters",
        distance_threshold, n_clusters,
    )
    return subcluster_labels, linkage_matrix


def hdbscan_refinement(
    subcluster_centers: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Density-based clustering on BIRCH subcluster centroids using HDBSCAN.

    Returns
    -------
    subcluster_labels : np.ndarray, shape (M,)
        Cluster label per subcluster center. Noise is labeled as -1.
    info : dict
        Summary diagnostics including cluster and noise counts.
    """
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError(
            "HDBSCAN clustering requested but package is missing. "
            "Install with: pip install hdbscan"
        ) from exc

    n_sub = len(subcluster_centers)
    if n_sub < 2:
        logger.warning(
            "Only %d subcluster(s) — too few for HDBSCAN. Returning a single cluster.",
            n_sub,
        )
        return np.zeros(n_sub, dtype=np.intp), {
            "n_subclusters": int(n_sub),
            "n_clusters": 1 if n_sub > 0 else 0,
            "n_noise": 0,
            "noise_ratio": 0.0,
        }

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )
    labels = clusterer.fit_predict(subcluster_centers)

    n_noise = int(np.sum(labels == -1))
    non_noise = labels[labels >= 0]
    n_clusters = int(len(np.unique(non_noise))) if len(non_noise) else 0
    info = {
        "n_subclusters": int(n_sub),
        "n_clusters": int(n_clusters),
        "n_noise": int(n_noise),
        "noise_ratio": float(n_noise / n_sub),
    }
    logger.info(
        "HDBSCAN on %d subclusters -> %d clusters, noise=%d (%.2f%%)",
        n_sub,
        n_clusters,
        n_noise,
        info["noise_ratio"] * 100.0,
    )
    return labels.astype(np.intp), info


def search_hdbscan_params(
    subcluster_centers: np.ndarray,
    min_cluster_size_grid: list[int],
    min_samples_grid: list[int],
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    scoring: str = "relative_validity",
    n_jobs: int = 1,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Grid search HDBSCAN hyperparameters on subcluster centroids.

    Parameters
    ----------
    subcluster_centers : np.ndarray
        BIRCH centroid embeddings.
    min_cluster_size_grid : list[int]
        Candidate min_cluster_size values.
    min_samples_grid : list[int]
        Candidate min_samples values.
    metric : str
        HDBSCAN metric.
    cluster_selection_method : str
        HDBSCAN cluster selection method.
    scoring : str
        One of ``"dbcv"`` or ``"relative_validity"``.
    n_jobs : int
        Number of worker threads for combination-level parallelism.

    Returns
    -------
    best_params : dict[str, Any]
        Best hyperparameters and score.
    results_df : pd.DataFrame
        Per-combination diagnostics and score.
    """
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError(
            "HDBSCAN search requested but package is missing. "
            "Install with: pip install hdbscan"
        ) from exc

    if scoring not in {"dbcv", "relative_validity"}:
        raise ValueError(
            f"Invalid HDBSCAN search scoring '{scoring}'. "
            "Valid: ['dbcv', 'relative_validity']"
        )

    combos = [
        (int(min_cluster_size), int(min_samples))
        for min_cluster_size in min_cluster_size_grid
        for min_samples in min_samples_grid
    ]
    if not combos:
        raise ValueError("HDBSCAN search grids are empty")

    def _score_combo(min_cluster_size: int, min_samples: int) -> dict[str, Any]:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
        )
        labels = clusterer.fit_predict(subcluster_centers)

        n_sub = len(labels)
        n_noise = int(np.sum(labels == -1))
        non_noise = labels[labels >= 0]
        n_clusters = int(len(np.unique(non_noise))) if len(non_noise) else 0
        noise_ratio = float(n_noise / n_sub) if n_sub else 0.0

        score = np.nan
        if scoring == "relative_validity":
            score = float(getattr(clusterer, "relative_validity_", np.nan))
        elif n_clusters >= 2:
            score = float(
                hdbscan.validity.validity_index(
                    subcluster_centers,
                    labels,
                    metric=metric,
                )
            )

        return {
            "min_cluster_size": int(min_cluster_size),
            "min_samples": int(min_samples),
            "score": float(score) if np.isfinite(score) else np.nan,
            "scoring": scoring,
            "n_clusters": int(n_clusters),
            "n_noise": int(n_noise),
            "noise_ratio": float(noise_ratio),
        }

    n_jobs_eff = max(1, int(n_jobs))
    if n_jobs_eff == 1 or len(combos) == 1:
        rows = [_score_combo(mcs, ms) for mcs, ms in combos]
    else:
        max_workers = min(n_jobs_eff, len(combos))
        logger.info(
            "HDBSCAN search using %d worker threads over %d combinations",
            max_workers,
            len(combos),
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            rows = list(executor.map(lambda p: _score_combo(p[0], p[1]), combos))

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        raise RuntimeError("HDBSCAN search produced no results")

    ranked = results_df.sort_values(
        by=["score", "noise_ratio", "min_cluster_size", "min_samples"],
        ascending=[False, True, True, True],
        na_position="last",
    )
    best = ranked.iloc[0]
    best_params = {
        "min_cluster_size": int(best["min_cluster_size"]),
        "min_samples": int(best["min_samples"]),
        "metric": metric,
        "cluster_selection_method": cluster_selection_method,
        "scoring": scoring,
        "best_score": float(best["score"]) if np.isfinite(best["score"]) else np.nan,
        "best_n_clusters": int(best["n_clusters"]),
        "best_noise_ratio": float(best["noise_ratio"]),
    }
    logger.info(
        "HDBSCAN search best: min_cluster_size=%d min_samples=%d score=%.4f (%s)",
        best_params["min_cluster_size"],
        best_params["min_samples"],
        best_params["best_score"] if np.isfinite(best_params["best_score"]) else float("nan"),
        scoring,
    )
    return best_params, results_df


def map_to_frame_labels(
    birch_labels: np.ndarray,
    subcluster_labels: np.ndarray,
) -> np.ndarray:
    """Map BIRCH per-frame subcluster indices to final cluster IDs."""
    return subcluster_labels[birch_labels]
