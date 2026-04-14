"""
Dimensionality reduction utilities for behavior clustering.

Supports optional UMAP projection and grid search over UMAP hyperparameters
validated with DBCV using train-segment cross-validation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


@dataclass
class UmapSearchConfig:
    n_components: int = 15
    metric: str = "cosine"
    random_state: int = 42
    n_neighbors_grid: tuple[int, ...] = (15, 30, 60, 90, 120)
    min_dist_grid: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2)
    n_splits: int = 5
    n_jobs: int = 1


def _import_umap():
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP is required but not installed. Install with: pip install umap-learn"
        ) from exc
    return umap


def _import_hdbscan():
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError(
            "HDBSCAN is required for DBCV validation. Install with: pip install hdbscan"
        ) from exc
    return hdbscan


def fit_umap(
    X_train: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
):
    """Fit a UMAP reducer on training features."""
    umap = _import_umap()
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit(X_train)


def transform_umap(reducer: Any, X: np.ndarray) -> np.ndarray:
    """Apply a fitted UMAP reducer."""
    return np.asarray(reducer.transform(X), dtype=np.float32)


def save_reducer(reducer: Any, path) -> None:
    joblib.dump(reducer, path)
    logger.info("Reducer saved -> %s", path)


def _segment_index_sets(segment_boundaries: list[tuple[int, int]]) -> list[np.ndarray]:
    idx_sets: list[np.ndarray] = []
    for start, end in segment_boundaries:
        idx_sets.append(np.arange(start, end, dtype=np.int64))
    return idx_sets


def _dbcv_score(X: np.ndarray, labels: np.ndarray) -> float:
    hdbscan = _import_hdbscan()
    cluster_labels = np.unique(labels[labels >= 0])
    if len(cluster_labels) < 2:
        return float("nan")
    return float(hdbscan.validity.validity_index(X, labels, metric="euclidean"))


def _score_fold(
    X: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> float:
    hdbscan = _import_hdbscan()

    reducer = fit_umap(
        X[train_indices],
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    X_val_red = transform_umap(reducer, X[val_indices])

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X_val_red)
    return _dbcv_score(X_val_red, labels)


def search_umap_dbcv(
    X_train: np.ndarray,
    segment_boundaries: list[tuple[int, int]],
    cfg: UmapSearchConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Grid search UMAP over n_neighbors and min_dist using DBCV validation.

    Cross-validation splits are constructed at segment level.

    Returns
    -------
    best_params : dict
        Selected parameter set with best mean DBCV score.
    results_df : pd.DataFrame
        Per-combination scores and fold statistics.
    """
    seg_indices = _segment_index_sets(segment_boundaries)
    n_segments = len(seg_indices)
    if n_segments < 3:
        raise ValueError("UMAP DBCV search requires at least 3 training segments")

    n_splits = min(cfg.n_splits, n_segments)
    if n_splits < 2:
        raise ValueError("At least 2 folds are required for UMAP DBCV search")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
    fold_splits = list(kf.split(np.arange(n_segments)))
    combos = [
        (int(n_neighbors), float(min_dist))
        for n_neighbors in cfg.n_neighbors_grid
        for min_dist in cfg.min_dist_grid
    ]

    def _eval_combo(n_neighbors: int, min_dist: float) -> dict[str, Any]:
        fold_scores: list[float] = []
        for fold_idx, (seg_train_idx, seg_val_idx) in enumerate(fold_splits):
            train_indices = np.concatenate([seg_indices[i] for i in seg_train_idx])
            val_indices = np.concatenate([seg_indices[i] for i in seg_val_idx])
            score = _score_fold(
                X_train,
                train_indices,
                val_indices,
                n_components=cfg.n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=cfg.metric,
                random_state=cfg.random_state + fold_idx,
            )
            fold_scores.append(score)

        arr = np.array(fold_scores, dtype=np.float64)
        valid = arr[np.isfinite(arr)]
        return {
            "n_neighbors": int(n_neighbors),
            "min_dist": float(min_dist),
            "n_components": int(cfg.n_components),
            "mean_dbcv": float(np.mean(valid)) if len(valid) else np.nan,
            "std_dbcv": float(np.std(valid)) if len(valid) else np.nan,
            "valid_folds": int(len(valid)),
            "total_folds": int(n_splits),
        }

    n_jobs = max(1, int(cfg.n_jobs))
    if n_jobs == 1 or len(combos) == 1:
        rows = [_eval_combo(nn, md) for nn, md in combos]
    else:
        max_workers = min(n_jobs, len(combos))
        logger.info(
            "UMAP grid search using %d worker threads over %d combinations",
            max_workers,
            len(combos),
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            rows = list(executor.map(lambda p: _eval_combo(p[0], p[1]), combos))

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        raise RuntimeError("UMAP DBCV search produced no results")

    ranked = results_df.sort_values(
        by=["mean_dbcv", "min_dist", "n_neighbors"],
        ascending=[False, True, True],
        na_position="last",
    )
    best = ranked.iloc[0]
    best_params = {
        "n_components": int(best["n_components"]),
        "n_neighbors": int(best["n_neighbors"]),
        "min_dist": float(best["min_dist"]),
        "metric": cfg.metric,
        "random_state": cfg.random_state,
        "best_mean_dbcv": float(best["mean_dbcv"]),
    }

    logger.info(
        "UMAP search best: n_neighbors=%d min_dist=%.3f mean_dbcv=%.4f",
        best_params["n_neighbors"],
        best_params["min_dist"],
        best_params["best_mean_dbcv"],
    )
    return best_params, results_df
