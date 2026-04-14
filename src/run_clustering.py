#!/usr/bin/env python3
"""
Behavior clustering pipeline for LISBET embeddings.

Pipeline: configurable smoothing -> robust scaling -> optional UMAP reduction
-> BIRCH subclustering -> configurable final clustering (agglomerative or
HDBSCAN), with evaluation and figure generation.

Usage
-----
    python -m post_training.behavior_clustering.src.run_clustering \\
        --embeddings-dir  /path/to/embeddings \\
        --output-dir      /path/to/output \\
        --config          configs/default.yaml \\
        --log-level       INFO
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Allow running as a script or as ``python -m``
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    __package__ = "post_training.behavior_clustering.src"

from .config import apply_config, load_config, validate_config
from .clustering import (
    assign_labels_gpu,
    fit_birch,
    hdbscan_refinement,
    hierarchical_refinement,
    map_to_frame_labels,
    search_hdbscan_params,
    save_birch,
)
from .data import load_embeddings
from .evaluation import (
    analyze_cross_video_clusters,
    compute_birch_radii,
    compute_temporal_coherence,
    summarize_labels,
    sweep_distance_thresholds,
)
from .preprocessing import fit_scaler, save_scaler, scale_data
from .reduction import UmapSearchConfig, fit_umap, save_reducer, search_umap_dbcv, transform_umap
from .smoothing import exponential_smoothing, median_smoothing, smooth_segments
from .visualization import (
    plot_birch_radii,
    plot_cluster_segment_heatmap,
    plot_cluster_sizes,
    plot_cross_video,
    plot_dendrogram,
    plot_merge_distances,
    plot_metrics_vs_threshold,
    plot_noise_distribution,
    plot_smoothing_effect,
    plot_temporal_coherence,
    plot_hdbscan_search_heatmap,
    plot_umap_search_heatmap,
)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--embeddings-dir",
        required=True,
        type=Path,
        help=(
            "Root directory containing 'train/' and 'test/' subdirectories, "
            "each with segment folders holding features_lisbet_embedding.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for all outputs (created if absent).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML/JSON config file (default: built-in defaults).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────


def setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level),
        stream=sys.stdout,
    )


# ─────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level)
    logger = logging.getLogger("behavior_clustering")
    t0 = time.time()

    # ── Config ───────────────────────────────────────────────
    cfg_raw = load_config(args.config)
    validate_config(cfg_raw)
    s = apply_config(cfg_raw)
    logger.info("Settings: %s", json.dumps(s, indent=2, default=str))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    stage_root = out / "stages"
    artifact_mode = s["artifact_mode"]
    save_stage_data = bool(s["save_stage_data"])
    save_stage_plots = bool(s["save_stage_plots"])
    if artifact_mode == "minimal":
        save_stage_data = False
        save_stage_plots = False
    elif artifact_mode == "verbose":
        save_stage_data = True
        save_stage_plots = True
    if save_stage_data or save_stage_plots:
        stage_root.mkdir(exist_ok=True)

    # Save config snapshot
    config_snapshot = out / "config_used.yaml"
    with open(config_snapshot, "w") as f:
        yaml.dump(s, f, default_flow_style=False, sort_keys=False)
    if args.config is not None:
        shutil.copy2(args.config, out / "config_original.yaml")

    emb_root = Path(args.embeddings_dir)

    # ── 1. Load data ─────────────────────────────────────────
    logger.info("Loading embeddings …")
    train_segs, train_names = load_embeddings(emb_root / "train")
    test_segs, test_names = load_embeddings(emb_root / "test")

    # ── 2. Smooth ────────────────────────────────────────────
    alpha = s["smoothing_alpha"]
    smooth_method = s["smoothing_method"]
    median_window = s["smoothing_median_window"]
    logger.info(
        "Smoothing (method=%s, alpha=%.2f, median_window=%d) …",
        smooth_method,
        alpha,
        median_window,
    )
    df_train, train_bounds = smooth_segments(
        train_segs,
        train_names,
        alpha=alpha,
        method=smooth_method,
        median_window=median_window,
    )
    df_test, test_bounds = smooth_segments(
        test_segs,
        test_names,
        alpha=alpha,
        method=smooth_method,
        median_window=median_window,
    )

    df_train.to_csv(out / "train_smoothed_features.csv", index=False)
    df_test.to_csv(out / "test_smoothed_features.csv", index=False)
    if save_stage_data:
        stage_dir = _stage_dir(stage_root, "02_smoothed")
        df_train.to_csv(stage_dir / "train_smoothed_features.csv", index=False)
        df_test.to_csv(stage_dir / "test_smoothed_features.csv", index=False)

    # Smoothing-effect figure (first train segment with enough frames)
    if save_stage_plots and len(train_segs[0]) > 200:
        if smooth_method == "ema":
            smoothed_example = exponential_smoothing(train_segs[0], alpha)
            method_name = "EMA"
        elif smooth_method == "median":
            smoothed_example = median_smoothing(train_segs[0], median_window)
            method_name = "Median"
        else:
            smoothed_example = train_segs[0]
            method_name = "No"
        plot_smoothing_effect(
            train_segs[0],
            smoothed_example,
            fig_dir,
            segment_name=train_names[0],
            method_name=method_name,
        )

    # ── 3. Scale ─────────────────────────────────────────────
    meta_cols = {"index", "segment_name", "segment_id"}
    feat_cols = [c for c in df_train.columns if c not in meta_cols]
    X_train = df_train[feat_cols].to_numpy(dtype=np.float32)
    X_test = df_test[feat_cols].to_numpy(dtype=np.float32)

    logger.info("Scaling with RobustScaler …")
    scaler = fit_scaler(X_train)
    X_train_s, X_test_s = scale_data(scaler, X_train, X_test)
    save_scaler(scaler, out / "scaler.joblib")
    np.save(out / "X_train_scaled.npy", X_train_s)
    np.save(out / "X_test_scaled.npy", X_test_s)
    if save_stage_data:
        stage_dir = _stage_dir(stage_root, "03_scaled")
        np.save(stage_dir / "X_train_scaled.npy", X_train_s)
        np.save(stage_dir / "X_test_scaled.npy", X_test_s)

    # ── 4. Optional UMAP reduction ──────────────────────────
    reduction_method = s["reduction_method"]
    X_train_cluster = X_train_s
    X_test_cluster = X_test_s
    umap_search_df = pd.DataFrame()
    selected_umap_params: dict[str, Any] = {}

    if reduction_method == "umap":
        logger.info("UMAP reduction enabled …")
        if s["umap_search_enabled"]:
            logger.info("Running UMAP grid search with DBCV validation …")
            umap_search_n_jobs = (
                s["reduction_validation_n_jobs"] if not _cuda_available() else 1
            )
            search_cfg = UmapSearchConfig(
                n_components=s["umap_n_components"],
                metric=s["umap_metric"],
                random_state=s["umap_random_state"],
                n_neighbors_grid=tuple(s["umap_search_n_neighbors_grid"]),
                min_dist_grid=tuple(s["umap_search_min_dist_grid"]),
                n_splits=s["reduction_validation_n_splits"],
                n_jobs=umap_search_n_jobs,
            )
            logger.info("UMAP search workers: %d", umap_search_n_jobs)
            selected_umap_params, umap_search_df = search_umap_dbcv(
                X_train_s,
                train_bounds,
                search_cfg,
            )
            umap_search_df.to_csv(out / "umap_grid_search_results.csv", index=False)
            with open(out / "umap_grid_search_best.json", "w") as f:
                json.dump(selected_umap_params, f, indent=2)
            if save_stage_data:
                stage_dir = _stage_dir(stage_root, "04_reduced")
                umap_search_df.to_csv(stage_dir / "umap_grid_search_results.csv", index=False)
            if save_stage_plots:
                plot_umap_search_heatmap(umap_search_df, fig_dir)
        else:
            selected_umap_params = {
                "n_components": s["umap_n_components"],
                "n_neighbors": s["umap_n_neighbors"],
                "min_dist": s["umap_min_dist"],
                "metric": s["umap_metric"],
                "random_state": s["umap_random_state"],
            }

        reducer = fit_umap(
            X_train_s,
            n_components=int(selected_umap_params["n_components"]),
            n_neighbors=int(selected_umap_params["n_neighbors"]),
            min_dist=float(selected_umap_params["min_dist"]),
            metric=str(selected_umap_params["metric"]),
            random_state=int(selected_umap_params["random_state"]),
        )
        X_train_cluster = transform_umap(reducer, X_train_s)
        X_test_cluster = transform_umap(reducer, X_test_s)
        save_reducer(reducer, out / "umap_reducer.joblib")
        np.save(out / "X_train_reduced.npy", X_train_cluster)
        np.save(out / "X_test_reduced.npy", X_test_cluster)
        if save_stage_data:
            stage_dir = _stage_dir(stage_root, "04_reduced")
            np.save(stage_dir / "X_train_reduced.npy", X_train_cluster)
            np.save(stage_dir / "X_test_reduced.npy", X_test_cluster)

    # ── 5. BIRCH ─────────────────────────────────────────────
    logger.info("BIRCH subclustering …")
    brc = fit_birch(
        X_train_cluster,
        threshold=s["birch_threshold"],
        branching_factor=s["birch_branching_factor"],
        batch_size=s["birch_batch_size"],
    )
    save_birch(brc, out / "birch_model.joblib")
    centroids = brc.subcluster_centers_.astype(np.float32)
    if save_stage_data:
        stage_dir = _stage_dir(stage_root, "05_birch")
        np.save(stage_dir / "birch_subcluster_centroids.npy", centroids)

    # ── 6. GPU label assignment ──────────────────────────────
    logger.info("GPU label assignment …")
    birch_labels_train = assign_labels_gpu(
        X_train_cluster, centroids, batch_size=s["gpu_batch_size"],
    )
    birch_labels_test = assign_labels_gpu(
        X_test_cluster, centroids, batch_size=s["gpu_batch_size"],
    )
    np.savetxt(
        out / "birch_labels_train.csv", birch_labels_train,
        delimiter=",", header="cluster_id", comments="", fmt="%d",
    )
    np.savetxt(
        out / "birch_labels_test.csv", birch_labels_test,
        delimiter=",", header="cluster_id", comments="", fmt="%d",
    )
    if save_stage_data:
        stage_dir = _stage_dir(stage_root, "06_birch_labels")
        np.savetxt(
            stage_dir / "birch_labels_train.csv", birch_labels_train,
            delimiter=",", header="cluster_id", comments="", fmt="%d",
        )
        np.savetxt(
            stage_dir / "birch_labels_test.csv", birch_labels_test,
            delimiter=",", header="cluster_id", comments="", fmt="%d",
        )

    # ── 7. Final clustering ──────────────────────────────────
    clustering_method = s["clustering_method"]
    linkage_mat = np.empty((0, 4))
    hdbscan_info: dict[str, Any] = {}
    hdbscan_search_df = pd.DataFrame()
    selected_hdbscan_params: dict[str, Any] = {}

    if clustering_method == "agglomerative":
        logger.info("Hierarchical refinement …")
        sub_labels, linkage_mat = hierarchical_refinement(
            centroids,
            distance_threshold=s["distance_threshold"],
            linkage_method=s["linkage_method"],
        )
        if linkage_mat.size > 0:
            np.save(out / "linkage_matrix.npy", linkage_mat)
            pd.DataFrame(
                linkage_mat, columns=["cluster1", "cluster2", "distance", "n_points"],
            ).to_csv(out / "linkage_matrix.csv", index=False)
            if save_stage_data:
                stage_dir = _stage_dir(stage_root, "07_clustered")
                np.save(stage_dir / "linkage_matrix.npy", linkage_mat)
        else:
            logger.warning("No linkage matrix (single subcluster). Skipping linkage save.")
    elif clustering_method == "hdbscan":
        logger.info("HDBSCAN refinement …")
        if s["hdbscan_search_enabled"]:
            logger.info(
                "Running HDBSCAN grid search (scoring=%s, workers=%d) …",
                s["hdbscan_search_scoring"],
                s["hdbscan_search_n_jobs"],
            )
            selected_hdbscan_params, hdbscan_search_df = search_hdbscan_params(
                centroids,
                min_cluster_size_grid=s["hdbscan_search_min_cluster_size_grid"],
                min_samples_grid=s["hdbscan_search_min_samples_grid"],
                metric=s["hdbscan_metric"],
                cluster_selection_method=s["hdbscan_cluster_selection_method"],
                scoring=s["hdbscan_search_scoring"],
                n_jobs=s["hdbscan_search_n_jobs"],
            )
            hdbscan_search_df.to_csv(out / "hdbscan_grid_search_results.csv", index=False)
            with open(out / "hdbscan_grid_search_best.json", "w") as f:
                json.dump(selected_hdbscan_params, f, indent=2)
            if save_stage_data:
                stage_dir = _stage_dir(stage_root, "07_clustered")
                hdbscan_search_df.to_csv(
                    stage_dir / "hdbscan_grid_search_results.csv", index=False,
                )

            hdb_min_cluster_size = int(selected_hdbscan_params["min_cluster_size"])
            hdb_min_samples = int(selected_hdbscan_params["min_samples"])
        else:
            hdb_min_cluster_size = s["hdbscan_min_cluster_size"]
            hdb_min_samples = s["hdbscan_min_samples"]

        sub_labels, hdbscan_info = hdbscan_refinement(
            centroids,
            min_cluster_size=hdb_min_cluster_size,
            min_samples=hdb_min_samples,
            metric=s["hdbscan_metric"],
            cluster_selection_method=s["hdbscan_cluster_selection_method"],
        )
        with open(out / "hdbscan_subcluster_summary.json", "w") as f:
            json.dump(hdbscan_info, f, indent=2)
    else:
        raise ValueError(f"Unsupported clustering method: {clustering_method}")

    final_train = map_to_frame_labels(birch_labels_train, sub_labels)
    final_test = map_to_frame_labels(birch_labels_test, sub_labels)

    # ── 7. Save frame-level mappings ─────────────────────────
    df_train["cluster_id"] = final_train
    df_test["cluster_id"] = final_test

    df_train[["index", "segment_name", "segment_id", "cluster_id"]].to_csv(
        out / "mapping_cluster_frame_train.csv", index=False,
    )
    df_test[["index", "segment_name", "segment_id", "cluster_id"]].to_csv(
        out / "mapping_cluster_frame_test.csv", index=False,
    )
    if save_stage_data:
        stage_dir = _stage_dir(stage_root, "08_frame_labels")
        df_train[["index", "segment_name", "segment_id", "cluster_id"]].to_csv(
            stage_dir / "mapping_cluster_frame_train.csv", index=False,
        )
        df_test[["index", "segment_name", "segment_id", "cluster_id"]].to_csv(
            stage_dir / "mapping_cluster_frame_test.csv", index=False,
        )

    # ── 9. Evaluation ────────────────────────────────────────
    logger.info("Running evaluation …")

    # 8a. Quality metrics sweep
    chosen_row = pd.DataFrame()
    if clustering_method == "agglomerative" and linkage_mat.size > 0:
        metrics_df = sweep_distance_thresholds(
            centroids, linkage_mat, s["eval_distance_thresholds"],
        )
        metrics_df.to_csv(out / "clustering_quality_metrics.csv", index=False)

        # Chosen-threshold metrics
        chosen_row = metrics_df[
            metrics_df["distance_threshold"] == s["distance_threshold"]
        ]
        if not chosen_row.empty:
            chosen_row.to_csv(out / "clustering_quality_chosen.csv", index=False)
    else:
        metrics_df = pd.DataFrame()
        logger.warning("Skipping distance-threshold sweep for clustering method: %s", clustering_method)

    # 8b. BIRCH radii
    radii, sizes = compute_birch_radii(X_train_cluster, birch_labels_train, centroids)
    pd.DataFrame({"radius": radii, "size": sizes}).to_csv(
        out / "birch_subcluster_radii.csv", index=False,
    )

    # 8c. Temporal coherence
    train_coh = compute_temporal_coherence(final_train, train_bounds)
    test_coh = compute_temporal_coherence(final_test, test_bounds)
    _save_coherence(train_coh, out / "temporal_coherence_train.csv")
    _save_coherence(test_coh, out / "temporal_coherence_test.csv")

    # 8d. Cross-video validation
    cross_train = analyze_cross_video_clusters(final_train, train_bounds)
    cross_test = analyze_cross_video_clusters(final_test, test_bounds)
    cross_train["segment_info"].to_csv(
        out / "cross_video_train.csv", index=False,
    )
    cross_test["segment_info"].to_csv(
        out / "cross_video_test.csv", index=False,
    )
    if save_stage_data:
        stage_dir = _stage_dir(stage_root, "09_evaluation")
        pd.DataFrame({"radius": radii, "size": sizes}).to_csv(
            stage_dir / "birch_subcluster_radii.csv", index=False,
        )
        if not metrics_df.empty:
            metrics_df.to_csv(stage_dir / "clustering_quality_metrics.csv", index=False)
        cross_train["segment_info"].to_csv(stage_dir / "cross_video_train.csv", index=False)
        cross_test["segment_info"].to_csv(stage_dir / "cross_video_test.csv", index=False)

    # ── 10. Figures ──────────────────────────────────────────
    logger.info("Generating figures …")
    plot_birch_radii(radii, s["birch_threshold"], fig_dir)
    if clustering_method == "agglomerative" and linkage_mat.size > 0:
        plot_merge_distances(linkage_mat, s["distance_threshold"], fig_dir)
        plot_dendrogram(linkage_mat, s["distance_threshold"], fig_dir)
    if not metrics_df.empty:
        plot_metrics_vs_threshold(metrics_df, fig_dir)
    plot_cluster_sizes(birch_labels_train, final_train, fig_dir)
    plot_temporal_coherence(
        train_coh, test_coh,
        birch_labels_train, birch_labels_test,
        final_train, final_test,
        train_bounds, test_bounds,
        fig_dir,
    )
    plot_cross_video(cross_train, len(train_bounds), fig_dir)
    plot_cluster_segment_heatmap(cross_train, fig_dir)
    if clustering_method == "hdbscan":
        plot_noise_distribution(final_train, fig_dir, "hdbscan_noise_train")
        plot_noise_distribution(final_test, fig_dir, "hdbscan_noise_test")
    if save_stage_plots:
        stage_fig_dir = _stage_dir(stage_root, "10_figures")
        if not umap_search_df.empty:
            plot_umap_search_heatmap(umap_search_df, stage_fig_dir)
        if not hdbscan_search_df.empty:
            plot_hdbscan_search_heatmap(hdbscan_search_df, stage_fig_dir)
    if not hdbscan_search_df.empty:
        plot_hdbscan_search_heatmap(hdbscan_search_df, fig_dir)

    # ── 11. Summary JSON ─────────────────────────────────────
    elapsed = time.time() - t0
    train_label_summary = summarize_labels(final_train)
    test_label_summary = summarize_labels(final_test)
    n_clusters = train_label_summary["n_clusters"]
    summary: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "embeddings_dir": str(emb_root),
        "n_train_segments": len(train_segs),
        "n_test_segments": len(test_segs),
        "n_train_frames": int(X_train_cluster.shape[0]),
        "n_test_frames": int(X_test_cluster.shape[0]),
        "n_features": int(X_train_cluster.shape[1]),
        "preprocessing_smoothing_method": smooth_method,
        "smoothing_alpha": s["smoothing_alpha"],
        "smoothing_median_window": s["smoothing_median_window"],
        "reduction_method": reduction_method,
        "clustering_method": clustering_method,
        "artifact_mode": artifact_mode,
        "n_birch_subclusters": int(len(centroids)),
        "n_final_clusters": n_clusters,
        "distance_threshold": s["distance_threshold"],
        "birch_threshold": s["birch_threshold"],
        "train_noise_ratio": train_label_summary["noise_ratio"],
        "test_noise_ratio": test_label_summary["noise_ratio"],
        "train_stability": round(train_coh["stability_score"], 4),
        "test_stability": round(test_coh["stability_score"], 4),
        "train_mean_bout_frames": round(train_coh["mean_bout_length"], 2),
        "test_mean_bout_frames": round(test_coh["mean_bout_length"], 2),
        "fps": s["fps"],
        "train_mean_bout_seconds": round(
            train_coh["mean_bout_length"] / s["fps"], 2,
        ),
    }

    if not chosen_row.empty:
        row = chosen_row.iloc[0]
        summary["silhouette_score"] = round(float(row["silhouette_score"]), 4)
        summary["davies_bouldin_index"] = round(float(row["davies_bouldin_index"]), 4)
        summary["calinski_harabasz_score"] = round(
            float(row["calinski_harabasz_score"]), 2,
        )

    if reduction_method == "umap":
        summary["umap_selected_params"] = selected_umap_params
    if clustering_method == "hdbscan":
        summary["hdbscan_subcluster_summary"] = hdbscan_info
    if selected_hdbscan_params:
        summary["hdbscan_selected_params"] = selected_hdbscan_params
    if not hdbscan_search_df.empty:
        summary["hdbscan_search_trials"] = int(len(hdbscan_search_df))
    if not umap_search_df.empty:
        summary["umap_search_trials"] = int(len(umap_search_df))

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Pipeline complete in %.1f s — %d clusters, "
        "stability=%.3f (train) / %.3f (test)",
        elapsed, n_clusters,
        train_coh["stability_score"], test_coh["stability_score"],
    )


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def _save_coherence(coh: dict, path: Path) -> None:
    """Persist temporal-coherence segment-level and summary metrics."""
    seg_df = coh["segment_metrics"]
    # Append summary row
    summary_row = pd.DataFrame([{
        "segment": "SUMMARY",
        "length": seg_df["length"].sum(),
        "switches": seg_df["switches"].sum(),
        "switch_rate": coh["switch_rate"],
        "stability": coh["stability_score"],
    }])
    pd.concat([seg_df, summary_row], ignore_index=True).to_csv(path, index=False)


def _stage_dir(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


if __name__ == "__main__":
    main()
