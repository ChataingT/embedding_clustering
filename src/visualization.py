"""
Publication-quality figure generation for the behavior clustering pipeline.

Every plot function saves both PNG (300 dpi) and PDF (vector) side by side.
Matplotlib ``'seaborn-v0_8-whitegrid'`` is used as the base style.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

logger = logging.getLogger(__name__)

# ── Styling ──────────────────────────────────────────────────

_STYLE = "seaborn-v0_8-whitegrid"


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    """Save figure as PNG + PDF."""
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved figure: %s", name)


# ── 1. Smoothing effect ─────────────────────────────────────

def plot_smoothing_effect(
    original: np.ndarray,
    smoothed: np.ndarray,
    out_dir: Path,
    segment_name: str = "segment_0",
    frame_range: tuple[int, int] = (100, 200),
    method_name: str = "EMA",
) -> None:
    """Compare original vs smoothed embeddings on 3 dimensions."""
    with plt.style.context(_STYLE):
        n_dims = original.shape[1]
        dims = [0, min(50, n_dims - 1), min(100, n_dims - 1)]
        lo, hi = frame_range

        fig, axes = plt.subplots(3, 1, figsize=(15, 8))
        for i, d in enumerate(dims):
            axes[i].plot(original[lo:hi, d], alpha=0.5, label="Original", lw=1)
            axes[i].plot(smoothed[lo:hi, d], label="Smoothed", lw=2)
            axes[i].set_ylabel(f"Dimension {d}", fontsize=12)
            axes[i].legend(fontsize=14)
            axes[i].grid(True, alpha=0.3)
        axes[-1].set_xlabel("Frame", fontsize=12)
        fig.suptitle(
            f"{method_name} Smoothing Effect ({segment_name})",
            fontsize=16, fontweight="bold",
        )
        fig.tight_layout()
        _save(fig, out_dir, "smoothing_effect")


# ── 2. BIRCH radius distribution ────────────────────────────

def plot_birch_radii(
    radii: np.ndarray,
    threshold: float,
    out_dir: Path,
) -> None:
    """Histogram of subcluster radii with threshold line."""
    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(radii, bins=30, kde=True, ax=ax)
        ax.axvline(threshold, color="red", ls="--", lw=2,
                   label=f"BIRCH threshold ({threshold})")
        ax.set_title("BIRCH Subcluster Radii Distribution", fontsize=14,
                      fontweight="bold")
        ax.set_xlabel("Radius", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend(fontsize=12)
        fig.tight_layout()
        _save(fig, out_dir, "birch_subcluster_radii")


# ── 3. Merge-distance curve ─────────────────────────────────

def plot_merge_distances(
    linkage_matrix: np.ndarray,
    distance_threshold: float,
    out_dir: Path,
) -> None:
    """Line plot of ward merge distances with cut-threshold."""
    with plt.style.context(_STYLE):
        distances = linkage_matrix[:, 2]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(distances, lw=1)
        ax.axhline(distance_threshold, color="red", ls="--", lw=2,
                    label=f"Distance threshold = {distance_threshold}")
        ax.set_title("Merge Distances in Hierarchical Clustering",
                      fontsize=16, fontweight="bold")
        ax.set_xlabel("Merge Step", fontsize=12)
        ax.set_ylabel("Ward Distance", fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, out_dir, "merge_distances")


# ── 4. Dendrogram ───────────────────────────────────────────

def plot_dendrogram(
    linkage_matrix: np.ndarray,
    distance_threshold: float,
    out_dir: Path,
    p: int = 300,
) -> None:
    """Truncated dendrogram with distance threshold line."""
    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(20, 10))
        dendrogram(
            linkage_matrix,
            truncate_mode="lastp",
            p=p,
            leaf_font_size=10,
            show_leaf_counts=True,
            color_threshold=distance_threshold,
            ax=ax,
        )
        ax.axhline(distance_threshold, color="red", ls="--", lw=2,
                    label=f"Distance threshold = {distance_threshold}")
        ax.set_title("Dendrogram with Distance Threshold",
                      fontsize=16, fontweight="bold")
        ax.set_xlabel("Subcluster", fontsize=14)
        ax.set_ylabel("Ward Distance", fontsize=14)
        ax.set_xticks([])
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        _save(fig, out_dir, "dendrogram")


# ── 5. Quality metrics vs distance threshold ────────────────

def plot_metrics_vs_threshold(
    metrics_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Three-panel line plot of clustering metrics vs distance threshold."""
    with plt.style.context(_STYLE):
        melted = metrics_df.melt(
            id_vars="distance_threshold",
            value_vars=[
                "silhouette_score",
                "davies_bouldin_index",
                "calinski_harabasz_score",
            ],
        )
        g = sns.FacetGrid(
            melted, col="variable", sharey=False, height=4, aspect=1.5,
        )
        g.map(sns.lineplot, "distance_threshold", "value", marker="o")
        g.set_axis_labels("Distance Threshold", "Metric Value")
        g.set_titles("{col_name}")
        g.figure.suptitle(
            "Clustering Quality vs Distance Threshold",
            fontsize=14, fontweight="bold", y=1.02,
        )
        g.tight_layout()
        _save(g.figure, out_dir, "metrics_vs_threshold")


# ── 6. Cluster size distribution ────────────────────────────

def plot_cluster_sizes(
    birch_labels: np.ndarray,
    final_labels: np.ndarray,
    out_dir: Path,
) -> None:
    """Side-by-side bar charts of BIRCH and hierarchical cluster sizes."""
    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        _, counts_b = np.unique(birch_labels, return_counts=True)
        axes[0].bar(range(len(counts_b)), sorted(counts_b, reverse=True))
        axes[0].set_title("BIRCH: Subcluster Size Distribution", fontsize=13,
                           fontweight="bold")
        axes[0].set_xlabel("Subcluster (sorted)")
        axes[0].set_ylabel("Frames")
        axes[0].grid(True, alpha=0.3)

        _, counts_h = np.unique(final_labels, return_counts=True)
        axes[1].bar(range(len(counts_h)), sorted(counts_h, reverse=True))
        axes[1].set_title("Hierarchical: Cluster Size Distribution", fontsize=13,
                           fontweight="bold")
        axes[1].set_xlabel("Cluster (sorted)")
        axes[1].set_ylabel("Frames")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        _save(fig, out_dir, "cluster_size_distributions")


# ── 7. Temporal coherence ───────────────────────────────────

def plot_temporal_coherence(
    train_coherence: dict,
    test_coherence: dict,
    train_birch_labels: np.ndarray,
    test_birch_labels: np.ndarray,
    train_final: np.ndarray,
    test_final: np.ndarray,
    train_boundaries: list[tuple[int, int]],
    test_boundaries: list[tuple[int, int]],
    out_dir: Path,
) -> None:
    """Four-panel figure: stability bars + cluster scatter for train/test."""
    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Train stability per segment
        sm_train = train_coherence["segment_metrics"]
        axes[0, 0].bar(sm_train["segment"], sm_train["stability"])
        axes[0, 0].axhline(train_coherence["stability_score"], color="r", ls="--",
                           label=f'Mean: {train_coherence["stability_score"]:.3f}')
        axes[0, 0].set_title("Train: Temporal Stability per Segment", fontweight="bold")
        axes[0, 0].set_xlabel("Segment")
        axes[0, 0].set_ylabel("Stability (1 − switch_rate)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Test stability per segment
        sm_test = test_coherence["segment_metrics"]
        axes[0, 1].bar(sm_test["segment"], sm_test["stability"])
        axes[0, 1].axhline(test_coherence["stability_score"], color="r", ls="--",
                           label=f'Mean: {test_coherence["stability_score"]:.3f}')
        axes[0, 1].set_title("Test: Temporal Stability per Segment", fontweight="bold")
        axes[0, 1].set_xlabel("Segment")
        axes[0, 1].set_ylabel("Stability (1 − switch_rate)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Train cluster scatter (first segment)
        first_len = min(1000, train_boundaries[0][1] - train_boundaries[0][0])
        axes[1, 0].scatter(
            range(first_len), train_final[:first_len],
            alpha=0.5, s=1, c=train_final[:first_len], cmap="tab20",
        )
        axes[1, 0].set_title("Train: Cluster Assignments (First Segment)",
                              fontweight="bold")
        axes[1, 0].set_xlabel("Frame")
        axes[1, 0].set_ylabel("Cluster ID")
        axes[1, 0].grid(True, alpha=0.3)

        # Test cluster scatter (first segment)
        first_len_t = min(1000, test_boundaries[0][1] - test_boundaries[0][0])
        axes[1, 1].scatter(
            range(first_len_t), test_final[:first_len_t],
            alpha=0.5, s=1, c=test_final[:first_len_t], cmap="tab20",
        )
        axes[1, 1].set_title("Test: Cluster Assignments (First Segment)",
                              fontweight="bold")
        axes[1, 1].set_xlabel("Frame")
        axes[1, 1].set_ylabel("Cluster ID")
        axes[1, 1].grid(True, alpha=0.3)

        fig.tight_layout()
        _save(fig, out_dir, "temporal_coherence")


# ── 8. Cross-video distribution ─────────────────────────────

def plot_cross_video(
    cross_info: dict,
    n_segments: int,
    out_dir: Path,
) -> None:
    """Cluster prevalence histogram + per-segment diversity bars."""
    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].hist(
            cross_info["cluster_prevalence"],
            bins=range(1, n_segments + 2),
            edgecolor="black", alpha=0.7,
        )
        axes[0].axvline(
            cross_info["cluster_prevalence"].mean(), color="r", ls="--",
            label=f'Mean: {cross_info["cluster_prevalence"].mean():.1f}',
        )
        axes[0].set_title("Cluster Prevalence Distribution", fontsize=13,
                           fontweight="bold")
        axes[0].set_xlabel("Number of Segments Containing Cluster")
        axes[0].set_ylabel("Number of Clusters")
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3, axis="y")

        axes[1].bar(range(n_segments), cross_info["segment_diversity"])
        axes[1].axhline(
            cross_info["segment_diversity"].mean(), color="r", ls="--",
            label=f'Mean: {cross_info["segment_diversity"].mean():.1f}',
        )
        axes[1].set_title("Cluster Diversity per Segment", fontsize=13,
                           fontweight="bold")
        axes[1].set_xlabel("Segment")
        axes[1].set_ylabel("Unique Clusters")
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        _save(fig, out_dir, "cross_video_distribution")


# ── 9. Cluster-segment co-occurrence heatmap ─────────────────

def plot_cluster_segment_heatmap(
    cross_info: dict,
    out_dir: Path,
    max_clusters: int = 50,
) -> None:
    """Heatmap of cluster presence across segments (if ≤ max_clusters)."""
    n_clusters = len(cross_info["unique_clusters"])
    if n_clusters > max_clusters:
        logger.info(
            "Skipping heatmap: %d clusters > max_clusters=%d",
            n_clusters, max_clusters,
        )
        return

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.heatmap(
            cross_info["cluster_segment_matrix"],
            cmap="YlOrRd",
            cbar_kws={"label": "Presence"},
            yticklabels=cross_info["unique_clusters"],
            ax=ax,
        )
        ax.set_title("Cluster–Segment Co-occurrence", fontsize=14,
                      fontweight="bold")
        ax.set_xlabel("Segment ID", fontsize=12)
        ax.set_ylabel("Cluster ID", fontsize=12)
        fig.tight_layout()
        _save(fig, out_dir, "cluster_segment_heatmap")


def plot_umap_search_heatmap(
    search_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Heatmap of mean DBCV scores over UMAP search grid."""
    if search_df.empty:
        return

    pivot = search_df.pivot_table(
        index="n_neighbors", columns="min_dist", values="mean_dbcv",
    )
    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(
            pivot,
            cmap="viridis",
            annot=True,
            fmt=".3f",
            cbar_kws={"label": "Mean DBCV"},
            ax=ax,
        )
        ax.set_title("UMAP Grid Search (DBCV)", fontsize=14, fontweight="bold")
        ax.set_xlabel("min_dist")
        ax.set_ylabel("n_neighbors")
        fig.tight_layout()
        _save(fig, out_dir, "umap_search_dbcv_heatmap")


def plot_noise_distribution(
    labels: np.ndarray,
    out_dir: Path,
    stem: str,
) -> None:
    """Bar chart of noise vs non-noise frame counts for HDBSCAN runs."""
    n_noise = int(np.sum(labels == -1))
    n_total = int(len(labels))
    n_non_noise = n_total - n_noise

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["clustered", "noise"], [n_non_noise, n_noise], color=["#4c72b0", "#dd8452"])
        ax.set_title("HDBSCAN Label Composition", fontsize=13, fontweight="bold")
        ax.set_ylabel("Frames")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        _save(fig, out_dir, stem)


def plot_hdbscan_search_heatmap(
    search_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Heatmap of HDBSCAN search scores over min_cluster_size x min_samples."""
    if search_df.empty:
        return

    pivot = search_df.pivot_table(
        index="min_cluster_size", columns="min_samples", values="score",
    )
    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(
            pivot,
            cmap="magma",
            annot=True,
            fmt=".3f",
            cbar_kws={"label": "Search Score"},
            ax=ax,
        )
        score_name = str(search_df["scoring"].iloc[0]) if "scoring" in search_df.columns else "score"
        ax.set_title(f"HDBSCAN Grid Search ({score_name})", fontsize=14, fontweight="bold")
        ax.set_xlabel("min_samples")
        ax.set_ylabel("min_cluster_size")
        fig.tight_layout()
        _save(fig, out_dir, "hdbscan_search_heatmap")
