# Behavior Clustering Module

Unsupervised discovery of behavioral states from LISBET embeddings using a
two-stage BIRCH pipeline with configurable final clustering.

---

## Overview

| Aspect          | Detail |
|-----------------|--------|
| **Input**       | LISBET embedding directories (`train/`, `test/`), each containing per-segment `features_lisbet_embedding.csv` files |
| **Output**      | Frame-level cluster assignments, quality metrics, temporal coherence analysis, cross-video validation, publication figures |
| **Method**      | Configurable smoothing (none/EMA/median) → RobustScaler → optional UMAP (128d to 15d) → BIRCH subclustering → agglomerative or HDBSCAN refinement |
| **GPU**         | Required for batched nearest-centroid assignment (PyTorch) |
| **Config**      | YAML file (see `configs/default.yaml`) |

## Dependencies

- Python 3.12+, NumPy, pandas, scikit-learn, SciPy, matplotlib, seaborn
- PyTorch (GPU), joblib, PyYAML
- Optional but required for new features: `umap-learn` (UMAP), `hdbscan` (HDBSCAN + DBCV)
- Module `GCCcore/13.3.0 Python/3.12.3 CUDA/12.8.0`

## Quick Start

### SLURM (recommended)

```bash
cd /srv/beegfs/scratch/shares/schaerm/schaer2/video_sam2_pose/humanLISBET-paper
sbatch run_clustering.sh
```

### Interactive

```bash
python -m post_training.behavior_clustering.src.run_clustering \
    --embeddings-dir  /path/to/embeddings \
    --output-dir      post_training/behavior_clustering/results \
    --config          post_training/behavior_clustering/configs/default.yaml \
    --log-level       INFO
```

## Input Directory Structure

```
embeddings/
├── train/
│   ├── segment_001/
│   │   └── features_lisbet_embedding.csv
│   ├── segment_002/
│   │   └── features_lisbet_embedding.csv
│   └── ...
└── test/
    ├── segment_101/
    │   └── features_lisbet_embedding.csv
    └── ...
```

Each CSV has an index column and D numeric embedding columns.

## CLI Arguments

| Argument           | Required | Default | Description |
|--------------------|----------|---------|-------------|
| `--embeddings-dir` | Yes      | —       | Root with `train/` and `test/` subdirs |
| `--output-dir`     | Yes      | —       | Output directory (created if absent) |
| `--config`         | No       | built-in defaults | YAML/JSON config file |
| `--log-level`      | No       | `INFO`  | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

## Configuration Parameters

| Section      | Key                  | Default | Description |
|--------------|----------------------|---------|-------------|
| `global`     | `random_state`       | 42      | Random seed |
| `global`     | `fps`                | 20      | Video frame rate (for bout-length reporting) |
| `preprocessing.smoothing` | `method` | `median` | Smoothing strategy (`none`, `ema`, `median`) |
| `preprocessing.smoothing` | `alpha` | 0.7 | EMA parameter when `method=ema` |
| `preprocessing.smoothing` | `median_window` | 5 | Median window when `method=median` |
| `reduction` | `method` | `none` | Dimensionality reduction method (`none`, `umap`) |
| `reduction.umap` | `n_components` | 15 | Reduced feature dimension |
| `reduction.umap.search` | `enabled` | `false` | Enables UMAP grid search |
| `reduction.validation` | `metric` | `dbcv` | Validation metric for UMAP search |
| `clustering` | `method` | `agglomerative` | Final clustering (`agglomerative`, `hdbscan`) |
| `clustering.hdbscan` | `min_cluster_size` | 10 | HDBSCAN minimum cluster size |
| `birch`      | `threshold`          | 9.0     | CF-tree leaf radius |
| `hierarchy`  | `distance_threshold` | 75.0    | Dendrogram cut height (agglomerative mode) |
| `gpu`        | `batch_size`         | 256     | Frames per GPU batch |
| `evaluation` | `distance_thresholds`| [60–95] | Thresholds for agglomerative metrics sweep |
| `artifacts`  | `mode` | `standard` | Intermediate save policy (`minimal`, `standard`, `verbose`) |

## Output Files

| File | Description |
|------|-------------|
| `mapping_cluster_frame_train.csv` | Frame index, segment name/id, cluster id (train) |
| `mapping_cluster_frame_test.csv`  | Same for test set |
| `clustering_quality_metrics.csv`  | Silhouette, DB, CH across distance thresholds |
| `clustering_quality_chosen.csv`   | Metrics at the chosen distance threshold |
| `temporal_coherence_train.csv`    | Per-segment stability + summary (train) |
| `temporal_coherence_test.csv`     | Same for test |
| `cross_video_train.csv`           | Cluster diversity per train segment |
| `cross_video_test.csv`            | Same for test |
| `birch_subcluster_radii.csv`      | RMS radius and size per subcluster |
| `summary.json`                    | Run metadata, key metrics, timestamps |
| `config_used.yaml`                | Exact config snapshot for reproducibility |
| `birch_model.joblib`              | Fitted BIRCH model |
| `scaler.joblib`                   | Fitted RobustScaler |
| `umap_reducer.joblib`             | Fitted UMAP reducer (if reduction enabled) |
| `umap_grid_search_results.csv`    | Grid-search scores over `n_neighbors` and `min_dist` |
| `umap_grid_search_best.json`      | Selected UMAP parameters and best DBCV score |
| `hdbscan_subcluster_summary.json` | HDBSCAN diagnostics on BIRCH centroids |
| `linkage_matrix.npy`              | Ward linkage matrix |
| `figures/*.{png,pdf}`             | Publication-quality figures (300 dpi) |
| `stages/*`                        | Optional per-stage snapshots and diagnostics |

## Module Layout

```
behavior_clustering/
├── __init__.py
├── README.md                  ← you are here
├── METHODS.md                 ← methodology for publication
├── configs/default.yaml
├── src/
│   ├── run_clustering.py      ← entry point
│   ├── config.py              ← YAML loader + validation
│   ├── data.py                ← embedding loading
│   ├── smoothing.py           ← smoothing methods (none/EMA/median)
│   ├── reduction.py           ← UMAP fit/transform/grid search (DBCV)
│   ├── preprocessing.py       ← RobustScaler
│   ├── clustering.py          ← BIRCH + GPU assignment + agglomerative/HDBSCAN
│   ├── evaluation.py          ← metrics, coherence, cross-video
│   └── visualization.py       ← all figures
├── scripts/
│   └── sweep_birch_threshold.sh
└── notebook_explorer/         ← exploratory notebooks
```

## Further Reading

| Topic | File |
|-------|------|
| Algorithm rationale & references | [METHODS.md](METHODS.md) |
| BIRCH threshold sensitivity sweep | [scripts/sweep_birch_threshold.sh](scripts/sweep_birch_threshold.sh) |
| Exploratory notebooks | [notebook_explorer/](notebook_explorer/) |
