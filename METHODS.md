# Behavior Clustering — Methods

This document describes the clustering methodology at a level suitable for a
publication methods section.

---

## 1. Overview

Behavioral states are discovered from LISBET self-supervised embeddings using a
configurable two-stage unsupervised clustering approach:

1. **BIRCH subclustering** compresses millions of frame-level embeddings into a
  compact set of subcluster centroids.
2. Final refinement is selected by configuration:
  - **Agglomerative (Ward/complete/average)** on subcluster centroids.
  - **HDBSCAN** on subcluster centroids for density-based clusters with
    optional noise labeling.

This pipeline is **scalable** (O(N) in BIRCH, O(M² log M) in Ward where
M ≪ N) and **deterministic** given fixed input ordering.

## 2. Preprocessing

### 2.1 Configurable Smoothing

Frame-level LISBET embeddings can exhibit high-frequency noise from minor pose
jitter that does not correspond to genuine behavioral transitions. The pipeline
supports three smoothing choices:

- **none**: no temporal smoothing
- **ema**: bidirectional exponential smoothing
- **median**: centered rolling median with odd window size

For each embedding dimension, the forward pass is:

$$s_t^{\text{fwd}} = \alpha \cdot x_t + (1 - \alpha) \cdot s_{t-1}^{\text{fwd}}$$

and the backward pass:

$$s_t^{\text{bwd}} = \alpha \cdot x_t + (1 - \alpha) \cdot s_{t+1}^{\text{bwd}}$$

The smoothed output is the average $\hat{x}_t = \frac{1}{2}(s_t^{\text{fwd}} + s_t^{\text{bwd}})$.

The smoothing parameter α controls the trade-off between noise reduction and
temporal fidelity. Lower values produce heavier smoothing. We select α by
monitoring the *temporal energy ratio*:

$$r(\alpha) = \frac{\bar{E}_{\text{smoothed}}}{\bar{E}_{\text{raw}}}, \quad \bar{E} = \frac{1}{T-1}\sum_{t=1}^{T-1} \|x_{t+1} - x_t\|_2$$

An energy ratio in the range 0.5–0.7 indicates that frame-level noise is
suppressed while behavior-timescale dynamics are preserved.

**Rationale.** Bidirectional smoothing avoids the phase lag inherent in
forward-only exponential smoothing. This is critical for behavioral annotation
where the precise onset/offset timing of states matters.

### 2.2 Robust Scaling

Embeddings are standardised using `RobustScaler` (median and interquartile
range). This is preferred over `StandardScaler` (mean/std) because:

- Behavioral embedding spaces contain heavy-tailed outliers from rare postures
  or atypical movement patterns.
- Outlier frames would inflate the standard deviation and compress the scale of
  the majority of data, distorting distance-based clustering.
- RobustScaler's median/IQR normalisation is insensitive to these extremes,
  producing a more faithful distance geometry.

The scaler is fitted on the training set only and applied to both train and
test to prevent information leakage.

### 2.3 Optional UMAP Reduction and DBCV Model Selection

Before BIRCH, embeddings may be reduced from 128 dimensions to 15 dimensions
using UMAP. If enabled, a grid search over `n_neighbors` and `min_dist` is
performed and validated by **DBCV** under train-segment cross-validation.

The selected parameter set maximizes mean DBCV score across folds. This step
can improve downstream cluster stability while reducing computational cost.

## 3. Clustering

### 3.1 Stage 1: BIRCH Subclustering

**BIRCH** (Balanced Iterative Reducing and Clustering using Hierarchies;
Zhang et al., 1996) builds a Clustering Feature (CF) tree that summarises the
data into subclusters. Each leaf node maintains sufficient statistics
(count, linear sum, squared sum) to represent its members compactly.

Key properties:
- **Single-pass, incremental:** data is processed in batches via `partial_fit`,
  enabling clustering of datasets that do not fit in memory.
- **Deterministic:** the CF-tree structure depends only on the insertion order,
  which is fixed by sorted directory iteration.
- The **threshold** parameter controls the maximum radius of a leaf
  subcluster — it is the primary granularity knob.

The BIRCH stage reduces N frames (typically 10⁵–10⁶) to M subclusters
(typically 10²–10³), making the subsequent hierarchical step tractable.

### 3.2 Stage 2: Ward Agglomerative Clustering

The M subcluster centroids from BIRCH are hierarchically merged using Ward's
minimum variance linkage (Ward, 1963; Murtagh & Contreras, 2012). Ward linkage
minimises the total within-cluster variance at each merge step:

$$\Delta(A, B) = \frac{n_A n_B}{n_A + n_B} \| \bar{x}_A - \bar{x}_B \|^2$$

The resulting dendrogram is cut at a user-specified distance threshold to
produce cluster labels.

**Rationale for Ward.** Ward linkage tends to produce compact, roughly
equally-sized clusters — a desirable property for behavioral states where
extreme size imbalance would indicate poor cluster separation. It is the
preferred linkage for Euclidean data when cluster compactness matters.

### 3.3 Label Assignment

Frame-level cluster labels are obtained in two steps:

1. Each frame is assigned to its nearest BIRCH subcluster centroid via batched
   squared-Euclidean distance on GPU (PyTorch).
2. The subcluster-level Ward labels are mapped to frames:
   `final_label[i] = ward_label[birch_label[i]]`.

This avoids recomputing distances against all clusters and scales linearly in
the number of frames.

## 4. Evaluation

### 4.1 Internal Clustering Quality

Three complementary metrics are computed on the subcluster centroids (not raw
frames, which would be O(N²)):

| Metric | Interpretation | Reference |
|--------|---------------|-----------|
| **Silhouette Score** | Cohesion vs separation; range [−1, 1]; higher is better | Rousseeuw, 1987 |
| **Davies-Bouldin Index** | Average cluster similarity; lower is better | Davies & Bouldin, 1979 |
| **Calinski-Harabasz Score** | Between/within dispersion ratio; higher is better | Calinski & Harabasz, 1974 |

Metrics are swept over a range of distance thresholds to visualise the
sensitivity of cluster quality to the dendrogram cut height.

### 4.2 Temporal Coherence

Behavioral states should form sustained *bouts* rather than flickering
frame-by-frame. We quantify this via:

- **Switch rate:** fraction of consecutive frames (excluding segment
  boundaries) that change cluster identity.
- **Stability score:** 1 − switch_rate (higher = more temporally coherent).
- **Bout length statistics:** mean, median, and standard deviation of
  consecutive runs of the same cluster label.

These are reported per video segment and as global summaries for both train
and test sets.

### 4.3 Cross-Video Cluster Generalisation

Good behavioral clusters should represent shared behaviors across multiple
video segments, not be idiosyncratic to a single recording. We analyse:

- **Cluster prevalence:** how many segments each cluster appears in.
- **Segment diversity:** how many unique clusters each segment contains.
- **Segment-specific clusters:** clusters appearing in only one segment
  (>30% of total clusters is flagged as potential overfitting).

## 5. Parameter Selection Guidelines

| Parameter | How to choose |
|-----------|---------------|
| Smoothing α | Monitor temporal energy ratio; target 0.5–0.7 |
| BIRCH threshold | Inspect subcluster count and radii distribution; should produce 100–2000 subclusters with radii below the threshold |
| Distance threshold | Inspect dendrogram, merge-distance curve, and metrics-vs-threshold plot; choose the elbow or the region maximising silhouette |
| BIRCH branching factor | Keep at 500 unless memory is constrained; higher = fewer tree levels |

## References

- Calinski, T. & Harabasz, J. (1974). A dendrite method for cluster analysis.
  *Communications in Statistics*, 3(1), 1–27.
- Davies, D.L. & Bouldin, D.W. (1979). A cluster separation measure.
  *IEEE TPAMI*, 1(2), 224–227.
- Murtagh, F. & Contreras, P. (2012). Algorithms for hierarchical clustering:
  an overview. *WIREs Data Mining and Knowledge Discovery*, 2(1), 86–97.
- Rousseeuw, P.J. (1987). Silhouettes: a graphical aid to the interpretation
  and validation of cluster analysis. *JCAM*, 20, 53–65.
- Ward, J.H. (1963). Hierarchical grouping to optimize an objective function.
  *JASA*, 58(301), 236–244.
- Zhang, T., Ramakrishnan, R. & Livny, M. (1996). BIRCH: an efficient data
  clustering method for very large databases. *ACM SIGMOD Record*, 25(2), 103–114.
