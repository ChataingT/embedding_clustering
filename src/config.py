"""
YAML/JSON configuration loader and validator for the behavior clustering pipeline.

Loads a config file, validates all keys and value types, and returns a flat
settings dict consumed by run_clustering.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Valid keys ───────────────────────────────────────────────

_VALID_TOP = {
    "global",
    "smoothing",      # backward-compatible legacy section
    "preprocessing",
    "reduction",
    "birch",
    "hierarchy",
    "clustering",
    "gpu",
    "evaluation",
    "artifacts",
}

_VALID_GLOBAL = {"random_state", "fps"}
_VALID_SMOOTHING = {"alpha"}
_VALID_PREPROCESSING = {"smoothing"}
_VALID_PREPROCESSING_SMOOTHING = {"method", "alpha", "median_window"}
_VALID_REDUCTION = {"method", "umap", "validation"}
_VALID_REDUCTION_UMAP = {
    "n_components",
    "n_neighbors",
    "min_dist",
    "metric",
    "random_state",
    "search",
}
_VALID_REDUCTION_UMAP_SEARCH = {
    "enabled",
    "n_neighbors_grid",
    "min_dist_grid",
}
_VALID_REDUCTION_VALIDATION = {"metric", "cv_mode", "n_splits", "n_jobs"}
_VALID_BIRCH = {"threshold", "branching_factor", "batch_size"}
_VALID_HIERARCHY = {"linkage_method", "distance_threshold"}
_VALID_CLUSTERING = {"method", "hdbscan"}
_VALID_CLUSTERING_HDBSCAN = {
    "min_cluster_size",
    "min_samples",
    "cluster_selection_method",
    "metric",
    "search",
}
_VALID_CLUSTERING_HDBSCAN_SEARCH = {
    "enabled",
    "min_cluster_size_grid",
    "min_samples_grid",
    "scoring",
    "n_jobs",
}
_VALID_GPU = {"batch_size"}
_VALID_EVALUATION = {"distance_thresholds"}
_VALID_ARTIFACTS = {"mode", "save_stage_data", "save_stage_plots"}

_VALID_LINKAGE = {"ward", "complete", "average"}
_VALID_SMOOTHING_METHODS = {"none", "ema", "median"}
_VALID_REDUCTION_METHODS = {"none", "umap"}
_VALID_REDUCTION_VALIDATION_METRICS = {"dbcv"}
_VALID_REDUCTION_VALIDATION_CV = {"train_segment_kfold"}
_VALID_CLUSTERING_METHODS = {"agglomerative", "hdbscan"}
_VALID_HDBSCAN_SELECTION = {"eom", "leaf"}
_VALID_HDBSCAN_SEARCH_SCORING = {"dbcv", "relative_validity"}
_VALID_ARTIFACT_MODES = {"minimal", "standard", "verbose"}

# ── Defaults ─────────────────────────────────────────────────

DEFAULTS: dict[str, Any] = {
    "random_state": 42,
    "fps": 20,
    "smoothing_method": "median",
    "smoothing_alpha": 0.7,
    "smoothing_median_window": 5,
    "reduction_method": "none",
    "umap_n_components": 15,
    "umap_n_neighbors": 30,
    "umap_min_dist": 0.1,
    "umap_metric": "euclidean",
    "umap_random_state": 42,
    "umap_search_enabled": False,
    "umap_search_n_neighbors_grid": [15, 30, 60, 90, 120],
    "umap_search_min_dist_grid": [0.0, 0.05, 0.1, 0.2],
    "reduction_validation_metric": "dbcv",
    "reduction_validation_cv_mode": "train_segment_kfold",
    "reduction_validation_n_splits": 5,
    "reduction_validation_n_jobs": 1,
    "birch_threshold": 9.0,
    "birch_branching_factor": 500,
    "birch_batch_size": 10_000,
    "linkage_method": "ward",
    "distance_threshold": 75.0,
    "clustering_method": "agglomerative",
    "hdbscan_min_cluster_size": 10,
    "hdbscan_min_samples": None,
    "hdbscan_cluster_selection_method": "eom",
    "hdbscan_metric": "euclidean",
    "hdbscan_search_enabled": False,
    "hdbscan_search_min_cluster_size_grid": [10, 20, 40, 80],
    "hdbscan_search_min_samples_grid": [5, 10, 20],
    "hdbscan_search_scoring": "relative_validity",
    "hdbscan_search_n_jobs": 1,
    "gpu_batch_size": 256,
    "eval_distance_thresholds": [60, 65, 70, 75, 80, 85, 90, 95],
    "artifact_mode": "standard",
    "save_stage_data": True,
    "save_stage_plots": True,
}


# ── Load ─────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    """Load a YAML or JSON config file. Returns empty dict for None."""
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")

    try:
        import yaml
        cfg = yaml.safe_load(text)
        logger.info("Config loaded (YAML): %s", path)
        return cfg if cfg is not None else {}
    except ImportError:
        logger.debug("PyYAML not available; trying JSON parser")
    except Exception as exc:
        raise ValueError(f"Failed to parse config as YAML: {exc}") from exc

    try:
        cfg = json.loads(text)
        logger.info("Config loaded (JSON): %s", path)
        return cfg
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse config: {exc}\n"
            "Install PyYAML for YAML support: pip install pyyaml"
        ) from exc


# ── Validate ─────────────────────────────────────────────────

def validate_config(cfg: dict) -> None:
    """Raise ValueError on the first invalid key or value."""
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping (dict).")

    unknown = set(cfg) - _VALID_TOP
    if unknown:
        raise ValueError(
            f"Unknown top-level config key(s): {sorted(unknown)}. "
            f"Valid: {sorted(_VALID_TOP)}"
        )

    _check_keys(cfg, "global", _VALID_GLOBAL)
    _check_keys(cfg, "smoothing", _VALID_SMOOTHING)
    _check_keys(cfg, "preprocessing", _VALID_PREPROCESSING)
    _check_nested_keys(
        cfg, "preprocessing", "smoothing", _VALID_PREPROCESSING_SMOOTHING,
    )

    _check_keys(cfg, "reduction", _VALID_REDUCTION)
    _check_nested_keys(cfg, "reduction", "umap", _VALID_REDUCTION_UMAP)
    _check_nested_keys(cfg, "reduction", "validation", _VALID_REDUCTION_VALIDATION)
    _check_nested_deeper_keys(
        cfg,
        section="reduction",
        subsection="umap",
        nested_key="search",
        valid=_VALID_REDUCTION_UMAP_SEARCH,
    )

    _check_keys(cfg, "birch", _VALID_BIRCH)
    _check_keys(cfg, "hierarchy", _VALID_HIERARCHY)
    _check_keys(cfg, "clustering", _VALID_CLUSTERING)
    _check_nested_keys(cfg, "clustering", "hdbscan", _VALID_CLUSTERING_HDBSCAN)
    _check_nested_deeper_keys(
        cfg,
        section="clustering",
        subsection="hdbscan",
        nested_key="search",
        valid=_VALID_CLUSTERING_HDBSCAN_SEARCH,
    )
    _check_keys(cfg, "gpu", _VALID_GPU)
    _check_keys(cfg, "evaluation", _VALID_EVALUATION)
    _check_keys(cfg, "artifacts", _VALID_ARTIFACTS)

    # Value-level checks
    hier = cfg.get("hierarchy", {})
    if "linkage_method" in hier and hier["linkage_method"] not in _VALID_LINKAGE:
        raise ValueError(
            f"Invalid linkage_method '{hier['linkage_method']}'. "
            f"Valid: {sorted(_VALID_LINKAGE)}"
        )

    smooth = cfg.get("smoothing", {})
    if "alpha" in smooth:
        a = smooth["alpha"]
        if not (0 < a <= 1):
            raise ValueError(f"smoothing.alpha must be in (0, 1], got {a}")

    p_sm = cfg.get("preprocessing", {}).get("smoothing", {})
    if "method" in p_sm and p_sm["method"] not in _VALID_SMOOTHING_METHODS:
        raise ValueError(
            f"Invalid preprocessing.smoothing.method '{p_sm['method']}'. "
            f"Valid: {sorted(_VALID_SMOOTHING_METHODS)}"
        )
    if "alpha" in p_sm:
        a = p_sm["alpha"]
        if not (0 < a <= 1):
            raise ValueError(f"preprocessing.smoothing.alpha must be in (0, 1], got {a}")
    if "median_window" in p_sm:
        w = int(p_sm["median_window"])
        if w < 1 or w % 2 == 0:
            raise ValueError(
                f"preprocessing.smoothing.median_window must be a positive odd integer, got {w}"
            )

    reduction = cfg.get("reduction", {})
    if "method" in reduction and reduction["method"] not in _VALID_REDUCTION_METHODS:
        raise ValueError(
            f"Invalid reduction.method '{reduction['method']}'. "
            f"Valid: {sorted(_VALID_REDUCTION_METHODS)}"
        )

    umap_cfg = reduction.get("umap", {})
    if "n_components" in umap_cfg and int(umap_cfg["n_components"]) < 2:
        raise ValueError("reduction.umap.n_components must be >= 2")
    if "n_neighbors" in umap_cfg and int(umap_cfg["n_neighbors"]) < 2:
        raise ValueError("reduction.umap.n_neighbors must be >= 2")
    if "min_dist" in umap_cfg and float(umap_cfg["min_dist"]) < 0:
        raise ValueError("reduction.umap.min_dist must be >= 0")

    search_cfg = umap_cfg.get("search", {})
    if "n_neighbors_grid" in search_cfg:
        nn = search_cfg["n_neighbors_grid"]
        if not isinstance(nn, list) or not nn or not all(int(x) >= 2 for x in nn):
            raise ValueError(
                "reduction.umap.search.n_neighbors_grid must be a non-empty list of integers >= 2"
            )
    if "min_dist_grid" in search_cfg:
        md = search_cfg["min_dist_grid"]
        if not isinstance(md, list) or not md or not all(float(x) >= 0 for x in md):
            raise ValueError(
                "reduction.umap.search.min_dist_grid must be a non-empty list of floats >= 0"
            )

    val_cfg = reduction.get("validation", {})
    if "metric" in val_cfg and val_cfg["metric"] not in _VALID_REDUCTION_VALIDATION_METRICS:
        raise ValueError(
            f"Invalid reduction.validation.metric '{val_cfg['metric']}'. "
            f"Valid: {sorted(_VALID_REDUCTION_VALIDATION_METRICS)}"
        )
    if "cv_mode" in val_cfg and val_cfg["cv_mode"] not in _VALID_REDUCTION_VALIDATION_CV:
        raise ValueError(
            f"Invalid reduction.validation.cv_mode '{val_cfg['cv_mode']}'. "
            f"Valid: {sorted(_VALID_REDUCTION_VALIDATION_CV)}"
        )
    if "n_splits" in val_cfg and int(val_cfg["n_splits"]) < 2:
        raise ValueError("reduction.validation.n_splits must be >= 2")
    if "n_jobs" in val_cfg and int(val_cfg["n_jobs"]) < 1:
        raise ValueError("reduction.validation.n_jobs must be >= 1")

    clustering = cfg.get("clustering", {})
    if "method" in clustering and clustering["method"] not in _VALID_CLUSTERING_METHODS:
        raise ValueError(
            f"Invalid clustering.method '{clustering['method']}'. "
            f"Valid: {sorted(_VALID_CLUSTERING_METHODS)}"
        )
    hdb = clustering.get("hdbscan", {})
    if "min_cluster_size" in hdb and int(hdb["min_cluster_size"]) < 2:
        raise ValueError("clustering.hdbscan.min_cluster_size must be >= 2")
    if "min_samples" in hdb and hdb["min_samples"] is not None and int(hdb["min_samples"]) < 1:
        raise ValueError("clustering.hdbscan.min_samples must be >= 1 when provided")
    if (
        "cluster_selection_method" in hdb
        and hdb["cluster_selection_method"] not in _VALID_HDBSCAN_SELECTION
    ):
        raise ValueError(
            "Invalid clustering.hdbscan.cluster_selection_method "
            f"'{hdb['cluster_selection_method']}'. "
            f"Valid: {sorted(_VALID_HDBSCAN_SELECTION)}"
        )
    hdb_search = hdb.get("search", {})
    if "min_cluster_size_grid" in hdb_search:
        mcg = hdb_search["min_cluster_size_grid"]
        if not isinstance(mcg, list) or not mcg or not all(int(x) >= 2 for x in mcg):
            raise ValueError(
                "clustering.hdbscan.search.min_cluster_size_grid must be a non-empty list of integers >= 2"
            )
    if "min_samples_grid" in hdb_search:
        msg = hdb_search["min_samples_grid"]
        if not isinstance(msg, list) or not msg or not all(int(x) >= 1 for x in msg):
            raise ValueError(
                "clustering.hdbscan.search.min_samples_grid must be a non-empty list of integers >= 1"
            )
    if (
        "scoring" in hdb_search
        and hdb_search["scoring"] not in _VALID_HDBSCAN_SEARCH_SCORING
    ):
        raise ValueError(
            "Invalid clustering.hdbscan.search.scoring "
            f"'{hdb_search['scoring']}'. "
            f"Valid: {sorted(_VALID_HDBSCAN_SEARCH_SCORING)}"
        )
    if "n_jobs" in hdb_search and int(hdb_search["n_jobs"]) < 1:
        raise ValueError("clustering.hdbscan.search.n_jobs must be >= 1")

    ev = cfg.get("evaluation", {})
    if "distance_thresholds" in ev:
        dt = ev["distance_thresholds"]
        if not isinstance(dt, list) or not all(isinstance(x, (int, float)) for x in dt):
            raise ValueError("evaluation.distance_thresholds must be a list of numbers")

    artifacts = cfg.get("artifacts", {})
    if "mode" in artifacts and artifacts["mode"] not in _VALID_ARTIFACT_MODES:
        raise ValueError(
            f"Invalid artifacts.mode '{artifacts['mode']}'. "
            f"Valid: {sorted(_VALID_ARTIFACT_MODES)}"
        )


def _check_keys(cfg: dict, section: str, valid: set[str]) -> None:
    if section not in cfg:
        return
    sub = cfg[section]
    if not isinstance(sub, dict):
        raise ValueError(f"'{section}' must be a mapping.")
    unknown = set(sub) - valid
    if unknown:
        raise ValueError(
            f"Unknown key(s) in {section}: {sorted(unknown)}. "
            f"Valid: {sorted(valid)}"
        )


def _check_nested_keys(cfg: dict, section: str, subsection: str, valid: set[str]) -> None:
    if section not in cfg:
        return
    sub = cfg[section]
    if subsection not in sub:
        return
    nested = sub[subsection]
    if not isinstance(nested, dict):
        raise ValueError(f"'{section}.{subsection}' must be a mapping.")
    unknown = set(nested) - valid
    if unknown:
        raise ValueError(
            f"Unknown key(s) in {section}.{subsection}: {sorted(unknown)}. "
            f"Valid: {sorted(valid)}"
        )


def _check_nested_deeper_keys(
    cfg: dict,
    section: str,
    subsection: str,
    nested_key: str,
    valid: set[str],
) -> None:
    if section not in cfg:
        return
    sub = cfg[section]
    if subsection not in sub:
        return
    sub_nested = sub[subsection]
    if nested_key not in sub_nested:
        return
    deep = sub_nested[nested_key]
    if not isinstance(deep, dict):
        raise ValueError(f"'{section}.{subsection}.{nested_key}' must be a mapping.")
    unknown = set(deep) - valid
    if unknown:
        raise ValueError(
            f"Unknown key(s) in {section}.{subsection}.{nested_key}: {sorted(unknown)}. "
            f"Valid: {sorted(valid)}"
        )


# ── Apply ────────────────────────────────────────────────────

def apply_config(cfg: dict) -> dict[str, Any]:
    """
    Merge a validated config dict with defaults.

    Returns a flat settings dict with all keys that the pipeline needs.
    """
    s = dict(DEFAULTS)

    g = cfg.get("global", {})
    if "random_state" in g:
        s["random_state"] = int(g["random_state"])
    if "fps" in g:
        s["fps"] = int(g["fps"])

    sm = cfg.get("smoothing", {})
    if "alpha" in sm:
        s["smoothing_alpha"] = float(sm["alpha"])

    p_sm = cfg.get("preprocessing", {}).get("smoothing", {})
    if "method" in p_sm:
        s["smoothing_method"] = str(p_sm["method"])
    if "alpha" in p_sm:
        s["smoothing_alpha"] = float(p_sm["alpha"])
    if "median_window" in p_sm:
        s["smoothing_median_window"] = int(p_sm["median_window"])

    red = cfg.get("reduction", {})
    if "method" in red:
        s["reduction_method"] = str(red["method"])
    red_umap = red.get("umap", {})
    if "n_components" in red_umap:
        s["umap_n_components"] = int(red_umap["n_components"])
    if "n_neighbors" in red_umap:
        s["umap_n_neighbors"] = int(red_umap["n_neighbors"])
    if "min_dist" in red_umap:
        s["umap_min_dist"] = float(red_umap["min_dist"])
    if "metric" in red_umap:
        s["umap_metric"] = str(red_umap["metric"])
    if "random_state" in red_umap:
        s["umap_random_state"] = int(red_umap["random_state"])

    red_search = red_umap.get("search", {})
    if "enabled" in red_search:
        s["umap_search_enabled"] = bool(red_search["enabled"])
    if "n_neighbors_grid" in red_search:
        s["umap_search_n_neighbors_grid"] = [int(x) for x in red_search["n_neighbors_grid"]]
    if "min_dist_grid" in red_search:
        s["umap_search_min_dist_grid"] = [float(x) for x in red_search["min_dist_grid"]]

    red_val = red.get("validation", {})
    if "metric" in red_val:
        s["reduction_validation_metric"] = str(red_val["metric"])
    if "cv_mode" in red_val:
        s["reduction_validation_cv_mode"] = str(red_val["cv_mode"])
    if "n_splits" in red_val:
        s["reduction_validation_n_splits"] = int(red_val["n_splits"])
    if "n_jobs" in red_val:
        s["reduction_validation_n_jobs"] = int(red_val["n_jobs"])

    b = cfg.get("birch", {})
    if "threshold" in b:
        s["birch_threshold"] = float(b["threshold"])
    if "branching_factor" in b:
        s["birch_branching_factor"] = int(b["branching_factor"])
    if "batch_size" in b:
        s["birch_batch_size"] = int(b["batch_size"])

    h = cfg.get("hierarchy", {})
    if "linkage_method" in h:
        s["linkage_method"] = str(h["linkage_method"])
    if "distance_threshold" in h:
        s["distance_threshold"] = float(h["distance_threshold"])

    c = cfg.get("clustering", {})
    if "method" in c:
        s["clustering_method"] = str(c["method"])
    c_hdb = c.get("hdbscan", {})
    if "min_cluster_size" in c_hdb:
        s["hdbscan_min_cluster_size"] = int(c_hdb["min_cluster_size"])
    if "min_samples" in c_hdb:
        s["hdbscan_min_samples"] = (
            None if c_hdb["min_samples"] is None else int(c_hdb["min_samples"])
        )
    if "cluster_selection_method" in c_hdb:
        s["hdbscan_cluster_selection_method"] = str(c_hdb["cluster_selection_method"])
    if "metric" in c_hdb:
        s["hdbscan_metric"] = str(c_hdb["metric"])
    c_hdb_search = c_hdb.get("search", {})
    if "enabled" in c_hdb_search:
        s["hdbscan_search_enabled"] = bool(c_hdb_search["enabled"])
    if "min_cluster_size_grid" in c_hdb_search:
        s["hdbscan_search_min_cluster_size_grid"] = [
            int(x) for x in c_hdb_search["min_cluster_size_grid"]
        ]
    if "min_samples_grid" in c_hdb_search:
        s["hdbscan_search_min_samples_grid"] = [
            int(x) for x in c_hdb_search["min_samples_grid"]
        ]
    if "scoring" in c_hdb_search:
        s["hdbscan_search_scoring"] = str(c_hdb_search["scoring"])
    if "n_jobs" in c_hdb_search:
        s["hdbscan_search_n_jobs"] = int(c_hdb_search["n_jobs"])

    gp = cfg.get("gpu", {})
    if "batch_size" in gp:
        s["gpu_batch_size"] = int(gp["batch_size"])

    ev = cfg.get("evaluation", {})
    if "distance_thresholds" in ev:
        s["eval_distance_thresholds"] = [float(x) for x in ev["distance_thresholds"]]

    art = cfg.get("artifacts", {})
    if "mode" in art:
        s["artifact_mode"] = str(art["mode"])
    if "save_stage_data" in art:
        s["save_stage_data"] = bool(art["save_stage_data"])
    if "save_stage_plots" in art:
        s["save_stage_plots"] = bool(art["save_stage_plots"])

    return s
