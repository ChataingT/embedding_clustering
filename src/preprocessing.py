"""
Feature scaling for the behavior clustering pipeline.

Uses RobustScaler (median / IQR) which is insensitive to outlier frames
caused by rare or atypical behaviours.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


def fit_scaler(X_train: np.ndarray) -> RobustScaler:
    """Fit a RobustScaler on the training data."""
    scaler = RobustScaler()
    scaler.fit(X_train)
    logger.info("RobustScaler fitted on %d samples × %d features", *X_train.shape)
    return scaler


def scale_data(
    scaler: RobustScaler,
    X_train: np.ndarray,
    X_test: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Transform train (and optionally test) arrays with a fitted scaler.

    Returns
    -------
    X_train_scaled : np.ndarray
    X_test_scaled : np.ndarray or None
    """
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test).astype(np.float32)
    return X_train_scaled, X_test_scaled


def save_scaler(scaler: RobustScaler, path: Path) -> None:
    """Persist the scaler to disk."""
    joblib.dump(scaler, path)
    logger.info("Scaler saved → %s", path)


def load_scaler(path: Path) -> RobustScaler:
    """Load a previously saved scaler."""
    scaler = joblib.load(path)
    logger.info("Scaler loaded ← %s", path)
    return scaler
