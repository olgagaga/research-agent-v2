"""
Feature engineering + resampling — AGENT-EDITABLE (tandem with config.yaml).

Contracts (keep them):
    class Preprocessor:
        fit(X_train:np.ndarray, y_train:np.ndarray) -> None
        transform(X:np.ndarray) -> np.ndarray   (2D, float)
    resample(X:np.ndarray, y:np.ndarray) -> (X, y)   # train set only

Baseline: identity preprocessing + no resampling. Prime levers:
standardisation, feature interactions, log transforms, and oversampling the
rare positive class (implement in numpy — no imblearn dependency).
"""

from __future__ import annotations

import numpy as np


class Preprocessor:
    """Identity transform. Fit stores nothing; transform returns X unchanged."""

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X = np.asarray(X_train, dtype=np.float32)
        # Ensure 2D array
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # Compute a robust heuristic for per-feature right-skewness
        # We consider a feature skewed if it's non-negative everywhere and p95 is >> median
        p50 = np.percentile(X, 50, axis=0)
        p95 = np.percentile(X, 95, axis=0)
        mins = X.min(axis=0)
        # avoid division by zero in the ratio test
        eps = 1e-6
        self.log_mask_ = (mins >= 0) & ((p95 / (p50 + eps)) > 10.0)
        # Apply log1p to skewed features for computing mean/std
        X_proc = X.copy()
        if self.log_mask_.any():
            X_proc[:, self.log_mask_] = np.log1p(X_proc[:, self.log_mask_])
        # Compute and store per-feature mean and std on the processed features
        self.mean_ = X_proc.mean(axis=0)
        self.std_ = X_proc.std(axis=0)
        # Guard against zero std to avoid division by zero at transform time
        self.std_[self.std_ == 0] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # Check that fit has been called
        if not hasattr(self, "mean_") or not hasattr(self, "std_"):
            raise RuntimeError("Preprocessor must be fitted before calling transform().")
        # Apply the same log1p mask if present
        X_proc = X.copy()
        if hasattr(self, "log_mask_") and self.log_mask_.any():
            # For safety cast mask to boolean ndarray
            mask = np.asarray(self.log_mask_, dtype=bool)
            # If input has fewer features than mask (shouldn't happen), truncate mask
            if mask.shape[0] != X_proc.shape[1]:
                mask = mask[: X_proc.shape[1]]
            X_proc[:, mask] = np.log1p(X_proc[:, mask])
        # Apply z-score using stored statistics and return float32
        X_scaled = (X_proc - self.mean_) / self.std_
        return X_scaled.astype(np.float32)


def resample(X: np.ndarray, y: np.ndarray):
    """No resampling in the baseline — return the train set unchanged."""
    return X, y
