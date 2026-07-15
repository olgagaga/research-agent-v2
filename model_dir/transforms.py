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
        # Compute and store per-feature mean and std
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        # Guard against zero std to avoid division by zero at transform time
        self.std_[self.std_ == 0] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # Check that fit has been called
        if not hasattr(self, "mean_") or not hasattr(self, "std_"):
            raise RuntimeError("Preprocessor must be fitted before calling transform().")
        # Apply z-score scaling and return float32 array
        X_scaled = (X - self.mean_) / self.std_
        return X_scaled.astype(np.float32)


def resample(X: np.ndarray, y: np.ndarray):
    """No resampling in the baseline — return the train set unchanged."""
    return X, y
