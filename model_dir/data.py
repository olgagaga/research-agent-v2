"""
Dataset loading — FIXED (the agent must not edit this).

Task: rare-event medical detection on the OpenML *mammography* dataset
(11,183 samples, 6 numeric features, ~2.3% positive = breast-cancer
microcalcifications).  A fixed, seeded train/val/test split guarantees every
experiment is scored on identical data — so score deltas reflect the agent's
change, not data reshuffling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

RANDOM_SEED = 1234
_DATASET = "mammography"


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def pos_rate(self) -> float:
        return float(self.y_train.mean())


def load_dataset() -> Dataset:
    """Load mammography and return a fixed, stratified 60/20/20 split.

    Downloads once via OpenML then caches under ~/scikit_learn_data.
    Raw features are returned untouched — all preprocessing lives in the
    agent-editable ``transforms.py``.
    """
    import warnings

    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = fetch_openml(_DATASET, version=1, as_frame=True)

    X = ds.data.to_numpy(dtype=np.float32)
    # Labels are '-1' (negative) / '1' (positive) → {0, 1}.
    y = (ds.target.astype(str).str.strip() == "1").to_numpy().astype(np.int64)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=RANDOM_SEED
    )  # 0.25 * 0.8 = 0.2

    log.info(
        "Loaded %s: train=%d val=%d test=%d  pos_rate(train)=%.4f",
        _DATASET, len(y_train), len(y_val), len(y_test), y_train.mean(),
    )
    return Dataset(X_train, y_train, X_val, y_val, X_test, y_test)
