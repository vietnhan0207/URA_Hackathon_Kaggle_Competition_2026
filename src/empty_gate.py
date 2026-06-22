"""OCR empty-gate: predict whether an image is textless (GT ocr empty) from cached
OCR features, so we emit '' instead of hallucinated text (CER 1.0 -> 0.0 on those).

Trained on labeled train rows that have OCR features. Small but free CER gain
(+~0.006 ocr_term on val at threshold 0.6).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


FEATURES = ["log_boxes", "mean_conf", "log_chars"]


def _featurize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n_chars = df.get("n_chars")
    if n_chars is None:
        n_chars = df["ocr_text"].fillna("").str.len()
    df["log_boxes"] = np.log1p(df["n_boxes"].fillna(0))
    df["log_chars"] = np.log1p(pd.Series(n_chars, index=df.index).fillna(0))
    df["mean_conf"] = df["mean_conf"].fillna(0.0)
    return df


class EmptyGate:
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.clf = None

    def fit(self, ocr_features: pd.DataFrame, gt_empty: pd.Series):
        X = _featurize(ocr_features)[FEATURES]
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, gt_empty)
        return self

    def is_empty(self, ocr_features: pd.DataFrame) -> np.ndarray:
        if self.clf is None:
            return np.zeros(len(ocr_features), dtype=bool)
        X = _featurize(ocr_features)[FEATURES]
        return self.clf.predict_proba(X)[:, 1] >= self.threshold
