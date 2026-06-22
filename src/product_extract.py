"""Product-name extraction.

Insight (see product_mining): GT product_name labels are inconsistent for the same
product (e.g. the Ha Long pate cluster appears as 'ĐỒ HỘP HẠ LONG', 'Pate Cột Đèn
Hải Phòng', ...). token-F1 is case-insensitive, so we:
  1. group training labels by lowercased token-set key,
  2. map every label to the MODAL surface form of its group (canonicalization),
  3. train a classifier OCR_text -> canonical form, with a separate "has product" gate.
Features fold diacritics (unidecode) so the classifier is robust to OCR diacritic errors.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from unidecode import unidecode
except Exception:  # pragma: no cover
    def unidecode(s): return s


def fold(s: str) -> str:
    """Diacritic-insensitive, lowercased, alnum-only — for matching & features."""
    s = unidecode(str(s)).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokkey(s: str) -> str:
    return " ".join(sorted(str(s).lower().split()))


class ProductExtractor:
    def __init__(self, min_class_count: int = 3, gate_threshold: float = 0.5,
                 max_features: int = 5000):
        self.min_class_count = min_class_count
        self.gate_threshold = gate_threshold
        self.max_features = max_features
        self.canonical: dict[str, str] = {}
        self._gate = None
        self._clf = None
        self.classes_ = []

    def _canonicalize(self, df: pd.DataFrame) -> pd.Series:
        # modal surface form per token-set group (computed on fit data)
        nonempty = df[df.product_name != ""]
        canon = {}
        for key, grp in nonempty.groupby(nonempty.product_name.map(tokkey)):
            canon[key] = grp.product_name.value_counts().idxmax()
        self.canonical = canon
        return df.product_name.map(lambda p: canon.get(tokkey(p), "") if p else "")

    def fit(self, df: pd.DataFrame):
        df = df[["ocr_text", "product_name"]].copy()
        df["ocr_text"] = df["ocr_text"].astype(str).str.strip()
        df["product_name"] = df["product_name"].astype(str).str.strip()
        df["folded"] = df["ocr_text"].map(fold)
        df["canon"] = self._canonicalize(df)

        # "has product?" gate (binary)
        self._gate = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                      max_features=self.max_features, min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=4.0)),
        ])
        self._gate.fit(df["folded"], (df["canon"] != "").astype(int))

        # multiclass over canonical forms with enough support
        pos = df[(df.folded != "") & (df.canon != "")]
        keep = pos["canon"].value_counts()
        keep = keep[keep >= self.min_class_count].index
        pos = pos[pos["canon"].isin(keep)]
        self.classes_ = sorted(pos["canon"].unique())
        self._clf = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                      max_features=self.max_features, min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=4.0)),
        ])
        if len(pos):
            self._clf.fit(pos["folded"], pos["canon"])
        return self

    def predict(self, ocr_text: str) -> str:
        t = "" if ocr_text is None else str(ocr_text).strip()
        if not t:
            return ""
        f = fold(t)
        if not f or self._gate is None:
            return ""
        classes = list(self._gate.classes_)
        if 1 not in classes:
            return ""
        p1 = self._gate.predict_proba([f])[0][classes.index(1)]
        if p1 < self.gate_threshold:
            return ""
        if self._clf is None or not self.classes_:
            return ""
        return str(self._clf.predict([f])[0])

    def predict_batch(self, texts) -> list[str]:
        return [self.predict(t) for t in texts]
