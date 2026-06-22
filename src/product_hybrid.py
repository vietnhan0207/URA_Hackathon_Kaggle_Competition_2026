"""Hybrid product head: friend's rules-first + canonicalize + evidence-gate,
with our trained classifier as the fallback (only when rules don't fire and OCR
supports the label). Drop-in replacement for ProductExtractor (predict/predict_batch).

Modes (set in __init__):
  use_classifier_fallback=True  -> rules first, then classifier-as-fallback
  use_classifier_fallback=False -> rules only (mirrors friend's rules-only path)
"""
from __future__ import annotations

import pandas as pd

from product_extract import ProductExtractor
from product_rules_friend import (
    extract_product,
    safe_product,
    product_supported_by_ocr,
    canonicalize_product_name,
)


class HybridProductExtractor:
    def __init__(self, min_class_count: int = 12, gate_threshold: float = 0.55,
                 use_classifier_fallback: bool = True, order: str = "rules_first"):
        # order: "rules_first"  -> rules, then classifier fallback
        #        "clf_first"     -> classifier, then rules fallback on empty (CV-safe)
        self.use_classifier_fallback = use_classifier_fallback
        self.order = order
        self._clf = ProductExtractor(min_class_count=min_class_count,
                                     gate_threshold=gate_threshold)

    def fit(self, df: pd.DataFrame):
        if self.use_classifier_fallback or self.order == "clf_first":
            self._clf.fit(df)
        return self

    def _rules(self, t):
        ruled = extract_product(t)
        if ruled:
            safe = safe_product(ruled, t)
            if safe:
                return safe
        return ""

    def _classifier(self, t):
        pred = self._clf.predict(t)
        if pred:
            safe = safe_product(pred, t)
            if safe:
                return safe
            return pred  # keep classifier label even if friend-evidence gate is strict
        return ""

    def predict(self, ocr_text: str) -> str:
        t = "" if ocr_text is None else str(ocr_text).strip()
        if not t:
            return ""

        if self.order == "clf_first":
            # Classifier first (our strength), RAW output untouched (no canonicalize
            # mangling). Rules only fill rows the classifier leaves empty.
            pred = self._clf.predict(t)
            if pred:
                return pred
            return self._rules(t)

        # rules_first: rules, then classifier fallback
        ruled = self._rules(t)
        if ruled:
            return ruled
        if self.use_classifier_fallback:
            return self._classifier(t)
        return ""

    def predict_batch(self, texts) -> list[str]:
        return [self.predict(t) for t in texts]
