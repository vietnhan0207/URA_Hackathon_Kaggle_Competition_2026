"""LB-tailored product head: calibrated dominant-family rules (most-specific
first, each emitting the train-optimal token-F1 string), with our classifier as
fallback for the long tail. Designed to maximize PUBLIC-LB product F1 on the
test set, which is ~53% concentrated in these families.

fit() recomputes each rule's emit string from the TRAINING fold only, so CV is
honest (no peeking at val labels).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

from product_extract import ProductExtractor
from scoring import token_f1


def fold(s):
    s = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


# (name, folded-OCR pattern) — ORDER MATTERS: most-specific first.
SIG_PATTERNS = [
    ("halong_canfoco_pate_cotden",
     r"(canfoco|canfuco|cafoco).*(pate|cot den).*(cot den|hai phong)|(pate|cot den).*(canfoco|canfuco)"),
    ("nan_optipro",   r"\bnan\b.*opti ?pro|opti ?pro.*\bnan\b"),
    ("nan_infinipro", r"\bnan\b.*infini ?pro|infini ?pro.*\bnan\b"),
    ("pate_cotden",   r"\bpate\b.*\b(cot den|hai phong)\b|\b(cot den|hai phong)\b.*\bpate\b|\bcot den\b"),
    ("halong_canfoco", r"\bcanfoco\b|\bcanfuco\b|\bcafoco\b|ha long canfoco|halong canfoco"),
    ("do_hop_ha_long", r"do hop ha long|do hop.*ha long|cong ty.*do hop.*ha long"),
    ("nan",           r"\bnan\b"),
    ("highlands",     r"highlands? coffee|highlands"),
    ("nestle",        r"\bnestle\b"),
]
MIN_PPROD = 0.55   # only deploy rules whose train precision clears this


class CalibratedRuleHead:
    def __init__(self, use_classifier_fallback=True, min_pprod=MIN_PPROD,
                 min_class_count=12, gate_threshold=0.55):
        self.use_clf = use_classifier_fallback
        self.min_pprod = min_pprod
        self._clf = ProductExtractor(min_class_count=min_class_count,
                                     gate_threshold=gate_threshold)
        self.rules = []   # (name, compiled_pat, emit_form, p_prod)

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        df["folded_ocr"] = df["ocr_text"].map(fold)
        df["prod"] = df["product_name"].astype(str).str.strip()
        remaining = df
        self.rules = []
        for name, pat in SIG_PATTERNS:
            m = remaining["folded_ocr"].str.contains(pat, regex=True, na=False)
            sub = remaining[m]
            if len(sub) >= 8:
                p_prod = (sub["prod"] != "").mean()
                forms = Counter(sub[sub["prod"] != ""]["prod"])
                cands = [f for f, c in forms.items() if c >= 3] or list(forms)
                gts = list(sub["prod"])
                best_form, best_val = "", -1.0
                for cand in cands:
                    v = np.mean([token_f1(g, cand) for g in gts])
                    if v > best_val:
                        best_val, best_form = v, cand
                empty_base = (sub["prod"] == "").mean()
                if best_form and (best_val - empty_base) > 0 and p_prod >= self.min_pprod:
                    self.rules.append((name, re.compile(pat), best_form, p_prod))
            remaining = remaining[~m]
        if self.use_clf:
            self._clf.fit(df[["image_id", "ocr_text", "product_name"]]
                          if "image_id" in df else df)
        return self

    def _rule(self, folded):
        for _name, pat, form, _p in self.rules:
            if pat.search(folded):
                return form
        return ""

    def predict(self, ocr_text):
        t = "" if ocr_text is None else str(ocr_text).strip()
        if not t:
            return ""
        f = fold(t)
        hit = self._rule(f)
        if hit:
            return hit
        if self.use_clf:
            return self._clf.predict(t)
        return ""

    def predict_batch(self, texts):
        return [self.predict(t) for t in texts]
