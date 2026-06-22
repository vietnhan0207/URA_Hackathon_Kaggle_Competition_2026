"""Honest CV of the LB-tailored calibrated head vs classifier vs friend-rules.
Fold-safe: emit strings refit on each training fold only.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from data import load_train_labels
from empty_gate import EmptyGate
from product_extract import ProductExtractor
from product_hybrid import HybridProductExtractor
from product_calibrated import CalibratedRuleHead
from run_ocr import cache_path
from scoring import cer, token_f1


def _gkey(text, image_id):
    t = " ".join(str(text).lower().split())
    return "empty_" + str(image_id) if not t else hashlib.md5(t.encode("utf-8")).hexdigest()


def make(kind):
    if kind == "classifier":
        return ProductExtractor(min_class_count=12, gate_threshold=0.55)
    if kind == "calib_p55":
        return CalibratedRuleHead(use_classifier_fallback=False, min_pprod=0.55)
    if kind == "calib_p70":
        return CalibratedRuleHead(use_classifier_fallback=False, min_pprod=0.70)
    if kind == "calib_p85":
        return CalibratedRuleHead(use_classifier_fallback=False, min_pprod=0.85)
    if kind == "calib_p55+strictclf":   # rules + strict long-tail classifier fallback
        return CalibratedRuleHead(use_classifier_fallback=True, min_pprod=0.55,
                                  gate_threshold=0.75)
    raise ValueError(kind)


labels = load_train_labels()
ocr = pd.read_parquet(cache_path("vietocr_ft", "all"))
df = labels.merge(ocr, on="image_id", suffixes=("_gt", "_ocr")).reset_index(drop=True)
df["ocr_text_ocr"] = df["ocr_text_ocr"].fillna("")
groups = [_gkey(t, i) for i, t in zip(df.image_id, df.ocr_text_gt)]
gkf = GroupKFold(n_splits=5)

print("Full composite on real vietocr_ft OCR (5-fold, empty-gate on):\n")
print(f"{'head':<20}{'composite':>10}{'F1':>8}{'ocr_term':>10}{'p_fill':>8}")
for kind in ("classifier", "calib_p55", "calib_p70", "calib_p85", "calib_p55+strictclf"):
    pooled_gt, pooled_pred, pooled_cer = [], [], []
    fills = []
    for tr_idx, va_idx in gkf.split(df, groups=groups):
        trd, vad = df.iloc[tr_idx], df.iloc[va_idx]
        head = make(kind).fit(
            trd.rename(columns={"ocr_text_gt": "ocr_text"})[["image_id", "ocr_text", "product_name"]])
        tg = trd.copy(); tg["gt_empty"] = (tg.ocr_text_gt.str.strip() == "").astype(int)
        eg = EmptyGate(threshold=0.6).fit(tg.rename(columns={"ocr_text_ocr": "ocr_text"}), tg.gt_empty)
        mask = eg.is_empty(vad.rename(columns={"ocr_text_ocr": "ocr_text"}))
        ocr_in = vad.ocr_text_ocr.where(~np.asarray(mask), "")
        preds = head.predict_batch(ocr_in)
        pooled_pred += list(preds); pooled_gt += list(vad.product_name)
        pooled_cer += [cer(g, p) for g, p in zip(vad.ocr_text_gt, ocr_in)]
        fills.append(np.mean([1 if str(p).strip() else 0 for p in preds]))
    f1 = np.mean([token_f1(g, p) for g, p in zip(pooled_gt, pooled_pred)])
    ot = 1 - np.mean(pooled_cer)
    print(f"{kind:<20}{0.6*f1+0.4*ot:>10.4f}{f1:>8.4f}{ot:>10.4f}{np.mean(fills):>8.3f}")
