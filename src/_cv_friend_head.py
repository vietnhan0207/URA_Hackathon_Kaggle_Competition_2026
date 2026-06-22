"""Head-to-head: friend's HybridProductExtractor (0.6616 RapidOCR pipeline) vs our
CalibratedRuleHead, BOTH predicting on OUR vietocr_ft OCR. Same OCR -> ocr_term is
identical, so any composite gap is PURE product token-F1. If his head wins, porting
it onto our (better-CER) VietOCR could beat 0.6685.

Fold-safe: every head refit on the training fold only.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from data import load_train_labels
from empty_gate import EmptyGate
from product_calibrated import CalibratedRuleHead
from run_ocr import cache_path
from scoring import cer, token_f1

import friend_dispatcher as fd


def _norm(p):  # friend emits MISSING_PRODUCT=" " for empty; treat as ""
    s = "" if p is None else str(p)
    return "" if s.strip() == "" else s


def _gkey(text, image_id):
    t = " ".join(str(text).lower().split())
    return "empty_" + str(image_id) if not t else hashlib.md5(t.encode("utf-8")).hexdigest()


labels = load_train_labels()
ocr = pd.read_parquet(cache_path("vietocr_ft", "all"))
df = labels.merge(ocr, on="image_id", suffixes=("_gt", "_ocr")).reset_index(drop=True)
df["ocr_text_ocr"] = df["ocr_text_ocr"].fillna("")
groups = [_gkey(t, i) for i, t in zip(df.image_id, df.ocr_text_gt)]
gkf = GroupKFold(n_splits=5)

ours_gt, ours_pred, friend_pred, pooled_cer = [], [], [], []
for tr_idx, va_idx in gkf.split(df, groups=groups):
    trd, vad = df.iloc[tr_idx], df.iloc[va_idx]
    tr_fit = trd.rename(columns={"ocr_text_gt": "ocr_text"})[["image_id", "ocr_text", "product_name"]]

    # ours: calibrated + empty-gate (exactly the 0.6685 product head)
    head = CalibratedRuleHead(use_classifier_fallback=True, min_pprod=0.55,
                              gate_threshold=0.75).fit(tr_fit)
    tg = trd.copy(); tg["gt_empty"] = (tg.ocr_text_gt.str.strip() == "").astype(int)
    eg = EmptyGate(threshold=0.6).fit(tg.rename(columns={"ocr_text_ocr": "ocr_text"}), tg.gt_empty)
    mask = eg.is_empty(vad.rename(columns={"ocr_text_ocr": "ocr_text"}))
    ocr_in = vad.ocr_text_ocr.where(~np.asarray(mask), "")
    ours_pred += [_norm(p) for p in head.predict_batch(ocr_in)]

    # friend: his full HybridProductExtractor (own internal gating), raw val OCR
    fh = fd.HybridProductExtractor(random_state=42).fit(tr_fit)
    friend_pred += [_norm(fh.predict(t)) for t in vad.ocr_text_ocr]

    ours_gt += list(vad.product_name)
    pooled_cer += [cer(g, p) for g, p in zip(vad.ocr_text_gt, ocr_in)]

f1_ours = np.mean([token_f1(g, p) for g, p in zip(ours_gt, ours_pred)])
f1_fr = np.mean([token_f1(g, p) for g, p in zip(ours_gt, friend_pred)])
ot = 1 - np.mean(pooled_cer)   # identical OCR for both
fill_ours = np.mean([1 if str(p).strip() else 0 for p in ours_pred])
fill_fr = np.mean([1 if str(p).strip() else 0 for p in friend_pred])

print("Both heads on OUR vietocr_ft OCR (5-fold, fold-safe). ocr_term identical.\n")
print(f"{'head':<26}{'composite':>10}{'prod_F1':>9}{'ocr_term':>10}{'p_fill':>8}")
print(f"{'ours (calib+gate)':<26}{0.6*f1_ours+0.4*ot:>10.4f}{f1_ours:>9.4f}{ot:>10.4f}{fill_ours:>8.3f}")
print(f"{'friend HybridExtractor':<26}{0.6*f1_fr+0.4*ot:>10.4f}{f1_fr:>9.4f}{ot:>10.4f}{fill_fr:>8.3f}")
print(f"\nprod-F1 delta (friend - ours): {f1_fr - f1_ours:+.4f}  "
      f"-> composite delta {0.6*(f1_fr - f1_ours):+.4f}")
