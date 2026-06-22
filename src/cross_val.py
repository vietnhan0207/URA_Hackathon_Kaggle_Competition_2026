"""Grouped k-fold cross-validation of the full composite — a trustworthy estimate
that predicts the private LB far better than our single 20% val split.

- OCR/CER half: uses the cached OCR (fixed per image, no training).
- Product half: properly CV'd — product head trained on k-1 folds' GT text,
  predicted on the held-out fold's REAL OCR text (matches the deployed pipeline).
- Grouped by normalized OCR-GT text so near-duplicate thumbnails stay in one fold.

Returns pooled composite (best estimate) + per-fold mean/std (variance ~ public/private gap).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from data import load_train_labels
from empty_gate import EmptyGate
from product_extract import ProductExtractor
from run_ocr import cache_path
from scoring import cer, token_f1


def _group_key(text: str, image_id: str) -> str:
    t = " ".join(str(text).lower().split())
    return "empty_" + str(image_id) if not t else hashlib.md5(t.encode("utf-8")).hexdigest()


def cv_composite(engine: str = "vietocr_ft", min_class_count: int = 12,
                 gate_threshold: float = 0.55, empty_gate: bool = False,
                 gate_thr: float = 0.6, n_folds: int = 5) -> dict:
    labels = load_train_labels()
    ocr = pd.read_parquet(cache_path(engine, "all"))
    ocr["n_chars"] = ocr["ocr_text"].fillna("").str.len()
    df = labels.merge(ocr, on="image_id", suffixes=("_gt", "_ocr")).reset_index(drop=True)
    df["ocr_text_ocr"] = df["ocr_text_ocr"].fillna("")
    groups = [_group_key(t, i) for i, t in zip(df.image_id, df.ocr_text_gt)]

    gkf = GroupKFold(n_splits=n_folds)
    pooled_pred, pooled_gt, pooled_cer = [], [], []
    fold_scores = []

    for tr_idx, va_idx in gkf.split(df, groups=groups):
        trd, vad = df.iloc[tr_idx], df.iloc[va_idx]
        ext = ProductExtractor(min_class_count=min_class_count, gate_threshold=gate_threshold)
        ext.fit(trd.rename(columns={"ocr_text_gt": "ocr_text"})[["image_id", "ocr_text", "product_name"]])

        ocr_in = vad["ocr_text_ocr"].copy()
        if empty_gate:
            tg = trd.copy()
            tg["gt_empty"] = (tg["ocr_text_gt"].str.strip() == "").astype(int)
            eg = EmptyGate(threshold=gate_thr).fit(
                tg.rename(columns={"ocr_text_ocr": "ocr_text"}), tg["gt_empty"])
            mask = eg.is_empty(vad.rename(columns={"ocr_text_ocr": "ocr_text"}))
            ocr_in = ocr_in.where(~np.asarray(mask), "")

        preds = ext.predict_batch(ocr_in)
        f1s = [token_f1(g, p) for g, p in zip(vad["product_name"], preds)]
        cers = [cer(g, p) for g, p in zip(vad["ocr_text_gt"], ocr_in)]
        fold_scores.append(0.6 * np.mean(f1s) + 0.4 * (1 - np.mean(cers)))
        pooled_pred += list(preds)
        pooled_gt += list(vad["product_name"])
        pooled_cer += cers

    f1 = float(np.mean([token_f1(g, p) for g, p in zip(pooled_gt, pooled_pred)]))
    ocr_term = 1 - float(np.mean(pooled_cer))
    return {
        "composite": round(0.6 * f1 + 0.4 * ocr_term, 4),
        "f1": round(f1, 4), "ocr_term": round(ocr_term, 4),
        "fold_mean": round(float(np.mean(fold_scores)), 4),
        "fold_std": round(float(np.std(fold_scores)), 4),
        "folds": [round(float(s), 4) for s in fold_scores],
    }
