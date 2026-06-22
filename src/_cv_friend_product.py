"""A/B CV: our classifier vs friend's rules-only vs hybrid (rules->classifier).

Two views:
  (1) ORACLE product F1 on GT OCR  -> isolates product-head quality (what the
      friend's Cell 10 reports). GroupKFold by GT-text hash.
  (2) FULL composite on real vietocr_ft OCR -> deployed-pipeline estimate, with
      the SAME empty-gate as cross_val.cv_composite.
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
from run_ocr import cache_path
from scoring import cer, token_f1


def _gkey(text, image_id):
    t = " ".join(str(text).lower().split())
    return "empty_" + str(image_id) if not t else hashlib.md5(t.encode("utf-8")).hexdigest()


def _make_head(kind):
    if kind == "classifier":
        return ProductExtractor(min_class_count=12, gate_threshold=0.55)
    if kind == "rules":
        return HybridProductExtractor(use_classifier_fallback=False)
    if kind == "hybrid":
        return HybridProductExtractor(min_class_count=12, gate_threshold=0.55,
                                      use_classifier_fallback=True, order="rules_first")
    if kind == "clf_first":
        return HybridProductExtractor(min_class_count=12, gate_threshold=0.55,
                                      order="clf_first")
    raise ValueError(kind)


def oracle_product_cv(n_folds=5):
    """Product F1 using GT OCR as input (no OCR errors). Isolates the head."""
    labels = load_train_labels()
    groups = [_gkey(t, i) for i, t in zip(labels.image_id, labels.ocr_text)]
    gkf = GroupKFold(n_splits=n_folds)
    out = {}
    for kind in ("classifier", "rules", "hybrid", "clf_first"):
        pooled_gt, pooled_pred = [], []
        for tr_idx, va_idx in gkf.split(labels, groups=groups):
            trd, vad = labels.iloc[tr_idx], labels.iloc[va_idx]
            head = _make_head(kind).fit(trd[["image_id", "ocr_text", "product_name"]])
            preds = head.predict_batch(vad["ocr_text"])  # GT OCR in
            pooled_pred += list(preds)
            pooled_gt += list(vad["product_name"])
        f1 = float(np.mean([token_f1(g, p) for g, p in zip(pooled_gt, pooled_pred)]))
        fill = float(np.mean([1 if str(p).strip() else 0 for p in pooled_pred]))
        out[kind] = {"oracle_f1": round(f1, 4), "fill": round(fill, 3)}
    return out


def full_composite_cv(engine="vietocr_ft", n_folds=5, empty_gate=True, gate_thr=0.6):
    """Deployed-pipeline composite on real OCR text (matches cross_val.cv_composite)."""
    labels = load_train_labels()
    ocr = pd.read_parquet(cache_path(engine, "all"))
    df = labels.merge(ocr, on="image_id", suffixes=("_gt", "_ocr")).reset_index(drop=True)
    df["ocr_text_ocr"] = df["ocr_text_ocr"].fillna("")
    groups = [_gkey(t, i) for i, t in zip(df.image_id, df.ocr_text_gt)]
    gkf = GroupKFold(n_splits=n_folds)

    out = {}
    for kind in ("classifier", "rules", "hybrid", "clf_first"):
        pooled_gt, pooled_pred, pooled_cer = [], [], []
        fold_scores = []
        for tr_idx, va_idx in gkf.split(df, groups=groups):
            trd, vad = df.iloc[tr_idx], df.iloc[va_idx]
            head = _make_head(kind).fit(
                trd.rename(columns={"ocr_text_gt": "ocr_text"})[["image_id", "ocr_text", "product_name"]])

            ocr_in = vad["ocr_text_ocr"].copy()
            if empty_gate:
                tg = trd.copy()
                tg["gt_empty"] = (tg["ocr_text_gt"].str.strip() == "").astype(int)
                eg = EmptyGate(threshold=gate_thr).fit(
                    tg.rename(columns={"ocr_text_ocr": "ocr_text"}), tg["gt_empty"])
                mask = eg.is_empty(vad.rename(columns={"ocr_text_ocr": "ocr_text"}))
                ocr_in = ocr_in.where(~np.asarray(mask), "")

            preds = head.predict_batch(ocr_in)
            f1s = [token_f1(g, p) for g, p in zip(vad["product_name"], preds)]
            cers = [cer(g, p) for g, p in zip(vad["ocr_text_gt"], ocr_in)]
            fold_scores.append(0.6 * np.mean(f1s) + 0.4 * (1 - np.mean(cers)))
            pooled_pred += list(preds)
            pooled_gt += list(vad["product_name"])
            pooled_cer += cers

        f1 = float(np.mean([token_f1(g, p) for g, p in zip(pooled_gt, pooled_pred)]))
        ocr_term = 1 - float(np.mean(pooled_cer))
        out[kind] = {
            "composite": round(0.6 * f1 + 0.4 * ocr_term, 4),
            "f1": round(f1, 4), "ocr_term": round(ocr_term, 4),
            "fold_mean": round(float(np.mean(fold_scores)), 4),
            "fold_std": round(float(np.std(fold_scores)), 4),
        }
    return out


if __name__ == "__main__":
    print("=== (1) ORACLE product F1 on GT OCR (isolates product head) ===")
    for k, v in oracle_product_cv().items():
        print(f"  {k:>10}: F1 {v['oracle_f1']:.4f} | fill {v['fill']:.3f}")

    print("\n=== (2) FULL composite on real vietocr_ft OCR (deployed estimate) ===")
    for k, v in full_composite_cv().items():
        print(f"  {k:>10}: composite {v['composite']:.4f} | F1 {v['f1']:.4f} | "
              f"ocr_term {v['ocr_term']:.4f} | folds {v['fold_mean']:.4f}±{v['fold_std']:.4f}")
