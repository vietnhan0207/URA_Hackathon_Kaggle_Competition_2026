"""Offline evaluation helpers for OCR caches and full submissions.

Phase 2 use: score an engine's cached OCR (CER term) on the val split, and
compare engines head-to-head. Product F1 comes later (Phase 4); here we isolate
OCR quality so engine choice is driven by CER alone.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from data import load_split_ids, load_train_labels
from run_ocr import cache_path
from scoring import cer, composite_score, per_row_scores


def load_ocr_cache(engine: str, split: str) -> pd.DataFrame:
    path = cache_path(engine, split)
    if not path.exists():
        raise FileNotFoundError(f"No cache: {path} (run run_ocr.py first)")
    return pd.read_parquet(path)


def score_ocr_on_split(engine: str, split: str = "val") -> dict:
    """CER term for one engine's cached OCR on a labeled split (train/val)."""
    labels = load_train_labels()
    keep = load_split_ids(split) if split in ("train", "val") else set(labels.image_id)
    gt = labels[labels.image_id.isin(keep)][["image_id", "ocr_text", "product_name"]].copy()

    cache = load_ocr_cache(engine, split if cache_path(engine, split).exists() else "all")
    cache = cache[cache.image_id.isin(keep)]

    # Score only images actually OCR'd (supports partial bake-off caches).
    gt = gt[gt.image_id.isin(set(cache.image_id))]
    merged = gt.merge(cache[["image_id", "ocr_text"]], on="image_id",
                      suffixes=("_gt", "_pred"), how="left")
    merged["ocr_text_pred"] = merged["ocr_text_pred"].fillna("")
    merged["row_cer"] = merged.apply(lambda r: cer(r["ocr_text_gt"], r["ocr_text_pred"]), axis=1)

    avg_cer = merged["row_cer"].mean()
    # split CER by whether GT text is empty (gate diagnostics)
    gt_empty = merged["ocr_text_gt"].str.strip() == ""
    return {
        "engine": engine,
        "n": len(merged),
        "avg_cer": round(float(avg_cer), 4),
        "ocr_term": round(float(1 - avg_cer), 4),
        "cer_on_nonempty_gt": round(float(merged.loc[~gt_empty, "row_cer"].mean()), 4),
        "cer_on_empty_gt": round(float(merged.loc[gt_empty, "row_cer"].mean()), 4),
        "pred_fill": round(float((merged["ocr_text_pred"].str.strip() != "").mean()), 4),
    }


def compare_engines(engines: list[str], split: str = "val") -> pd.DataFrame:
    rows = []
    for e in engines:
        try:
            rows.append(score_ocr_on_split(e, split))
        except FileNotFoundError as ex:
            rows.append({"engine": e, "n": 0, "avg_cer": None, "note": str(ex)})
    return pd.DataFrame(rows).sort_values("ocr_term", ascending=False, na_position="last")


def score_submission(pred_df: pd.DataFrame, split: str = "val") -> dict:
    """Full composite (product + OCR) for a prediction df on a labeled split."""
    labels = load_train_labels()
    keep = load_split_ids(split) if split in ("train", "val") else set(labels.image_id)
    gt = labels[labels.image_id.isin(keep)][["image_id", "ocr_text", "product_name"]]
    pred = pred_df[pred_df.image_id.isin(keep)][["image_id", "ocr_text", "product_name"]]
    return composite_score(gt, pred, return_components=True)
