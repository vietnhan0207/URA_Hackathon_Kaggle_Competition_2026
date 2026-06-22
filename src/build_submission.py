"""Assemble, validate, and export submission.csv from a test OCR cache + product head.

Usage (env `ura`):
    python build_submission.py --engine paddleocr --tag v1
    python build_submission.py --engine paddleocr --tag v1 --train-on all   # OCR'd train text

- ocr_text  : the cached OCR for each test image (optionally empty-gated)
- product   : ProductExtractor prediction on that OCR text
Runs AC-1..AC-7, writes UTF-8 + QUOTE_ALL, empty -> " ".
"""
from __future__ import annotations

import argparse
import csv

import pandas as pd

import config
from data import load_sample_submission, load_test_ids, load_train_labels
from empty_gate import EmptyGate
from product_extract import ProductExtractor
from run_ocr import cache_path


def write_submission_csv(df: pd.DataFrame, path) -> None:
    out = df[["image_id", "ocr_text", "product_name"]].copy()
    for col in ("ocr_text", "product_name"):
        out[col] = out[col].fillna("").astype(str).str.strip()
        out.loc[out[col] == "", col] = " "   # Kaggle rejects blank cells
    out.to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)


def validate(sub: pd.DataFrame, sample: pd.DataFrame) -> bool:
    expected, got = set(sample.image_id), set(sub.image_id)
    checks = {
        "AC-1 row count": len(sub) == len(sample),
        "AC-2 no extra ids": len(got - expected) == 0,
        "AC-3 no missing ids": len(expected - got) == 0,
        "AC-4 no dup ids": not sub.image_id.duplicated().any(),
        "AC-5 no nulls": not sub[["image_id", "ocr_text", "product_name"]].isnull().any().any(),
        "AC-6 no newline/tab": not sub.ocr_text.str.contains(r"[\n\t]", regex=True, na=False).any(),
        "AC-7 columns": list(sub.columns[:3]) == ["image_id", "ocr_text", "product_name"],
    }
    ok = True
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    return ok


def build(engine: str, tag: str, train_on: str = "gt",
          min_class_count: int = 5, gate_threshold: float = 0.45,
          empty_gate: bool = False, gate_thr: float = 0.6) -> pd.DataFrame:
    labels = load_train_labels()
    # SUBMISSION trains on ALL labels (no val holdout needed for the final model).
    tr = labels

    # product head training text: clean GT (best), or OCR'd train text
    if train_on == "all":
        ocr_all = pd.read_parquet(cache_path(engine, "all"))[["image_id", "ocr_text"]]
        tr = tr[["image_id", "product_name"]].merge(ocr_all, on="image_id", how="inner")
    ext = ProductExtractor(min_class_count=min_class_count, gate_threshold=gate_threshold).fit(tr)

    # optional trained empty-gate: learn "is textless" from cached train OCR features
    gate = None
    if empty_gate:
        feats = labels.merge(pd.read_parquet(cache_path(engine, "all")), on="image_id")
        feats["gt_empty"] = (feats["ocr_text_x"].fillna("").str.strip() == "").astype(int)
        gate = EmptyGate(threshold=gate_thr).fit(
            feats.rename(columns={"ocr_text_y": "ocr_text"}), feats["gt_empty"])

    test_ids = load_test_ids()
    ocr_test = pd.read_parquet(cache_path(engine, "test"))
    n_missing = len(set(test_ids.image_id) - set(ocr_test.image_id))
    if n_missing:
        print(f"  WARNING: {n_missing} test images missing from OCR cache "
              f"(forced empty OCR). Finish run_ocr --split test before final submission.")
    sub = test_ids.merge(ocr_test, on="image_id", how="left")
    sub["ocr_text"] = sub["ocr_text"].fillna("")
    if "mean_conf" not in sub:
        sub["mean_conf"], sub["n_boxes"] = 0.0, 0

    if gate is not None:
        mask = gate.is_empty(sub)
        sub.loc[mask, "ocr_text"] = ""
        print(f"  empty-gate: zeroed {int(mask.sum())} test OCR rows")

    sub["product_name"] = ext.predict_batch(sub["ocr_text"])
    sub = sub[["image_id", "ocr_text", "product_name"]]

    sample = load_sample_submission()
    print(f"Validating submission '{tag}' ({engine}, train_on={train_on})...")
    if not validate(sub, sample):
        raise SystemExit("Validation FAILED")
    sub = sub.set_index("image_id").reindex(sample["image_id"]).reset_index()

    out_path = config.SUBMISSIONS_DIR / f"submission_{tag}.csv"
    write_submission_csv(sub, out_path)
    ocr_fill = (sub.ocr_text.str.strip() != "").mean()
    prod_fill = (sub.product_name.str.strip() != "").mean()
    print(f"\nSaved {out_path}")
    print(f"rows={len(sub)} | OCR fill={ocr_fill:.1%} | product fill={prod_fill:.1%}")
    return sub


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="paddleocr")
    p.add_argument("--tag", required=True)
    p.add_argument("--train-on", default="gt", choices=["gt", "all"])
    p.add_argument("--min-class-count", type=int, default=5)
    p.add_argument("--gate-threshold", type=float, default=0.45)
    p.add_argument("--empty-gate", action="store_true")
    p.add_argument("--gate-thr", type=float, default=0.6)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    build(a.engine, a.tag, train_on=a.train_on, min_class_count=a.min_class_count,
          gate_threshold=a.gate_threshold, empty_gate=a.empty_gate, gate_thr=a.gate_thr)
