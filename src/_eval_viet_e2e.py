"""End-to-end val composite with VietOCR OCR text, across product configs.
Trains product head on GT train-split, evaluates on val (VietOCR text + empty-gate).
Compares to PaddleOCR end-to-end for the same configs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from data import load_train_labels, load_split_ids
from scoring import composite_score
from product_extract import ProductExtractor
from run_ocr import cache_path

labels = load_train_labels()
tr_ids, va_ids = load_split_ids("train"), load_split_ids("val")
tr = labels[labels.image_id.isin(tr_ids)]


def eval_engine(engine):
    allc = pd.read_parquet(cache_path(engine, "all"))
    allc["n_chars"] = allc["ocr_text"].fillna("").str.len()
    cached = set(allc.image_id)
    va = labels[labels.image_id.isin(va_ids & cached)].merge(
        allc[["image_id", "ocr_text", "mean_conf", "n_boxes", "n_chars"]],
        on="image_id", suffixes=("_gt", "_ocr"))
    # empty-gate (thr 0.6) trained on this engine's features
    m = labels.merge(allc, on="image_id")
    m["log_boxes"] = np.log1p(m["n_boxes"]); m["log_chars"] = np.log1p(m["n_chars"])
    m["gt_empty"] = (m["ocr_text_x"].fillna("").str.strip() == "").astype(int)
    gf = ["log_boxes", "mean_conf", "log_chars"]
    g = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
        m[m.image_id.isin(tr_ids & cached)][gf], m[m.image_id.isin(tr_ids & cached)]["gt_empty"])
    vg = m[m.image_id.isin(va_ids & cached)].copy()
    gmask = dict(zip(vg.image_id, g.predict_proba(vg[gf])[:, 1] >= 0.6))
    ocr_gated = va.apply(lambda r: "" if gmask.get(r.image_id) else r.ocr_text_ocr, axis=1)
    gt = va[["image_id", "ocr_text_gt", "product_name"]].rename(columns={"ocr_text_gt": "ocr_text"})

    print(f"\n=== {engine} (val {len(va)}) ===")
    for mc, gthr in [(5, 0.45), (8, 0.50), (12, 0.55)]:
        ext = ProductExtractor(min_class_count=mc, gate_threshold=gthr).fit(tr)
        sub = pd.DataFrame({"image_id": va.image_id, "ocr_text": ocr_gated,
                            "product_name": ext.predict_batch(ocr_gated)})
        c = composite_score(gt, sub, return_components=True)
        print(f"  min{mc}/gate{gthr}: composite {c['composite']} "
              f"(F1 {c['product_f1']}, ocr_term {c['ocr_term']})")


eval_engine("paddleocr")
eval_engine("vietocr")
eval_engine("vietocr_ft")
