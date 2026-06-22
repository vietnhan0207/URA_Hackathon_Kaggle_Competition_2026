"""Sweep ProductExtractor hyperparams on REAL PaddleOCR val text (cached subset).
Trains on train-split clean GT text (best per earlier finding); predicts on real OCR.
Reports product F1 + end-to-end composite (with empty-gate)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from data import load_train_labels, load_split_ids
from scoring import composite_score, token_f1
from product_extract import ProductExtractor
from run_ocr import cache_path

labels = load_train_labels()
allc = pd.read_parquet(cache_path("paddleocr", "all"))
allc["n_chars"] = allc["ocr_text"].fillna("").str.len()
cached = set(allc.image_id)
tr_ids, va_ids = load_split_ids("train"), load_split_ids("val")

tr = labels[labels.image_id.isin(tr_ids)]  # train on clean GT (full train split)
va = labels[labels.image_id.isin(va_ids & cached)].merge(
    allc[["image_id", "ocr_text", "mean_conf", "n_boxes", "n_chars"]],
    on="image_id", suffixes=("_gt", "_ocr"))

# empty-gate (thr 0.6)
m = labels.merge(allc, on="image_id")
m["log_boxes"] = np.log1p(m["n_boxes"]); m["log_chars"] = np.log1p(m["n_chars"])
m["gt_empty"] = (m["ocr_text_x"].fillna("").str.strip() == "").astype(int)
gf = ["log_boxes", "mean_conf", "log_chars"]
gate = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
    m[m.image_id.isin(tr_ids & cached)][gf], m[m.image_id.isin(tr_ids & cached)]["gt_empty"])
vg = m[m.image_id.isin(va_ids & cached)].copy()
gmask = dict(zip(vg.image_id, gate.predict_proba(vg[gf])[:, 1] >= 0.6))
ocr_gated = va.apply(lambda r: "" if gmask.get(r.image_id) else r.ocr_text_ocr, axis=1)

gt = va[["image_id", "ocr_text_gt", "product_name"]].rename(columns={"ocr_text_gt": "ocr_text"})
print(f"cached val {len(va)}\n{'min_cnt':>7} {'gate':>5} {'classes':>7} {'prod_F1':>8} {'composite':>9}")
best = None
for mc in (12, 16, 20, 28):
    ext = ProductExtractor(min_class_count=mc, gate_threshold=0.45).fit(tr)
    for gthr in (0.55, 0.62, 0.70):
        ext.gate_threshold = gthr
        preds = ext.predict_batch(ocr_gated)
        f1 = float(pd.Series([token_f1(g, p) for g, p in zip(va.product_name, preds)]).mean())
        sub = pd.DataFrame({"image_id": va.image_id, "ocr_text": ocr_gated, "product_name": preds})
        comp = composite_score(gt, sub)
        print(f"{mc:>7} {gthr:>5.2f} {len(ext.classes_):>7} {f1:>8.4f} {comp:>9.4f}")
        if best is None or comp > best[0]:
            best = (comp, mc, gthr, f1)
print(f"\nBEST composite {best[0]:.4f} @ min_class_count={best[1]} gate={best[2]} (F1={best[3]:.4f})")
