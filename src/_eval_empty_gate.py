"""Tune an OCR empty-gate: predict GT-empty-ocr from cached OCR features
(n_boxes, mean_conf, n_chars) and measure CER-term gain on the fixed val split.

Gating a true-empty-GT row (we'd otherwise emit hallucinated text) flips CER 1.0->0.0;
gating a true-NONempty row flips its CER ->1.0. So we want high precision on "empty".
Sweeps the decision threshold to maximize val ocr_term.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from data import load_train_labels, load_split_ids
from scoring import cer
from run_ocr import cache_path

labels = load_train_labels()
allc = pd.read_parquet(cache_path("paddleocr", "all"))
df = labels.merge(allc, on="image_id", how="inner", suffixes=("_gt", "_ocr"))
df["ocr_text_ocr"] = df["ocr_text_ocr"].fillna("")
df["n_chars"] = df["ocr_text_ocr"].str.len()
df["gt_empty"] = (df["ocr_text_gt"].str.strip() == "").astype(int)

feat = ["n_boxes", "mean_conf", "n_chars"]
df["log_chars"] = np.log1p(df["n_chars"])
df["log_boxes"] = np.log1p(df["n_boxes"])
feat = ["log_boxes", "mean_conf", "log_chars"]

tr = df[df.image_id.isin(load_split_ids("train"))]
va = df[df.image_id.isin(load_split_ids("val"))].copy()
print(f"train {len(tr)} (empty {tr.gt_empty.mean():.1%}) | val {len(va)} (empty {va.gt_empty.mean():.1%})")

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(tr[feat], tr["gt_empty"])
va["p_empty"] = clf.predict_proba(va[feat])[:, 1]


def ocr_term(pred_text):
    return 1 - np.mean([cer(g, p) for g, p in zip(va["ocr_text_gt"], pred_text)])


base = ocr_term(va["ocr_text_ocr"])
print(f"\nval ocr_term (no gate): {base:.4f}")
print(f"{'thresh':>7} {'gated':>6} {'true_empty_gated':>16} {'false_gated':>11} {'ocr_term':>9}")
best = (base, None)
for thr in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
    gate = va["p_empty"] >= thr
    pred = va["ocr_text_ocr"].where(~gate, "")
    t = ocr_term(pred)
    true_empty_gated = int((gate & (va.gt_empty == 1)).sum())
    false_gated = int((gate & (va.gt_empty == 0)).sum())
    print(f"{thr:>7.2f} {int(gate.sum()):>6} {true_empty_gated:>16} {false_gated:>11} {t:>9.4f}")
    if t > best[0]:
        best = (t, thr)
print(f"\nBEST ocr_term {best[0]:.4f} @ threshold {best[1]} "
      f"(vs {base:.4f} no-gate; +{best[0]-base:.4f})")
