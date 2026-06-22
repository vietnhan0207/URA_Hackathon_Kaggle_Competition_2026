"""Evaluate product extraction on val using GT ocr_text (oracle OCR).
Isolates the product head: composite_ceiling = 0.6*F1 + 0.4 (CER=0 with oracle).
Sweeps gate threshold + min_class_count.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from data import load_train_labels, load_split_ids
from scoring import token_f1
from product_extract import ProductExtractor

labels = load_train_labels()
tr = labels[labels.image_id.isin(load_split_ids("train"))].reset_index(drop=True)
va = labels[labels.image_id.isin(load_split_ids("val"))].reset_index(drop=True)


def product_f1(pred_series, gt_series):
    return float(pd.Series([token_f1(g, p) for g, p in zip(gt_series, pred_series)]).mean())


# Reference: predict all empty
empty_f1 = product_f1(["" for _ in range(len(va))], va.product_name)
print(f"val rows: {len(va)} | empty-product baseline F1 = {empty_f1:.4f} "
      f"(composite ceiling {0.6*empty_f1+0.4:.4f})\n")

print(f"{'min_cnt':>7} {'gate':>5} {'n_classes':>9} {'F1':>7} {'ceiling':>8} {'fill':>6}")
best = None
for min_cnt in (2, 3, 5):
    ext = ProductExtractor(min_class_count=min_cnt, gate_threshold=0.5).fit(tr)
    # gate sweep reuses the same fitted model (threshold applied at predict time)
    for gate in (0.35, 0.45, 0.55, 0.65):
        ext.gate_threshold = gate
        preds = ext.predict_batch(va.ocr_text)
        f1 = product_f1(preds, va.product_name)
        fill = sum(1 for p in preds if p) / len(preds)
        ceiling = 0.6 * f1 + 0.4
        print(f"{min_cnt:>7} {gate:>5.2f} {len(ext.classes_):>9} {f1:>7.4f} {ceiling:>8.4f} {fill:>6.1%}")
        if best is None or f1 > best[0]:
            best = (f1, min_cnt, gate, len(ext.classes_))

print(f"\nBEST: F1={best[0]:.4f} (min_class_count={best[1]}, gate={best[2]}, "
      f"{best[3]} classes) -> oracle-OCR composite ceiling {0.6*best[0]+0.4:.4f}")
