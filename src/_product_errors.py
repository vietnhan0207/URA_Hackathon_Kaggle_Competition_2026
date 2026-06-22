"""Product error breakdown on val (oracle GT text, to isolate product logic from OCR).
Categorizes every val row to show WHERE F1 is lost -> guides the next improvement."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from data import load_train_labels, load_split_ids
from scoring import token_f1
from product_extract import ProductExtractor

labels = load_train_labels()
tr = labels[labels.image_id.isin(load_split_ids("train"))]
va = labels[labels.image_id.isin(load_split_ids("val"))].reset_index(drop=True)

ext = ProductExtractor(min_class_count=12, gate_threshold=0.55).fit(tr)
va["pred"] = ext.predict_batch(va["ocr_text"])   # oracle OCR
va["f1"] = [token_f1(g, p) for g, p in zip(va["product_name"], va["pred"])]

gt_empty = va["product_name"].str.strip() == ""
pred_empty = va["pred"].str.strip() == ""

cats = {
    "both empty (perfect)":      (gt_empty & pred_empty),
    "FALSE POS (GT empty, pred not)": (gt_empty & ~pred_empty),
    "FALSE NEG (GT has, pred empty)": (~gt_empty & pred_empty),
    "both non-empty":            (~gt_empty & ~pred_empty),
}
print(f"val rows: {len(va)} | mean F1: {va['f1'].mean():.4f}\n")
print(f"{'category':<34} {'count':>6} {'meanF1':>7} {'F1 lost':>8}")
for name, mask in cats.items():
    n = int(mask.sum())
    mf1 = va.loc[mask, "f1"].mean() if n else 0
    lost = (1 - va.loc[mask, "f1"]).sum()   # F1 points lost vs perfect
    print(f"{name:<34} {n:>6} {mf1:>7.3f} {lost:>8.1f}")

# among both-non-empty, how many exact vs partial vs zero
bne = va[~gt_empty & ~pred_empty]
print(f"\nAmong both-non-empty ({len(bne)}):")
print(f"  exact match (F1=1):   {(bne.f1==1).sum()}")
print(f"  partial (0<F1<1):     {((bne.f1>0)&(bne.f1<1)).sum()}")
print(f"  wrong (F1=0):         {(bne.f1==0).sum()}")

print("\n10 FALSE POSITIVES (GT empty but we predicted a product):")
for _, r in va[gt_empty & ~pred_empty].head(10).iterrows():
    print(f"  pred='{r['pred']}'  | ocr: {r['ocr_text'][:70]}")
