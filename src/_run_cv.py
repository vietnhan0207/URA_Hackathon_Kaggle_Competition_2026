"""Trustworthy CV report: does each config gain hold under 5-fold CV (predicts private)?
Compares engines + product configs. fold_std ~ the public/private wobble to expect."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_val import cv_composite

print("=== ENGINE comparison (product min12/gate0.55 + empty-gate) ===")
for eng in ["paddleocr", "vietocr", "vietocr_ft"]:
    try:
        r = cv_composite(engine=eng, min_class_count=12, gate_threshold=0.55, empty_gate=True)
        print(f"{eng:>12}: CV {r['composite']} | F1 {r['f1']} | ocr_term {r['ocr_term']} "
              f"| folds {r['fold_mean']}±{r['fold_std']}")
    except Exception as e:
        print(f"{eng:>12}: skip ({e})")

print("\n=== PRODUCT CONFIG on FT (does conservative + empty-gate hold under CV?) ===")
configs = [
    ("min5/gate0.45, no gate",  dict(min_class_count=5,  gate_threshold=0.45, empty_gate=False)),
    ("min12/gate0.55, no gate", dict(min_class_count=12, gate_threshold=0.55, empty_gate=False)),
    ("min12/gate0.55, +gate",   dict(min_class_count=12, gate_threshold=0.55, empty_gate=True)),
    ("min20/gate0.60, +gate",   dict(min_class_count=20, gate_threshold=0.60, empty_gate=True)),
]
for name, cfg in configs:
    r = cv_composite(engine="vietocr_ft", **cfg)
    print(f"{name:>26}: CV {r['composite']} | F1 {r['f1']} | ocr_term {r['ocr_term']} "
          f"| folds {r['fold_mean']}±{r['fold_std']}")
