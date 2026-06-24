"""Full engine + detector bake-off comparison.

Computes CER, diacritic density, OCR-term (1-CER), and product F1
for every parquet file in cache/, then prints a clean Markdown table.

Run from project root:
    python src/_compare_engines.py

Output: prints table to stdout + writes presentation/engine_comparison.json
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))

import config
from data import load_train_labels
from product_calibrated import CalibratedRuleHead
from scoring import cer, token_f1

CACHE = config.CACHE_DIR
ROOT  = CACHE.parent

# ── Load ground truth ──────────────────────────────────────────────────────────
labels = load_train_labels()
labels["ocr_text"]     = labels["ocr_text"].fillna("")
labels["product_name"] = labels["product_name"].fillna("").str.strip()
gt = labels.set_index("image_id")

def diac_frac(s):
    if not s: return 0.0
    nfd = unicodedata.normalize("NFD", str(s))
    letters = sum(1 for c in nfd if c.isalpha())
    marks   = sum(1 for c in nfd if unicodedata.category(c) == "Mn")
    return marks / letters if letters else 0.0

# ── Engines to evaluate ────────────────────────────────────────────────────────
# (display_name, parquet_path, detector, notes)
ENGINES = [
    # End-to-end systems (own detector + recogniser)
    ("EasyOCR (E2E)",
     CACHE / "ocr_easyocr_val.parquet",
     "CRAFT (built-in)", "CRAFT-based E2E; val only"),

    ("RapidOCR (E2E)",
     CACHE / "ocr_rapidocr_val.parquet",
     "DBNet ONNX (built-in)", "Lightweight ONNX E2E; val only"),

    ("PaddleOCR (E2E)",
     CACHE / "ocr_paddleocr_all.parquet",
     "PP-OCR det (built-in)", "PP-OCR v3 E2E"),

    # VietOCR variants — all use CRAFT or swapped detector
    ("VietOCR base",
     CACHE / "ocr_vietocr_all.parquet",
     "CRAFT", "Pre-trained vgg_transformer; no fine-tuning"),

    ("VietOCR-FT (ours) ★",
     CACHE / "ocr_vietocr_ft_all.parquet",
     "CRAFT", "Fine-tuned on ext. Vi dataset + competition train"),

    ("VietOCR-FTdb",
     CACHE / "ocr_vietocr_ftdb_all.parquet",
     "RapidOCR DB ONNX", "CRAFT swapped → DB detector + upscale small thumbnails"),

    ("VietOCR-FTup",
     CACHE / "ocr_vietocr_ftup_all.parquet",
     "CRAFT + upscale", "CRAFT pipeline + upscale small thumbnails"),

    ("VietOCR-FTup2",
     CACHE / "ocr_vietocr_ftup2_all.parquet",
     "CRAFT + upscale v2", "CRAFT pipeline + stronger upscale"),
]

# ── Fit calibrated head on all training data (for oracle product F1 comparison) ─
print("Fitting CalibratedRuleHead on all training labels …")
head = CalibratedRuleHead(use_classifier_fallback=True,
                          min_pprod=0.55, gate_threshold=0.75)
head.fit(labels[["image_id", "ocr_text", "product_name"]])

# ── Evaluate each engine ───────────────────────────────────────────────────────
results = []

for name, path, detector, notes in ENGINES:
    if not Path(path).exists():
        print(f"  SKIP {name} — parquet not found: {path}")
        continue

    eng = pd.read_parquet(path)[["image_id", "ocr_text"]].copy()
    eng["ocr_text"] = eng["ocr_text"].fillna("")
    eng = eng.set_index("image_id")

    # Common images with GT
    common = sorted(set(gt.index) & set(eng.index))
    if not common:
        print(f"  SKIP {name} — no common image IDs with GT")
        continue

    gt_sub  = gt.loc[common]
    eng_sub = eng.loc[common]

    # CER vs GT ocr_text
    cers = [cer(gt_sub.loc[i, "ocr_text"], eng_sub.loc[i, "ocr_text"])
            for i in common]
    mean_cer   = float(np.mean(cers))
    ocr_term   = 1.0 - mean_cer

    # Diacritic density of engine output
    eng_diac   = float(np.mean([diac_frac(eng_sub.loc[i, "ocr_text"]) for i in common]))
    gt_diac    = float(np.mean([diac_frac(gt_sub.loc[i, "ocr_text"])  for i in common]))

    # Product F1: run our calibrated head on this engine's OCR text
    preds      = head.predict_batch([eng_sub.loc[i, "ocr_text"] for i in common])
    gts_prod   = [gt_sub.loc[i, "product_name"] for i in common]
    prod_f1    = float(np.mean([token_f1(g, p) for g, p in zip(gts_prod, preds)]))

    composite  = 0.6 * prod_f1 + 0.4 * ocr_term

    results.append({
        "name":      name,
        "detector":  detector,
        "n":         len(common),
        "cer":       round(mean_cer,   4),
        "ocr_term":  round(ocr_term,   4),
        "eng_diac":  round(eng_diac,   4),
        "gt_diac":   round(gt_diac,    4),
        "prod_f1":   round(prod_f1,    4),
        "composite": round(composite,  4),
        "notes":     notes,
    })
    print(f"  {name:<28} n={len(common):4d}  CER={mean_cer:.4f}  "
          f"diac={eng_diac:.3f}  prodF1={prod_f1:.4f}  comp={composite:.4f}")

# ── Print Markdown table ───────────────────────────────────────────────────────
gt_diac_ref = results[0]["gt_diac"] if results else 0.257
print(f"\nGT diacritic reference: {gt_diac_ref:.3f}\n")

header = ("| Engine | Detector | n | CER ↓ | OCR-term ↑ | "
          "Diac density (eng) | Prod F1 ↑ | Composite ↑ | Notes |")
sep    = "|:--|:--|--:|--:|--:|--:|--:|--:|:--|"
print(header)
print(sep)
for r in sorted(results, key=lambda x: x["cer"]):
    star = " **★**" if "★" in r["name"] else ""
    print(f"| {r['name']}{star} | {r['detector']} | {r['n']:,} | "
          f"{r['cer']:.4f} | {r['ocr_term']:.4f} | "
          f"{r['eng_diac']:.3f} | {r['prod_f1']:.4f} | "
          f"{r['composite']:.4f} | {r['notes']} |")

# ── CV vs LB table ─────────────────────────────────────────────────────────────
print("\n\n## CV vs LB progression\n")
CV_LB = [
    ("v2",        "PaddleOCR + product head + empty-gate",             None,   0.5890),
    ("v3",        "PaddleOCR + product head (min12/gate0.55, all labels)", None, 0.5803),
    ("v4",        "VietOCR base + product head (min12/gate0.55)",      None,   0.5898),
    ("v6",        "VietOCR-FT + product head (min5/gate0.45)",         0.590,  0.6030),
    ("v7",        "VietOCR-FT + product head (min12/gate0.55)",        0.597,  0.6217),
    ("v8",        "VietOCR-FT + product head + empty-gate",            0.597,  0.6232),
    ("ftup2",     "VietOCR-FTup2 + product head (min12/gate0.55)",     None,   0.6209),
    ("v10",       "VietOCR-FT + CalibratedRuleHead (5-fold CV)",       0.6142, 0.6685),
    ("mix",       "Our OCR + teammate product (post-hoc)",              None,   0.6959),
]
print("| Tag | Config | CV (train) | Kaggle LB | CV→LB gap |")
print("|:--|:--|--:|--:|--:|")
for tag, cfg, cv, lb in CV_LB:
    cv_str  = f"{cv:.4f}" if cv else "—"
    gap_str = f"+{lb-cv:.4f}" if cv else "n/a"
    print(f"| {tag} | {cfg} | {cv_str} | {lb:.4f} | {gap_str} |")

# ── Save JSON for embedding in presentation ────────────────────────────────────
out = ROOT / "presentation" / "engine_comparison.json"
payload = {
    "gt_diac_ref": gt_diac_ref,
    "engines": results,
    "cv_lb": [{"tag": t, "config": c, "cv": cv, "lb": lb}
              for t, c, cv, lb in CV_LB],
}
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nSaved → {out}")
