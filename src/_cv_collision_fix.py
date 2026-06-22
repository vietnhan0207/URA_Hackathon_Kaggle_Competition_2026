"""Empirically test whether fixing the dominant-family collisions beats the
current calibrated head (CV 0.6142). Fold-safe: emit strings refit per fold.
Tries several SIG_PATTERNS orderings/guards via monkeypatch.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import product_calibrated as PC
from data import load_train_labels
from empty_gate import EmptyGate
from product_calibrated import CalibratedRuleHead
from run_ocr import cache_path
from scoring import cer, token_f1

BASE = PC.SIG_PATTERNS

# V1: Highlands BEFORE the Ha Long rules (win the Highlands-GT mixed images)
V1 = [BASE[8]] + BASE[:8]  # nestle? no -- careful, BASE[8] is nestle. recompute below

# build by name for clarity
by = {n: (n, p) for n, p in BASE}
order_base = [n for n, _ in BASE]

VARIANTS = {
    "baseline": order_base,
    # highlands hoisted above all Ha Long / pate / nan rules
    "highlands_first": (["highlands"] +
                        [n for n in order_base if n != "highlands"]),
    # highlands above Ha Long but below the explicit pate/canfoco combos
    "highlands_mid": ["halong_canfoco_pate_cotden", "nan_optipro", "nan_infinipro",
                      "pate_cotden", "highlands", "halong_canfoco", "do_hop_ha_long",
                      "nan", "nestle"],
    # nestle BEFORE bare nan (plain-Nestle images stop emitting NAN variants)
    "nestle_before_nan": ["halong_canfoco_pate_cotden", "nan_optipro", "nan_infinipro",
                          "pate_cotden", "halong_canfoco", "do_hop_ha_long",
                          "highlands", "nestle", "nan"],
    # both
    "highlands_first+nestle_before_nan":
        ["highlands", "halong_canfoco_pate_cotden", "nan_optipro", "nan_infinipro",
         "pate_cotden", "halong_canfoco", "do_hop_ha_long", "nestle", "nan"],
}


def _gkey(text, image_id):
    t = " ".join(str(text).lower().split())
    return "empty_" + str(image_id) if not t else hashlib.md5(t.encode("utf-8")).hexdigest()


labels = load_train_labels()
ocr = pd.read_parquet(cache_path("vietocr_ft", "all"))
df = labels.merge(ocr, on="image_id", suffixes=("_gt", "_ocr")).reset_index(drop=True)
df["ocr_text_ocr"] = df["ocr_text_ocr"].fillna("")
groups = [_gkey(t, i) for i, t in zip(df.image_id, df.ocr_text_gt)]
gkf = GroupKFold(n_splits=5)

print("Fold-safe CV composite on real vietocr_ft OCR (empty-gate on):\n")
print(f"{'variant':<36}{'composite':>10}{'F1':>8}{'ocr_term':>10}")
for vname, order in VARIANTS.items():
    PC.SIG_PATTERNS = [by[n] for n in order]
    pooled_gt, pooled_pred, pooled_cer = [], [], []
    for tr_idx, va_idx in gkf.split(df, groups=groups):
        trd, vad = df.iloc[tr_idx], df.iloc[va_idx]
        head = CalibratedRuleHead(use_classifier_fallback=True, min_pprod=0.55,
                                  gate_threshold=0.75).fit(
            trd.rename(columns={"ocr_text_gt": "ocr_text"})[["image_id", "ocr_text", "product_name"]])
        tg = trd.copy(); tg["gt_empty"] = (tg.ocr_text_gt.str.strip() == "").astype(int)
        eg = EmptyGate(threshold=0.6).fit(tg.rename(columns={"ocr_text_ocr": "ocr_text"}), tg.gt_empty)
        mask = eg.is_empty(vad.rename(columns={"ocr_text_ocr": "ocr_text"}))
        ocr_in = vad.ocr_text_ocr.where(~np.asarray(mask), "")
        preds = head.predict_batch(ocr_in)
        pooled_pred += list(preds); pooled_gt += list(vad.product_name)
        pooled_cer += [cer(g, p) for g, p in zip(vad.ocr_text_gt, ocr_in)]
    f1 = np.mean([token_f1(g, p) for g, p in zip(pooled_gt, pooled_pred)])
    ot = 1 - np.mean(pooled_cer)
    star = "  <-- baseline" if vname == "baseline" else ""
    print(f"{vname:<36}{0.6*f1+0.4*ot:>10.4f}{f1:>8.4f}{ot:>10.4f}{star}")

PC.SIG_PATTERNS = BASE
