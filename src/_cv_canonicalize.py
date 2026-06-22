"""Does friend's product CANONICALIZER (curated alias map + train-frequency) lift
OUR calibrated head's product F1? Fold-safe: head + train-freq map built on the
train fold only; manual aliases are external spelling rules (host-documented).
"""
from __future__ import annotations

import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# import friend's canonicalizer from the cloned repo
REPO = Path(__file__).resolve().parents[2] / "_friend_repo"
sys.path.insert(0, str(REPO))
from src.postprocessing.product_canonicalizer import ProductCanonicalizer, normalize_product_key

from data import load_train_labels
from empty_gate import EmptyGate
from product_calibrated import CalibratedRuleHead
from run_ocr import cache_path
from scoring import cer, token_f1

# manual aliases (host-documented spelling normalization)
ALIAS_YAML = REPO / "configs" / "product_aliases.yaml"
aliases = {}
for line in ALIAS_YAML.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if ":" in line and line.startswith('"'):
        k, v = line.split(":", 1)
        aliases[k.strip().strip('"')] = v.strip().strip('"')
print(f"loaded {len(aliases)} manual aliases")

HIS_REF = pd.read_csv(
    Path(__file__).resolve().parents[1] / "submissions" / "submission_old_ocr_latest_product_pipeline.csv",
    keep_default_na=False, dtype=str)


def build_map(train_products, reference_products=None):
    cbk = defaultdict(Counter)
    for p in train_products.fillna("").astype(str):
        if p.strip():
            k = normalize_product_key(p)
            if k:
                cbk[k][p] += 1
    if reference_products is not None:
        for p in reference_products.fillna("").astype(str):
            if p.strip():
                k = normalize_product_key(p)
                if k:
                    cbk[k][p] += 10
    return {k: c.most_common(1)[0][0] for k, c in cbk.items()}


def _gkey(text, image_id):
    t = " ".join(str(text).lower().split())
    return "empty_" + str(image_id) if not t else hashlib.md5(t.encode("utf-8")).hexdigest()


labels = load_train_labels()
ocr = pd.read_parquet(cache_path("vietocr_ft", "all"))
df = labels.merge(ocr, on="image_id", suffixes=("_gt", "_ocr")).reset_index(drop=True)
df["ocr_text_ocr"] = df["ocr_text_ocr"].fillna("")
groups = [_gkey(t, i) for i, t in zip(df.image_id, df.ocr_text_gt)]
gkf = GroupKFold(n_splits=5)

gt, base, canon_clean, canon_ref = [], [], [], []
pooled_cer = []
for tr_idx, va_idx in gkf.split(df, groups=groups):
    trd, vad = df.iloc[tr_idx], df.iloc[va_idx]
    head = CalibratedRuleHead(use_classifier_fallback=True, min_pprod=0.55,
                              gate_threshold=0.75).fit(
        trd.rename(columns={"ocr_text_gt": "ocr_text"})[["image_id", "ocr_text", "product_name"]])
    tg = trd.copy(); tg["gt_empty"] = (tg.ocr_text_gt.str.strip() == "").astype(int)
    eg = EmptyGate(threshold=0.6).fit(tg.rename(columns={"ocr_text_ocr": "ocr_text"}), tg.gt_empty)
    mask = eg.is_empty(vad.rename(columns={"ocr_text_ocr": "ocr_text"}))
    ocr_in = vad.ocr_text_ocr.where(~np.asarray(mask), "")
    preds = list(head.predict_batch(ocr_in))

    cz_clean = ProductCanonicalizer(build_map(trd["product_name"]), alias_mapping=aliases)
    cz_ref = ProductCanonicalizer(build_map(trd["product_name"], HIS_REF["product_name"]), alias_mapping=aliases)

    base += preds
    canon_clean += [cz_clean.canonicalize(p).canonical for p in preds]
    canon_ref += [cz_ref.canonicalize(p).canonical for p in preds]
    gt += list(vad.product_name)
    pooled_cer += [cer(g, p) for g, p in zip(vad.ocr_text_gt, ocr_in)]

ot = 1 - np.mean(pooled_cer)
for name, pred in [("baseline (no canon)", base),
                   ("+ canon: train+aliases (CLEAN)", canon_clean),
                   ("+ canon: train+aliases+hisRef", canon_ref)]:
    f1 = np.mean([token_f1(g, p) for g, p in zip(gt, pred)])
    print(f"{name:<34} prod_F1 {f1:.4f} | composite {0.6*f1+0.4*ot:.4f}")
