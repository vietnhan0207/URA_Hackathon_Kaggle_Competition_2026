"""Mine train+real-OCR for TARGETED product-rule opportunities that could beat
our classifier WITHOUT adding false positives.

Approach (all on 5-fold CV with REAL vietocr_ft OCR, the deployed setting):
  - Pool out-of-fold classifier predictions.
  - Bucket every row: false-neg (GT has product, we predict empty),
    wrong (both non-empty, F1=0), false-pos (GT empty, we predict).
  - For false-negs, see whether the friend's rules would recover them, and at
    what false-positive cost on empty-GT rows. Net F1 delta = the only thing
    that matters.
  - Also: which GT products are most often missed, and are they dropped by our
    min_class_count=12 filter (i.e. rule-only candidates)?
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from data import load_train_labels
from empty_gate import EmptyGate
from product_extract import ProductExtractor, tokkey
from product_rules_friend import extract_product, safe_product
from run_ocr import cache_path
from scoring import token_f1


def _gkey(text, image_id):
    t = " ".join(str(text).lower().split())
    return "empty_" + str(image_id) if not t else hashlib.md5(t.encode("utf-8")).hexdigest()


labels = load_train_labels()
ocr = pd.read_parquet(cache_path("vietocr_ft", "all"))
df = labels.merge(ocr, on="image_id", suffixes=("_gt", "_ocr")).reset_index(drop=True)
df["ocr_text_ocr"] = df["ocr_text_ocr"].fillna("")
groups = [_gkey(t, i) for i, t in zip(df.image_id, df.ocr_text_gt)]

# pooled out-of-fold predictions with the SAME empty-gate as deployed
gkf = GroupKFold(n_splits=5)
rows = []
for tr_idx, va_idx in gkf.split(df, groups=groups):
    trd, vad = df.iloc[tr_idx], df.iloc[va_idx]
    ext = ProductExtractor(min_class_count=12, gate_threshold=0.55).fit(
        trd.rename(columns={"ocr_text_gt": "ocr_text"})[["image_id", "ocr_text", "product_name"]])
    tg = trd.copy()
    tg["gt_empty"] = (tg["ocr_text_gt"].str.strip() == "").astype(int)
    eg = EmptyGate(threshold=0.6).fit(tg.rename(columns={"ocr_text_ocr": "ocr_text"}), tg["gt_empty"])
    mask = eg.is_empty(vad.rename(columns={"ocr_text_ocr": "ocr_text"}))
    ocr_in = vad["ocr_text_ocr"].where(~np.asarray(mask), "")
    clf_pred = ext.predict_batch(ocr_in)
    rule_pred = [safe_product(extract_product(t), t) for t in ocr_in]
    for gt_p, cp, rp, otext in zip(vad["product_name"], clf_pred, rule_pred, ocr_in):
        rows.append({"gt": gt_p.strip(), "clf": cp.strip(), "rule": rp.strip(), "ocr": otext})

r = pd.DataFrame(rows)
r["clf_f1"] = [token_f1(g, p) for g, p in zip(r["gt"], r["clf"])]

gt_empty = r["gt"] == ""
clf_empty = r["clf"] == ""

print(f"CV rows: {len(r)} | classifier mean F1: {r['clf_f1'].mean():.4f}\n")

# ---- 1) Where is F1 lost? ----
buckets = {
    "false-neg (GT has, clf empty)": (~gt_empty & clf_empty),
    "wrong   (both non-empty, F1=0)": (~gt_empty & ~clf_empty & (r["clf_f1"] == 0)),
    "partial (0<F1<1)":              (~gt_empty & ~clf_empty & (r["clf_f1"] > 0) & (r["clf_f1"] < 1)),
    "false-pos (GT empty, clf has)": (gt_empty & ~clf_empty),
}
print(f"{'bucket':<34}{'count':>6}{'F1 lost':>9}")
for name, m in buckets.items():
    print(f"{name:<34}{int(m.sum()):>6}{(1 - r.loc[m, 'clf_f1']).sum():>9.1f}")

# ---- 2) Could the friend's rule RECOVER false-negs, and at what FP cost? ----
fn = r[~gt_empty & clf_empty].copy()
fn["rule_f1"] = [token_f1(g, p) for g, p in zip(fn["gt"], fn["rule"])]
recovered = fn[fn["rule"] != ""]
recover_gain = (recovered["rule_f1"] - 0).sum()  # clf gave 0 here
# FP cost: empty-GT rows where rule fires (token_f1 = 0, loses 1 each vs clf's correct 1.0 only if clf was empty there)
fp_rows = r[gt_empty & (r["rule"] != "")]
# clf already correct (empty) on those, rule would break them
fp_cost = float((fp_rows["clf"] == "").sum())  # each becomes F1=0 instead of 1.0
print(f"\n--- Friend-rule as fallback on classifier-empty rows ---")
print(f"false-neg rows: {len(fn)} | rule recovers (non-empty): {len(recovered)}")
print(f"  F1 gained on recovered rows : +{recover_gain:.1f}")
print(f"  empty-GT rows the rule fires : {len(fp_rows)}  (FP cost ~ -{fp_cost:.1f} F1)")
print(f"  NET F1 delta (rows)         : {recover_gain - fp_cost:+.1f}  -> /{len(r)} = {(recover_gain - fp_cost)/len(r):+.4f}")

# ---- 3) Most-missed GT products (recall), and are they classifier-eligible? ----
elig = labels[labels.product_name.str.strip() != ""].copy()
elig_counts = Counter(elig["product_name"].map(lambda p: tokkey(p)))
miss = r[(~gt_empty) & (r["clf_f1"] < 1)]
by_prod = miss.groupby("gt").size().sort_values(ascending=False).head(18)
print(f"\n--- Top GT products we miss/partial (real OCR), with train support ---")
print(f"{'GT product':<38}{'misses':>7}{'train_n':>8}{'<12?':>6}")
for prod, cnt in by_prod.items():
    n = elig_counts.get(tokkey(prod), 0)
    print(f"{str(prod)[:37]:<38}{cnt:>7}{n:>8}{'  RULE' if n < 12 else '':>6}")
