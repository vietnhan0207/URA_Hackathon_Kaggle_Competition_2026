"""Calibrate dominant-family rules on TRAIN so the LB-aggressive labeling is
data-grounded, not guessed.

For each candidate signature (regex on diacritic-folded OCR text), measure on
the FULL train set:
  - support    : # train images whose OCR matches the signature
  - p_product  : P(GT product non-empty | signature)   -> how safe to fire
  - best_form  : output string maximizing expected token-F1 over the GT products
                 of the matched images (this is the LB-optimal emit string)
  - exp_f1     : expected token-F1 if we emit best_form on every matched image
                 (vs leaving empty), i.e. the per-image value of the rule
  - emit_gain  : exp_f1 over matched images minus the all-empty baseline value
                 (all-empty value = P(GT empty) on matched, since empty==empty=1)

A rule is worth firing aggressively when emit_gain > 0 (emitting beats staying
empty in expectation over its matched test images).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

import numpy as np

from data import load_train_labels
from scoring import token_f1


def fold(s):
    s = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


labels = load_train_labels()
labels["folded_ocr"] = labels["ocr_text"].map(fold)
labels["prod"] = labels["product_name"].str.strip()

# candidate signatures, most-specific first (order matters at apply time)
SIGS = [
    ("halong_canfoco_pate_cotden", r"(canfoco|canfuco|cafoco).*(pate|cot den).*(cot den|hai phong)|"
                                   r"(pate|cot den).*(canfoco|canfuco)"),
    ("pate_cotden",                r"\bpate\b.*\b(cot den|hai phong)\b|\b(cot den|hai phong)\b.*\bpate\b|"
                                   r"\bcot den\b"),
    ("halong_canfoco",             r"\bcanfoco\b|\bcanfuco\b|\bcafoco\b|ha long canfoco|halong canfoco"),
    ("do_hop_ha_long",             r"do hop ha long|do hop.*ha long|cong ty.*do hop.*ha long"),
    ("nan_optipro",                r"\bnan\b.*opti ?pro|opti ?pro.*\bnan\b"),
    ("nan_infinipro",              r"\bnan\b.*infini ?pro|infini ?pro.*\bnan\b"),
    ("nan",                        r"\bnan\b"),
    ("milo",                       r"\bmilo\b"),
    ("highlands",                  r"highlands? coffee|highlands"),
    ("nestle",                     r"\bnestle\b"),
    ("vinamilk",                   r"\bvinamilk\b"),
]

_pp = (labels["prod"] != "").mean()
print(f"TRAIN {len(labels)} rows | overall P(product)={_pp:.3f}\n")
print("SEQUENTIAL calibration (most-specific first; each rule sees only residual):\n")
print(f"{'signature':<28}{'supp':>5}{'p_prod':>7}{'emit_gain':>10}  best_form")
print("-" * 92)

remaining = labels.copy()
rules = []
for name, pat in SIGS:
    m = remaining["folded_ocr"].str.contains(pat, regex=True, na=False)
    sub = remaining[m]
    supp = len(sub)
    if supp == 0:
        print(f"{name:<28}{supp:>5}      -         -    (no residual support)")
        continue
    p_prod = (sub["prod"] != "").mean()
    forms = Counter(sub[sub["prod"] != ""]["prod"])
    # only consider emit strings that occur >=3x (avoid 1-off concatenations)
    cands = [f for f, c in forms.items() if c >= 3] or list(forms)
    best_form, best_val = "", -1.0
    gts = list(sub["prod"])
    for cand in cands:
        val = np.mean([token_f1(g, cand) for g in gts])
        if val > best_val:
            best_val, best_form = val, cand
    empty_baseline = (sub["prod"] == "").mean()
    emit_gain = best_val - empty_baseline
    rules.append((name, pat, round(p_prod, 3), best_form, round(emit_gain, 3), supp))
    flag = "  <-- DROP" if emit_gain <= 0 else ""
    print(f"{name:<28}{supp:>5}{p_prod:>7.3f}{emit_gain:>+10.3f}  {best_form[:42]}{flag}")
    remaining = remaining[~m]   # consume matched rows

print(f"\nresidual unmatched: {len(remaining)} rows | P(product) in residual = "
      f"{(remaining['prod'] != '').mean():.3f}")
print("\nThese best_form strings + p_prod are the deployable LB-tailored rules.")
print("Compare to friend's outputs: canfoco-alone -> 'Đồ Hộp Hạ Long' (he emits 'Ha Long Canfoco').")

# emit a ready-to-paste rule table
print("\n# CALIBRATED_RULES = [(name, pattern, emit_form, p_prod, support), ...]")
for name, pat, p_prod, form, gain, supp in rules:
    if gain > 0:
        print(f'  ({name!r}, r"{pat}", {form!r}, {p_prod}, {supp}),')
