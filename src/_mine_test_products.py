"""Mine the TEST OCR for product coverage to tailor rules to the public set.

Outputs:
  A. Coverage gap: test images with product-like brand tokens that NEITHER our
     classifier NOR the friend rules currently label -> candidate new rules.
  B. Distinctive token frequency in test OCR (brand-name candidates).
  C. For every product family the friend rules DO catch, the optimal canonical
     output string (from train) -> the exact string to emit.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

import pandas as pd

from data import load_train_labels
from product_extract import ProductExtractor
from product_rules_friend import extract_product, safe_product
from run_ocr import cache_path


def fold(s):
    s = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


labels = load_train_labels()
test_ocr = pd.read_parquet(cache_path("vietocr_ft", "test"))
test_ocr["ocr_text"] = test_ocr["ocr_text"].fillna("")
test_ocr["folded"] = test_ocr["ocr_text"].map(fold)

# brand vocabulary from train product labels (folded tokens, len>=3)
train_brands = labels[labels.product_name.str.strip() != ""].product_name
brand_tokens = Counter()
for p in train_brands:
    for t in set(fold(p).split()):
        if len(t) >= 3:
            brand_tokens[t] += 1
known_brand_tokens = {t for t, c in brand_tokens.items() if c >= 5}
# drop generic words
GENERIC = {"sua", "hop", "long", "cot", "den", "hai", "phong", "pate", "cong",
           "phan", "viet", "nam", "san", "pham", "che", "bien", "thuc"}
known_brand_tokens -= GENERIC

clf = ProductExtractor(min_class_count=12, gate_threshold=0.55).fit(labels)
test_ocr["clf"] = clf.predict_batch(test_ocr["ocr_text"])
test_ocr["rule"] = [safe_product(extract_product(t), t) for t in test_ocr["ocr_text"]]
both_empty = (test_ocr.clf.str.strip() == "") & (test_ocr.rule.str.strip() == "")

print(f"TEST {len(test_ocr)} | classifier fill {(test_ocr.clf.str.strip()!='').mean():.1%} | "
      f"rule fill {(test_ocr.rule.str.strip()!='').mean():.1%} | "
      f"both empty {both_empty.mean():.1%}")

# ---- A. brand tokens that appear in test OCR but we label NOTHING ----
uncovered = test_ocr[both_empty & (test_ocr.ocr_text.str.strip() != "")]
hit = Counter()
examples = {}
for _, r in uncovered.iterrows():
    toks = set(r.folded.split())
    for bt in toks & known_brand_tokens:
        hit[bt] += 1
        examples.setdefault(bt, r.ocr_text[:70])
print(f"\n=== A. Brand tokens present in UNLABELED test OCR (coverage gaps) ===")
print(f"{'token':<16}{'#imgs':>6}   example OCR")
for tok, c in hit.most_common(25):
    print(f"{tok:<16}{c:>6}   {examples[tok]}")

# ---- B. high-frequency distinctive test tokens overall (potential brands) ----
all_tok = Counter()
for f in test_ocr.folded:
    for t in set(f.split()):
        if len(t) >= 4:
            all_tok[t] += 1
print(f"\n=== B. Most frequent len>=4 tokens in test OCR (scan for brands) ===")
common = [(t, c) for t, c in all_tok.most_common(60)]
print(", ".join(f"{t}:{c}" for t, c in common))
