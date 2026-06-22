"""Explain the friend's 0.6495: is the TEST set concentrated in the few products
his rules hard-code, and would his exact-canonical output beat our classifier's
partial labels on exactly those?

We have no test product labels, but we DO have:
  - train product distribution (concentration)
  - our test OCR text -> count dominant-product signatures in test
  - per-image: what our classifier outputs vs what his rules output, on test OCR
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

from data import load_train_labels, load_test_ids
from product_extract import ProductExtractor, fold
from product_rules_friend import extract_product, safe_product
from run_ocr import cache_path

labels = load_train_labels()

# ---- 1) TRAIN product distribution (how concentrated?) ----
nz = labels[labels.product_name.str.strip() != ""]
print(f"TRAIN: {len(labels)} rows | {len(nz)} have product ({len(nz)/len(labels):.1%})")
top = nz.product_name.str.strip().value_counts().head(12)
print("\nTop train products:")
for name, c in top.items():
    print(f"  {c:>4} ({c/len(nz):>5.1%})  {name[:50]}")

# ---- 2) TEST OCR: signature counts for the dominant families ----
test_ocr = pd.read_parquet(cache_path("vietocr_ft", "test"))
test_ocr["folded"] = test_ocr["ocr_text"].fillna("").map(fold)
N = len(test_ocr)
sigs = {
    "ha long / canfoco": r"canfoco|canfuco|canfood|ha long",
    "pate / cot den":    r"pate|cot den|hai phong",
    "nestle / nan":      r"nestle|\bnan\b",
    "milo":              r"milo",
    "vinamilk":          r"vinamilk",
}
print(f"\nTEST: {N} images | OCR fill {(test_ocr.ocr_text.str.strip()!='').mean():.1%}")
print("Dominant-product signatures present in TEST OCR text:")
any_dom = test_ocr["folded"].str.contains("|".join(sigs.values()), regex=True)
for label, pat in sigs.items():
    m = test_ocr["folded"].str.contains(pat, regex=True)
    print(f"  {label:<20}: {int(m.sum()):>4} ({m.mean():>5.1%} of test)")
print(f"  >>> ANY dominant sig : {int(any_dom.sum()):>4} ({any_dom.mean():.1%} of test)")

# ---- 3) On test OCR: classifier label vs friend-rule canonical label ----
clf = ProductExtractor(min_class_count=12, gate_threshold=0.55).fit(labels)
test_ocr["clf"] = clf.predict_batch(test_ocr["ocr_text"].fillna(""))
test_ocr["rule"] = [safe_product(extract_product(t), t) for t in test_ocr["ocr_text"].fillna("")]

both = test_ocr[(test_ocr.clf.str.strip() != "") & (test_ocr.rule.str.strip() != "")]
diff = both[both.clf.str.strip() != both.rule.str.strip()]
# where rule label has MORE tokens than classifier (i.e. fuller canonical form)
rule_fuller = diff[diff.rule.str.split().map(len) > diff.clf.str.split().map(len)]
print(f"\nOn test, both-nonempty rows: {len(both)}")
print(f"  classifier vs rule DIFFER : {len(diff)}")
print(f"  rule gives FULLER label   : {len(rule_fuller)}  (token-F1 upgrade candidates)")
print("\nExamples where rule label is fuller than classifier (clf -> rule):")
for _, r in rule_fuller.head(15).iterrows():
    print(f"  [{r.clf[:28]:<28}] -> [{r.rule[:38]}]")

# how many test rows would change under full rules_first vs our classifier
rules_first = []
for t in test_ocr["ocr_text"].fillna(""):
    rp = safe_product(extract_product(t), t)
    rules_first.append(rp if rp else clf.predict(t))
chg = sum(1 for a, b in zip(rules_first, test_ocr["clf"]) if str(a).strip() != str(b).strip())
print(f"\nrules_first vs classifier on test: {chg} product cells change")
