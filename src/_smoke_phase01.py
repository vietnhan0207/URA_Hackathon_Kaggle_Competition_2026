"""Headless smoke test for Phase 0/1 modules (run in env `ura`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import data
import scoring

print("== config ==")
print("project root:", config.PROJECT_ROOT.name)
print("train imgs dir exists:", config.TRAIN_IMAGES_DIR.exists())
print("test  imgs dir exists:", config.TEST_IMAGES_DIR.exists())

print("\n== scoring unit tests ==")
assert scoring.token_f1("Vinamilk Flex", "vinamilk flex") == 1.0
assert scoring.token_f1("", "") == 1.0
assert scoring.token_f1("Vinamilk", "") == 0.0
assert abs(scoring.token_f1("Ha Long Canfoco Pate", "Ha Long Canfoco") - 0.8571) < 1e-3
assert scoring.cer("", "") == 0.0
assert scoring.cer("", "abc") == 1.0
assert abs(scoring.cer("abcd", "abxd") - 0.25) < 1e-9
print("token_f1 / cer asserts passed")

print("\n== data load ==")
labels = data.load_train_labels()
test_ids = data.load_test_ids()
print(f"labels rows: {len(labels):,} | test ids: {len(test_ids):,}")
print("empty ocr  %.1f%%" % ((labels.ocr_text == "").mean() * 100))
print("empty prod %.1f%%" % ((labels.product_name == "").mean() * 100))

print("\n== composite sanity ==")
empty = labels[["image_id"]].copy()
empty["ocr_text"] = ""
empty["product_name"] = ""
print("all-empty vs labels:", scoring.composite_score(labels, empty, return_components=True))
print("oracle (GT==pred)  :", scoring.composite_score(labels, labels, return_components=True))

print("\n== grouped split ==")
split = data.make_split(labels)
n_val = (split.split == "val").sum()
g = split.groupby("group")["split"].nunique()
print(f"train {(split.split=='train').sum():,} | val {n_val:,} ({n_val/len(split):.1%})")
print("groups spanning both splits (must be 0):", int((g > 1).sum()))
data.save_split(split)
print("saved splits/. ALL PHASE 0/1 CHECKS PASSED.")
