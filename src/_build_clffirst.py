"""Build submission with the CV-validated clf_first product head (composite 0.6017
vs 0.5992 classifier). Classifier output untouched; friend's rules recover only
the rows the classifier leaves empty. OCR = existing vietocr_ft test cache.
"""
import csv

import config
from build_submission import validate, write_submission_csv
from data import load_sample_submission, load_test_ids, load_train_labels
from empty_gate import EmptyGate
from product_hybrid import HybridProductExtractor
from run_ocr import cache_path
import pandas as pd

ENGINE, TAG = "vietocr_ft", "v9_clffirst"
MIN_CC, GATE_THR_CLF, GATE_THR_EMPTY = 12, 0.55, 0.6

labels = load_train_labels()
head = HybridProductExtractor(min_class_count=MIN_CC, gate_threshold=GATE_THR_CLF,
                              order="clf_first").fit(labels[["image_id", "ocr_text", "product_name"]])

# trained empty-gate (same as deployed CV)
feats = labels.merge(pd.read_parquet(cache_path(ENGINE, "all")), on="image_id")
feats["gt_empty"] = (feats["ocr_text_x"].fillna("").str.strip() == "").astype(int)
gate = EmptyGate(threshold=GATE_THR_EMPTY).fit(
    feats.rename(columns={"ocr_text_y": "ocr_text"}), feats["gt_empty"])

test_ids = load_test_ids()
ocr_test = pd.read_parquet(cache_path(ENGINE, "test"))
n_missing = len(set(test_ids.image_id) - set(ocr_test.image_id))
print(f"test missing from OCR cache: {n_missing}")
sub = test_ids.merge(ocr_test, on="image_id", how="left")
sub["ocr_text"] = sub["ocr_text"].fillna("")
if "mean_conf" not in sub:
    sub["mean_conf"], sub["n_boxes"] = 0.0, 0

mask = gate.is_empty(sub)
sub.loc[mask, "ocr_text"] = ""
print(f"empty-gate zeroed {int(mask.sum())} test rows")

sub["product_name"] = head.predict_batch(sub["ocr_text"])
sub = sub[["image_id", "ocr_text", "product_name"]]

sample = load_sample_submission()
print(f"Validating '{TAG}'...")
assert validate(sub, sample), "validation failed"
sub = sub.set_index("image_id").reindex(sample["image_id"]).reset_index()

out_path = config.SUBMISSIONS_DIR / f"submission_{TAG}.csv"
write_submission_csv(sub, out_path)
print(f"\nSaved {out_path}")
print(f"rows={len(sub)} | OCR fill={(sub.ocr_text.str.strip()!='').mean():.1%} | "
      f"product fill={(sub.product_name.str.strip()!='').mean():.1%}")

# diff vs v8 (current best) so we know how much actually changed
import pandas as pd2
v8 = pd.read_csv(config.SUBMISSIONS_DIR / "submission_v8.csv", keep_default_na=False, dtype=str)
m = v8.merge(sub, on="image_id", suffixes=("_v8", "_new"))
prod_changed = (m["product_name_v8"].str.strip() != m["product_name_new"].str.strip()).sum()
ocr_changed = (m["ocr_text_v8"].str.strip() != m["ocr_text_new"].str.strip()).sum()
adds = ((m["product_name_v8"].str.strip() == "") & (m["product_name_new"].str.strip() != "")).sum()
removes = ((m["product_name_v8"].str.strip() != "") & (m["product_name_new"].str.strip() == "")).sum()
print(f"\nvs v8: product changed={prod_changed} (adds={adds}, removes={removes}) | ocr changed={ocr_changed}")
