"""Build the LB-tailored submission: calibrated dominant-family rules + strict
long-tail classifier fallback, on the cached vietocr_ft TEST OCR.
CV composite 0.6142 (vs v8 0.5992) -> projected LB ~0.638.
"""
import pandas as pd

import config
from build_submission import validate, write_submission_csv
from data import load_sample_submission, load_test_ids, load_train_labels
from empty_gate import EmptyGate
from product_calibrated import CalibratedRuleHead
from run_ocr import cache_path

ENGINE, TAG = "vietocr_ft", "v10_calibrated"

labels = load_train_labels()
head = CalibratedRuleHead(use_classifier_fallback=True, min_pprod=0.55,
                          gate_threshold=0.75).fit(
    labels[["image_id", "ocr_text", "product_name"]])

print(f"Deployed {len(head.rules)} calibrated rules:")
for name, _pat, form, p in head.rules:
    print(f"  {name:<28} p_prod={p:.3f} -> '{form}'")

# empty-gate (same as CV/deployed)
feats = labels.merge(pd.read_parquet(cache_path(ENGINE, "all")), on="image_id")
feats["gt_empty"] = (feats["ocr_text_x"].fillna("").str.strip() == "").astype(int)
gate = EmptyGate(threshold=0.6).fit(
    feats.rename(columns={"ocr_text_y": "ocr_text"}), feats["gt_empty"])

test_ids = load_test_ids()
ocr_test = pd.read_parquet(cache_path(ENGINE, "test"))
sub = test_ids.merge(ocr_test, on="image_id", how="left")
sub["ocr_text"] = sub["ocr_text"].fillna("")
if "mean_conf" not in sub:
    sub["mean_conf"], sub["n_boxes"] = 0.0, 0
mask = gate.is_empty(sub)
sub.loc[mask, "ocr_text"] = ""
print(f"\nempty-gate zeroed {int(mask.sum())} test OCR rows")

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

# product distribution of what we emit
print("\nTop emitted products:")
vc = sub[sub.product_name.str.strip() != ""].product_name.value_counts().head(12)
for n, c in vc.items():
    print(f"  {c:>4}  {n}")

# diff vs v8
v8 = pd.read_csv(config.SUBMISSIONS_DIR / "submission_v8.csv", keep_default_na=False, dtype=str)
m = v8.merge(sub, on="image_id", suffixes=("_v8", "_new"))
chg = (m.product_name_v8.str.strip() != m.product_name_new.str.strip()).sum()
adds = ((m.product_name_v8.str.strip() == "") & (m.product_name_new.str.strip() != "")).sum()
rem = ((m.product_name_v8.str.strip() != "") & (m.product_name_new.str.strip() == "")).sum()
swap = ((m.product_name_v8.str.strip() != "") & (m.product_name_new.str.strip() != "")
        & (m.product_name_v8.str.strip() != m.product_name_new.str.strip())).sum()
ocrchg = (m.ocr_text_v8.str.strip() != m.ocr_text_new.str.strip()).sum()
print(f"\nvs v8: product changed={chg} (adds={adds}, removes={rem}, swaps={swap}) | ocr changed={ocrchg}")
