"""Mine the product_name space on the TRAIN split only (no val leakage).

token-F1 is case-insensitive set F1, so we group product_name variants by their
lowercased token-set key and pick the MODAL surface form per group. Predicting the
modal form maximizes expected F1 across the variant spellings in GT.
Writes a UTF-8 report.
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from data import load_train_labels, load_split_ids


def tokkey(s: str) -> str:
    return " ".join(sorted(str(s).lower().split()))


labels = load_train_labels()
train_ids = load_split_ids("train")
tr = labels[labels.image_id.isin(train_ids)]
prod = tr[tr.product_name != ""]

# group variants by token-set key -> modal surface form + total count
groups = defaultdict(Counter)
for p in prod.product_name:
    groups[tokkey(p)][p] += 1

rows = []
for key, forms in groups.items():
    total = sum(forms.values())
    modal = forms.most_common(1)[0][0]
    rows.append((total, modal, len(forms), dict(forms)))
rows.sort(reverse=True)

out = Path(__file__).resolve().parents[1] / "cache" / "product_mining.txt"
lines = []
lines.append(f"TRAIN split: {len(tr)} rows | {len(prod)} with product | "
             f"{prod.product_name.nunique()} raw distinct | {len(groups)} token-set groups")
lines.append(f"empty product rate (train): {(tr.product_name=='').mean():.1%}")
cov = sum(r[0] for r in rows[:30]) / len(prod)
lines.append(f"top-30 token-groups cover {cov:.1%} of all product rows\n")
lines.append("RANK  COUNT  MODAL_FORM  (#variants)  variants")
for i, (total, modal, nvar, forms) in enumerate(rows[:50], 1):
    lines.append(f"{i:>3}  {total:>4}  {modal!r}  ({nvar})  {forms if nvar>1 else ''}")

out.write_text("\n".join(lines), encoding="utf-8")
print("wrote", out)
print("\n".join(lines[:6]))
