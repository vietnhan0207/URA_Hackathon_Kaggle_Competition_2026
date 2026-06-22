"""Compare VietOCR vs PaddleOCR on TEST (no labels) — fill, length, diacritic density,
and sample rows. Diacritic density is a proxy for Vietnamese recognition quality."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from run_ocr import cache_path

viet = pd.read_parquet(cache_path("vietocr", "test"))[["image_id", "ocr_text"]].set_index("image_id")["ocr_text"].fillna("")
padd = pd.read_parquet(cache_path("paddleocr", "test"))[["image_id", "ocr_text"]].set_index("image_id")["ocr_text"].fillna("")
ids = sorted(set(viet.index) & set(padd.index))

DIAC = set("ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
           "ĂÂĐÊÔƠƯÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ")


def stats(series):
    txt = [series[i] for i in ids]
    fill = sum(1 for t in txt if t.strip()) / len(txt)
    avglen = sum(len(t) for t in txt) / len(txt)
    total_chars = sum(len(t) for t in txt)
    diac = sum(sum(1 for c in t if c in DIAC) for t in txt)
    return fill, avglen, (diac / total_chars if total_chars else 0)


for name, s in [("PaddleOCR", padd), ("VietOCR", viet)]:
    fill, avglen, dd = stats(s)
    print(f"{name:>10}: fill {fill:.1%} | avg len {avglen:.0f} | diacritic density {dd:.3%}")

print("\n--- sample rows (first 6 non-empty in both) ---")
shown = 0
for i in ids:
    if viet[i].strip() and padd[i].strip() and shown < 6:
        print(f"[{i}]")
        print("  PADDLE :", padd[i][:110])
        print("  VIETOCR:", viet[i][:110])
        shown += 1
