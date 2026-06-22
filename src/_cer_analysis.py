"""Analyze CER composition for a cached engine on val: best/worst rows + an
estimate of how much CER comes from EXTRA predicted tokens vs MISSED/wrong tokens.
Writes a UTF-8 report file (avoids Windows console mojibake).
Usage: python _cer_analysis.py <engine>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from data import load_train_labels
from scoring import cer
from run_ocr import cache_path

engine = sys.argv[1] if len(sys.argv) > 1 else "paddleocr"
_snap = cache_path(engine, "val").parent / "_snap_paddle_val.parquet"
cache = pd.read_parquet(_snap if _snap.exists() else cache_path(engine, "val"))
gt = load_train_labels().set_index("image_id")

rows = []
for _, r in cache.iterrows():
    g = str(gt.loc[r.image_id, "ocr_text"])
    p = str(r.ocr_text)
    gt_tok = set(g.lower().split())
    p_tok = set(p.lower().split())
    extra = p_tok - gt_tok       # predicted but not in GT (noise)
    missed = gt_tok - p_tok      # in GT but not predicted
    rows.append({
        "image_id": r.image_id, "cer": cer(g, p),
        "gt_len": len(g), "pred_len": len(p),
        "n_extra_tok": len(extra), "n_missed_tok": len(missed),
        "gt": g, "pred": p,
    })
df = pd.DataFrame(rows).sort_values("cer")

out = Path(__file__).resolve().parents[1] / "cache" / f"cer_analysis_{engine}.txt"
lines = []
lines.append(f"ENGINE={engine}  n={len(df)}  mean_cer={df.cer.mean():.3f}")
lines.append(f"empty GT rows: {(df.gt_len==0).sum()} | empty pred rows: {(df.pred_len==0).sum()}")
lines.append(f"mean extra tokens/img: {df.n_extra_tok.mean():.2f} | mean missed tokens/img: {df.n_missed_tok.mean():.2f}")
lines.append(f"rows where pred longer than GT: {(df.pred_len>df.gt_len).mean():.1%}")
lines.append("\n===== 8 BEST (low CER) =====")
for _, r in df.head(8).iterrows():
    lines.append(f"[{r['cer']:.2f}] {r['image_id']}  extra={r['n_extra_tok']} missed={r['n_missed_tok']}")
    lines.append(f"   GT  : {r['gt'][:140]}")
    lines.append(f"   PRED: {r['pred'][:140]}")
lines.append("\n===== 12 WORST (high CER, non-empty GT) =====")
worst = df[df.gt_len > 0].sort_values("cer", ascending=False).head(12)
for _, r in worst.iterrows():
    lines.append(f"[{r['cer']:.2f}] {r['image_id']}  extra={r['n_extra_tok']} missed={r['n_missed_tok']}")
    lines.append(f"   GT  : {r['gt'][:140]}")
    lines.append(f"   PRED: {r['pred'][:140]}")

out.write_text("\n".join(lines), encoding="utf-8")
print("wrote", out)
print("\n".join(lines[:5]))
