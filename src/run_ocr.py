"""Batch OCR runner with parquet caching + resume.

Caches results so we never re-run OCR to iterate on product extraction.

Usage (in env `ura`):
    python run_ocr.py --engine paddleocr --split test  --gpu 1
    python run_ocr.py --engine paddleocr --split train --gpu 1
    python run_ocr.py --engine easyocr   --split val   --gpu 1 --limit 50   # quick check

Cache file: cache/ocr_<engine>_<split>.parquet
Columns: image_id, ocr_text, raw_text, mean_conf, n_boxes, n_chars
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
from data import load_split_ids, load_test_ids, load_train_labels
from ocr_engines import get_engine


def _ids_and_dir(split: str) -> tuple[list[str], Path]:
    if split == "test":
        ids = load_test_ids()["image_id"].tolist()
        return ids, config.TEST_IMAGES_DIR
    if split in ("train", "val", "all"):
        labels = load_train_labels()
        if split == "all":
            ids = labels["image_id"].tolist()
        else:
            keep = load_split_ids(split)
            ids = [i for i in labels["image_id"] if i in keep]
        return ids, config.TRAIN_IMAGES_DIR
    raise ValueError(f"bad split: {split}")


def cache_path(engine: str, split: str) -> Path:
    return config.CACHE_DIR / f"ocr_{engine}_{split}.parquet"


def run(engine_name: str, split: str, gpu: bool = True,
        limit: int | None = None, save_every: int = 100,
        **engine_kwargs) -> pd.DataFrame:
    ids, img_dir = _ids_and_dir(split)
    if limit:
        ids = ids[:limit]
    out_path = cache_path(engine_name, split)

    done: dict[str, dict] = {}
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        done = {r["image_id"]: r for r in prev.to_dict("records")}
        print(f"Resuming: {len(done):,} already cached in {out_path.name}")

    pending = [i for i in ids if i not in done]
    print(f"Engine={engine_name} split={split} gpu={gpu} | "
          f"pending {len(pending):,} / {len(ids):,}")
    if not pending:
        print("Nothing to do.")
        return pd.DataFrame(list(done.values()))

    engine = get_engine(engine_name, gpu=gpu, **engine_kwargs) \
        if engine_name != "tesseract" else get_engine(engine_name, **engine_kwargs)

    records = list(done.values())
    t0 = time.perf_counter()
    for k, img_id in enumerate(tqdm(pending, desc=f"{engine_name}/{split}")):
        path = img_dir / f"{img_id}.jpg"
        try:
            r = engine.transcribe(path)
            rec = {"image_id": img_id, "ocr_text": r.text, "raw_text": r.raw_text,
                   "mean_conf": round(r.mean_conf, 4), "n_boxes": r.n_boxes,
                   "n_chars": r.n_chars}
        except Exception as e:  # never lose the whole run on one bad image
            rec = {"image_id": img_id, "ocr_text": "", "raw_text": "",
                   "mean_conf": 0.0, "n_boxes": 0, "n_chars": 0}
            tqdm.write(f"  [warn] {img_id}: {type(e).__name__}: {e}")
        records.append(rec)
        if (k + 1) % save_every == 0:
            pd.DataFrame(records).to_parquet(out_path, index=False)

    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)
    dt = time.perf_counter() - t0
    per = dt / max(len(pending), 1)
    print(f"\nSaved {len(df):,} rows -> {out_path}")
    print(f"Time {dt:.1f}s | {per*1000:.0f} ms/img | "
          f"OCR fill {(df.ocr_text.str.strip() != '').mean():.1%}")
    return df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True,
                   choices=["easyocr", "paddleocr", "tesseract", "rapidocr"])
    p.add_argument("--split", required=True, choices=["train", "val", "test", "all"])
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save-every", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.engine, a.split, gpu=bool(a.gpu), limit=a.limit, save_every=a.save_every)
