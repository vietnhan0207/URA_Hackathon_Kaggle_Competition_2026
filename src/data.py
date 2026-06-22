"""Data loading + train/val split.

Key concern: TikTok thumbnails repeat (cover / origin_cover / music_cover /
dynamic_cover of the same video share near-identical text). A naive random
split leaks those across train/val and inflates the score. We group by the
normalized ocr_text so identical-text images stay on the same side.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from config import (
    RANDOM_SEED,
    SAMPLE_SUBMISSION_CSV,
    SPLITS_DIR,
    TEST_CSV,
    TRAIN_LABELS_CSV,
    VAL_FRACTION,
)


def load_train_labels() -> pd.DataFrame:
    """image_id, ocr_text, product_name (empty strings preserved, not NaN)."""
    df = pd.read_csv(TRAIN_LABELS_CSV, keep_default_na=False, dtype=str)
    df["ocr_text"] = df["ocr_text"].astype(str).str.strip()
    df["product_name"] = df["product_name"].astype(str).str.strip()
    return df


def load_test_ids() -> pd.DataFrame:
    return pd.read_csv(TEST_CSV, keep_default_na=False, dtype=str)


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_SUBMISSION_CSV, keep_default_na=False, dtype=str)


def _group_key(text: str, image_id: str = "") -> str:
    """Group near-identical thumbnails by their normalized text.

    Empty-text rows each get a UNIQUE key (their own group, keyed by image_id) so
    they spread proportionally across the split instead of colliding into one giant
    group (which would put all empties on one side and bias val).
    """
    t = " ".join(str(text).lower().split())
    if not t:
        return "empty_" + str(image_id)
    return hashlib.md5(t.encode("utf-8")).hexdigest()


def make_split(df: pd.DataFrame | None = None,
               val_fraction: float = VAL_FRACTION,
               seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Return df with an added 'split' column ('train'/'val'), grouped by text."""
    if df is None:
        df = load_train_labels()
    df = df.copy()
    df["group"] = [_group_key(t, iid) for iid, t in zip(df["image_id"], df["ocr_text"])]

    # Assign whole groups to val until we reach ~val_fraction of rows.
    # Empty-text rows (unique groups) are assigned per-row to hit the target.
    rng = pd.Series(df["group"].unique())
    rng = rng.sample(frac=1.0, random_state=seed).tolist()

    group_sizes = df.groupby("group").size().to_dict()
    target_val = int(len(df) * val_fraction)
    val_groups, running = set(), 0
    for g in rng:
        if running >= target_val:
            break
        val_groups.add(g)
        running += group_sizes[g]

    df["split"] = df["group"].map(lambda g: "val" if g in val_groups else "train")
    return df


def save_split(df_split: pd.DataFrame) -> None:
    for name in ("train", "val"):
        ids = df_split.loc[df_split["split"] == name, "image_id"]
        ids.to_csv(SPLITS_DIR / f"{name}_ids.txt", index=False, header=False)


def load_split_ids(name: str) -> set[str]:
    path = SPLITS_DIR / f"{name}_ids.txt"
    return set(path.read_text(encoding="utf-8").split())
