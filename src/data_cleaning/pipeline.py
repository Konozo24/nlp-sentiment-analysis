"""Shared machinery for running a model-specific cleaning pipeline.

Each preprocess_*.py defines a STEPS list of atomic helpers from utils.py;
this module runs those steps over the merged dataset and writes the
model-ready CSV to data/processed/.
"""

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from .base_cleaning import CLEANED_PATH, PROCESSED_DIR

Step = Callable[[str], str]


def run_steps(text: str, steps: list[Step]) -> str:
    for step in steps:
        text = step(text)
    return text


def preprocess_dataset(steps: list[Step], out_name: str, cleaned_path: Path = CLEANED_PATH) -> pd.DataFrame:
    """Apply a step chain to the cleaned dataset and save the result.

    Rows whose text becomes empty after cleaning (e.g. a tweet that was
    only a URL) are dropped, as are duplicates created by the cleaning.
    """
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"{cleaned_path} not found — run 'python -m src.data_cleaning.base_cleaning' first."
        )

    df = pd.read_csv(cleaned_path, encoding="utf-8")
    df["tweet"] = df["tweet"].fillna("").astype(str)
    df["clean_text"] = df["tweet"].map(lambda t: run_steps(t, steps))

    before = len(df)
    df = df[df["clean_text"].str.len() > 0]
    df = df.drop_duplicates(subset="clean_text", keep="first").reset_index(drop=True)
    print(f"{out_name}: {before} rows -> {len(df)} after cleaning")

    out_path = PROCESSED_DIR / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved to {out_path}")
    return df
