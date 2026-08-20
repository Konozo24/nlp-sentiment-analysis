"""Shared structural cleaning of merged_tweets.csv -> cleaned_tweets.csv.

Structural only (drop empties, strip 'RT @user:', normalize whitespace, dedup);
each model's preprocess_*.py adds its own text cleaning on top. Also stamps the
train/val/test `split` per tweet id here, before model-specific dedup, so all
three models share the same test set.

Run:  python -m src.data_cleaning.base_cleaning
"""

from pathlib import Path

import pandas as pd

from .splits import attach_split_column, summarise
from .utils import normalize_whitespace, remove_rt_prefix, unescape_html

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MERGED_PATH = PROCESSED_DIR / "merged_tweets.csv"
CLEANED_PATH = PROCESSED_DIR / "cleaned_tweets.csv"


def structural_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning that every model wants, with no text mutation
    beyond whitespace/RT-prefix normalization."""
    df = df.copy()
    df["tweet"] = df["tweet"].fillna("").astype(str)

    # Strip the 'RT @user:' prefix so a retweet and its original tweet
    # become identical strings and collapse in the dedup step below.
    # HTML entities ('&amp;', '&lt;3') are decoded so no model ever sees
    # literal 'amp'/'lt' tokens.
    df["tweet"] = df["tweet"].map(unescape_html).map(remove_rt_prefix).map(normalize_whitespace)

    df = df[df["tweet"].str.len() > 0]
    df = df.drop_duplicates(subset="tweet", keep="first")
    return df.reset_index(drop=True)


def build_cleaned_dataset(merged_path: Path = MERGED_PATH, out_path: Path = CLEANED_PATH) -> pd.DataFrame:
    if not merged_path.exists():
        raise FileNotFoundError(f"{merged_path} not found — run 'python scripts/merge_datasets.py' first.")

    df = pd.read_csv(merged_path, encoding="utf-8")
    if "tweet" not in df.columns:
        raise ValueError(f"{merged_path} has no 'tweet' column (columns: {list(df.columns)})")

    before = len(df)
    df = structural_clean(df)
    print(f"Cleaned {before} rows -> {len(df)} after removing empties/duplicates")

    before_lang = len(df)
    df = df[df["lang"] == "en"].reset_index(drop=True)
    print(f"Filtered {before_lang} rows -> {len(df)} keeping lang == 'en'")

    # stamp the shared train/val/test assignment before any model sees the data
    df = attach_split_column(df)
    summarise(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig adds a BOM so Excel displays emoji/non-ASCII correctly
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved cleaned dataset to {out_path}")
    return df


if __name__ == "__main__":
    build_cleaned_dataset()
