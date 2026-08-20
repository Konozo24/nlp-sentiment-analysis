"""Dataset partitioning and split allocation module.

Generates and attaches deterministic, stratified train/validation/test 
split labels based on unique tweet IDs to prevent partition drift.

Usage:
    $ python -m src.data_cleaning.splits
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_PATH = PROJECT_ROOT / "data" / "processed" / "splits.csv"

SEED = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15
STRATIFY_ON = "sentiment"


def build_splits(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """
    Assign every row a split label. Returns a frame of just `id` and `split`.
    Generate a stratified 70/15/15 train, validation, and test split.
    """

    holdout = VAL_SIZE + TEST_SIZE
    train_df, rest = train_test_split(
        df, test_size=holdout, stratify=df[STRATIFY_ON], random_state=seed
    )
    val_df, test_df = train_test_split(
        rest, test_size=TEST_SIZE / holdout, stratify=rest[STRATIFY_ON], random_state=seed
    )

    labelled = pd.concat(
        [
            pd.DataFrame({"id": train_df["id"], "split": "train"}),
            pd.DataFrame({"id": val_df["id"], "split": "val"}),
            pd.DataFrame({"id": test_df["id"], "split": "test"}),
        ]
    )
    return labelled.sort_values("id").reset_index(drop=True)


def attach_split_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `split` column to df, reusing splits.csv so the split never moves.

    Ids present in df but missing from the file are
    assigned and appended rather than triggering a full reshuffle.
    """
    if "id" not in df.columns:
        raise ValueError(f"cannot assign splits: no 'id' column (columns: {list(df.columns)})")

    if SPLITS_PATH.exists(): 
        splits = pd.read_csv(SPLITS_PATH, encoding="utf-8")
        missing = df[~df["id"].isin(splits["id"])]
        if not missing.empty:
            print(f"{len(missing)} new ids since splits.csv was written and assigning them")
            splits = pd.concat([splits, build_splits(missing)]).sort_values("id")
            splits.to_csv(SPLITS_PATH, index=False, encoding="utf-8")
    else:
        print(f"No {SPLITS_PATH.name} yet - creating the shared split")
        splits = build_splits(df)
        SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
        splits.to_csv(SPLITS_PATH, index=False, encoding="utf-8")

    merged = df.merge(splits, on="id", how="left")
    if merged["split"].isna().any():
        raise ValueError(f"{merged['split'].isna().sum()} rows got no split label")
    return merged


def summarise(df: pd.DataFrame) -> None:
    """Print split sizes and the class balance within each stratification check."""
    print(f"\nTotal rows: {len(df):,}")
    for name in ("train", "val", "test"):
        part = df[df["split"] == name]
        share = len(part) / len(df)
        balance = (part[STRATIFY_ON].value_counts(normalize=True) * 100).round(1).to_dict()
        print(f"  {name:5s} {len(part):6,} ({share:.1%})  {balance}")


if __name__ == "__main__":
    from .base_cleaning import CLEANED_PATH

    if not CLEANED_PATH.exists():
        raise SystemExit(
            f"{CLEANED_PATH} not found, run 'python -m src.data_cleaning.base_cleaning' first."
        )
    cleaned = pd.read_csv(CLEANED_PATH, encoding="utf-8")
    summarise(attach_split_column(cleaned))
