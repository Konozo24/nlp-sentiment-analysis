"""Print label distributions for the cleaned, hand-labelled dataset.

Run:  python scripts/dataset_characteristics.py
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_tweets.csv"
LABEL_COLUMNS = ["sentiment", "emotion", "topic"]


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, encoding="utf-8", usecols=LABEL_COLUMNS)

    print(f"Dataset: {DATA_PATH}")
    print(f"Rows: {len(df):,}")

    for column in LABEL_COLUMNS:
        counts = df[column].value_counts()
        percentages = df[column].value_counts(normalize=True).mul(100)

        result = pd.DataFrame(
            {
                "Count": counts,
                "Percentage": percentages.round(2),
            }
        )

        print(f"\n===== {column.upper()} =====")
        print(result.to_string())


if __name__ == "__main__":
    main()