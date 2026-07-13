"""Merge the 4 World Cup tweet datasets (2014, 2018, 2022, 2026) into one CSV.

Output: data/processed/merged_tweets.csv with columns: id, tweet, date, year
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/merged_tweets.csv")


def load_year(year: int) -> pd.DataFrame:
    path = RAW_DIR / f"worldcup{year}_tweets.csv"

    df = pd.read_csv(path)
    df = df.rename(columns={"rawContent": "tweet"})
    if "lang" not in df.columns:  # 2022 Kaggle file has no lang column
        df["lang"] = pd.NA
    df = df[["tweet", "date", "lang"]]
    df["year"] = year

    # normalize timestamps to a single UTC format
    df["date"] = pd.to_datetime(df["date"], utc=True, format="mixed", errors="coerce")

    return df[["tweet", "date", "lang", "year"]]


def main() -> None:
    frames = [load_year(y) for y in (2014, 2018, 2022, 2026)]
    merged = pd.concat(frames, ignore_index=True)

    before = len(merged)
    merged = merged.dropna(subset=["tweet"])
    merged = merged[merged["tweet"].str.strip() != ""]
    merged = merged.drop_duplicates(subset=["tweet"])
    print(f"Dropped {before - len(merged)} empty/duplicate tweets")

    # simple sequential id: 1..N
    merged = merged.reset_index(drop=True)
    merged.insert(0, "id", merged.index + 1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig adds a BOM so Excel displays emoji/non-ASCII correctly
    merged.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved {len(merged)} rows -> {OUT_PATH}")
    print(merged["year"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
