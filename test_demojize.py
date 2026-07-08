import csv
from pathlib import Path

from src.data_cleaning.text_cleaning import demojize_emoji

CSV_PATH = Path(__file__).resolve().parent / "data" / "raw" / "worldcup2018_tweets.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
    reader = csv.DictReader(csv_file)
    for i, row in enumerate(reader):
        if i >= 10:
            break
        raw_content = row.get("rawContent", "")
        lang = row.get("lang", "en") or "en"
        demojized = demojize_emoji(raw_content, language=lang)
        print(f"ROW {i + 1}")
        print("ORIGINAL:", raw_content)
        print("DEMOJIZED:", demojized)
        print("LANG:", lang)
        print("-")
