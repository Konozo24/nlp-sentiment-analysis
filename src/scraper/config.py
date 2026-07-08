from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR    = PROJECT_ROOT / "data"
RAW_DIR     = DATA_DIR / "raw"
ACCOUNTS_DB = DATA_DIR / "accounts.db"
OUTPUT_CSV  = RAW_DIR / "world_cup_fifa_2018_tweets.csv"

SEARCH_QUERIES = [
    "FIFA World Cup 2018",
    "#FIFAWorldCup2018",
    "#Russia2018",
]

LIMIT_PER_QUERY = 700
DATE_SINCE: str | None = "2026-01-01"
DATE_UNTIL: str | None = None   # e.g. "2024-12-31"
EXCLUDE_RETWEETS = True           # appends -is:retweet to every query
EXCLUDE_OFFICIAL_ACCOUNT = True   # appends -from:MyRapidKL; removes their own announcements

CSV_COLUMNS = [
    "id",
    "rawContent",
    "date",
    "username",
    "likeCount",
    "retweetCount",
    "replyCount",
    "lang",
]
