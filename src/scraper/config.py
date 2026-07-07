from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR    = PROJECT_ROOT / "data"
RAW_DIR     = DATA_DIR / "raw"
ACCOUNTS_DB = DATA_DIR / "accounts.db"
OUTPUT_CSV  = RAW_DIR / "worldcup_tweets.csv"

SEARCH_QUERIES = [
    "#WorldCup2026",
]

LIMIT_PER_QUERY = 600
DATE_SINCE: str | None
DATE_UNTIL: str | None  # e.g. "2024-12-31"
EXCLUDE_RETWEETS = True           # appends -is:retweet to every query
EXCLUDE_OFFICIAL_ACCOUNT = False  # set True + EXCLUDE_ACCOUNT below to drop one account's own posts
EXCLUDE_ACCOUNT: str | None = None  # e.g. "MyRapidKL"

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
