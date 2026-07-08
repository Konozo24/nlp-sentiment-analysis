from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR    = PROJECT_ROOT / "data"
RAW_DIR     = DATA_DIR / "raw"
ACCOUNTS_DB = DATA_DIR / "accounts.db"
OUTPUT_CSV  = RAW_DIR / "worldcup2018_tweets.csv"

SEARCH_QUERIES = [
    "FIFA World Cup 2018",
]

LIMIT_PER_QUERY = 10000
DATE_SINCE: str | None = "2018-06-01"
DATE_UNTIL: str | None = "2018-07-31"   # World Cup 2018 date range
EXCLUDE_RETWEETS = True           # appends -is:retweet to every query
OFFICIAL_ACCOUNT_TO_EXCLUDE: str | None = None  # set to username to exclude tweets from that account

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
