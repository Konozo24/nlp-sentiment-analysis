# Rapid KL Twitter Scraper

Web crawler for collecting Rapid KL-related tweets from Twitter/X as a dataset for NLP sentiment analysis. Built with [twscrape](https://github.com/vladkens/twscrape).

---

## File Structure

```
src/scraper/
├── __init__.py       — makes the package importable from project root
├── config.py         — all settings (queries, paths, limits, filters)
├── scraper.py        — main scraper logic: search → deduplicate → save to CSV
└── add_accounts.py   — one-time account setup helper
```

---

## Setup (One-Time)

Before scraping, you need to add at least one Twitter/X account to authenticate requests.

```bash
venv\Scripts\activate
python -m src.scraper.add_accounts
```

When prompted:
1. Choose **Mode 1** (cookie-based, recommended)
2. Enter any label as the username (e.g. `myaccount`)
3. Paste your cookies string in format: `auth_token=abc123; ct0=xyz456`

**How to get cookies:**
> Open twitter.com → F12 → Application tab → Cookies → `https://twitter.com` → copy `auth_token` and `ct0` values

---

## Running the Scraper

```bash
python -m src.scraper.scraper
```

Run from the project root. Output is saved to `data/raw/rapidkl_tweets.csv`.

Re-running the script is safe — it only appends new tweets and skips any already collected.

---

## How It Works

```
main()
 │
 ├─ 1. load_seen_ids(csv_path)
 │      Reads existing CSV → builds a set of already-collected tweet IDs
 │      Prevents duplicates across multiple runs
 │
 ├─ 2. API(accounts.db)
 │      Loads Twitter account(s) for authentication
 │
 ├─ 3. Opens CSV in append mode; writes header only on first run
 │
 └─ 4. For each query in SEARCH_QUERIES:
         │
         ├─ build_query()
         │    Constructs the final Twitter search string:
         │    e.g. "to:MyRapidKL -is:retweet -from:MyRapidKL"
         │
         ├─ api.search(query, limit=500)
         │    Sends query to Twitter, paginates automatically
         │
         ├─ For each tweet returned:
         │    - Skip if ID already seen (deduplication)
         │    - Write row to CSV
         │
         └─ Flush CSV to disk after each query
              (partial results are saved even if a later query fails)
```

Each query is wrapped in a `try/except` — if one query hits a rate limit or fails, the scraper moves on to the next query instead of crashing.

---

## Search Queries

Defined in `src/scraper/config.py`:

| Query | Purpose |
|---|---|
| `to:MyRapidKL` | Direct complaints/praise to the official account — highest sentiment signal |

### Automatic Filters Applied to Every Query

| Filter | Effect |
|---|---|
| `-is:retweet` | Excludes retweets — keeps only original tweets for cleaner text diversity |
| `-from:MyRapidKL` | Excludes Rapid KL's own posts — their announcements are neutral, not opinionated |

The same tweet matched by multiple queries is only saved once (deduplicated by tweet ID).

---

## CSV Output   

Saved to `data/raw/rapidkl_tweets.csv`.

| Column | Description |
|---|---|
| `id` | Unique tweet ID |
| `rawContent` | Full tweet text — primary input for sentiment labeling |
| `date` | Timestamp in ISO 8601 format |
| `username` | Twitter handle of the author |
| `likeCount` | Number of likes |
| `retweetCount` | Number of retweets |
| `replyCount` | Number of replies |
| `lang` | Language detected by Twitter (`en`, `ms`, etc.) |

---

## Configuration

All settings are in `src/scraper/config.py`:

| Setting | Default | Description |
|---|---|---|
| `LIMIT_PER_QUERY` | `700` | Max tweets to collect per search term |
| `DATE_SINCE` | `None` | Filter tweets after this date, e.g. `"2024-01-01"` |
| `DATE_UNTIL` | `None` | Filter tweets before this date, e.g. `"2024-12-31"` |
| `EXCLUDE_RETWEETS` | `True` | Append `-is:retweet` to all queries |
| `EXCLUDE_OFFICIAL_ACCOUNT` | `True` | Append `-from:MyRapidKL` to all queries |
| `OUTPUT_CSV` | `data/raw/rapidkl_tweets.csv` | Output file path |
| `ACCOUNTS_DB` | `data/accounts.db` | twscrape session database |
