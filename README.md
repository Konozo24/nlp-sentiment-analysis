# NLP Sentiment Analysis — Rapid KL Tweets

Collects tweets about Rapid KL (Malaysian public transit) from Twitter/X and runs NLP sentiment analysis on them.

See [scraper.md](scraper.md) for full scraper documentation.

---

## Quickstart (Scraper)

**1. Set up the environment**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**2. Add your Twitter account** (one-time, each teammate does this once)
```bash
python -m src.scraper.add_accounts
```
Choose **Mode 1** (cookie-based). Get your cookies from:
> twitter.com → F12 → Application → Cookies → `https://twitter.com` → copy `auth_token` and `ct0`

Paste them as: `auth_token=abc123; ct0=xyz456`

**3. Run the scraper**
```bash
python -m src.scraper.scraper
```

Output is saved to `data/raw/rapidkl_tweets.csv`. Re-running is safe — it skips tweets already collected.
