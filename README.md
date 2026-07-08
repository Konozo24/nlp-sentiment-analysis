# NLP Sentiment Analysis — World Cup 2018 Tweets

Collects tweets about the 2018 FIFA World Cup from Twitter/X and runs NLP sentiment analysis on them.

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

Output is saved to `data/raw/worldcup2018_tweets.csv`. Re-running is safe — it skips tweets already collected.

---

## Text cleaning utilities

This project now includes a simple emoji demojization helper in `src/data_cleaning/utils.py`.

Example usage:

```python
from src.data_cleaning.utils import demojize_emoji

text = "I love ronaldao 🏐"
cleaned = demojize_emoji(text, language="en")
print(cleaned)
```

Currently available helper:
- `demojize_emoji(text, language="en")`

More data-cleaning helpers can be added later as needed.
