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

## Modelling pipeline and fair evaluation

Three models are compared on four tasks (sentiment, emotion, topic, ner). For the
comparison to mean anything, the row set and the split are frozen **once**, upstream of
every model:

```bash
python -m src.data_cleaning.base_cleaning        # clean, dedup, stamp split  -> 27,582 rows
python -m src.data_cleaning.preprocess_svm       # pure text rewrite, row count unchanged
python -m src.data_cleaning.preprocess_bilstm
python -m src.data_cleaning.preprocess_robertacnn

python -m src.models.svm.train                   # then .evaluate for each model
python scripts/compare_models.py                 # -> data/models/comparison.md
```

What makes the numbers comparable:

- **One row set.** Dedup runs in `base_cleaning.py` on `utils.canonical_key()` — a
  normalisation harsher than any single model's — *before* the split is stamped. The
  per-model `preprocess_*.py` stages are pure text rewrites and may not add or drop a row
  (`pipeline.py` raises if they do). All three `*_input.csv` files therefore carry identical
  ids, and all three models are scored on the same 4,130 test tweets.
- **Group-aware split.** Near-duplicate tweets are clustered by `utils.group_key()` and kept
  on one side of the 70/15/15 partition, so a tweet cannot appear in train with its near-twin
  in test.
- **One metric definition.** Every headline number comes from `src/models/metrics.py`:
  Accuracy + macro P/R/F1 per classification task, and entity-type presence F1 for NER — the
  only NER framing all three models share. `compare_models.py` refuses to build a table if
  the models report different test-set sizes.
- **Matched class weighting.** All three use `min(n / (k * count), 10.0)` inverse-frequency
  weights, on CPU and GPU alike.

Known method difference to state in the report: the SVM does not use the `val` slice
(LinearSVC has no early stopping), while the BiLSTM and RoBERTa-CNN early-stop on it.

---

## Text cleaning utilities

`src/data_cleaning/utils.py` holds the atomic text helpers. Each is pure — string in,
string out — and the `preprocess_*.py` pipelines compose them into per-model chains.

```python
from src.data_cleaning.utils import demojize_to_token

text = "I love ronaldao 🏐"
cleaned = demojize_to_token(text, language="en")
print(cleaned)
```

Two helpers are not part of any model's chain and exist only to decide the shared row set:
`canonical_key()` (dedup key) and `group_key()` (near-duplicate cluster key). See the
section above.
