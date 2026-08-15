"""Minimal cleaning for TRABSA (Twitter-RoBERTa + Attention + BiLSTM) — Ming.

TRABSA's encoder, cardiffnlp/twitter-roberta-base, is a transformer with a
subword tokenizer pretrained on ~58M real tweets — it already handles
casing, punctuation, and emoji as signal. Aggressive normalization (as used
for the GloVe-based baseline) would throw that signal away and mismatch the
encoder's pretraining distribution. So we only:
  - replace links with the literal token 'http' (not delete them — this
    matches cardiffnlp/twitter-roberta-base's own pretraining preprocessing),
  - replace @mentions with a generic '@user' token (same convention),
  - keep the '#' word, emoji, casing, and punctuation as-is.

Run:  python -m src.data_cleaning.preprocess_trabsa
"""

from .pipeline import preprocess_dataset, run_steps
from .utils import (
    normalize_whitespace,
    remove_hashtag_symbol,
    replace_links_with_token,
    replace_mentions_with_token,
)

STEPS = [
    replace_links_with_token,
    replace_mentions_with_token,
    remove_hashtag_symbol,
    normalize_whitespace,
]


def clean_for_trabsa(text: str) -> str:
    return run_steps(text, STEPS)


if __name__ == "__main__":
    preprocess_dataset(STEPS, "trabsa_input.csv")
