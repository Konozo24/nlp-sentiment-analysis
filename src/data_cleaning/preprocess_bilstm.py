"""
Run:  python -m src.data_cleaning.preprocess_bilstm
"""

from functools import partial

from .pipeline import preprocess_dataset, run_steps
from .utils import (
    demojize_to_token,
    lowercase,
    normalize_whitespace,
    remove_hashtag_symbol,
    remove_links,
    remove_unicode_punctuation,
    replace_mentions_with_token,
)

# 'usermention' rather than '@user': remove_punctuation would strip the '@'
# and leave the ordinary English word 'user', colliding with real usages.
_replace_mentions = partial(replace_mentions_with_token, token="usermention")

STEPS = [
    remove_links,
    _replace_mentions,
    remove_hashtag_symbol,
    # unicode-aware, so the curly quotes Twitter inserts don't survive to
    # fragment the vocabulary. MUST run before demojize: '_' and '-' are
    # punctuation, so cleaning afterwards would split 'red_heart' in two.
    remove_unicode_punctuation,
    demojize_to_token,
    lowercase,
    normalize_whitespace,
]


def clean_for_bilstm(text: str) -> str:
    return run_steps(text, STEPS)


if __name__ == "__main__":
    preprocess_dataset(STEPS, "bilstm_input.csv")
