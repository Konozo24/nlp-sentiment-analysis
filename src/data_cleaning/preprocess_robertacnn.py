"""
Run:  python -m src.data_cleaning.preprocess_robertacnn
"""

from .pipeline import preprocess_dataset, run_steps
from .utils import (
    normalize_whitespace,
    remove_hashtag_symbol,
    remove_rt_prefix,
    replace_links_with_token,
    replace_mentions_with_token,
    unescape_html,
)

STEPS = [
    unescape_html,          # '&amp;' -> '&' before anything else touches the text
    remove_rt_prefix,       # strip leading 'RT @user:' retweet markers
    replace_links_with_token,
    replace_mentions_with_token,
    remove_hashtag_symbol,
    normalize_whitespace,
]


def clean_for_robertacnn(text: str) -> str:
    return run_steps(text, STEPS)


if __name__ == "__main__":
    preprocess_dataset(STEPS, "robertacnn_input.csv")