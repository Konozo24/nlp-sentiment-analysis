"""Unit tests for the cleaning helpers and the three model pipelines.

Run:  python -m pytest tests/ -v      (or)      python -m tests.test_cleaning
"""

from src.data_cleaning.preprocess_bilstm import clean_for_bilstm
from src.data_cleaning.preprocess_svm import clean_for_svm
from src.data_cleaning.preprocess_xlmr import clean_for_xlmr
from src.data_cleaning.utils import (
    demojize_to_token,
    normalize_whitespace,
    remove_hashtag_symbol,
    remove_links,
    remove_mentions,
    remove_punctuation,
    remove_rt_prefix,
    replace_mentions_with_token,
)


def test_remove_links():
    assert remove_links("gol! https://t.co/abc123 what a match") == "gol!  what a match"
    assert remove_links("see www.fifa.com now") == "see  now"


def test_remove_rt_prefix():
    assert remove_rt_prefix("RT @fifa: What a goal!") == "What a goal!"
    assert remove_rt_prefix("normal tweet") == "normal tweet"


def test_remove_mentions():
    assert remove_mentions("@fifa great match @espn!") == " great match !"


def test_replace_mentions_with_token():
    assert replace_mentions_with_token("@fifa great match") == "@user great match"


def test_remove_hashtag_symbol_keeps_word():
    assert remove_hashtag_symbol("#WorldCup2026 is here") == "WorldCup2026 is here"


def test_demojize_to_token_keeps_single_token():
    text = demojize_to_token("goal ❤️")
    assert "red_heart" in text
    assert ":" not in text


def test_normalize_whitespace():
    assert normalize_whitespace("  too   many\n\nspaces ") == "too many spaces"


def test_remove_punctuation():
    assert remove_punctuation("Gooal!!! #1?") == "Gooal 1"


def test_svm_pipeline_is_aggressive():
    out = clean_for_svm("RT @fifa: The BEST goal!!! ❤️ https://t.co/x #WorldCup 2026")
    assert out == "rt best goal red_heart worldcup"


def test_bilstm_pipeline_keeps_stopwords():
    out = clean_for_bilstm("This is NOT a good game! https://t.co/x")
    assert out == "this is not a good game"


def test_xlmr_pipeline_is_minimal():
    out = clean_for_xlmr("@fifa The BEST goal!!! ❤️ https://t.co/x #WorldCup")
    assert out == "@user The BEST goal!!! ❤️ WorldCup"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(failures)
