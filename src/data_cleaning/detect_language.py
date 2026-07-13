"""Fill in missing 'lang' values in cleaned_tweets.csv using fastText.

Some source datasets have no language labels. Instead of guessing, we use
fastText's lid.176 model (176 languages, character n-gram based):

  1. VALIDATE: run the detector over every row that already has a Twitter
     lang label and report the agreement rate — this measures how well the
     detector performs on OUR tweets before we trust it.
  2. FILL: detect language for rows where lang is missing. Detections below
     the confidence threshold (or on too-short text) are labelled 'und'
     rather than guessed.

Run:  python -m src.data_cleaning.detect_language
"""

import urllib.request
from pathlib import Path

import emoji
import fasttext
import pandas as pd

from .base_cleaning import CLEANED_PATH
from .utils import normalize_whitespace, remove_hashtag_symbol, remove_links, remove_mentions

MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "models" / "lid.176.ftz"

# Detection quality knobs: below these, we say 'und' instead of guessing.
# 0.50 is safe here: validation against Twitter's own labels shows 98.9%
# English/non-English agreement, and World Cup tweets are dense with
# proper nouns (player/country names) that depress confidence on
# otherwise-clear English text.
CONFIDENCE_THRESHOLD = 0.50
MIN_CHARS = 15

# Twitter uses some legacy ISO codes; map them to what fastText emits
# so the validation comparison doesn't count pure naming differences as errors.
TWITTER_TO_ISO = {"in": "id", "iw": "he"}


def load_model(model_path: Path = MODEL_PATH):
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading fastText lid.176 model to {model_path} ...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return fasttext.load_model(str(model_path))


def strip_for_detection(text: str) -> str:
    """Remove tweet noise that carries no language signal.

    URLs, @mentions and emoji confuse the detector; hashtag WORDS are kept
    (only the '#' is dropped) because they are often real words.
    """
    text = remove_links(text)
    text = remove_mentions(text)
    text = remove_hashtag_symbol(text)
    text = emoji.replace_emoji(text, replace=" ")
    return normalize_whitespace(text)


def detect_lang(model, text: str) -> tuple[str, float]:
    """Return (iso_code, confidence). 'und' when the text is too short or
    the model is not confident enough to trust."""
    stripped = strip_for_detection(str(text))
    if len(stripped) < MIN_CHARS:
        return "und", 0.0
    labels, probs = model.predict(stripped.replace("\n", " "))
    lang, prob = labels[0].removeprefix("__label__"), float(probs[0])
    if prob < CONFIDENCE_THRESHOLD:
        return "und", prob
    return lang, prob


def validate_against_twitter_labels(model, df: pd.DataFrame) -> None:
    """Measure detector accuracy on rows Twitter already labelled."""
    labelled = df[df["lang"].notna() & ~df["lang"].isin(["und", "qme", "qht", "qam", "zxx"])].copy()
    detected = labelled["tweet"].map(lambda t: detect_lang(model, t))
    labelled["detected"] = detected.map(lambda r: r[0])
    labelled["twitter_iso"] = labelled["lang"].replace(TWITTER_TO_ISO)

    confident = labelled[labelled["detected"] != "und"]
    coverage = len(confident) / len(labelled)
    agreement = (confident["detected"] == confident["twitter_iso"]).mean()
    en_agreement = (
        (confident["detected"] == "en") == (confident["twitter_iso"] == "en")
    ).mean()

    print(f"Validation on {len(labelled)} Twitter-labelled tweets:")
    print(f"  confident detections: {coverage:.1%} (rest 'und': too short/low confidence)")
    print(f"  exact language agreement: {agreement:.1%}")
    print(f"  English vs non-English agreement: {en_agreement:.1%}")


def fill_missing_lang(cleaned_path: Path = CLEANED_PATH) -> pd.DataFrame:
    df = pd.read_csv(cleaned_path, encoding="utf-8")
    if "lang" not in df.columns:
        raise ValueError(f"{cleaned_path} has no 'lang' column — re-run base_cleaning first.")

    model = load_model()
    validate_against_twitter_labels(model, df)

    missing = df["lang"].isna()
    print(f"\nDetecting language for {missing.sum()} unlabelled rows ...")
    results = df.loc[missing, "tweet"].map(lambda t: detect_lang(model, t))
    df.loc[missing, "lang"] = results.map(lambda r: r[0])

    filled = df.loc[missing, "lang"]
    print(f"Filled: {(filled != 'und').sum()} detected, {(filled == 'und').sum()} left as 'und'")
    print(filled.value_counts().head(10).to_string())

    # utf-8-sig adds a BOM so Excel displays emoji/non-ASCII correctly
    df.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved back to {cleaned_path}")
    return df


if __name__ == "__main__":
    fill_missing_lang()
