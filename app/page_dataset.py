"""The dataset every model trains on, drawn once.

The label columns (year, sentiment, emotion, topic, ner, split) are identical
in all three *_input.csv files - only `tweet` differs, because each model gets
its own cleaning stage. So the distributions live here instead of being drawn
three times, and the per-model text is shown side by side instead.
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from app.registry import PROJECT_ROOT, ModelSpec

SENTIMENT_COLORS = {"positive": "orange", "neutral": "blue", "negative": "red"}

# the shared cleaning stage every preprocess_*.py starts from
BASE_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_tweets.csv"
BASE_COLUMN = "Cleaned base"

LABEL_COLUMNS = ["year", "sentiment", "emotion", "topic"]
SAMPLE_ROWS = 8


def render(specs: list[ModelSpec]) -> None:
    st.title("Dataset")
    st.caption(
        "2014-2026 FIFA World Cup tweets - English only, cleaned, deduplicated, "
        "and human-annotated for sentiment, emotion, topic, and NER."
    )

    _render_distributions(specs[0])
    st.divider()
    _render_preprocessing(specs)


def _render_distributions(spec: ModelSpec) -> None:
    df = _read_labels(str(spec.data_path))
    if df is None:
        st.error(f"No dataset found at {spec.data_path}. Run that model's preprocess_*.py first.")
        return

    st.write(f"**{len(df):,} tweets.**")

    year_col, sentiment_col = st.columns(2)
    with year_col:
        st.write("**By year**")
        st.bar_chart(df["year"].value_counts().sort_index())
    with sentiment_col:
        st.write("**Sentiment distribution**")
        _render_pie(df["sentiment"].value_counts(), colors=SENTIMENT_COLORS)

    emotion_col, topic_col = st.columns(2)
    with emotion_col:
        st.write("**Emotion distribution**")
        st.bar_chart(df["emotion"].value_counts())
    with topic_col:
        st.write("**Topic distribution**")
        st.bar_chart(df["topic"].value_counts())


def _render_preprocessing(specs: list[ModelSpec]) -> None:
    st.subheader("How each model cleans the same tweet")
    st.caption(
        "Why there are three input files. SVM and BiLSTM lowercase, strip punctuation, "
        "and demojize (differently - `soccerball` vs `soccer_ball`), because their "
        "features are bag-of-words and a fixed vocabulary. RoBERTa-base keeps casing, "
        "punctuation, and emoji, because its tokenizer was pretrained on raw tweets."
    )

    columns = {}
    base = _read_tweets(str(BASE_PATH))
    if base is not None:
        columns[BASE_COLUMN] = base
    for spec in specs:
        tweets = _read_tweets(str(spec.data_path))
        if tweets is not None:
            columns[spec.label] = tweets

    if not columns:
        st.error("No `*_input.csv` files found. Run the preprocessing pipeline first.")
        return

    st.dataframe(pd.DataFrame(columns).sample(SAMPLE_ROWS, random_state=1))


def _render_pie(counts: pd.Series, colors: dict[str, str] | None = None) -> None:
    data = counts.rename("count").rename_axis("class").reset_index()
    data["share"] = data["count"] / data["count"].sum()

    color = (
        alt.Color(
            "class:N",
            legend=alt.Legend(title=None),
            scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values())),
        )
        if colors
        else alt.Color("class:N", legend=alt.Legend(title=None))
    )
    chart = (
        alt.Chart(data)
        .mark_arc()
        .encode(
            theta=alt.Theta("count:Q", stack=True),
            color=color,
            tooltip=[
                alt.Tooltip("class:N", title="Class"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
    )
    st.altair_chart(chart)


@st.cache_data(show_spinner="Reading dataset...")
def _read_labels(data_path: str) -> pd.DataFrame | None:
    """Just the label columns - skipping tweet, date, and ner keeps this small."""
    path = Path(data_path)
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8", usecols=LABEL_COLUMNS)


@st.cache_data(show_spinner=False)
def _read_tweets(data_path: str) -> pd.Series | None:
    """One file's tweet text, indexed by id so the columns line up."""
    path = Path(data_path)
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8", usecols=["id", "tweet"]).set_index("id")["tweet"]
