"""One model's page: live demo, performance, dataset, and model notes.

Parameterized by ModelSpec - the three model pages are structurally
identical, so there is exactly one place to fix a rendering bug or add a
fifth tab.
"""

import pandas as pd
import streamlit as st

from app.confusion import render_confusion_section
from app.language import warn_if_not_english
from app.ner import format_entities, format_legend, has_entities
from app.registry import ModelSpec
from app.ui import (
    PREDICT_FAILED,
    SAMPLE_TWEETS,
    TASKS,
    render_dataset_tab,
    render_performance_tab,
    render_task_metrics,
    safe_load,
    safe_predict,
)

WRITE_MY_OWN = "(write my own)"
MAX_TWEET_CHARS = 500


def render(spec: ModelSpec) -> None:
    st.title(f"{spec.label} - FIFA World Cup tweet analysis")
    st.caption(spec.tagline)

    # on_change="rerun" makes .open available, so the tabs that read files off
    # disk only do that work while they are the visible tab
    demo, performance, dataset, about = st.tabs(
        ["Live demo", "Performance", "Dataset", "About the model"],
        on_change="rerun",
        key=f"{spec.key}_tabs",
    )

    with demo:
        _render_live_demo(spec)
    if performance.open:
        with performance:
            render_performance_tab(spec.model_dir)
            st.divider()
            render_confusion_section(spec.model_dir, key_prefix=spec.key)
    if dataset.open:
        with dataset:
            render_dataset_tab(spec.data_path, spec.label)
    with about:
        st.markdown(spec.about_md)


def _render_live_demo(spec: ModelSpec) -> None:
    sample_key, tweet_key = f"{spec.key}_sample", f"{spec.key}_tweet"

    def copy_sample_into_box() -> None:
        sample = st.session_state[sample_key]
        st.session_state[tweet_key] = "" if sample == WRITE_MY_OWN else sample

    st.selectbox(
        "Try a sample tweet, or write your own below",
        [WRITE_MY_OWN, *SAMPLE_TWEETS],
        key=sample_key,
        on_change=copy_sample_into_box,
    )
    # a keyed widget ignores `value=` after its first render, so the dropdown
    # has to fill the box through session state in the callback above
    tweet = st.text_area(
        "Tweet text",
        height=90,
        max_chars=MAX_TWEET_CHARS,
        placeholder="Type a World Cup tweet...",
        key=tweet_key,
    )
    analyze = st.button(
        "Analyze", type="primary", icon=":material/play_arrow:", key=f"{spec.key}_analyze"
    )

    if not tweet.strip():
        st.info("Pick a sample tweet or write your own, then click Analyze.")
        return
    if not analyze:
        return

    warn_if_not_english(tweet)

    bundle = safe_load(spec.label, spec.model_dir, spec.load)
    if bundle is None:
        return
    with st.spinner(f"Running {spec.label}..."):
        result = safe_predict(spec.label, spec.predict, tweet, bundle)

    if result is PREDICT_FAILED:
        return
    if result is None:
        st.warning("Nothing left to analyze after cleaning (e.g. a link-only tweet).")
        return

    _render_result(result)


def _render_result(result: dict) -> None:
    if result.get("words"):
        st.markdown("**Cleaned input:** `" + " ".join(result["words"]) + "`")

    render_task_metrics(result["tasks"])

    if result.get("oov"):
        st.info(
            "**Words not in the vocabulary:** "
            + ", ".join(f"`{word}`" for word in result["oov"])
            + " - vectors composed from character n-grams (or fell back to `<unk>` "
            "if no in-domain model was available), so they still carry some meaning."
        )

    distributions = {
        task: result["tasks"][task]["distribution"]
        for task in TASKS
        if result["tasks"][task].get("distribution") is not None
    }
    if distributions:
        with st.expander("Full class probabilities"):
            for task, distribution in distributions.items():
                st.write(f"**{task.capitalize()}**")
                st.bar_chart(pd.Series(distribution, name="probability"))

    st.markdown("**Named entities**")
    if has_entities(result["ner"]):
        st.markdown(format_entities(result["ner"]))
        st.caption(format_legend())
    else:
        st.write("No named entities detected.")
