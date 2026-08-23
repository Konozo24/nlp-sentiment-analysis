"""Renders any one model's full page (Live Demo / Performance / Dataset /
About) from its ModelSpec. One view, parameterized — the three models'
pages are structurally identical, so there is exactly one place to fix a
rendering bug or add a fifth tab.
"""

import pandas as pd
import streamlit as st

from app._common import (
    ENTITY_COLORS,
    SAMPLE_TWEETS,
    render_dataset_tab,
    render_ner_html,
    render_performance_tab,
    render_task_metrics,
)
from app.registry import ModelSpec

TASKS = ["sentiment", "emotion", "topic"]


def render(spec: ModelSpec) -> None:
    st.title(f"{spec.label} — FIFA World Cup Tweet Analysis")
    st.caption(spec.tagline)

    tab_demo, tab_perf, tab_data, tab_about = st.tabs(
        ["Live Demo", "Performance", "Dataset", "About the model"]
    )

    with tab_demo:
        _render_live_demo(spec)
    with tab_perf:
        render_performance_tab(spec.model_dir)
    with tab_data:
        render_dataset_tab(spec.data_path, spec.label)
    with tab_about:
        st.markdown(spec.about_md)


def _render_live_demo(spec: ModelSpec) -> None:
    choice_key, tweet_key = f"{spec.key}_choice", f"{spec.key}_tweet"

    def _apply_sample() -> None:
        choice = st.session_state[choice_key]
        st.session_state[tweet_key] = "" if choice == "(write my own)" else choice

    st.selectbox(
        "Try a sample tweet, or write your own below:",
        ["(write my own)"] + SAMPLE_TWEETS,
        key=choice_key,
        on_change=_apply_sample,
    )
    # Widgets with an explicit key ignore `value=` after their first render, so
    # populating the box from the dropdown has to go through session_state via
    # the on_change callback above, not through a value= argument here.
    tweet = st.text_area(
        "Tweet text",
        height=90,
        placeholder="Type a World Cup tweet...",
        key=tweet_key,
    )

    if not (st.button("Analyze", type="primary", key=f"{spec.key}_analyze") and tweet.strip()):
        if not tweet.strip():
            st.info("Pick a sample tweet or write your own, then click Analyze.")
        return

    bundle = spec.load()
    with st.spinner(f"Running {spec.label}..."):
        result = spec.predict(tweet, bundle)

    if result is None:
        st.warning("Nothing left to analyze after cleaning (e.g. a link-only tweet).")
        return

    if result.get("words"):
        st.markdown("**Cleaned input:** `" + " ".join(result["words"]) + "`")

    cols = st.columns(3)
    render_task_metrics(cols, TASKS, result["tasks"])

    if result.get("oov"):
        st.info(
            "**Words not in the vocabulary:** "
            + ", ".join(f"`{w}`" for w in result["oov"])
            + " — vectors composed from character n-grams (or fell back to `<unk>` "
            "if no in-domain model was available), so they still carry some meaning."
        )

    if any(info.get("distribution") is not None for info in result["tasks"].values()):
        with st.expander("Full class probabilities"):
            for task in TASKS:
                dist = result["tasks"][task].get("distribution")
                if dist is None:
                    continue
                st.write(f"**{task.capitalize()}**")
                st.bar_chart(pd.Series(dist, name="probability"))

    st.markdown("**Named entities**")
    if any(tag != "O" for _, tag in result["ner"]):
        st.markdown(render_ner_html(result["ner"]), unsafe_allow_html=True)
        legend = "  ".join(
            f'<span style="color:{c};font-weight:600;">■</span> {t}'
            for t, c in ENTITY_COLORS.items()
        )
        st.markdown(f"<div style='margin-top:6px;font-size:0.85em;'>{legend}</div>", unsafe_allow_html=True)
    else:
        st.write("No named entities detected.")
