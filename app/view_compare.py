"""Run one tweet through all three models side by side.

This is the app's answer to rubric requirement (f) — compare and evaluate
all models — made visible and interactive rather than left in a markdown
file. It's also the most expensive page: loading all three models at once
is ~600MB of checkpoints plus spaCy's transformer NER pipeline, so it's an
explicit sidebar choice, never the default.
"""

from pathlib import Path

import streamlit as st

from app._common import EMOJI_MAP, render_ner_html
from app.registry import ModelSpec

TASKS = ["sentiment", "emotion", "topic"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def render(specs: list[ModelSpec]) -> None:
    st.title("⚖️ Compare all three models")
    st.caption("One tweet, three models, side by side — sentiment / emotion / topic / NER.")

    tweet = st.text_area(
        "Tweet text", height=90, key="cmp_tweet", placeholder="Type any World Cup tweet..."
    )

    if st.button("Analyze with all 3 models", type="primary") and tweet.strip():
        results = {}
        for spec in specs:
            with st.spinner(f"Running {spec.label}..."):
                bundle = spec.load()
                results[spec.key] = (spec, spec.predict(tweet, bundle))

        missing = [spec.label for spec, res in results.values() if res is None]
        if missing:
            st.warning(
                f"{', '.join(missing)} had nothing left to analyze after cleaning "
                "(e.g. a link-only tweet)."
            )

        cols = st.columns(3)
        for col, (spec, res) in zip(cols, results.values(), strict=True):
            with col:
                st.subheader(f"{spec.icon} {spec.label}")
                if res is None:
                    st.write("—")
                    continue
                for task in TASKS:
                    info = res["tasks"][task]
                    icon = EMOJI_MAP.get(info["label"], "")
                    st.metric(task.capitalize(), f"{icon} {info['label']}", f"{info['confidence']:.0%}")
                st.markdown("**Entities**")
                if any(tag != "O" for _, tag in res["ner"]):
                    st.markdown(render_ner_html(res["ner"]), unsafe_allow_html=True)
                else:
                    st.write("None detected.")

        usable = {spec.label: res for spec, res in results.values() if res is not None}
        if len(usable) > 1:
            st.divider()
            st.subheader("Agreement across models")
            for task in TASKS:
                labels_by_model = {name: res["tasks"][task]["label"] for name, res in usable.items()}
                agree = len(set(labels_by_model.values())) == 1
                icon = "✅" if agree else "⚠️"
                detail = ", ".join(f"{name}={label}" for name, label in labels_by_model.items())
                st.write(f"{icon} **{task.capitalize()}**: {detail}")
    elif not tweet.strip():
        st.info("Type a tweet above, then click Analyze with all 3 models.")

    st.divider()
    st.subheader("Cross-model metrics (test split, 4,130 tweets)")
    md_path = PROJECT_ROOT / "data" / "models" / "comparison.md"
    if md_path.exists():
        st.markdown(md_path.read_text(encoding="utf-8"))
    else:
        st.error(f"{md_path} not found. Run scripts/compare_models.py first.")
