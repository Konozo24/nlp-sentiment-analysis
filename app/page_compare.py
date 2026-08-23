"""Run one tweet through all three models side by side.

The most expensive page in the app — it loads roughly 600MB of checkpoints
plus spaCy's transformer NER pipeline — so it is never the default page.
"""

from pathlib import Path

import streamlit as st

from app.language import warn_if_not_english
from app.ner import format_entities, has_entities
from app.registry import PROJECT_ROOT, ModelSpec
from app.ui import PREDICT_FAILED, TASKS, safe_load, safe_predict

COMPARISON_MD = PROJECT_ROOT / "data" / "models" / "comparison.md"
MAX_TWEET_CHARS = 500


def render(specs: list[ModelSpec]) -> None:
    st.title("Compare all three models")
    st.caption("One tweet, three models, side by side — sentiment, emotion, topic, and NER.")

    tweet = st.text_area(
        "Tweet text",
        height=90,
        max_chars=MAX_TWEET_CHARS,
        key="compare_tweet",
        placeholder="Type any World Cup tweet...",
    )
    analyze = st.button(
        "Analyze with all 3 models", type="primary", icon=":material/play_arrow:"
    )

    if not tweet.strip():
        st.info("Type a tweet above, then click Analyze with all 3 models.")
    elif analyze:
        warn_if_not_english(tweet)
        _render_comparison(tweet, specs)

    st.divider()
    _render_cross_model_metrics()


def _render_comparison(tweet: str, specs: list[ModelSpec]) -> None:
    results = []
    for spec in specs:
        with st.spinner(f"Running {spec.label}..."):
            bundle = safe_load(spec.label, spec.model_dir, spec.load)
            result = (
                safe_predict(spec.label, spec.predict, tweet, bundle)
                if bundle is not None
                else PREDICT_FAILED
            )
        results.append((spec, result))

    empty_after_cleaning = [spec.label for spec, result in results if result is None]
    if empty_after_cleaning:
        st.warning(
            f"{', '.join(empty_after_cleaning)} had nothing left to analyze after cleaning "
            "(e.g. a link-only tweet)."
        )

    for col, (spec, result) in zip(st.columns(len(results)), results, strict=True):
        with col, st.container(border=True):
            _render_model_column(spec, result)

    usable = {spec.label: result for spec, result in results if isinstance(result, dict)}
    _render_agreement(usable)


def _render_model_column(spec: ModelSpec, result: dict | None) -> None:
    st.subheader(spec.label)
    if result is PREDICT_FAILED:
        st.write("Failed — see error above.")
        return
    if result is None:
        st.write("—")
        return

    for task in TASKS:
        info = result["tasks"][task]
        st.metric(task.capitalize(), info["label"], f"{info['confidence']:.0%}")

    st.markdown("**Entities**")
    if has_entities(result["ner"]):
        st.markdown(format_entities(result["ner"]))
    else:
        st.write("None detected.")


def _render_agreement(results: dict[str, dict]) -> None:
    """Per task, whether the models that produced a label all picked the same one."""
    if len(results) < 2:
        return

    st.divider()
    st.subheader("Agreement across models")
    for task in TASKS:
        labels = {label: result["tasks"][task]["label"] for label, result in results.items()}
        verdict = "agree" if len(set(labels.values())) == 1 else "disagree"
        detail = ", ".join(f"{model}={label}" for model, label in labels.items())
        st.write(f"**{task.capitalize()}** ({verdict}): {detail}")


def _render_cross_model_metrics() -> None:
    st.subheader("Cross-model metrics (test split, 4,130 tweets)")
    if COMPARISON_MD.exists():
        st.markdown(_read_comparison(str(COMPARISON_MD)))
    else:
        st.error(f"{COMPARISON_MD} not found. Run scripts/compare_models.py first.")


@st.cache_data(show_spinner=False)
def _read_comparison(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")
