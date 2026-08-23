"""Rendering helpers shared by the model pages.

Model-agnostic on purpose: nothing here imports torch, sklearn, or any
model-specific module, so importing this file is always cheap.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.metrics import load_headlines, parse_entity_presence, parse_report

# the three sentence-level tasks, in the order every page shows them
TASKS = ("sentiment", "emotion", "topic")

TASK_ORDER = (*TASKS, "ner")

SAMPLE_TWEETS = [
    "Messi is on fire! What a goal to win it for Argentina 🔥🐐",
    "Absolute disgrace from the referee tonight, VAR ruined the match #WorldCup",
    "Kickoff in 10 minutes at Lusail Stadium, can't wait!",
    "that last-minute equaliser was absolutely bonkersss, I'm still shakinggg",
]


def safe_load(label: str, model_dir: Path, load: Callable[[], Any]) -> Any | None:
    """Load a model's artifacts, turning a missing or corrupt checkpoint into an
    actionable st.error instead of a traceback mid-demo."""
    try:
        return load()
    except (FileNotFoundError, OSError) as error:
        st.error(f"Couldn't load {label}'s artifacts from `{model_dir}`: {error}")
        return None


# sentinel distinguishing "predict() raised" from predict()'s own None, which
# means "nothing left to analyze after cleaning" - the two need different messages
PREDICT_FAILED = object()


def safe_predict(label: str, predict: Callable[[str, Any], dict | None], tweet: str, bundle: Any):
    """Run a model's predict(), turning any exception into an st.error instead of a
    crash - broad on purpose, since predict() is model-specific code this layer
    doesn't otherwise validate. Returns PREDICT_FAILED on error."""
    try:
        return predict(tweet, bundle)
    except Exception as error:  # noqa: BLE001 - last line of defense before the UI
        st.error(f"{label} failed to analyze this tweet ({error}).")
        return PREDICT_FAILED


def render_task_metrics(tasks: dict) -> None:
    """One metric per sentence-level task: predicted label plus confidence."""
    with st.container(horizontal=True):
        for task in TASKS:
            info = tasks[task]
            st.metric(task.capitalize(), info["label"], f"{info['confidence']:.0%} confidence")


def render_performance_tab(model_dir: Path) -> None:
    """Accuracy/precision/recall/F1, per-class tables, and entity-presence numbers.

    Headline numbers (the four requirement-f metrics) come from metrics.json;
    the per-class tables only exist in metrics.txt, so both are read.
    """
    report_text = _read_metrics(str(model_dir))
    if report_text is None:
        st.error(f"No metrics.txt found in {model_dir}. Run that model's evaluate.py first.")
        return

    oov_line = _find_line(report_text, "Out-of-vocabulary rate")
    if oov_line:
        st.caption(oov_line)

    headlines = load_headlines(model_dir)
    if headlines:
        _render_metrics_summary(headlines)

    st.divider()
    tasks = parse_report(report_text)
    for task in TASK_ORDER:
        _render_task_classes(task, tasks.get(task))

    _render_entity_presence(report_text)

    with st.expander("Raw evaluate.py output"):
        st.code(report_text)


def render_dataset_tab(data_path: Path, label: str) -> None:
    """This model's own view of the data: the tweet text after its cleaning stage.
    """
    df = _read_dataset(str(data_path))
    if df is None:
        st.error(f"No dataset found at {data_path}. Run that model's preprocess_*.py first.")
        return

    st.write(f"**{len(df):,} tweets** in `{data_path.name}` (English only, cleaned and deduplicated).")

    st.write(f"**Sample rows** (tweet text shown is already {label}-cleaned)")
    st.dataframe(
        df[["tweet", "sentiment", "emotion", "topic"]].sample(min(10, len(df)), random_state=1)
    )
    st.caption("Label distributions for the full dataset are on the **Dataset** page.")


@st.cache_data(show_spinner=False)
def _read_metrics(model_dir: str) -> str | None:
    path = Path(model_dir) / "metrics.txt"
    return path.read_text(encoding="utf-8") if path.exists() else None


@st.cache_data(show_spinner="Reading dataset...")
def _read_dataset(data_path: str) -> pd.DataFrame | None:
    """Only the columns the sample-rows table shows - the rest of the file is
    the ner and date columns, which this tab never draws."""
    path = Path(data_path)
    if not path.exists():
        return None
    return pd.read_csv(
        path, encoding="utf-8", usecols=["tweet", "sentiment", "emotion", "topic"]
    )


def _find_line(text: str, prefix: str) -> str | None:
    return next((line.strip() for line in text.splitlines() if prefix in line), None)


def _render_metrics_summary(headlines: dict) -> None:
    """Accuracy, macro precision/recall/F1, and weighted F1 - the metrics requirement (f)
    names - one row per sentence-level task. NER isn't a single-label task, so it keeps
    its own micro/macro/exact-match framing in the entity-presence section below."""
    rows = {
        task.capitalize(): {
            "Accuracy": headlines[task]["accuracy"],
            "Macro precision": headlines[task]["macro_precision"],
            "Macro recall": headlines[task]["macro_recall"],
            "Macro F1": headlines[task]["macro_f1"],
            "Weighted F1": headlines[task]["weighted_f1"],
        }
        for task in TASKS
        if task in headlines
    }
    if not rows:
        return

    df = pd.DataFrame(rows).T
    st.dataframe(
        df,
        column_config={
            col: st.column_config.NumberColumn(format="percent") for col in df.columns
        },
    )
    st.caption(
        "Macro is the headline average: classes are skewed and all three models train "
        "with inverse-frequency class weights, so macro matches the training objective."
    )


def _render_task_classes(task: str, task_metrics: dict | None) -> None:
    if not task_metrics or task_metrics["classes"].empty:
        return

    st.subheader(task.capitalize())
    df = task_metrics["classes"].set_index("class")
    table, chart = st.columns([2, 1])
    table.dataframe(df[["precision", "recall", "f1", "support"]])
    chart.bar_chart(df["f1"])

    entity_only = task_metrics["entity_only"]
    if entity_only:
        st.caption(
            f"Entity-only (excluding 'O'): precision {entity_only['precision']:.1%}, "
            f"recall {entity_only['recall']:.1%}, F1 {entity_only['f1']:.1%} - the more "
            "honest NER number, since most words are outside any entity."
        )


def _render_entity_presence(report_text: str) -> None:
    presence = parse_entity_presence(report_text)
    if not presence:
        return

    st.subheader("NER - entity-type presence")
    st.caption(
        "The one NER framing all three models share: did the tweet mention each "
        "entity type at all, regardless of exact wording."
    )
    with st.container(horizontal=True):
        st.metric("Micro F1", f"{presence['micro_f1']:.1%}")
        st.metric("Macro F1", f"{presence['macro_f1']:.1%}")
        st.metric("Exact match (all 4 types)", f"{presence['exact_match']:.1%}")
    if presence["per_type"]:
        st.dataframe(pd.DataFrame(presence["per_type"]).T)
