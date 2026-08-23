"""Rendering helpers shared by the model pages.

Model-agnostic on purpose: nothing here imports torch, sklearn, or any
model-specific module, so importing this file is always cheap.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from app.metrics import parse_entity_presence, parse_report

# the three sentence-level tasks, in the order every page shows them
TASKS = ("sentiment", "emotion", "topic")

TASK_ORDER = (*TASKS, "ner")

SAMPLE_TWEETS = [
    "Messi is on fire! What a goal to win it for Argentina 🔥🐐",
    "Absolute disgrace from the referee tonight, VAR ruined the match #WorldCup",
    "Kickoff in 10 minutes at Lusail Stadium, can't wait!",
    "that last-minute equaliser was absolutely bonkersss, I'm still shakinggg",
]


def render_task_metrics(tasks: dict) -> None:
    """One metric per sentence-level task: predicted label plus confidence."""
    with st.container(horizontal=True):
        for task in TASKS:
            info = tasks[task]
            st.metric(task.capitalize(), info["label"], f"{info['confidence']:.0%} confidence")


def render_performance_tab(model_dir: Path) -> None:
    """Accuracy, per-class tables, and entity-presence numbers from that model's metrics.txt."""
    report_text = _read_metrics(str(model_dir))
    if report_text is None:
        st.error(f"No metrics.txt found in {model_dir}. Run that model's evaluate.py first.")
        return

    oov_line = _find_line(report_text, "Out-of-vocabulary rate")
    if oov_line:
        st.caption(oov_line)

    tasks = parse_report(report_text)
    _render_accuracy_row(tasks)

    st.divider()
    for task in TASK_ORDER:
        _render_task_classes(task, tasks.get(task))

    _render_entity_presence(report_text)

    with st.expander("Raw evaluate.py output"):
        st.code(report_text)


def render_dataset_tab(data_path: Path, label: str) -> None:
    """Distribution charts and sample rows.

    Works for all three *_input.csv files because they share the same columns
    (id, tweet, date, lang, year, sentiment, emotion, topic, ner, split).
    """
    df = _read_dataset(str(data_path))
    st.write(f"**{len(df):,} tweets** in `{data_path.name}` (English only, cleaned and deduplicated).")

    charts = [
        ("By year", df["year"].value_counts().sort_index()),
        ("Sentiment distribution", df["sentiment"].value_counts()),
        ("Emotion distribution", df["emotion"].value_counts()),
        ("Topic distribution", df["topic"].value_counts()),
    ]
    for first, second in (charts[:2], charts[2:]):
        for col, (title, counts) in zip(st.columns(2), (first, second), strict=True):
            col.write(f"**{title}**")
            col.bar_chart(counts)

    st.write(f"**Sample rows** (tweet text shown is already {label}-cleaned)")
    st.dataframe(
        df[["tweet", "sentiment", "emotion", "topic"]].sample(min(10, len(df)), random_state=1)
    )


@st.cache_data(show_spinner=False)
def _read_metrics(model_dir: str) -> str | None:
    path = Path(model_dir) / "metrics.txt"
    return path.read_text(encoding="utf-8") if path.exists() else None


@st.cache_data(show_spinner="Reading dataset...")
def _read_dataset(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path, encoding="utf-8")


def _find_line(text: str, prefix: str) -> str | None:
    return next((line.strip() for line in text.splitlines() if prefix in line), None)


def _render_accuracy_row(tasks: dict) -> None:
    scored = [t for t in TASK_ORDER if tasks.get(t, {}).get("accuracy") is not None]
    if not scored:
        return
    with st.container(horizontal=True):
        for task in scored:
            st.metric(task.upper(), f"{tasks[task]['accuracy']:.1%}", "accuracy")


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
            f"recall {entity_only['recall']:.1%}, F1 {entity_only['f1']:.1%} — the more "
            "honest NER number, since most words are outside any entity."
        )


def _render_entity_presence(report_text: str) -> None:
    presence = parse_entity_presence(report_text)
    if not presence:
        return

    st.subheader("NER — entity-type presence")
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
