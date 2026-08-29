"""Confusion-matrix section for a model's Performance tab.

Reads predictions.csv (id, task, gold, pred), written by save_predictions()
in src/models/metrics.py.
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from app.ui import TASKS


def render_confusion_section(model_dir: Path, key_prefix: str) -> None:
    predictions = _read_predictions(str(model_dir))
    if predictions is None:
        return

    tasks_present = [task for task in TASKS if task in predictions["task"].unique()]
    if not tasks_present:
        return

    st.subheader("Confusion matrix")
    task = st.segmented_control(
        "Task",
        tasks_present,
        default=tasks_present[0],
        format_func=str.capitalize,
        key=f"{key_prefix}_confusion_task",
    )
    normalize = st.toggle(
        "Normalize by true class (recall)",
        value=True,
        key=f"{key_prefix}_confusion_normalize",
        help="Row-normalized shows what each true class gets predicted as, "
        "which is the more readable view when classes are this skewed.",
    )
    if task is None:
        return

    counts = _confusion_counts(predictions, task)
    _render_heatmap(counts, normalize=normalize)
    _render_top_confusions(counts)


@st.cache_data(show_spinner=False)
def _read_predictions(model_dir: str) -> pd.DataFrame | None:
    path = Path(model_dir) / "predictions.csv"
    return pd.read_csv(path, encoding="utf-8") if path.exists() else None


def _confusion_counts(predictions: pd.DataFrame, task: str) -> pd.DataFrame:
    """Square gold x pred count matrix, indexed and columned by every label seen."""
    task_rows = predictions[predictions["task"] == task]
    labels = sorted(set(task_rows["gold"]) | set(task_rows["pred"]))
    return pd.crosstab(task_rows["gold"], task_rows["pred"]).reindex(
        index=labels, columns=labels, fill_value=0
    )


def _render_heatmap(counts: pd.DataFrame, normalize: bool) -> None:
    labels = list(counts.index)
    display = counts.div(counts.sum(axis=1), axis=0) if normalize else counts

    long = display.reset_index().melt(id_vars="gold", var_name="pred", value_name="value")
    long["count"] = counts.reset_index().melt(id_vars="gold", var_name="pred", value_name="count")[
        "count"
    ]

    value_format = ".0%" if normalize else ".0f"
    # text needs a light color once the cell is dark enough to read against
    threshold = display.to_numpy().max() / 2

    base = alt.Chart(long).encode(
        x=alt.X("pred:N", title="Predicted", sort=labels),
        y=alt.Y("gold:N", title="Actual", sort=labels),
    )
    heatmap = base.mark_rect().encode(
        color=alt.Color("value:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=[
            alt.Tooltip("gold:N", title="Actual"),
            alt.Tooltip("pred:N", title="Predicted"),
            alt.Tooltip("count:Q", title="Count"),
            alt.Tooltip("value:Q", title="Share" if normalize else "Count", format=value_format),
        ],
    )
    labels_layer = base.mark_text(baseline="middle").encode(
        text=alt.Text("value:Q", format=value_format),
        color=alt.condition(
            alt.datum.value > threshold, alt.value("white"), alt.value("black")
        ),
    )

    chart = (heatmap + labels_layer).properties(height=450)
    
    left, _right = st.columns(2)
    with left:
        st.altair_chart(chart, use_container_width=True)


def _render_top_confusions(counts: pd.DataFrame, top_n: int = 2) -> None:
    """Caption the most common misclassifications - gold != pred, ranked by raw count."""
    mistakes = counts.copy()
    for label in mistakes.index:
        mistakes.loc[label, label] = 0

    ranked = mistakes.stack().rename("count").reset_index()  # columns: gold, pred, count
    ranked = ranked[ranked["count"] > 0].sort_values("count", ascending=False).head(top_n)
    if ranked.empty:
        return

    pairs = ", ".join(f"{row.gold} -> {row.pred} ({row.count})" for row in ranked.itertuples())
    st.caption(f"Most confused pairs (actual -> predicted): {pairs}")
