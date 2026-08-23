"""Shared rendering helpers used by every model view.

Kept model-agnostic on purpose: nothing here imports torch, sklearn, or any
model-specific module, so importing this file is always cheap.
"""

import html
import re

import pandas as pd
import streamlit as st

ENTITY_COLORS = {
    "PER": "#f97316",  # orange
    "ORG": "#3b82f6",  # blue
    "LOC": "#22c55e",  # green
    "EVENT": "#a855f7",  # purple
}

SAMPLE_TWEETS = [
    "Messi is on fire! What a goal to win it for Argentina 🔥🐐",
    "Absolute disgrace from the referee tonight, VAR ruined the match #WorldCup",
    "Kickoff in 10 minutes at Lusail Stadium, can't wait!",
    "that last-minute equaliser was absolutely bonkersss, I'm still shakinggg",
]


def render_ner_html(ner_tags: list[tuple[str, str]]) -> str:
    """Render words with colored background spans for detected entities (BIO -> chunks)."""
    spans = []
    i = 0
    while i < len(ner_tags):
        word, tag = ner_tags[i]
        if tag == "O":
            spans.append(html.escape(word))
            i += 1
            continue
        entity_type = tag.split("-", 1)[1]
        chunk = [word]
        i += 1
        while i < len(ner_tags) and ner_tags[i][1] == f"I-{entity_type}":
            chunk.append(ner_tags[i][0])
            i += 1
        color = ENTITY_COLORS.get(entity_type, "#94a3b8")
        text = html.escape(" ".join(chunk))
        spans.append(
            f'<span style="background:{color}25;border:1px solid {color};border-radius:4px;'
            f'padding:1px 5px;margin:0 1px;white-space:nowrap;">{text}'
            f'<span style="font-size:0.7em;font-weight:600;color:{color};'
            f'margin-left:4px;">{entity_type}</span>'
            f"</span>"
        )
    return " ".join(spans)


def parse_inline_bio(bio_str: str) -> list[tuple[str, str]]:
    """Adapter for the SVM's NER string format.

    format_bio_entities() (src/models/svm/ner_bio.py) emits each tagged token
    as its own 'word [TAG]' piece and each untagged token as a bare word, all
    joined by single spaces. This regex pulls the (word, tag) pairs back out,
    defaulting untagged words to 'O' so the result is the same shape
    render_ner_html() already consumes for the other two models.
    """
    pattern = re.compile(r"(\S+)(?:\s\[(\S+)\])?")
    return [(word, tag or "O") for word, tag in pattern.findall(bio_str) if word]


def render_task_metrics(cols, tasks: list[str], tasks_dict: dict) -> None:
    """One st.metric per task: label + confidence. Shared by single and compare views."""
    for col, task in zip(cols, tasks, strict=True):
        info = tasks_dict[task]
        label = info["label"]
        col.metric(task.capitalize(), label, f"{info['confidence']:.0%} confidence")


def parse_metrics(text: str) -> dict:
    """Turn evaluate.py's printed report back into per-task DataFrames."""
    tasks = {}
    blocks = re.split(r"=====\s*(\w+)\s*=====", text)[1:]  # alternating name, body
    for name, body in zip(blocks[0::2], blocks[1::2], strict=False):
        acc_match = re.search(r"Accuracy:\s*([\d.]+)", body)
        rows = []
        for line in body.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            # class rows: "<label...> precision recall f1 support" (label can be multi-word)
            p, r, f1, support = parts[-4:]
            label = " ".join(parts[: len(parts) - 4])
            try:
                rows.append(
                    {
                        "class": label,
                        "precision": float(p),
                        "recall": float(r),
                        "f1": float(f1),
                        "support": int(support),
                    }
                )
            except ValueError:
                continue
        df = pd.DataFrame(rows)
        if not df.empty:
            summary_rows = ["accuracy", "macro avg", "weighted avg"]
            df = df[~df["class"].isin(summary_rows)].reset_index(drop=True)
        entity_line = re.search(
            r"Entity tags only.*?precision ([\d.]+), recall ([\d.]+), F1 ([\d.]+)", body
        )
        tasks[name.lower()] = {
            "accuracy": float(acc_match.group(1)) if acc_match else None,
            "classes": df,
            "entity_only": (
                {
                    "precision": float(entity_line.group(1)),
                    "recall": float(entity_line.group(2)),
                    "f1": float(entity_line.group(3)),
                }
                if entity_line
                else None
            ),
        }
    return tasks


def parse_entity_presence(text: str) -> dict | None:
    """Extract the 'Entity-type presence' block — the one NER framing all
    three models share (SVM predicts type presence directly; the two
    sequence models' BIO output is collapsed into it). Present in identical
    format in every metrics.txt.
    """
    micro = re.search(r"Micro F1:\s*([\d.]+)", text)
    macro = re.search(r"Macro F1:\s*([\d.]+)", text)
    exact = re.search(r"Exact-match accuracy.*?:\s*([\d.]+)", text)
    if not (micro and macro and exact):
        return None
    per_type = {}
    for entity_type, precision, recall, f1 in re.findall(
        r"(PER|ORG|LOC|EVENT)\s+precision\s+([\d.]+)\s+recall\s+([\d.]+)\s+F1\s+([\d.]+)", text
    ):
        per_type[entity_type] = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
    return {
        "micro_f1": float(micro.group(1)),
        "macro_f1": float(macro.group(1)),
        "exact_match": float(exact.group(1)),
        "per_type": per_type,
    }


@st.cache_data(show_spinner=False)
def _read_metrics_text(model_dir_str: str) -> str | None:
    from pathlib import Path

    path = Path(model_dir_str) / "metrics.txt"
    return path.read_text(encoding="utf-8") if path.exists() else None


def render_performance_tab(model_dir) -> None:
    """Accuracy + per-class tables + entity-presence numbers, from that model's metrics.txt."""
    metrics_text = _read_metrics_text(str(model_dir))
    if metrics_text is None:
        st.error(f"No metrics.txt found at {model_dir}. Run that model's evaluate.py first.")
        return

    oov_line = re.search(r"Out-of-vocabulary rate.*", metrics_text)
    if oov_line:
        st.caption(oov_line.group(0))

    parsed = parse_metrics(metrics_text)
    task_order = ["sentiment", "emotion", "topic", "ner"]
    present = [t for t in task_order if t in parsed and parsed[t]["accuracy"] is not None]
    acc_cols = st.columns(len(present)) if present else []
    for col, task in zip(acc_cols, present, strict=True):
        col.metric(task.upper(), f"{parsed[task]['accuracy']:.1%}", "accuracy")

    st.divider()
    for task in task_order:
        if task not in parsed or parsed[task]["classes"].empty:
            continue
        st.subheader(task.capitalize())
        df = parsed[task]["classes"]
        left, right = st.columns([2, 1])
        left.dataframe(
            df.set_index("class")[["precision", "recall", "f1", "support"]], width="stretch"
        )
        right.bar_chart(df.set_index("class")["f1"])
        if parsed[task]["entity_only"]:
            e = parsed[task]["entity_only"]
            st.caption(
                f"Entity-only (excluding 'O'): precision {e['precision']:.1%}, "
                f"recall {e['recall']:.1%}, F1 {e['f1']:.1%} — the more honest NER number, "
                f"since most words are outside any entity."
            )

    presence = parse_entity_presence(metrics_text)
    if presence:
        st.subheader("NER — entity-type presence")
        st.caption(
            "The one NER framing all three models share: did the tweet mention each "
            "entity type at all, regardless of exact wording."
        )
        p1, p2, p3 = st.columns(3)
        p1.metric("Micro F1", f"{presence['micro_f1']:.1%}")
        p2.metric("Macro F1", f"{presence['macro_f1']:.1%}")
        p3.metric("Exact match (all 4 types)", f"{presence['exact_match']:.1%}")
        if presence["per_type"]:
            st.dataframe(pd.DataFrame(presence["per_type"]).T, width="stretch")

    with st.expander("Raw evaluate.py output"):
        st.code(metrics_text)


@st.cache_data(show_spinner="Reading dataset...")
def _read_dataset(data_path_str: str) -> pd.DataFrame:
    return pd.read_csv(data_path_str, encoding="utf-8")


def render_dataset_tab(data_path, label: str) -> None:
    """Distribution charts + sample rows, generalized across all three *_input.csv files
    since they share identical columns (id, tweet, date, lang, year, sentiment, emotion,
    topic, ner, split)."""
    df = _read_dataset(str(data_path))
    st.write(f"**{len(df):,} tweets** in `{data_path.name}` (English only, cleaned and deduplicated).")

    c1, c2 = st.columns(2)
    with c1:
        st.write("**By year**")
        st.bar_chart(df["year"].value_counts().sort_index())
    with c2:
        st.write("**Sentiment distribution**")
        st.bar_chart(df["sentiment"].value_counts())

    c3, c4 = st.columns(2)
    with c3:
        st.write("**Emotion distribution**")
        st.bar_chart(df["emotion"].value_counts())
    with c4:
        st.write("**Topic distribution**")
        st.bar_chart(df["topic"].value_counts())

    st.write(f"**Sample rows** (tweet text shown is already {label}-cleaned)")
    st.dataframe(
        df[["tweet", "sentiment", "emotion", "topic"]].sample(min(10, len(df)), random_state=1),
        width="stretch",
    )
