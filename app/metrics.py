"""Parsers for the metrics.txt each model's evaluate.py writes.

Pure functions over text - no Streamlit - so the report format lives in one
place and can be tested without running the app.
"""

import json
import re
from pathlib import Path

import pandas as pd

# rows evaluate.py appends after the per-class rows; not classes themselves
SUMMARY_ROWS = ("accuracy", "macro avg", "weighted avg")

_TASK_BLOCK = re.compile(r"=====\s*(\w+)\s*=====")
_ENTITY_ONLY = re.compile(
    r"Entity tags only.*?precision ([\d.]+), recall ([\d.]+), F1 ([\d.]+)"
)
_PER_TYPE = re.compile(
    r"(PER|ORG|LOC|EVENT)\s+precision\s+([\d.]+)\s+recall\s+([\d.]+)\s+F1\s+([\d.]+)"
)


def load_headlines(model_dir) -> dict | None:
    """Read metrics.json: the structured accuracy/precision/recall/F1 behind metrics.txt."""
    path = Path(model_dir) / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_report(text: str) -> dict[str, dict]:
    """Split the report into each task's per-class table and entity-only F1."""
    tasks = {}
    blocks = _TASK_BLOCK.split(text)[1:]  # alternating name, body
    for name, body in zip(blocks[0::2], blocks[1::2], strict=False):
        entity_only = _ENTITY_ONLY.search(body)
        tasks[name.lower()] = {
            "classes": _parse_class_table(body),
            "entity_only": (
                {
                    "precision": float(entity_only.group(1)),
                    "recall": float(entity_only.group(2)),
                    "f1": float(entity_only.group(3)),
                }
                if entity_only
                else None
            ),
        }
    return tasks


def parse_entity_presence(text: str) -> dict | None:
    """Extract the 'Entity-type presence' block, the one NER framing all three models share.

    The SVM predicts entity-type presence directly; the two sequence models'
    BIO output is collapsed into it. Every metrics.txt carries this block in
    the same format, so it is the only cross-model NER comparison available.
    """
    micro = re.search(r"Micro F1:\s*([\d.]+)", text)
    macro = re.search(r"Macro F1:\s*([\d.]+)", text)
    exact = re.search(r"Exact-match accuracy.*?:\s*([\d.]+)", text)
    if not (micro and macro and exact):
        return None

    return {
        "micro_f1": float(micro.group(1)),
        "macro_f1": float(macro.group(1)),
        "exact_match": float(exact.group(1)),
        "per_type": {
            entity_type: {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
            for entity_type, precision, recall, f1 in _PER_TYPE.findall(text)
        },
    }


def _parse_class_table(body: str) -> pd.DataFrame:
    """Read sklearn's classification report rows: '<label...> precision recall f1 support'."""
    rows = []
    for line in body.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        precision, recall, f1, support = parts[-4:]
        try:
            rows.append(
                {
                    "class": " ".join(parts[:-4]),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "support": int(support),
                }
            )
        except ValueError:
            continue  # label row is not numeric, e.g. the header

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[~df["class"].isin(SUMMARY_ROWS)].reset_index(drop=True)
