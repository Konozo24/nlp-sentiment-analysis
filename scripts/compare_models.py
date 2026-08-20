"""Build one comparison table from the three models' metrics.json files.

Prints a table per task — sentiment, emotion, topic, and NER entity-type
presence — and writes them to data/models/comparison.md.

Exits with a message if any model has not been evaluated yet, or if the models
disagree on the test-set size or the label sets, since a table built from those
would not be comparable.

Run:  python scripts/compare_models.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "data" / "models"
OUT_PATH = MODELS_DIR / "comparison.md"

MODELS = {"svm": "SVM (TF-IDF)", "bilstm": "BiLSTM", "robertacnn": "RoBERTa-CNN"}
TASKS = ["sentiment", "emotion", "topic"]
COLUMNS = [
    ("accuracy", "Accuracy"),
    ("macro_precision", "Macro P"),
    ("macro_recall", "Macro R"),
    ("macro_f1", "Macro F1"),
    ("weighted_f1", "Weighted F1"),
]


def load_metrics() -> dict[str, dict]:
    metrics, missing = {}, []
    for key in MODELS:
        path = MODELS_DIR / key / "metrics.json"
        if path.exists():
            metrics[key] = json.loads(path.read_text(encoding="utf-8"))
        else:
            # ASCII only: the Windows console codepage mangles non-ASCII here
            missing.append(f"  {path}  -> run 'python -m src.models.{key}.evaluate'")
    if missing:
        sys.exit("No metrics.json for:\n" + "\n".join(missing))
    return metrics


def check_comparable(metrics: dict[str, dict]) -> int:
    """Return the shared test-set size, or exit if the models are not comparable.

    Checks that all models report the same n_test and the same label list for
    every task.
    """
    sizes = {key: value["n_test"] for key, value in metrics.items()}
    if len(set(sizes.values())) != 1:
        sys.exit(
            f"Models were scored on different numbers of test rows: {sizes}.\n"
            "The comparison would be meaningless. Re-run:\n"
            "  python -m src.data_cleaning.base_cleaning\n"
            "  python -m src.data_cleaning.preprocess_{svm,bilstm,robertacnn}\n"
            "then retrain and re-evaluate."
        )

    for task in TASKS:
        label_sets = {key: tuple(value[task]["labels"]) for key, value in metrics.items()}
        if len(set(label_sets.values())) != 1:
            sys.exit(f"Models disagree on the {task} label set: {label_sets}")

    return next(iter(sizes.values()))


def table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def main():
    metrics = load_metrics()
    n_test = check_comparable(metrics)

    parts = [
        "# Model comparison",
        "",
        f"All three models scored on the same {n_test:,} test tweets "
        "(shared row set and group-aware split; see src/data_cleaning/base_cleaning.py).",
        "",
    ]

    for task in TASKS:
        rows = [
            [MODELS[key]] + [f"{metrics[key][task][field]:.4f}" for field, _ in COLUMNS]
            for key in MODELS
        ]
        parts += [f"## {task.capitalize()}", "", table(rows, ["Model"] + [name for _, name in COLUMNS]), ""]

    ner_columns = [("micro_f1", "Micro F1"), ("macro_f1", "Macro F1"),
                   ("exact_match_accuracy", "Exact match")]
    rows = [
        [MODELS[key]] + [f"{metrics[key]['ner'][field]:.4f}" for field, _ in ner_columns]
        for key in MODELS
    ]
    parts += [
        "## NER (entity-type presence)",
        "",
        table(rows, ["Model"] + [name for _, name in ner_columns]),
        "",
        "_Entity-type presence is the only NER framing the three models share. The BiLSTM's "
        "and RoBERTa-CNN's token-level BIO scores are in their own metrics.txt and are not "
        "comparable with the SVM's._",
        "",
    ]

    report = "\n".join(parts)
    print(report)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
