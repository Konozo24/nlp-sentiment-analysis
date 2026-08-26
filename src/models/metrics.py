"""Metrics shared by every model's evaluate.py, so the scores compare.

Every number in data/models/comparison.md comes from here.

  sentiment / emotion / topic   single_label_report()
      Accuracy and macro P/R/F1, plus weighted F1. Macro is the headline: the
      classes are skewed and every model trains with inverse-frequency class
      weights, so macro matches the training objective.

  ner                           entity_presence_report()
      Which of PER/ORG/LOC/EVENT a tweet mentions, as a four-way multi-label
      problem - the one framing every model can express. The sequence models
      collapse their BIO output into it with types_from_bio(); their
      token-level scores stay in their own evaluate.py, where the differing
      tokenisations make them incomparable.

Each evaluator calls save_metrics() to write metrics.json for
scripts/compare_models.py.
"""

import json

import pandas as pd

from src.data_cleaning.utils import canonical_key
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

ENTITY_TYPES = ["PER", "ORG", "LOC", "EVENT"]


def types_from_bio(tags: list[str]) -> list[str]:
    """['B-PER','I-PER','O','B-ORG'] -> ['PER','ORG'] — one BIO sequence's types."""
    return sorted({tag.split("-", 1)[1] for tag in tags if tag != "O" and "-" in tag})


def findable_entity_types(tweet: str, ner_value) -> list[str]:
    """Entity types both annotated for a tweet AND findable in its text.

    Present means the entity's words appear as a run in the tweet, with both
    sides normalised by canonical_key() - not by any one model's cleaning, so
    every model is scored against the same gold. The ~5% of annotated entities
    that never appear in the text are dropped rather than counted as misses.

        findable_entity_types("messi scores for argentina",
                              "PER: Messi | ORG: FIFA")   ->  ["PER"]
    """
    words = canonical_key(tweet).split()
    types = set()
    for entity_type, name in _parse_entities(ner_value):
        entity_words = canonical_key(name).split()
        if not entity_words:
            continue
        span = len(entity_words)
        if any(words[i : i + span] == entity_words for i in range(len(words) - span + 1)):
            types.add(entity_type)
    return sorted(types)


def _parse_entities(ner_value) -> list[tuple[str, str]]:
    """'ORG: FIFA, Adidas | PER: Messi' -> [('ORG','FIFA'), ('ORG','Adidas'), ('PER','Messi')]"""
    if pd.isna(ner_value) or str(ner_value).strip().lower() == "none":
        return []
    entities = []
    for part in str(ner_value).split("|"):
        if ":" not in part:
            continue
        entity_type, names = part.split(":", 1)
        entities += [
            (entity_type.strip().upper(), name.strip()) for name in names.split(",") if name.strip()
        ]
    return entities


def single_label_report(y_true, y_pred, label_names: list[str]) -> tuple[str, dict]:
    """Score one single-label task: sklearn's per-class text report, and a
    dict of accuracy, macro P/R/F1, weighted F1 and the label list.

    Passing every class in `label_names` to sklearn means one the model never
    predicts scores zero instead of being dropped from the macro average.
    """
    label_ids = list(range(len(label_names)))
    text = classification_report(
        y_true, y_pred, labels=label_ids, target_names=label_names,
        digits=4, zero_division=0,
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, average="macro", zero_division=0
    )
    headline = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)
        ),
        "labels": list(label_names),
    }
    return text, headline


def entity_presence_report(true_types, pred_types) -> tuple[str, dict]:
    """Score entity-type presence as a four-way multi-label problem.

    Both arguments are one entity-type list per tweet, e.g. [['PER','ORG'],
    [], ['LOC']]; gold normally comes from findable_entity_types().

    Returns a text block and a dict of micro F1, macro F1, exact-match accuracy
    (all four types right at once) and per-type P/R/F1.
    """
    binarizer = MultiLabelBinarizer(classes=ENTITY_TYPES).fit([ENTITY_TYPES])
    true = binarizer.transform(true_types)
    pred = binarizer.transform(pred_types)

    headline = {
        "micro_f1": float(f1_score(true, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "exact_match_accuracy": float(accuracy_score(true, pred)),
        "per_type": {},
    }
    lines = [
        "Entity-type presence (comparable across models)",
        f"  Micro F1: {headline['micro_f1']:.4f}",
        f"  Macro F1: {headline['macro_f1']:.4f}",
        f"  Exact-match accuracy (all four types right): {headline['exact_match_accuracy']:.4f}",
    ]
    for index, entity_type in enumerate(ENTITY_TYPES):
        precision = float(precision_score(true[:, index], pred[:, index], zero_division=0))
        recall = float(recall_score(true[:, index], pred[:, index], zero_division=0))
        f1 = float(f1_score(true[:, index], pred[:, index], zero_division=0))
        headline["per_type"][entity_type] = {
            "precision": precision, "recall": recall, "f1": f1,
        }
        lines.append(
            f"  {entity_type:<5} precision {precision:.4f}  recall {recall:.4f}  F1 {f1:.4f}"
        )
    return "\n".join(lines), headline


def save_metrics(model_name: str, n_test: int, headlines: dict, model_dir) -> None:
    """Write metrics.json into model_dir for scripts/compare_models.py.

    `headlines` maps a task name to the dict single_label_report() or
    entity_presence_report() returned. n_test is stored too, because
    compare_models.py checks it matches before tabulating.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {"model": model_name, "n_test": int(n_test), **headlines}
    (model_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def save_predictions(model_dir, ids, gold: dict[str, list[str]], pred: dict[str, list[str]]) -> None:
    """Write predictions.csv (id, task, gold, pred) for the single-label tasks.

    One row per (tweet, task), with label names rather than ids so the file
    reads without the encoder that produced them.
    """
    assert gold.keys() == pred.keys(), "gold and pred must cover the same tasks"
    rows = []
    for task in gold:
        assert len(ids) == len(gold[task]) == len(pred[task]), (
            f"row-count mismatch on task {task!r}: "
            f"{len(ids)} ids, {len(gold[task])} gold, {len(pred[task])} pred"
        )
        rows += [
            {"id": row_id, "task": task, "gold": g, "pred": p}
            for row_id, g, p in zip(ids, gold[task], pred[task], strict=True)
        ]

    model_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(model_dir / "predictions.csv", index=False, encoding="utf-8")


__all__ = [
    "ENTITY_TYPES",
    "entity_presence_report",
    "findable_entity_types",
    "save_metrics",
    "save_predictions",
    "single_label_report",
    "types_from_bio",
]
