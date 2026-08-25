"""Score the trained RobertaBase on the test split.

Prints a report and writes it to metrics.txt, plus metrics.json for
scripts/compare_models.py. The headline numbers come from
src/models/metrics.py, shared with the other two models; the token-level BIO
report below them diagnoses the CRF and is specific to this model.

Run:  python -m src.models.robertabase.evaluate
"""

from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

import torch

from src.models.metrics import (
    entity_presence_report,
    findable_entity_types,
    save_metrics,
    save_predictions,
    single_label_report,
    types_from_bio,
)

from .config import BATCH_SIZE, MODEL_DIR, TASKS
from .data import load_and_split, make_batches
from .predict import DEVICE, load_model


def main():
    _, _, test_df = load_and_split()
    model, tokenizer, labels = load_model()

    all_tasks = [*TASKS, "ner"]
    true = {task: [] for task in all_tasks}
    pred = {task: [] for task in all_tasks}
    # grouped by tweet, which the entity-presence metric needs
    per_tweet_tags: list[list[int]] = []

    with torch.no_grad():
        for batch in make_batches(test_df, tokenizer, labels, BATCH_SIZE):
            batch = {name: tensor.to(DEVICE) for name, tensor in batch.items()}
            predictions = model(
                batch["input_ids"], batch["attention_mask"], batch["word_index"], batch["word_mask"]
            )
            for task in TASKS:
                true[task] += batch[task].tolist()
                pred[task] += predictions[task].argmax(dim=-1).tolist()
            true["ner"] += batch["ner"][batch["word_mask"]].tolist()
            decoded = model.crf.decode(predictions["ner"], mask=batch["word_mask"])
            pred["ner"] += [tag for sample in decoded for tag in sample]
            per_tweet_tags += decoded

    lines = [
        "========== MODEL EVALUATION (RoBERTa-base) ==========",
        f"Test rows: {len(test_df)}",
    ]
    headlines = {}
    gold_labels: dict[str, list[str]] = {}
    pred_labels: dict[str, list[str]] = {}

    for task in TASKS:
        report, headline = single_label_report(true[task], pred[task], labels[task])
        headlines[task] = headline
        gold_labels[task] = [labels[task][i] for i in true[task]]
        pred_labels[task] = [labels[task][i] for i in pred[task]]
        lines.extend([
            f"\n===== {task.upper()} =====",
            f"Accuracy: {headline['accuracy']:.4f}  "
            f"macro P {headline['macro_precision']:.4f}  "
            f"R {headline['macro_recall']:.4f}  F1 {headline['macro_f1']:.4f}",
            report,
        ])

    predicted_types = [
        types_from_bio([labels["ner"][tag] for tag in tags]) for tags in per_tweet_tags
    ]
    true_types = [
        findable_entity_types(tweet, ner)
        for tweet, ner in zip(test_df["tweet"], test_df["ner"], strict=True)
    ]
    presence_report, headline = entity_presence_report(true_types, predicted_types)
    headlines["ner"] = headline
    lines.extend([f"\n===== NER =====", presence_report])

    lines.append("\n--- token-level BIO detail (this model only) ---")
    lines.append(f"Token accuracy incl. 'O': {accuracy_score(true['ner'], pred['ner']):.4f}")
    lines.append(
        classification_report(
            true["ner"], pred["ner"], labels=range(len(labels["ner"])),
            target_names=labels["ner"], digits=4, zero_division=0,
        )
    )
    entity_tags = [i for i, name in enumerate(labels["ner"]) if name != "O"]
    p, r, f1, _ = precision_recall_fscore_support(
        true["ner"], pred["ner"], labels=entity_tags, average="macro", zero_division=0
    )
    lines.append(f"Entity tags only (excluding O): precision {p:.4f}, recall {r:.4f}, F1 {f1:.4f}")

    report = "\n".join(lines)
    print(report)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "metrics.txt").write_text(report, encoding="utf-8")
    save_metrics("robertabase", len(test_df), headlines, MODEL_DIR)
    save_predictions(MODEL_DIR, list(test_df["id"]), gold_labels, pred_labels)
    print(f"\nSaved to {MODEL_DIR / 'metrics.txt'}, metrics.json, and predictions.csv")


if __name__ == "__main__":
    main()
