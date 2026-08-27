"""Score the trained BiLSTM on the test split.

Writes metrics.txt to read, metrics.json for scripts/compare_models.py, and
predictions.csv for error analysis. Headline numbers come from
src/models/metrics.py, shared with the other models so the scores compare.

The OOV rate and the token-level BIO report are printed for this model alone -
neither has an SVM equivalent.

Run:  python -m src.models.bilstm.evaluate
"""


from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

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
from .data import load_and_split, make_loader, unk_rate
from .predict import DEVICE, load_model


@torch.no_grad()
def collect_predictions(model, loader) -> tuple[dict, dict, list]:
    """Run the model over a loader and gather its predictions.

    true, pred      one flat list of label ids per task, plus 'ner'
    per_tweet_tags  the same NER tags, but grouped by tweet - the shape the
                    entity-presence metric needs
    """
    all_tasks = [*TASKS, "ner"]
    true = {task: [] for task in all_tasks}
    pred = {task: [] for task in all_tasks}
    per_tweet_tags: list[list[int]] = []

    for batch in loader:
        batch = {name: tensor.to(DEVICE) for name, tensor in batch.items()}
        predictions = model(batch["input_ids"], batch["mask"])
        for task in TASKS:
            true[task] += batch[task].tolist()
            pred[task] += predictions[task].argmax(dim=-1).tolist()
        # real words only - padding id 0 is a real tag (B-EVENT), so scoring
        # it would count thousands of phantom entities
        true["ner"] += batch["ner"][batch["mask"]].tolist()
        decoded = model.crf.decode(predictions["ner"], mask=batch["mask"])
        pred["ner"] += [tag for sample in decoded for tag in sample]
        per_tweet_tags += decoded

    return true, pred, per_tweet_tags


def main():
    # run the model over the test split
    _, _, test_df = load_and_split()
    model, vocab, labels = load_model()

    loader = make_loader(test_df, vocab, labels, BATCH_SIZE, pin_memory=DEVICE.type == "cuda")
    true, pred, per_tweet_tags = collect_predictions(model, loader)

    lines = [
        "========== MODEL EVALUATION (BiLSTM) ==========",
        f"Test rows: {len(test_df)}",
        f"Out-of-vocabulary rate on the test split: {unk_rate(test_df, vocab):.2%}",
    ]
    headlines = {}
    gold_labels: dict[str, list[str]] = {}
    pred_labels: dict[str, list[str]] = {}

    # sentiment, emotion, topic
    for task in TASKS:
        report, headline = single_label_report(true[task], pred[task], labels[task])
        headlines[task] = headline
        gold_labels[task] = [labels[task][i] for i in true[task]]
        pred_labels[task] = [labels[task][i] for i in pred[task]]
        lines.extend([
            f"\n===== {task.upper()} =====",
            (
                f"Accuracy: {headline['accuracy']:.4f}  "
                f"macro P {headline['macro_precision']:.4f}  "
                f"R {headline['macro_recall']:.4f}  F1 {headline['macro_f1']:.4f}"
            ),
            report,
        ])

    # NER, scored as entity presence
    # collapse each tweet's tags into the set of types it mentions - the one
    # framing all four models share, so these numbers compare across them
    predicted_types = [
        types_from_bio([labels["ner"][tag] for tag in tags]) for tags in per_tweet_tags
    ]
    true_types = [
        findable_entity_types(tweet, ner)
        for tweet, ner in zip(test_df["tweet"], test_df["ner"], strict=True)
    ]
    presence_report, headline = entity_presence_report(true_types, predicted_types)
    headlines["ner"] = headline
    lines.extend(["\n===== NER =====", presence_report])

    # NER, scored per token
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

    # save result
    report = "\n".join(lines)
    print(report)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "metrics.txt").write_text(report, encoding="utf-8")
    save_metrics("bilstm", len(test_df), headlines, MODEL_DIR)
    save_predictions(MODEL_DIR, list(test_df["id"]), gold_labels, pred_labels)
    print(f"\nSaved to {MODEL_DIR / 'metrics.txt'}, metrics.json, and predictions.csv")


if __name__ == "__main__":
    main()
