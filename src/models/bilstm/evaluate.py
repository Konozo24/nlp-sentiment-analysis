"""Evaluate the BiLSTM on the test split — the same tweets the other two models are scored on.

Prints Accuracy, Precision, Recall and F1 per task. NER is scored word by
word; since most words are 'O', the entity-only average is also printed as
the more honest NER number.

The out-of-vocabulary rate is printed alongside, because it is the number that
explains the embedding choice: it is what fastText's character n-grams rescue
and what a GloVe table would simply lose.

Run:  python -m src.models.bilstm.evaluate
"""

# sklearn must import before torch, or Windows raises a heap-corruption crash
from sklearn.metrics import (  # noqa: I001
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

import torch

from .config import BATCH_SIZE, MODEL_DIR, TASKS
from .data import load_and_split, make_loader, unk_rate
from .predict import DEVICE, load_model


@torch.no_grad()
def collect_predictions(model, loader) -> tuple[dict, dict]:
    """Run the whole test set and gather true/predicted ids per task."""
    all_tasks = [*TASKS, "ner"]
    true = {task: [] for task in all_tasks}
    pred = {task: [] for task in all_tasks}

    for batch in loader:
        batch = {name: tensor.to(DEVICE) for name, tensor in batch.items()}
        predictions = model(batch["input_ids"], batch["mask"])
        for task in TASKS:
            true[task] += batch[task].tolist()
            pred[task] += predictions[task].argmax(dim=-1).tolist()
        # only score real words: padding slots carry no label.
        true["ner"] += batch["ner"][batch["mask"]].tolist()
        decoded = model.crf.decode(predictions["ner"], mask=batch["mask"])
        pred["ner"] += [tag for sample in decoded for tag in sample]

    return true, pred


def main():
    _, _, test_df = load_and_split()
    model, vocab, labels = load_model()  # already in eval mode

    all_tasks = [*TASKS, "ner"]
    loader = make_loader(test_df, vocab, labels, BATCH_SIZE, pin_memory=DEVICE.type == "cuda")
    true, pred = collect_predictions(model, loader)

    lines = [
        f"Out-of-vocabulary rate on the test split: {unk_rate(test_df, vocab):.2%}",
    ]
    for task in all_tasks:
        lines.append(f"\n===== {task.upper()} =====")
        lines.append(f"Accuracy: {accuracy_score(true[task], pred[task]):.4f}")
        lines.append(
            classification_report(
                true[task],
                pred[task],
                labels=range(len(labels[task])),
                target_names=labels[task],
                digits=4,
                zero_division=0,
            )
        )
        if task == "ner":
            entity_tags = [i for i, name in enumerate(labels["ner"]) if name != "O"]
            p, r, f1, _ = precision_recall_fscore_support(
                true["ner"], pred["ner"], labels=entity_tags, average="macro", zero_division=0
            )
            lines.append(
                f"Entity tags only (excluding O): "
                f"precision {p:.4f}, recall {r:.4f}, F1 {f1:.4f}"
            )

    report = "\n".join(lines)
    print(report)
    (MODEL_DIR / "metrics.txt").write_text(report, encoding="utf-8")
    print(f"\nSaved to {MODEL_DIR / 'metrics.txt'}")


if __name__ == "__main__":
    main()
