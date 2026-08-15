"""Evaluate TRABSA on the test split — the same tweets the BiLSTM was scored on.

Prints Accuracy, Precision, Recall and F1 per task. NER is scored word by
word; since most words are 'O', the entity-only average is also printed as
the more honest NER number.

Run:  python -m src.models.trabsa.evaluate
"""

# sklearn must import before torch, or Windows raises a heap-corruption crash
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

import torch

from .config import BATCH_SIZE, MODEL_DIR, TASKS
from .data import load_and_split, make_batches
from .predict import DEVICE, load_model


def main():
    _, _, test_df = load_and_split()
    model, tokenizer, labels = load_model()

    all_tasks = [*TASKS, "ner"]
    true = {task: [] for task in all_tasks}
    pred = {task: [] for task in all_tasks}
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

    lines = []
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
            lines.append(f"Entity tags only (excluding O): precision {p:.4f}, recall {r:.4f}, F1 {f1:.4f}")

    report = "\n".join(lines)
    print(report)
    (MODEL_DIR / "metrics.txt").write_text(report, encoding="utf-8")
    print(f"\nSaved to {MODEL_DIR / 'metrics.txt'}")


if __name__ == "__main__":
    main()
