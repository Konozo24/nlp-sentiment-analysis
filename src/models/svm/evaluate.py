"""Score the trained SVM artifacts on the test split.

Prints a report and writes it to metrics.txt, plus metrics.json for
scripts/compare_models.py. All four task metrics come from
src/models/metrics.py, shared with the other two models.

Run:  python -m src.models.svm.evaluate
"""

from src.models.metrics import (
    entity_presence_report,
    findable_entity_types,
    save_metrics,
    save_predictions,
    single_label_report,
)

from .config import ENTITY_TYPES, MODEL_DIR, TASKS
from .data import encode_targets, load_and_split
from .model import GPU_AVAILABLE, to_numpy
from .predict import load_model


def evaluate(models, vectorizer, encoders, binarizer, test_df):
    texts = test_df["tweet"]
    if GPU_AVAILABLE:
        import cudf
        texts = cudf.Series(texts.reset_index(drop=True))
    features = vectorizer.transform(texts)
    single_targets, _ = encode_targets(test_df, encoders, binarizer)

    lines = ["========== MODEL EVALUATION (SVM) ==========", f"Test rows: {len(test_df)}"]
    headlines = {}
    gold_labels: dict[str, list[str]] = {}
    pred_labels: dict[str, list[str]] = {}

    for task in TASKS:
        predictions = to_numpy(models[task].predict(features))
        report, headline = single_label_report(
            single_targets[task], predictions, list(encoders[task].classes_)
        )
        headlines[task] = headline
        gold_labels[task] = list(encoders[task].inverse_transform(single_targets[task]))
        pred_labels[task] = list(encoders[task].inverse_transform(predictions))
        lines.extend([
            f"\n===== {task.upper()} =====",
            f"Accuracy: {headline['accuracy']:.4f}  "
            f"macro P {headline['macro_precision']:.4f}  "
            f"R {headline['macro_recall']:.4f}  F1 {headline['macro_f1']:.4f}",
            report,
        ])

    # the one-vs-rest head already predicts entity-type presence, so its output
    # feeds the shared metric directly
    ner_predictions = to_numpy(models["ner"].predict(features))
    predicted_types = [
        [ENTITY_TYPES[i] for i, present in enumerate(row) if present] for row in ner_predictions
    ]
    true_types = [
        findable_entity_types(tweet, ner)
        for tweet, ner in zip(test_df["tweet"], test_df["ner"], strict=True)
    ]
    report, headline = entity_presence_report(true_types, predicted_types)
    headlines["ner"] = headline
    lines.extend([f"\n===== NER =====", report])

    return "\n".join(lines), headlines, gold_labels, pred_labels


def main():
    models, vectorizer, encoders, binarizer = load_model()
    _, test_df = load_and_split()
    report, headlines, gold_labels, pred_labels = evaluate(
        models, vectorizer, encoders, binarizer, test_df
    )
    print(report)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "metrics.txt").write_text(report, encoding="utf-8")
    save_metrics("svm", len(test_df), headlines, MODEL_DIR)
    save_predictions(MODEL_DIR, list(test_df["id"]), gold_labels, pred_labels)
    print(f"\nSaved to {MODEL_DIR / 'metrics.txt'}, metrics.json, and predictions.csv")


if __name__ == "__main__":
    main()
