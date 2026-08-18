"""Evaluate trained SVM artifacts on the reproducible test split."""

from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

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
    single_targets, ner_targets = encode_targets(test_df, encoders, binarizer)
    lines = ["========== MODEL EVALUATION =========="]
    for task in TASKS:
        predictions = to_numpy(models[task].predict(features))
        lines.extend([f"\n----- {task.upper()} -----", f"Accuracy: {accuracy_score(single_targets[task], predictions):.4f}", classification_report(single_targets[task], predictions, target_names=encoders[task].classes_, zero_division=0)])
    predictions = to_numpy(models["ner"].predict(features))
    lines.extend(["----- NER (entity-type presence, multi-label) -----", f"Exact-match accuracy: {accuracy_score(ner_targets, predictions):.4f}", f"Micro F1: {f1_score(ner_targets, predictions, average='micro', zero_division=0):.4f}", f"Macro F1: {f1_score(ner_targets, predictions, average='macro', zero_division=0):.4f}"])
    for index, entity_type in enumerate(ENTITY_TYPES):
        lines.append(f"{entity_type}: precision {precision_score(ner_targets[:, index], predictions[:, index], zero_division=0):.4f}, recall {recall_score(ner_targets[:, index], predictions[:, index], zero_division=0):.4f}")
    return "\n".join(lines)


def main():
    models, vectorizer, encoders, binarizer = load_model()
    _, test_df = load_and_split()
    report = evaluate(models, vectorizer, encoders, binarizer, test_df)
    print(report)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "metrics.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
