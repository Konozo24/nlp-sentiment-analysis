"""Train the TF-IDF + LinearSVC baseline.

Run: ``python -m src.models.svm.train``
"""

import joblib

from .config import ARTIFACTS, ENTITY_TYPES, MODEL_DIR, TASKS
from .data import build_label_encoders, build_vectorizer, encode_targets, load_and_split
from .evaluate import evaluate
from .model import DEVICE, GPU_AVAILABLE, train_models


def main():
    print(f"Training on: {DEVICE}")
    train_df, test_df = load_and_split()
    encoders, binarizer = build_label_encoders(train_df)
    for task in TASKS:
        print(f"{task:<10} classes: {len(encoders[task].classes_)}")
    print("NER types:", ", ".join(ENTITY_TYPES))

    vectorizer = build_vectorizer()
    texts = train_df["tweet"]
    if GPU_AVAILABLE:
        import cudf
        texts = cudf.Series(texts.reset_index(drop=True))
    train_features = vectorizer.fit_transform(texts)
    print(f"TF-IDF features: {train_features.shape[1]}")
    single_targets, ner_targets = encode_targets(train_df, encoders, binarizer)
    models = train_models(train_features, single_targets, ner_targets)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, artifact in zip(ARTIFACTS, (vectorizer, encoders, binarizer, models)):
        joblib.dump(artifact, MODEL_DIR / name)
    report = evaluate(models, vectorizer, encoders, binarizer, test_df)
    print(report)
    (MODEL_DIR / "metrics.txt").write_text(report, encoding="utf-8")
    print(f"Saved artifacts and metrics to {MODEL_DIR}")


if __name__ == "__main__":
    main()
