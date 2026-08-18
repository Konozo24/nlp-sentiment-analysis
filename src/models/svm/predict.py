"""Predict SVM labels for a tweet.

Run: ``python -m src.models.svm.predict`` or pass a tweet as arguments.
"""

import sys

import joblib
import numpy as np

from .config import ARTIFACTS, LEGACY_MODEL_DIR, MODEL_DIR, TASKS
from .model import GPU_AVAILABLE
from .ner_bio import format_bio_entities


def _artifact_dir():
    """Prefer the new data/models location, fall back to pre-refactor files."""
    if all((MODEL_DIR / name).exists() for name in ARTIFACTS):
        return MODEL_DIR
    if all((LEGACY_MODEL_DIR / name).exists() for name in ARTIFACTS):
        return LEGACY_MODEL_DIR
    raise FileNotFoundError("No SVM artifacts found. Run: python -m src.models.svm.train")


def load_model():
    directory = _artifact_dir()
    return tuple(joblib.load(directory / name) for name in ("svm_models.pkl", "tfidf.pkl", "label_encoders.pkl", "ner_binarizer.pkl"))


def confidence_from_scores(scores):
    scores = np.asarray(scores).reshape(-1).astype(float)
    if scores.size == 1:
        probability = 1 / (1 + np.exp(-scores[0]))
        return max(probability, 1 - probability)
    shifted = scores - scores.max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    return probabilities.max()


def predict(text, models=None, vectorizer=None, encoders=None, binarizer=None):
    if models is None:
        models, vectorizer, encoders, binarizer = load_model()
    texts = [text]
    if GPU_AVAILABLE:
        import cudf
        texts = cudf.Series(texts)
    features = vectorizer.transform(texts)
    result = {}
    for task in TASKS:
        predicted = models[task].predict(features)[0]
        result[task] = encoders[task].inverse_transform([predicted])[0]
        result[f"{task}_confidence"] = confidence_from_scores(models[task].decision_function(features))
    selected = np.asarray(models["ner"].predict(features)[0]).astype(bool)
    scores = np.asarray(models["ner"].decision_function(features)).reshape(-1).astype(float)
    probabilities = 1 / (1 + np.exp(-scores))
    result["ner_confidence"] = probabilities[selected].max() if selected.any() else (1 - probabilities).max()
    result["ner_bio"] = format_bio_entities(text)
    return result


def main():
    models, vectorizer, encoders, binarizer = load_model()
    if len(sys.argv) > 1:
        tweets = [" ".join(sys.argv[1:])]
    else:
        tweets = None
    while True:
        tweet = tweets.pop() if tweets else input("Enter a tweet (empty to quit): ").strip()
        if not tweet:
            break
        result = predict(tweet, models, vectorizer, encoders, binarizer)
        for task in TASKS:
            print(f"{task.title():9s}: {result[task]} ({result[f'{task}_confidence']:.0%} sure)")
        print(f"NER      : {result['ner_bio']} ({result['ner_confidence']:.0%} sure)")
        if tweets is not None:
            break


if __name__ == "__main__":
    main()
