import joblib
import numpy as np
import spacy
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent

# Load trained artifacts
vectorizer = joblib.load(MODEL_DIR / "tfidf.pkl")
label_encoders = joblib.load(MODEL_DIR / "label_encoders.pkl")   # sentiment/emotion/topic
ner_binarizer = joblib.load(MODEL_DIR / "ner_binarizer.pkl")     # PER/ORG/LOC/EVENT
svm_models = joblib.load(MODEL_DIR / "svm_models.pkl")           # dict of LinearSVC/OvR

SINGLE_LABEL_COLUMNS = ["sentiment", "emotion", "topic"]
NER_LABEL_MAP = {
    "PERSON": "PER",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "EVENT": "EVENT",
}

# spaCy handles the actual entity TEXT extraction (who/what/where).
# The SVM only decides WHICH entity types are present.
nlp = spacy.load("en_core_web_trf")


def format_bio_entities(text):
    """Show each named entity using BIO tags, e.g. Cristiano [B-PER]."""
    doc = nlp(text)
    tags = ["O"] * len(doc)
    for entity in doc.ents:
        entity_type = NER_LABEL_MAP.get(entity.label_)
        if entity_type is None:
            continue
        for token_i in range(entity.start, entity.end):
            prefix = "B" if token_i == entity.start else "I"
            tags[token_i] = f"{prefix}-{entity_type}"
    return " ".join(
        f"{token.text} [{tag}]" if tag != "O" else token.text
        for token, tag in zip(doc, tags)
    )


def confidence_from_scores(scores):
    """Convert SVM decision scores into a display-only confidence estimate."""
    scores = np.asarray(scores).reshape(-1).astype(float)
    if scores.size == 1:
        probability = 1 / (1 + np.exp(-scores[0]))
        return max(probability, 1 - probability)
    shifted = scores - scores.max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    return probabilities.max()


def ner_confidence(scores, selected):
    probabilities = 1 / (1 + np.exp(-np.asarray(scores).reshape(-1).astype(float)))
    return probabilities[selected].max() if selected.any() else (1 - probabilities).max()


def predict(text):
    x = vectorizer.transform([text])

    result = {}
    for col in SINGLE_LABEL_COLUMNS:
        pred_idx = svm_models[col].predict(x)[0]
        result[col] = label_encoders[col].inverse_transform([pred_idx])[0]
        result[f"{col}_confidence"] = confidence_from_scores(
            svm_models[col].decision_function(x)
        )

    # Multi-label entity-type prediction from the SVM
    ner_pred = svm_models["ner"].predict(x)[0]
    selected_types = ner_pred.astype(bool)
    result["ner_bio"] = format_bio_entities(text)
    result["ner_confidence"] = ner_confidence(
        svm_models["ner"].decision_function(x), selected_types
    )

    return result


if __name__ == "__main__":
    print("========== Tweet Classifier (SVM) ==========")

    while True:
        comment = input("\nEnter a tweet (type 'exit' to quit): ")

        if comment.lower() == "exit":
            print("Program ended.")
            break

        prediction = predict(comment)

        print("\n========== RESULT ==========")
        print("Tweet     :", comment)
        print(f"Sentiment : {prediction['sentiment']} ({prediction['sentiment_confidence']:.0%} sure)")
        print(f"Emotion   : {prediction['emotion']} ({prediction['emotion_confidence']:.0%} sure)")
        print(f"Topic     : {prediction['topic']} ({prediction['topic_confidence']:.0%} sure)")
        print(f"NER       : {prediction['ner_bio']} ({prediction['ner_confidence']:.0%} sure)")
