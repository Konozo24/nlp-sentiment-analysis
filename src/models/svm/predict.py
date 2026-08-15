import joblib
import spacy
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent

# Load trained artifacts
vectorizer = joblib.load(MODEL_DIR / "tfidf.pkl")
label_encoders = joblib.load(MODEL_DIR / "label_encoders.pkl")   # sentiment/emotion/topic
ner_binarizer = joblib.load(MODEL_DIR / "ner_binarizer.pkl")     # PER/ORG/LOC/EVENT
svm_models = joblib.load(MODEL_DIR / "svm_models.pkl")           # dict of LinearSVC/OvR

SINGLE_LABEL_COLUMNS = ["sentiment", "emotion", "topic"]

# spaCy handles the actual entity TEXT extraction (who/what/where).
# The SVM only decides WHICH entity types are present.
nlp = spacy.load("en_core_web_trf")


def extract_entities(text):
    doc = nlp(text)

    grouped = {"PER": [], "ORG": [], "LOC": [], "EVENT": []}

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            grouped["PER"].append(ent.text)
        elif ent.label_ == "ORG":
            grouped["ORG"].append(ent.text)
        elif ent.label_ in ("GPE", "LOC"):
            grouped["LOC"].append(ent.text)
        elif ent.label_ == "EVENT":
            grouped["EVENT"].append(ent.text)

    return grouped


def format_entities(grouped, allowed_types=None):
    """Format entities, optionally restricted to the types the SVM flagged as present."""
    parts = []
    for t in ("PER", "ORG", "LOC", "EVENT"):
        if allowed_types is not None and t not in allowed_types:
            continue
        if grouped[t]:
            parts.append(f"{t}: " + ", ".join(sorted(set(grouped[t]))))
    return " | ".join(parts) if parts else "None"


def predict(text):
    x = vectorizer.transform([text])

    result = {}
    for col in SINGLE_LABEL_COLUMNS:
        pred_idx = svm_models[col].predict(x)[0]
        result[col] = label_encoders[col].inverse_transform([pred_idx])[0]

    # Multi-label entity-type prediction from the SVM
    ner_pred = svm_models["ner"].predict(x)[0]
    predicted_types = ner_binarizer.classes_[ner_pred.astype(bool)]

    # spaCy fills in the actual entity text
    entities = extract_entities(text)

    result["ner_types_svm"] = list(predicted_types)
    result["ner_svm_filtered"] = format_entities(entities, allowed_types=set(predicted_types))
    result["ner_spacy_full"] = format_entities(entities)  # unfiltered, for comparison

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
        print("Tweet          :", comment)
        print("Sentiment      :", prediction["sentiment"])
        print("Emotion        :", prediction["emotion"])
        print("Topic          :", prediction["topic"])
        print("NER types (SVM):", prediction["ner_types_svm"] or "None")
        print("NER (filtered) :", prediction["ner_svm_filtered"])
        print("NER (spaCy all):", prediction["ner_spacy_full"])