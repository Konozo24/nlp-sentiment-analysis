import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# ==========================
# Device (GPU via RAPIDS cuML if available, else CPU sklearn)
# ==========================
# LinearSVC on sparse TF-IDF doesn't get much from the GPU (it's already
# fast on CPU), but we still try cuML first so this drops into a CUDA box
# automatically. Falls back cleanly to scikit-learn if cuML/GPU isn't there.

try:
    import cupy as cp
    from cuml.svm import LinearSVC as cuLinearSVC
    from cuml.feature_extraction.text import TfidfVectorizer as cuTfidfVectorizer
    from cuml.multiclass import OneVsRestClassifier as cuOneVsRestClassifier

    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    GPU_AVAILABLE = False

if GPU_AVAILABLE:
    device = "cuda"
    print(f"Using device : cuda (RAPIDS cuML)")
    print(f"GPU          : {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    TfidfVectorizer = cuTfidfVectorizer
    LinearSVC = cuLinearSVC
    OneVsRestClassifier = cuOneVsRestClassifier
else:
    device = "cpu"
    print("Using device : cpu (scikit-learn) — cuML/GPU not available")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.multiclass import OneVsRestClassifier

# ==========================
# Paths
# ==========================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR.parent.parent / "data" / "processed" / "svm_input.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

# ==========================
# Load Dataset
# ==========================

print("Loading dataset...")

data = pd.read_csv(DATASET_PATH)
data["ner"] = data["ner"].fillna("none")

X = data["tweet"]

# sentiment / emotion / topic stay single-label multiclass.
# ner is handled separately below as multi-label entity-type detection.
SINGLE_LABEL_COLUMNS = ["sentiment", "emotion", "topic"]
ENTITY_TYPES = ["PER", "ORG", "LOC", "EVENT"]

# ==========================
# Encode single-label targets
# ==========================

encoders = {}
y_single = pd.DataFrame(index=data.index)

for col in SINGLE_LABEL_COLUMNS:
    enc = LabelEncoder()
    y_single[col] = enc.fit_transform(data[col])
    encoders[col] = enc
    print(f"{col:<10} classes: {enc.classes_.size}")

# ==========================
# Encode NER as multi-label entity-type presence
# ==========================
# Raw ner values look like: "PER: Messi, Ronaldo | ORG: FIFA | LOC: Qatar"
# Encoding the whole string as ONE class (old approach) explodes into
# thousands of near-unique classes -> unlearnable. Instead we predict,
# independently, whether each entity TYPE is present in the tweet.


def parse_entity_types(ner_str):
    if ner_str == "none":
        return []
    return [seg.split(":")[0].strip().upper() for seg in ner_str.split("|")]


data["ner_types"] = data["ner"].apply(parse_entity_types)

mlb = MultiLabelBinarizer(classes=ENTITY_TYPES)
y_ner = mlb.fit_transform(data["ner_types"])  # shape (n, 4), one column per type

print("\nNER entity-type prevalence (share of tweets containing each type):")
for i, t in enumerate(ENTITY_TYPES):
    print(f"  {t:<6}: {y_ner[:, i].mean():.3f}")

# ==========================
# Train/Test Split
# ==========================
# Split once, apply the same row indices to every target so all labels
# share the same train/test tweets.

idx_train, idx_test = train_test_split(
    data.index,
    test_size=0.2,
    random_state=42,
    stratify=data["sentiment"],
)

X_train_text, X_test_text = X.loc[idx_train], X.loc[idx_test]

if GPU_AVAILABLE:
    import cudf
    X_train_text = cudf.Series(X_train_text.reset_index(drop=True))
    X_test_text = cudf.Series(X_test_text.reset_index(drop=True))

y_train_single = y_single.loc[idx_train]
y_test_single = y_single.loc[idx_test]

y_train_ner = y_ner[data.index.get_indexer(idx_train)]
y_test_ner = y_ner[data.index.get_indexer(idx_test)]

# ==========================
# TF-IDF feature extraction
# ==========================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print(f"\nTF-IDF features: {X_train.shape[1]}")


def to_numpy(arr):
    """cuML/cupy results -> numpy, no-op on CPU."""
    return cp.asnumpy(arr) if GPU_AVAILABLE else np.asarray(arr)


def svc_kwargs():
    # cuML's LinearSVC doesn't accept class_weight; keep it CPU-only.
    return {} if GPU_AVAILABLE else {"class_weight": "balanced"}


# ==========================
# Train SVM: sentiment / emotion / topic (multiclass)
# ==========================

svm_models = {}

for col in SINGLE_LABEL_COLUMNS:
    print(f"\nTraining LinearSVC for '{col}'...")
    clf = LinearSVC(**svc_kwargs())
    y_col = y_train_single[col].to_numpy()
    if GPU_AVAILABLE:
        y_col = cp.asarray(y_col)
    clf.fit(X_train, y_col)
    svm_models[col] = clf

# ==========================
# Train SVM: NER (multi-label entity-type presence)
# ==========================

print("\nTraining LinearSVC (One-vs-Rest) for 'ner'...")
y_train_ner_fit = cp.asarray(y_train_ner) if GPU_AVAILABLE else y_train_ner
ner_clf = OneVsRestClassifier(LinearSVC(**svc_kwargs()))
ner_clf.fit(X_train, y_train_ner_fit)
svm_models["ner"] = ner_clf

# ==========================
# Evaluation
# ==========================

print("\n========== MODEL EVALUATION ==========\n")

for col in SINGLE_LABEL_COLUMNS:
    print(f"----- {col.upper()} -----")
    preds = to_numpy(svm_models[col].predict(X_test))
    print("Accuracy:", accuracy_score(y_test_single[col], preds))
    print(
        classification_report(
            y_test_single[col],
            preds,
            target_names=encoders[col].classes_,
            zero_division=0,
        )
    )

print("----- NER (entity-type presence, multi-label) -----")
ner_preds = to_numpy(svm_models["ner"].predict(X_test))

# Subset accuracy = exact match across all 4 entity-type flags for a tweet
subset_acc = accuracy_score(y_test_ner, ner_preds)
print(f"Exact-match accuracy (all 4 types correct): {subset_acc:.3f}")

# Micro F1: pool TP/FP/FN across all 4 entity types FIRST, then compute one
# F1 from those totals. Every individual prediction counts equally, so
# common types (e.g. PER/ORG) dominate the score. Answers: "overall, what
# fraction of predictions are correct?"
print(f"Micro F1 : {f1_score(y_test_ner, ner_preds, average='micro', zero_division=0):.3f}")

# Macro F1: compute F1 separately per entity type, then average those 4
# numbers. Each type counts equally regardless of how rare it is, so a
# rare-but-hard type (e.g. EVENT) drags the score down just as much as a
# common one. Answers: "how well does the model do on EVERY type, even
# the rare ones?" A macro F1 much lower than micro F1 is a red flag that
# one or more types are being handled badly while staying hidden behind
# strong performance on the common types.
print(f"Macro F1 : {f1_score(y_test_ner, ner_preds, average='macro', zero_division=0):.3f}")
print()
print(f"  {'TYPE':<7}{'Precision':>11}{'Recall':>10}{'F1':>9}{'Accuracy':>11}")
print("  " + "-" * 48)
for i, t in enumerate(ENTITY_TYPES):
    p = precision_score(y_test_ner[:, i], ner_preds[:, i], zero_division=0)
    r = recall_score(y_test_ner[:, i], ner_preds[:, i], zero_division=0)
    f1 = f1_score(y_test_ner[:, i], ner_preds[:, i], zero_division=0)
    acc = accuracy_score(y_test_ner[:, i], ner_preds[:, i])
    print(f"  {t:<7}{p:>11.3f}{r:>10.3f}{f1:>9.3f}{acc:>11.3f}")

print(
    "\nNote: this predicts WHICH entity types are present per tweet, not the exact "
    "entity text. The actual entity strings (who/what/where) still come from spaCy "
    "in predict.py."
)

# ==========================
# Save Model
# ==========================

joblib.dump(vectorizer, OUTPUT_DIR / "tfidf.pkl")
joblib.dump(encoders, OUTPUT_DIR / "label_encoders.pkl")
joblib.dump(mlb, OUTPUT_DIR / "ner_binarizer.pkl")
joblib.dump(svm_models, OUTPUT_DIR / "svm_models.pkl")

print("\nModel saved successfully!")
print("Artifacts: tfidf.pkl, label_encoders.pkl, ner_binarizer.pkl, svm_models.pkl")
