"""Dataset loading, splitting, label encoding, and TF-IDF feature creation."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from .config import DATA_PATH, ENTITY_TYPES, SEED, TASKS, TEST_SIZE, TFIDF_KWARGS
from .model import make_vectorizer


def parse_entity_types(ner_value) -> list[str]:
    """Extract entity types from ``PER: Messi | ORG: FIFA`` annotations."""
    if pd.isna(ner_value) or str(ner_value).strip().lower() == "none":
        return []
    return [part.split(":", 1)[0].strip().upper() for part in str(ner_value).split("|") if ":" in part]


def load_and_split():
    """Load the cleaned SVM input and make the reproducible 80/20 split."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8").dropna(subset=["tweet", *TASKS]).reset_index(drop=True)
    df["ner"] = df["ner"].fillna("none")
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=SEED, stratify=df["sentiment"])
    print(f"Split: train {len(train_df)}, test {len(test_df)}")
    return train_df.copy(), test_df.copy()


def build_label_encoders(train_df):
    encoders = {task: LabelEncoder().fit(train_df[task].astype(str)) for task in TASKS}
    return encoders, MultiLabelBinarizer(classes=ENTITY_TYPES).fit([ENTITY_TYPES])


def encode_targets(df, encoders, binarizer):
    return ({task: encoders[task].transform(df[task].astype(str)) for task in TASKS}, binarizer.transform(df["ner"].map(parse_entity_types)))


def build_vectorizer():
    return make_vectorizer(TFIDF_KWARGS)
