"""Dataset loading, splitting, label encoding, and TF-IDF feature creation."""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from .config import DATA_PATH, ENTITY_TYPES, TASKS, TFIDF_KWARGS
from .model import make_vectorizer


def parse_entity_types(ner_value) -> list[str]:
    """Extract entity types from ``PER: Messi | ORG: FIFA`` annotations."""
    if pd.isna(ner_value) or str(ner_value).strip().lower() == "none":
        return []
    return [part.split(":", 1)[0].strip().upper() for part in str(ner_value).split("|") if ":" in part]


def load_and_split():
    """Load the cleaned SVM input and return the shared train/test split.

    Split labels come from data/processed/splits.csv (70/15/15). The 'val'
    slice is unused - LinearSVC has no early stopping.
    """
    df = pd.read_csv(DATA_PATH, encoding="utf-8").dropna(subset=["tweet", *TASKS]).reset_index(drop=True)
    df["ner"] = df["ner"].fillna("none")

    if "split" not in df.columns:
        raise ValueError(
            f"{DATA_PATH} has no 'split' column — re-run "
            "'python -m src.data_cleaning.base_cleaning' then "
            "'python -m src.data_cleaning.preprocess_svm'"
        )

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Split: train {len(train_df)}, test {len(test_df)} (val slice unused)")
    return train_df, test_df


def build_label_encoders(train_df):
    encoders = {task: LabelEncoder().fit(train_df[task].astype(str)) for task in TASKS}
    return encoders, MultiLabelBinarizer(classes=ENTITY_TYPES).fit([ENTITY_TYPES])


def encode_targets(df, encoders, binarizer):
    return ({task: encoders[task].transform(df[task].astype(str)) for task in TASKS}, binarizer.transform(df["ner"].map(parse_entity_types)))


def build_vectorizer():
    return make_vectorizer(TFIDF_KWARGS)
