"""Dataset loading, splitting, label encoding, and TF-IDF feature creation."""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from src.models.metrics import findable_entity_types

from .config import DATA_PATH, ENTITY_TYPES, TASKS, TFIDF_KWARGS
from .model import make_vectorizer


def load_and_split():
    """Load the cleaned SVM input and return the shared train/test split.

    Split labels come from data/processed/splits.csv (70/15/15). The 'val'
    slice is unused - LinearSVC has no early stopping.
    """
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df = df[df["lang"] == "en"]
    df = df.dropna(subset=["tweet", *TASKS]).reset_index(drop=True)
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
    """Encode targets for one dataframe.

    Returns a (single_targets, ner_targets) pair: an integer label array per
    task in TASKS, and a binary matrix over ENTITY_TYPES marking which entity
    types each tweet mentions.

    The NER target comes from findable_entity_types(), the same gold the shared
    evaluation metric uses, so the model trains on what it is scored on.
    """
    entity_types = [
        findable_entity_types(tweet, ner) for tweet, ner in zip(df["tweet"], df["ner"], strict=True)
    ]
    return (
        {task: encoders[task].transform(df[task].astype(str)) for task in TASKS},
        binarizer.transform(entity_types),
    )


def build_vectorizer():
    return make_vectorizer(TFIDF_KWARGS)
