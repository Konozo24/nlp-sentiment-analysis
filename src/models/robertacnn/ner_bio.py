"""Turn the 'ner' column into one BIO tag per word, for the RobertaCNN pipeline.

Thin wrapper around the shared core in src.data_cleaning.ner_bio_core —
see that module for the tagging logic. Only the clean function differs
between team members' models.

Run:  python -m src.models.robertacnn.ner_bio
"""

import pandas as pd
import string

from src.data_cleaning.preprocess_robertacnn import clean_for_robertacnn

from .config import DATA_PATH

def parse_entities(raw) -> list[tuple[str, str]]:
    if pd.isna(raw):
        return []
    entities = []
    for part in str(raw).split("|"):
        if ":" not in part:
            continue
        etype, names = part.split(":", 1)
        for name in names.split(","):
            if name.strip():
                entities.append((etype.strip().upper(), name.strip()))
    return entities


def _match_key(word: str) -> str:
    return word.strip(string.punctuation).lower()


def tag_sentence(words: list[str], entities: list[tuple[str, str]], clean_fn) -> tuple[list[str], int]:
    tags = ["O"] * len(words)
    match_words = [_match_key(w) for w in words]

    cleaned = [(etype, [_match_key(w) for w in clean_fn(name).split()]) for etype, name in entities]
    cleaned = [(etype, ew) for etype, ew in cleaned if ew]
    cleaned.sort(key=lambda e: len(e[1]), reverse=True)

    found_count = 0
    for etype, entity_words in cleaned:
        n = len(entity_words)
        found = False
        for i in range(len(match_words) - n + 1):
            if match_words[i : i + n] == entity_words and all(t == "O" for t in tags[i : i + n]):
                tags[i] = "B-" + etype
                for j in range(i + 1, i + n):
                    tags[j] = "I-" + etype
                found = True
        found_count += found

    return tags, found_count


def add_bio_tags(df: pd.DataFrame, clean_fn) -> pd.DataFrame:
    df = df.copy()
    all_tags, total, matched = [], 0, 0
    for text, raw_ner in zip(df["tweet"], df["ner"]):
        words = str(text).split()
        entities = parse_entities(raw_ner)
        tags, found = tag_sentence(words, entities, clean_fn)
        all_tags.append(" ".join(tags))
        total += len(entities)
        matched += found
    df["bio_tags"] = all_tags
    if total:
        print(f"NER: matched {matched}/{total} entities ({matched / total:.1%})")
    return df


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df = add_bio_tags(df, clean_for_robertacnn)
    print(df[["tweet", "bio_tags"]].head(3).to_string())