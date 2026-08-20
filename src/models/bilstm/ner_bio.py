"""Turn the 'ner' column into one BIO tag per word of the (already cleaned) tweet.

The CSV stores entities as text like:
    "PER: Lionel Messi | ORG: FIFA, Adidas"
but the model needs one tag per word:
    words: ["messi", "scores", "for", "fifa"]
    tags:  ["B-PER", "O",      "O",   "B-ORG"]

B- marks the first word of an entity, I- the following words, O = not an
entity. Each entity name is cleaned with the SAME clean_for_bilstm() applied
to the tweets, so both sides are lowercased and stripped of punctuation and
the strings line up directly. Entities we cannot find in the tweet are skipped.

Run:  python -m src.models.bilstm.ner_bio      (prints how many entities matched)
"""

import pandas as pd

from src.data_cleaning.preprocess_bilstm import clean_for_bilstm

from .config import DATA_PATH


def parse_entities(raw) -> list[tuple[str, str]]:
    """'ORG: FIFA, Adidas | PER: Messi' -> [('ORG','FIFA'), ('ORG','Adidas'), ('PER','Messi')]"""
    if pd.isna(raw):
        return []
    entities = []
    for part in str(raw).split("|"):  # "ORG: FIFA, Adidas"
        if ":" not in part:
            continue
        etype, names = part.split(":", 1)  # "ORG", " FIFA, Adidas"
        for name in names.split(","):
            if name.strip():
                entities.append((etype.strip().upper(), name.strip()))
    return entities


def tag_sentence(words: list[str], entities: list[tuple[str, str]]) -> tuple[list[str], int]:
    """Find each entity's words inside the tweet's words and tag them B-/I-.

    Longer entities are placed first so 'fifa world cup' wins over 'fifa'.
    Returns (tags, how many entities were found).
    """
    tags = ["O"] * len(words)

    # clean each entity the same way the tweet was cleaned, then sort longest first.
    cleaned = [(etype, clean_for_bilstm(name).split()) for etype, name in entities]
    cleaned = [(etype, entity_words) for etype, entity_words in cleaned if entity_words]
    cleaned.sort(key=lambda e: len(e[1]), reverse=True)

    found_count = 0
    for etype, entity_words in cleaned:
        n = len(entity_words)
        found = False
        for i in range(len(words) - n + 1):
            # match here, and don't overwrite an entity already tagged
            if words[i : i + n] == entity_words and all(t == "O" for t in tags[i : i + n]):
                tags[i] = "B-" + etype
                for j in range(i + 1, i + n):
                    tags[j] = "I-" + etype
                found = True
        found_count += found

    return tags, found_count


def add_bio_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'bio_tags' column: space-joined tags, one per word of the tweet."""
    df = df.copy()
    all_tags = []
    total = matched = 0

    for text, raw_ner in zip(df["tweet"], df["ner"], strict=True):
        words = str(text).split()
        entities = parse_entities(raw_ner)
        tags, found = tag_sentence(words, entities)
        all_tags.append(" ".join(tags))
        total += len(entities)
        matched += found

    df["bio_tags"] = all_tags
    if total:
        print(f"NER: matched {matched}/{total} entities ({matched / total:.1%})")
    return df


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df = add_bio_tags(df)
    print(df[["tweet", "bio_tags"]].head(3).to_string())
