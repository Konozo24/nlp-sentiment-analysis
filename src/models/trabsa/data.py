"""Load the data, split it, and turn tweets into RoBERTa subword batches,
keeping one slot per WORD.

RoBERTa splits words into subwords ('messi' -> 'mess' + 'i'), but the BIO
tags from ner_bio.py are one-per-word. So after the encoder runs we keep
only the FIRST subword of each word as that word's vector — the standard
alignment trick for transformer sequence labelling. Every batch therefore
carries:

  input_ids / attention_mask : subword level, what the encoder reads
  word_index                 : where each word's first subword sits
  word_mask                  : which word slots are real (not padding)
  sentiment/emotion/topic    : one label per tweet
  ner                        : one tag per word
"""

import json

from sklearn.model_selection import train_test_split

import pandas as pd
import torch

from .config import DATA_PATH, MAX_LEN, MAX_SUBWORDS, SEED, TASKS
from .ner_bio import add_bio_tags


def load_and_split():
    """Read the CSV, add BIO tags, and split 70/15/15 (always the same split)."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df = df[df["lang"] == "en"]  # project scope: English tweets only
    df = df.dropna(subset=["tweet", *TASKS]).reset_index(drop=True)
    df = add_bio_tags(df)

    # First cut 30% off, then split that 30% in half -> 70/15/15.
    train_df, rest = train_test_split(df, test_size=0.30, stratify=df["sentiment"], random_state=SEED)
    val_df, test_df = train_test_split(rest, test_size=0.50, stratify=rest["sentiment"], random_state=SEED)
    print(f"Split: train {len(train_df)}, val {len(val_df)}, test {len(test_df)}")
    return train_df, val_df, test_df


def build_labels(train_df) -> dict[str, list[str]]:
    """For each task, the sorted list of class names (position = class number)."""
    labels = {task: sorted(set(train_df[task].astype(str))) for task in TASKS}
    labels["ner"] = sorted({tag for tags in train_df["bio_tags"] for tag in tags.split()})
    return labels


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def encode_batch(word_lists, tag_lists, tokenizer, ner_ids):
    """Tokenize a batch of word-lists and map each word to its first subword."""
    enc = tokenizer(
        word_lists,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_SUBWORDS,
        padding=True,
        return_tensors="pt",
    )

    # For each tweet, the subword position where each surviving word starts.
    first_subword = []
    for i in range(len(word_lists)):
        start_of = {}
        for position, word in enumerate(enc.word_ids(i)):
            if word is not None and word not in start_of:
                start_of[word] = position
        first_subword.append([start_of[w] for w in sorted(start_of)])

    longest = max(len(f) for f in first_subword)
    word_index = torch.zeros(len(word_lists), longest, dtype=torch.long)
    word_mask = torch.zeros(len(word_lists), longest, dtype=torch.bool)
    ner = torch.zeros(len(word_lists), longest, dtype=torch.long)

    for i, positions in enumerate(first_subword):
        n = len(positions)
        word_index[i, :n] = torch.tensor(positions)
        word_mask[i, :n] = True
        # tags line up with words, so truncate them the same way
        ner[i, :n] = torch.tensor([ner_ids[t] for t in tag_lists[i][:n]])

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "word_index": word_index,
        "word_mask": word_mask,
        "ner": ner,
    }


def make_batches(df, tokenizer, labels, batch_size, shuffle=False):
    """Yield one ready-to-run batch of tensors at a time."""
    df = df.sample(frac=1, random_state=SEED) if shuffle else df
    ner_ids = {tag: i for i, tag in enumerate(labels["ner"])}
    class_ids = {task: {name: i for i, name in enumerate(labels[task])} for task in TASKS}

    for start in range(0, len(df), batch_size):
        rows = df.iloc[start : start + batch_size]
        word_lists = [str(text).split()[:MAX_LEN] for text in rows["tweet"]]
        tag_lists = [str(tags).split()[:MAX_LEN] for tags in rows["bio_tags"]]

        batch = encode_batch(word_lists, tag_lists, tokenizer, ner_ids)
        for task in TASKS:
            batch[task] = torch.tensor([class_ids[task][str(v)] for v in rows[task]])
        yield batch


__all__ = [
    "build_labels", "encode_batch", "load_and_split", "load_json", "make_batches", "save_json",
]
