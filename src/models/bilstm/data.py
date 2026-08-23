"""Load the data, split it, and serve it as padded batches of word ids.

Contrast with the transformer members' data.py: theirs must map subwords back
to words (RoBERTa splits 'messi' into 'mess'+'i', but the BIO tags from
ner_bio.py are one-per-word, so they keep the first subword of each word and
carry a word_index/word_mask pair through every batch).

Here tokenisation IS word-level - one embedding lookup per whitespace token -
so that alignment machinery disappears entirely. A batch is just:

  input_ids  : (batch, words)  one vocabulary id per word (0 = <pad>, 1 = <unk>)
  mask       : (batch, words)  True where the slot is a real word
  sentiment/emotion/topic : (batch,)        one label per tweet
  ner        : (batch, words)  one BIO tag id per word

Tweets vary in length, so padding happens in collate_fn - the standard PyTorch
place for it - rather than being baked into a hand-rolled batch generator.
"""

import json  # noqa: I001 - import order below is intentional (see next line)

# sklearn must import before torch, or Windows raises a heap-corruption crash
import sklearn  # noqa: F401

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import DATA_PATH, EMBEDDING_PATH, MAX_LEN, TASKS, VOCAB_PATH
from .ner_bio import add_bio_tags

PAD_ID, UNK_ID = 0, 1


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the CSV, add BIO tags, and return the shared train/val/test split.

    Split labels come from data/processed/splits.csv (70/15/15).
    """
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df = df[df["lang"] == "en"]  # project scope: English tweets only
    df = df.dropna(subset=["tweet", *TASKS]).reset_index(drop=True)

    if "split" not in df.columns:
        raise ValueError(
            f"{DATA_PATH} has no 'split' column - re-run "
            "'python -m src.data_cleaning.base_cleaning' then "
            "'python -m src.data_cleaning.preprocess_bilstm'"
        )

    df = add_bio_tags(df)
    parts = [df[df["split"] == name].reset_index(drop=True) for name in ("train", "val", "test")]
    train_df, val_df, test_df = parts
    print(f"Split: train {len(train_df)}, val {len(val_df)}, test {len(test_df)}")
    return train_df, val_df, test_df


def load_vocab() -> dict[str, int]:
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"{VOCAB_PATH} not found - run 'python scripts/build_embeddings.py' first."
        )
    return json.loads(VOCAB_PATH.read_text(encoding="utf-8"))


def load_embeddings() -> torch.Tensor:
    if not EMBEDDING_PATH.exists():
        raise FileNotFoundError(
            f"{EMBEDDING_PATH} not found - run 'python scripts/build_embeddings.py' first."
        )
    return torch.from_numpy(np.load(EMBEDDING_PATH))


def build_labels(train_df: pd.DataFrame) -> dict[str, list[str]]:
    """For each task, the sorted list of class names (position = class number)."""
    labels = {task: sorted(set(train_df[task].astype(str))) for task in TASKS}
    labels["ner"] = sorted({tag for tags in train_df["bio_tags"] for tag in tags.split()})
    return labels


def save_json(obj, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Dataset / collation
# --------------------------------------------------------------------------- #


def encode_words(
    words: list[str],
    tags: list[str],
    vocab: dict[str, int],
    ner_ids: dict[str, int],
) -> dict[str, torch.Tensor]:
    """One tweet -> one unpadded example. Words and tags line up one-to-one."""
    n = min(len(words), MAX_LEN)
    return {
        "input_ids": torch.tensor([vocab.get(w, UNK_ID) for w in words[:n]], dtype=torch.long),
        "ner": torch.tensor([ner_ids[t] for t in tags[:n]], dtype=torch.long),
    }


class TweetDataset(Dataset):
    """Tweets as tensors, tokenised once up front rather than once per epoch.

    Labels are optional so the same class serves inference, where there are
    none - predict.py builds a single-item batch through the same collate_fn,
    which keeps training and inference on one code path.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vocab: dict[str, int],
        labels: dict[str, list[str]],
        with_targets: bool = True,
    ) -> None:
        self.with_targets = with_targets
        ner_ids = {tag: i for i, tag in enumerate(labels["ner"])}
        class_ids = {task: {name: i for i, name in enumerate(labels[task])} for task in TASKS}

        self.examples: list[dict[str, torch.Tensor]] = []
        for row in df.itertuples(index=False):
            words = str(row.tweet).split()
            tags = str(row.bio_tags).split()
            example = encode_words(words, tags, vocab, ner_ids)
            if with_targets:
                for task in TASKS:
                    example[task] = torch.tensor(class_ids[task][str(getattr(row, task))])
            self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad a list of variable-length examples into rectangular tensors.

    pad_sequence pads to the longest tweet IN THIS BATCH, not to MAX_LEN, so
    short batches stay small and the LSTM does less wasted work.
    """
    input_ids = nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch], batch_first=True, padding_value=PAD_ID
    )
    ner = nn.utils.rnn.pad_sequence(
        [item["ner"] for item in batch], batch_first=True, padding_value=0
    )
    collated = {"input_ids": input_ids, "mask": input_ids != PAD_ID, "ner": ner}

    for task in TASKS:
        if task in batch[0]:
            collated[task] = torch.stack([item[task] for item in batch])
    return collated


def make_loader(
    df: pd.DataFrame,
    vocab: dict[str, int],
    labels: dict[str, list[str]],
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Build a DataLoader over the tweets.

    shuffle=True draws a fresh permutation each epoch; reproducibility comes
    from seeding the global RNG once in train.py, not from fixing the order.

    num_workers defaults to 0 because the dataset is already tensors in memory,
    so worker processes would only add Windows spawn overhead.
    """
    return DataLoader(
        TweetDataset(df, vocab, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def unk_rate(df: pd.DataFrame, vocab: dict[str, int]) -> float:
    """Share of tokens with no vocabulary entry - reported in the README/paper.

    Worth measuring per variant: it is the single number that justifies using
    fastText over GloVe, since fastText composes a vector for anything.
    """
    total = misses = 0
    for text in df["tweet"].astype(str):
        for word in text.split()[:MAX_LEN]:
            total += 1
            misses += word not in vocab
    return misses / max(total, 1)


__all__ = [
    "PAD_ID",
    "UNK_ID",
    "TweetDataset",
    "build_labels",
    "collate_fn",
    "encode_words",
    "load_and_split",
    "load_embeddings",
    "load_json",
    "load_vocab",
    "make_loader",
    "save_json",
    "unk_rate",
]
