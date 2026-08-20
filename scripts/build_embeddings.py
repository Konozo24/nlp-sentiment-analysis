"""Build the static word-embedding matrix the BiLSTM reads.

Why two sources instead of one
------------------------------
No embedding released before 2026 can know 2026 slang, so we attack the
problem from both ends and concatenate the results per token:

  cols   0-299  pretrained fastText (cc.en.300, Common Crawl)
                General English semantics, and — because fastText composes a
                vector from character n-grams — a usable vector even for words
                it never saw. 'goooaaal' resolves through 'goo','ooa','aal'
                instead of collapsing to <unk>, which is what GloVe/Word2Vec
                would do to most of Twitter.

  cols 300-399  in-domain fastText, trained here on our own World Cup tweets
                The corpus holds ~10k tweets from 2026. Whatever current slang
                means, it means it *here* — no external corpus can supply that,
                because the usage postdates every published embedding.

Each half is L2-normalised before joining so neither dominates the input scale
of the BiLSTM.

Steps (each caches its output; re-running is cheap)
--------------------------------------------------
  vocab       token -> row index, from the cleaned corpus
  indomain    train gensim FastText on our tweets      -> indomain.npy + .model
  pretrained  pull vectors for our vocab out of cc.en.300.bin -> pretrained.npy
  assemble    normalise, concatenate, write embeddings.npy

The 'pretrained' step needs ~8GB of free RAM to hold cc.en.300.bin. It is a
separate step precisely so that cost is paid once, in its own process, and
never again — the cached pretrained.npy is only ~40MB.

Usage
-----
  python scripts/build_embeddings.py                      # all four steps
  python scripts/build_embeddings.py --only pretrained    # just the heavy step
"""

import argparse
import gzip
import json
import shutil
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings"

CC_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
CC_BIN = EMBEDDING_DIR / "cc.en.300.bin"

PRETRAINED_DIM = 300
INDOMAIN_DIM = 100

PAD, UNK = "<pad>", "<unk>"


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #


def load_corpus() -> list[list[str]]:
    """The TRAINING tweets only, cleaned and tokenised.

    It is tempting to train the embeddings on every tweet we have — embedding
    training needs no labels, so the validation and test text is just free
    extra material. Resist it. Building the vocabulary from text the model
    will later be scored on drives the reported out-of-vocabulary rate to
    literally 0%, which hides the very problem this project is about and would
    not survive a careful reader.

    Restricting to the training split costs some in-domain vector quality and
    buys two things worth more: an honest OOV number for the results section,
    and a live demonstration that fastText composes usable vectors for words
    the model genuinely never saw.
    """
    from src.models.bilstm.data import load_and_split

    train_df, _, _ = load_and_split()
    sentences = [tokens for text in train_df["tweet"].astype(str) if (tokens := text.split())]

    print(f"Corpus: {len(sentences):,} training tweets, {sum(len(s) for s in sentences):,} tokens")
    return sentences


def build_vocab(sentences: list[list[str]], min_count: int) -> dict[str, int]:
    """token -> row index, with <pad> at 0 and <unk> at 1.

    min_count defaults to 1. With a lookup-table embedding you would raise it,
    because rare words get randomly initialised vectors that are pure noise.
    Here every entry is *composed* by fastText from character n-grams, so a
    rare token still gets a meaningful vector — keeping it strictly reduces the
    <unk> rate at no cost beyond ~30MB of matrix.
    """
    counts = Counter(token for sentence in sentences for token in sentence)
    kept = sorted((t for t, c in counts.items() if c >= min_count), key=lambda t: (-counts[t], t))
    vocab = {PAD: 0, UNK: 1}
    for token in kept:
        vocab[token] = len(vocab)
    print(
        f"Vocab: {len(vocab):,} entries "
        f"(min_count={min_count}, {len(counts):,} distinct tokens seen)"
    )
    return vocab


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def step_vocab(min_count: int) -> dict[str, int]:
    sentences = load_corpus()
    vocab = build_vocab(sentences, min_count)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    write_json(EMBEDDING_DIR / "vocab.json", vocab)
    write_json(EMBEDDING_DIR / "corpus.json", sentences)
    print(f"Saved {EMBEDDING_DIR / 'vocab.json'}")
    return vocab


def step_indomain() -> np.ndarray:
    from gensim.models import FastText

    vocab = load_vocab()
    sentences = json.loads((EMBEDDING_DIR / "corpus.json").read_text(encoding="utf-8"))

    print(f"Training in-domain fastText ({INDOMAIN_DIM}d) on {len(sentences):,} tweets...")
    model = FastText(
        vector_size=INDOMAIN_DIM,
        window=5,
        min_count=2,  # a word seen once teaches the *word* nothing, but its
        # character n-grams still train from every other word
        sg=1,  # skip-gram: better than CBOW on small corpora
        min_n=3,
        max_n=6,
        # gensim's default is 2,000,000 n-gram buckets, sized for corpora
        # thousands of times larger than ours. At 1.1M tokens that is mostly
        # empty rows, and it produces an 800MB file the demo app has to load.
        # 200k buckets keeps collisions negligible at this vocabulary size and
        # brings the model down to ~80MB.
        bucket=200_000,
        workers=4,
        seed=42,
    )
    model.build_vocab(corpus_iterable=sentences)
    model.train(corpus_iterable=sentences, total_examples=len(sentences), epochs=20)
    model.save(str(EMBEDDING_DIR / "indomain_ft.model"))

    matrix = np.zeros((len(vocab), INDOMAIN_DIM), dtype=np.float32)
    for token, index in vocab.items():
        if token in (PAD, UNK):
            continue
        matrix[index] = model.wv[token]  # composed from n-grams if not in wv.key_to_index

    np.save(EMBEDDING_DIR / "indomain.npy", matrix)
    print(f"Saved {EMBEDDING_DIR / 'indomain.npy'} {matrix.shape}")
    return matrix


def download_pretrained() -> None:
    """Fetch cc.en.300.bin.gz (~4.2GB) and decompress it (~7GB) — once."""
    if CC_BIN.exists():
        return
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    archive = EMBEDDING_DIR / "cc.en.300.bin.gz"

    if not archive.exists():
        print(f"Downloading {CC_URL}\n  (~4.2GB, one time only)")

        def progress(block_count, block_size, total_size):
            done = block_count * block_size
            if total_size > 0:
                pct = min(100.0, done * 100.0 / total_size)
                print(f"\r  {done / 1e9:.2f}/{total_size / 1e9:.2f} GB ({pct:.1f}%)", end="")

        tmp = archive.with_suffix(".gz.part")
        urllib.request.urlretrieve(CC_URL, tmp, reporthook=progress)
        tmp.rename(archive)
        print()

    print("Decompressing (~7GB on disk)...")
    with gzip.open(archive, "rb") as src, open(CC_BIN, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1 << 24)
    archive.unlink()
    print(f"Ready: {CC_BIN}")


def step_pretrained() -> np.ndarray:
    from gensim.models.fasttext import load_facebook_vectors

    vocab = load_vocab()
    download_pretrained()

    print("Loading cc.en.300.bin (needs ~8GB RAM, takes a few minutes)...")
    kv = load_facebook_vectors(str(CC_BIN))

    matrix = np.zeros((len(vocab), PRETRAINED_DIM), dtype=np.float32)
    exact = 0
    for token, index in vocab.items():
        if token in (PAD, UNK):
            continue
        # kv[token] works whether or not the token is in kv.key_to_index —
        # a miss is composed from character n-grams. That is the whole reason
        # this project uses fastText rather than GloVe or Word2Vec.
        matrix[index] = kv[token]
        exact += token in kv.key_to_index

    covered = len(vocab) - 2
    print(f"Pretrained: {exact:,}/{covered:,} tokens known exactly ({exact / covered:.1%}); "
          f"the remaining {covered - exact:,} were composed from subwords instead of dropped")

    np.save(EMBEDDING_DIR / "pretrained.npy", matrix)
    print(f"Saved {EMBEDDING_DIR / 'pretrained.npy'} {matrix.shape}")
    print(f"\n{CC_BIN} ({CC_BIN.stat().st_size / 1e9:.1f}GB) is no longer needed — safe to delete.")
    return matrix


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-8)


def step_assemble() -> None:
    """Normalise both halves, join them, and write the matrix the model loads.

    Each half is L2-normalised BEFORE concatenating. Without that the two
    sources arrive on different scales and whichever has larger norms would
    dominate the projection layer's input regardless of which is more useful.
    """
    matrices = []
    for name in ("pretrained", "indomain"):
        path = EMBEDDING_DIR / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run: python scripts/build_embeddings.py --only {name}"
            )
        matrices.append(l2_normalise(np.load(path)))

    matrix = np.concatenate(matrices, axis=1)

    matrix[0] = 0.0  # <pad> must be exactly zero
    matrix[1] = matrix[2:].mean(axis=0)  # <unk> = centroid of the real vocabulary

    out_path = EMBEDDING_DIR / "embeddings.npy"
    np.save(out_path, matrix)
    print(f"Saved {out_path} {matrix.shape}  (300d pretrained + 100d in-domain)")


def load_vocab() -> dict[str, int]:
    path = EMBEDDING_DIR / "vocab.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run: python scripts/build_embeddings.py --only vocab"
        )
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", choices=["vocab", "indomain", "pretrained", "assemble"],
                        help="run a single step instead of the whole chain")
    parser.add_argument(
        "--min-count", type=int, default=1, help="minimum token frequency for the vocab"
    )
    args = parser.parse_args()

    if args.only == "vocab":
        step_vocab(args.min_count)
    elif args.only == "indomain":
        step_indomain()
    elif args.only == "pretrained":
        step_pretrained()
    elif args.only == "assemble":
        step_assemble()
    else:
        step_vocab(args.min_count)
        step_indomain()
        step_pretrained()
        step_assemble()


if __name__ == "__main__":
    main()
