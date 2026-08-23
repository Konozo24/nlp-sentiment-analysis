"""Try the trained BiLSTM on any tweet you type.

Run:  python -m src.models.bilstm.predict                       (interactive)
      python -m src.models.bilstm.predict "Messi is on fire!"   (one-off)
"""

import sys

import numpy as np
import sklearn  # noqa: F401 - must import before torch (Windows heap-corruption crash otherwise)
import torch

from src.data_cleaning.preprocess_bilstm import clean_for_bilstm

from .config import INDOMAIN_MODEL_PATH, MAX_LEN, MODEL_DIR, TASKS
from .data import (
    UNK_ID,
    collate_fn,
    encode_words,
    load_embeddings,
    load_json,
    load_vocab,
)
from .model import BiLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# which columns of the concatenated vector the in-domain model owns. For the
# 'concat' variant the pretrained half occupies the first 300 columns.
INDOMAIN_DIM = 100


def load_indomain_model():
    """The small fastText model kept for run-time OOV composition.

    Returns None if it is missing - prediction still works, unknown words just
    fall back to <unk> as they would in any ordinary embedding model.
    """
    if not INDOMAIN_MODEL_PATH.exists():
        return None
    from gensim.models import FastText

    return FastText.load(str(INDOMAIN_MODEL_PATH))


def load_model():
    """Rebuild the trained model from the checkpoint train.py saved.

    Also used by evaluate.py and the Streamlit app. weights_only=False is
    required because the checkpoint carries the label lists alongside the
    tensors; the file is one we produced ourselves, not untrusted input.
    """
    labels = load_json(MODEL_DIR / "labels.json")
    vocab = load_vocab()
    embeddings = load_embeddings()

    model = BiLSTM.from_labels(labels, embeddings)
    checkpoint = torch.load(MODEL_DIR / "best_model.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()  # dropout off for inference
    return model, vocab, labels


def embed_words(words, vocab, embeddings, indomain_model):
    """One vector per word, composing anything the vocabulary is missing.

    Returns (tensor of shape (1, len(words), dim), list of the OOV words).
    """
    dim = embeddings.size(1)
    vectors = torch.empty(len(words), dim)
    oov = []

    for i, word in enumerate(words):
        if word in vocab:
            vectors[i] = embeddings[vocab[word]]
            continue

        oov.append(word)
        vectors[i] = embeddings[UNK_ID]
        if indomain_model is None:
            continue

        # compose the in-domain half from character n-grams and L2-normalise it
        # the same way build_embeddings.py normalised the stored rows, so the
        # composed vector lands on the same scale as everything around it.
        composed = np.asarray(indomain_model.wv[word], dtype=np.float32)
        composed /= max(float(np.linalg.norm(composed)), 1e-8)
        vectors[i, dim - INDOMAIN_DIM :] = torch.from_numpy(composed)

    return vectors.unsqueeze(0), oov


@torch.no_grad()
def predict_structured(tweet: str, model, vocab, labels, indomain_model=None):
    """Run the model and return the results as data (used by predict() and the app)."""
    words = clean_for_bilstm(tweet).split()[:MAX_LEN]
    if not words:
        return None

    # single-item batch through the same collate_fn training uses, so there is
    # one padding/masking code path rather than two that can drift apart.
    ner_ids = {tag: i for i, tag in enumerate(labels["ner"])}
    batch = collate_fn([encode_words(words, ["O"] * len(words), vocab, ner_ids)])
    input_ids = batch["input_ids"].to(DEVICE)
    mask = batch["mask"].to(DEVICE)

    override, oov = embed_words(words, vocab, model.embedding.weight.detach().cpu(), indomain_model)
    out = model(input_ids, mask, embeddings_override=override.to(DEVICE))

    result = {"words": words, "oov": oov, "tasks": {}}
    for task in TASKS:
        probs = torch.softmax(out[task][0], dim=-1)
        result["tasks"][task] = {
            "label": labels[task][probs.argmax().item()],
            "confidence": probs.max().item(),
            "distribution": {labels[task][i]: probs[i].item() for i in range(len(labels[task]))},
        }

    tag_ids = model.crf.decode(out["ner"], mask=mask)[0]
    result["ner"] = [(word, labels["ner"][tag]) for word, tag in zip(words, tag_ids, strict=True)]
    return result


def predict(tweet: str, model, vocab, labels, indomain_model=None):
    result = predict_structured(tweet, model, vocab, labels, indomain_model)
    if result is None:
        print("  (nothing left after cleaning)")
        return

    for task in TASKS:
        info = result["tasks"][task]
        print(f"  {task:9s}: {info['label']}  ({info['confidence']:.0%} sure)")

    tagged = [f"{w}[{tag}]" if tag != "O" else w for w, tag in result["ner"]]
    print(f"  entities : {' '.join(tagged)}")

    if result["oov"]:
        composed = "composed from subwords" if indomain_model else "fell back to <unk>"
        print(f"  new words: {', '.join(result['oov'])}  ({composed})")


if __name__ == "__main__":
    model, vocab, labels = load_model()
    indomain_model = load_indomain_model()
    if len(sys.argv) > 1:
        predict(" ".join(sys.argv[1:]), model, vocab, labels, indomain_model)
    else:
        print("Type a tweet and press Enter (empty line to quit):")
        while True:
            tweet = input("> ").strip()
            if not tweet:
                break
            predict(tweet, model, vocab, labels, indomain_model)
