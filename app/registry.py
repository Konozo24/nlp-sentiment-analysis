"""One ModelSpec per model — the single place a new model page gets added.

Kept import-light on purpose: nothing heavy (torch, transformers, spacy) is
imported at module scope. Every loader/predictor does its heavy imports
inside its own function body, so picking SVM in the sidebar never pulls in
torch, and Streamlit's rerun-per-interaction only pays that cost once thanks
to @st.cache_resource.
"""

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from app._common import parse_inline_bio

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    tagline: str
    data_path: Path
    model_dir: Path
    about_md: str
    load: Callable[[], Any]
    predict: Callable[[str, Any], dict | None]


# --------------------------------------------------------------------------- #
# SVM (TF-IDF + LinearSVC)
# --------------------------------------------------------------------------- #

_spacy_patched = False


def _patch_spacy_cache() -> None:
    """format_bio_entities() (src/models/svm/ner_bio.py) calls spacy.load()
    fresh on every single prediction — a multi-second disk load each time.
    This wraps spacy.load in an LRU cache, once, at the app layer. Changes no
    model output, only avoids re-reading the same weights every keystroke."""
    global _spacy_patched
    if _spacy_patched:
        return
    import spacy

    spacy.load = functools.lru_cache(maxsize=4)(spacy.load)
    _spacy_patched = True


@st.cache_resource(show_spinner="Loading SVM model (first run only)...")
def _load_svm():
    from src.models.svm.predict import load_model

    return load_model()


def _predict_svm(tweet: str, bundle):
    from src.models.svm.config import TASKS
    from src.models.svm.predict import predict as svm_predict

    _patch_spacy_cache()
    models, vectorizer, encoders, binarizer = bundle
    result = svm_predict(tweet, models, vectorizer, encoders, binarizer)
    tasks = {
        task: {
            "label": result[task],
            "confidence": result[f"{task}_confidence"],
            "distribution": None,  # LinearSVC exposes a max confidence, not a full distribution
        }
        for task in TASKS
    }
    return {"tasks": tasks, "ner": parse_inline_bio(result["ner_bio"])}


SVM_SPEC = ModelSpec(
    key="svm",
    label="SVM (TF-IDF)",
    tagline="TF-IDF features → LinearSVC per task — the classical-ML baseline.",
    data_path=PROJECT_ROOT / "data" / "processed" / "svm_input.csv",
    model_dir=PROJECT_ROOT / "data" / "models" / "svm",
    about_md="""
### Architecture

```
tweet
    |
TF-IDF (1-2 grams, sublinear tf)
    |
LinearSVC, one-vs-rest per task   (+ spaCy en_core_web_trf for NER)
```

### Why TF-IDF + a linear SVM

This is the **classical era** model in the group's three-way comparison —
no embeddings, no neural network anywhere in it. Each word (or word pair)
becomes one sparse feature weighted by how distinctive it is across the
corpus, and a separate LinearSVC is trained per task on that fixed
representation.

That representation has no notion of word order or context: "not good" and
"good" share the token "good" with equal weight regardless of the negation
next to it. It's a meaningful lower bound for the other two models to beat,
and — per the group's comparison table — it still holds its own, especially
where the training set is small relative to the vocabulary.

### Confidence, without native probabilities

`LinearSVC` produces a margin (`decision_function`), not a probability. The
confidence shown here comes from turning that margin into one via a sigmoid
(binary tasks) or a softmax over the one-vs-rest margins (multi-class) — an
approximation, not a calibrated probability, which is why this page has no
full class-probability breakdown the way the two neural models do.

### NER is a different model entirely

Named-entity recognition here doesn't come from the TF-IDF/SVM pipeline at
all — it runs spaCy's `en_core_web_trf` (a transformer NER pipeline) and maps
its entity types onto this project's PER/ORG/LOC/EVENT scheme. The first
prediction in a session is noticeably slower while that pipeline loads from
disk; every prediction after is fast.
""",
    load=_load_svm,
    predict=_predict_svm,
)


# --------------------------------------------------------------------------- #
# BiLSTM (fastText embeddings + BiLSTM + attention + CRF)
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner="Loading BiLSTM model (first run only)...")
def _load_bilstm():
    import sklearn  # noqa: F401 — must import before torch (Windows heap-corruption crash otherwise)

    from src.models.bilstm.predict import load_indomain_model, load_model

    model, vocab, labels = load_model()
    indomain_model = load_indomain_model()
    return model, vocab, labels, indomain_model


def _predict_bilstm(tweet: str, bundle):
    from src.models.bilstm.predict import predict_structured

    model, vocab, labels, indomain_model = bundle
    return predict_structured(tweet, model, vocab, labels, indomain_model)


def _bilstm_about_md() -> str:
    from src.models.bilstm.config import DROPOUT, HIDDEN_SIZE, PROJ_DIM

    return f"""
### Architecture

```
word ids
    |
frozen embedding table  (fastText: pretrained 300d + in-domain 100d)
    |
Linear -> ReLU -> Dropout      (trainable projection, {PROJ_DIM}d)
    |
BiLSTM ({HIDDEN_SIZE} units/direction, dropout {DROPOUT})  --> one vector per word
    |                                   |
attention-weighted sum                  --> NER head --> CRF (tag per word)
    |
sentiment / emotion / topic heads
```

### Why static embeddings, and why fastText

This is the **deep-learning era** model in the group's three-way comparison:
SVM + TF-IDF (classical), this BiLSTM (deep learning), RoBERTa-CNN
(transformer). There is deliberately no transformer anywhere in it — the
whole point is to show what the era before transformers could and could not do.

Every word here gets **one fixed vector regardless of context**. 'Fire' in
'he's on fire' and 'the manager got fired up' start from the same point; only
the BiLSTM's surrounding context can separate them. That limitation is exactly
what the transformer era removed, which makes this a meaningful baseline
rather than a weaker copy of Jason's model.

The embedding table itself combines two sources:

- **Pretrained fastText** (`cc.en.300`, Common Crawl) — general English meaning.
- **In-domain fastText**, trained here on our own ~58k World Cup tweets —
  including ~10k from 2026. Current slang means what it means *in this corpus*;
  no published embedding can contain it, because the usage postdates them all.

### Handling words nobody has seen

Both halves are fastText, which represents a word as the sum of its character
n-grams. A word missing from the vocabulary is therefore **composed** rather
than discarded: `bonkersss` is reached through `bonk`, `onke`, `kers`. GloVe
and Word2Vec cannot do this at all — an unseen token gets nothing.

Try it in the Live Demo tab: invent a word and watch it still get a vector.
"""


BILSTM_SPEC = ModelSpec(
    key="bilstm",
    label="BiLSTM",
    tagline="fastText word embeddings → BiLSTM → Attention → 4 task heads.",
    data_path=PROJECT_ROOT / "data" / "processed" / "bilstm_input.csv",
    model_dir=PROJECT_ROOT / "data" / "models" / "bilstm",
    about_md=_bilstm_about_md(),
    load=_load_bilstm,
    predict=_predict_bilstm,
)


# --------------------------------------------------------------------------- #
# RoBERTa-CNN (fine-tuned twitter-roberta-base + CNN pooling + CRF)
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner="Loading RoBERTa-CNN (first run only — ~500MB checkpoint)...")
def _load_robertacnn():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # already cached locally; never touch the network
    import sklearn  # noqa: F401 — must import before torch

    from src.models.robertacnn.predict import load_model

    return load_model()


def _predict_robertacnn(tweet: str, bundle):
    from src.models.robertacnn.predict import predict_structured

    model, tokenizer, labels = bundle
    return predict_structured(tweet, model, tokenizer, labels)


ROBERTACNN_SPEC = ModelSpec(
    key="robertacnn",
    label="RoBERTa-CNN",
    tagline="cardiffnlp/twitter-roberta-base (fine-tuned) → CNN pooling → 4 task heads + CRF.",
    data_path=PROJECT_ROOT / "data" / "processed" / "robertacnn_input.csv",
    model_dir=PROJECT_ROOT / "data" / "models" / "robertacnn",
    about_md="""
### Architecture

```
tweet
    |
cardiffnlp/twitter-roberta-base   (fine-tuned, subword tokens)
    |
gather first-subword-per-word --> one vector per word
    |                                       |
multi-kernel CNN + max-pool     same-padding CNN (no pooling)
    |                                       |
sentiment / emotion / topic          NER head --> CRF (tag per word)
```

### The transformer-era model

This is the **transformer era** model in the group's three-way comparison —
SVM + TF-IDF (classical), BiLSTM + fastText (deep learning, no transformer),
RoBERTa-CNN (this page). Unlike the BiLSTM's frozen, context-independent word
vectors, every word's representation here depends on the whole tweet around
it, because the encoder itself is fine-tuned end-to-end on this dataset.

### Why CNN pooling, not the encoder's own [CLS] token

Rather than reading off RoBERTa's pooled `[CLS]` representation, this model
takes the encoder's per-word hidden states and runs them through a small
multi-kernel 1D CNN + max-pool for sentence-level tasks (sentiment/emotion/
topic), and a separate same-padding CNN that keeps one vector per word (no
pooling) to feed the NER CRF. Hyperparameters intentionally match the
BiLSTM's, so any performance gap traces to the architecture — a fine-tuned
transformer encoder vs. frozen fastText — not to a different training recipe.

### The cost of fine-tuning on ~19k tweets

Full fine-tuning of a 110M-parameter encoder on a training set this size
overfits quickly — this model's own training run converges in the first few
epochs, well before the BiLSTM does, and gains little from training longer.
That's a real, expected trade-off of the transformer era, not a bug: more
representational power, less data to justify it than a larger corpus would.
""",
    load=_load_robertacnn,
    predict=_predict_robertacnn,
)


MODEL_SPECS: list[ModelSpec] = [SVM_SPEC, BILSTM_SPEC, ROBERTACNN_SPEC]
