"""One ModelSpec per model - the single place a new model page is added.

Kept import-light on purpose: nothing heavy (torch, transformers) is
imported at module scope. Every loader and predictor does its heavy imports
inside its own body, so opening the SVM page never pulls in torch, and
@st.cache_resource means the app pays each load cost only once.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABOUT_DIR = Path(__file__).parent / "about"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    icon: str
    tagline: str
    data_path: Path
    model_dir: Path
    about_md: str
    load: Callable[[], Any]
    predict: Callable[[str, Any], dict | None]


def _about(name: str) -> str:
    return (ABOUT_DIR / f"{name}.md").read_text(encoding="utf-8")


def _paths(key: str) -> dict[str, Path]:
    return {
        "data_path": PROJECT_ROOT / "data" / "processed" / f"{key}_input.csv",
        "model_dir": PROJECT_ROOT / "data" / "models" / key,
    }


# --------------------------------------------------------------------------- #
# SVM (TF-IDF + LinearSVC)
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner="Loading SVM model (first run only)...")
def _load_svm():
    from src.models.svm.predict import load_model

    return load_model()


def _predict_svm(tweet: str, bundle) -> dict | None:
    from src.models.svm.config import TASKS
    from src.models.svm.predict import predict as svm_predict

    models, vectorizer, encoders, ner_vectorizer = bundle
    result = svm_predict(tweet, models, vectorizer, encoders, ner_vectorizer)

    words = result["cleaned"].split()
    if not words:
        return None  # link-only tweet: nothing left for TF-IDF to score

    tasks = {
        task: {
            "label": result[task],
            "confidence": result[f"{task}_confidence"],
            "distribution": None,  # LinearSVC gives a margin, not a full distribution
        }
        for task in TASKS
    }
    return {"words": words, "tasks": tasks, "ner": result["ner"]}


SVM_SPEC = ModelSpec(
    key="svm",
    label="SVM (TF-IDF)",
    icon=":material/functions:",
    tagline="TF-IDF features -> LinearSVC per task - the classical-ML baseline.",
    about_md=_about("svm"),
    load=_load_svm,
    predict=_predict_svm,
    **_paths("svm"),
)


# --------------------------------------------------------------------------- #
# BiLSTM (fastText embeddings + BiLSTM + attention + CRF)
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner="Loading BiLSTM model (first run only)...")
def _load_bilstm():
    import sklearn  # noqa: F401 - must import before torch (windows heap corruption otherwise)

    from src.models.bilstm.predict import load_indomain_model, load_model

    model, vocab, labels = load_model()
    return model, vocab, labels, load_indomain_model()


def _predict_bilstm(tweet: str, bundle) -> dict | None:
    from src.models.bilstm.predict import predict_structured

    model, vocab, labels, indomain_model = bundle
    return predict_structured(tweet, model, vocab, labels, indomain_model)


def _bilstm_about() -> str:
    from src.models.bilstm.config import DROPOUT, HIDDEN_SIZE, PROJ_DIM

    return _about("bilstm").format(
        PROJ_DIM=PROJ_DIM, HIDDEN_SIZE=HIDDEN_SIZE, DROPOUT=DROPOUT
    )


BILSTM_SPEC = ModelSpec(
    key="bilstm",
    label="BiLSTM",
    icon=":material/linear_scale:",
    tagline="fastText word embeddings -> BiLSTM -> attention -> 4 task heads.",
    about_md=_bilstm_about(),
    load=_load_bilstm,
    predict=_predict_bilstm,
    **_paths("bilstm"),
)


# --------------------------------------------------------------------------- #
# RoBERTa-CNN (fine-tuned twitter-roberta-base + CNN pooling + CRF)
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner="Loading RoBERTa-CNN (first run only - ~500MB checkpoint)...")
def _load_robertacnn():
    import sklearn  # noqa: F401 - must import before torch

    from src.models.robertacnn.predict import load_model

    return load_model()


def _predict_robertacnn(tweet: str, bundle) -> dict | None:
    from src.models.robertacnn.predict import predict_structured

    model, tokenizer, labels = bundle
    return predict_structured(tweet, model, tokenizer, labels)


ROBERTACNN_SPEC = ModelSpec(
    key="robertacnn",
    label="RoBERTa-CNN",
    icon=":material/hub:",
    tagline="cardiffnlp/twitter-roberta-base (fine-tuned) -> CNN pooling -> 4 task heads + CRF.",
    about_md=_about("robertacnn"),
    load=_load_robertacnn,
    predict=_predict_robertacnn,
    **_paths("robertacnn"),
)


MODEL_SPECS: list[ModelSpec] = [SVM_SPEC, BILSTM_SPEC, ROBERTACNN_SPEC]
