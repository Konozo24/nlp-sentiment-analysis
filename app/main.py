"""World Cup tweet analysis demo — SVM / BiLSTM / RoBERTa-CNN + a compare mode.

Run:  streamlit run app/main.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # before any transformers import, process-wide

import sys
from pathlib import Path

import sklearn  # noqa: F401 — must import before torch (Windows heap-corruption crash otherwise)
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="World Cup Tweet Analysis", page_icon="⚽", layout="wide")

from app.registry import MODEL_SPECS  # noqa: E402 — light: adapters defer heavy imports

COMPARE = "⚖️ Compare all three"

with st.sidebar:
    st.title("⚽ World Cup Tweets")
    choice = st.selectbox(
        "Model",
        [f"{spec.icon} {spec.label}" for spec in MODEL_SPECS] + [COMPARE],
    )
    st.caption("Sentiment · Emotion · Topic · NER — 2014-2026 FIFA World Cup tweets.")

if choice == COMPARE:
    from app.view_compare import render

    render(MODEL_SPECS)
else:
    from app.view_single import render

    render(next(spec for spec in MODEL_SPECS if f"{spec.icon} {spec.label}" == choice))
