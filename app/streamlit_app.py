"""World Cup tweet analysis demo for SVM, BiLSTM, RoBERTa-base, plus a compare page.

Run:  streamlit run app/streamlit_app.py
"""

import sys
from functools import partial
from pathlib import Path

import sklearn  # noqa: F401 - must import before torch (windows heap corruption otherwise)
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.checkpoint_parts import restore_all

restore_all()

st.set_page_config(page_title="World Cup tweet analysis", page_icon="⚽", layout="wide")

from app import page_compare, page_dataset, page_model
from app.registry import MODEL_SPECS

model_pages = [
    st.Page(
        partial(page_model.render, spec),
        title=spec.label,
        icon=spec.icon,
        url_path=spec.key,
        default=index == 0,
    )
    for index, spec in enumerate(MODEL_SPECS)
]
dataset_page = st.Page(
    partial(page_dataset.render, MODEL_SPECS),
    title="Dataset",
    icon=":material/table_chart:",
    url_path="dataset",
)
compare_page = st.Page(
    partial(page_compare.render, MODEL_SPECS),
    title="Compare all three",
    icon=":material/compare_arrows:",
    url_path="compare",
)

page = st.navigation({"Models": model_pages, "Analysis": [dataset_page, compare_page]})

with st.sidebar:
    st.caption("Sentiment · emotion · topic · NER  \n(2014-2026 FIFA World Cup tweets)")

page.run()
