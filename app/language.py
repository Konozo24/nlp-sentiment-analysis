"""English-only guard for the live demo.

Uses the fastText language-id model already in the repo
(data/models/lid.176.ftz) rather than adding a new dependency. Fails open: if
the model can't be loaded, every input is treated as English so a missing
file never blocks analysis - it only turns off one warning.
"""

from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LID_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "lid.176.ftz"

# below this confidence the model is guessing, not detecting - stay quiet
CONFIDENCE_THRESHOLD = 0.5


@st.cache_resource(show_spinner=False)
def _load_lid_model():
    if not LID_MODEL_PATH.exists():
        return None
    import fasttext

    return fasttext.load_model(str(LID_MODEL_PATH))


def warn_if_not_english(text: str) -> None:
    """Show a warning if the text doesn't look like English.

    The training corpus is English-only (base_cleaning.py filters
    lang == 'en'), so a non-English tweet gets an unreliable prediction
    rather than an error - this makes that visible instead of silent.
    """
    model = _load_lid_model()
    if model is None:
        return

    cleaned = " ".join(text.split())  # fastText's predict() rejects embedded newlines
    if not cleaned:
        return

    (label,), (confidence,) = model.predict(cleaned, k=1)
    language = label.removeprefix("__label__")
    if language != "en" and confidence >= CONFIDENCE_THRESHOLD:
        st.warning(
            f"This looks like **{language}**, not English. The models were trained on "
            "English tweets only, so the result below may not be reliable."
        )
