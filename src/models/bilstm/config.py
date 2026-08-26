"""Settings for the BiLSTM. Change values here, not in the code.

BiLSTM = static word embeddings (fastText, pretrained + in-domain)
-> BiLSTM -> attention pooling -> one head per task (plus a CRF for NER).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "bilstm_input.csv"
EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings"

MODEL_DIR = PROJECT_ROOT / "data" / "models" / "bilstm"

# 400d word vectors built by scripts/build_embeddings.py
EMBEDDING_PATH = EMBEDDING_DIR / "embeddings.npy"
VOCAB_PATH = EMBEDDING_DIR / "vocab.json"
INDOMAIN_MODEL_PATH = EMBEDDING_DIR / "indomain_ft.model"

SEED = 42  # fixed so train/evaluate rebuild the same split
MAX_LEN = 64  # tweets longer than 64 words get cut off

FREEZE_EMBEDDINGS = True 
PROJ_DIM = 256  # embeddings are projected down to this before the LSTM
HIDDEN_SIZE = 256  # per LSTM direction
DROPOUT = 0.3

# flat LR, no warmup schedule - why: see train.py's module docstring
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 30  # the model is small and trains fast; let early stopping decide
PATIENCE = 5

TASKS = ["sentiment", "emotion", "topic"]  # sentence-level (NER handled separately)

__all__ = [
    "BATCH_SIZE",
    "DATA_PATH",
    "DROPOUT",
    "EMBEDDING_DIR",
    "EMBEDDING_PATH",
    "FREEZE_EMBEDDINGS",
    "HIDDEN_SIZE",
    "INDOMAIN_MODEL_PATH",
    "LR",
    "MAX_EPOCHS",
    "MAX_LEN",
    "MODEL_DIR",
    "PATIENCE",
    "PROJ_DIM",
    "SEED",
    "TASKS",
    "VOCAB_PATH",
    "WEIGHT_DECAY",
]
