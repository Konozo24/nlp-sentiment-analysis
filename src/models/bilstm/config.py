"""Settings for the BiLSTM. Change values here, not in the code.

BiLSTM = static word embeddings (fastText, pretrained + in-domain)
-> BiLSTM -> attention pooling -> one head per task (plus a CRF for NER).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "bilstm_input.csv"
EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings"

MODEL_DIR = PROJECT_ROOT / "data" / "models" / "bilstm"

# the embedding table, built by scripts/build_embeddings.py: one 400d vector
# per word, formed by concatenating
#   cols   0-299  pretrained fastText (cc.en.300, Common Crawl)
#   cols 300-399  in-domain fastText, trained on our own World Cup tweets
EMBEDDING_PATH = EMBEDDING_DIR / "embeddings.npy"
VOCAB_PATH = EMBEDDING_DIR / "vocab.json"
INDOMAIN_MODEL_PATH = EMBEDDING_DIR / "indomain_ft.model"

SEED = 42  # fixed so train/evaluate rebuild the same split
MAX_LEN = 64  # tweets longer than 64 words get cut off

# embeddings stay frozen: ~19k training rows cannot fine-tune a 33k x 400
# lookup table without simply memorising it. The projection below gives the
# model room to reshape the space without touching the vectors themselves.
FREEZE_EMBEDDINGS = True
PROJ_DIM = 256  # embeddings are projected down to this before the LSTM
HIDDEN_SIZE = 256  # per LSTM direction
DROPOUT = 0.3

# no pretrained weights to preserve here, so one ordinary learning rate for
# everything and no warmup — unlike the transformer models, which need a tiny
# encoder step and a warmup schedule to avoid washing their weights away.
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
