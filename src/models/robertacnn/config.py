"""Settings for RobertaCNN. Change values here, not in the code.

RobertaCNN = pretrained Twitter-RoBERTa encoder -> multi-kernel CNN pooling
-> one head per task (plus CNN+CRF for NER).

Hyperparameters below deliberately match src/models/trabsa/config.py so
RobertaCNN vs TRABSA is a controlled comparison — only the pooling
architecture (CNN vs BiLSTM+attention) differs.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "robertacnn_input.csv"
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "robertacnn"

SEED = 42
MAX_LEN = 64

ENCODER_NAME = "cardiffnlp/twitter-roberta-base"

MAX_SUBWORDS = 128
DROPOUT = 0.3

# CNN pooling head (sentiment/topic/emotion) — multi-kernel, max-pooled
NUM_FILTERS = 128
KERNEL_SIZES = (2, 3, 4)

# CNN token-enrichment head (NER) — same-padding, feeds the CRF
NER_FILTERS = 768
NER_KERNEL_SIZES = (3, 5)

FREEZE_ENCODER = False

BATCH_SIZE = 16
ENCODER_LR = 2e-5
HEAD_LR = 1e-3
WARMUP_RATIO = 0.1
MAX_EPOCHS = 6
PATIENCE = 2

TASKS = ["sentiment", "emotion", "topic"]

__all__ = [
    "BATCH_SIZE", "DATA_PATH", "DROPOUT", "ENCODER_LR", "ENCODER_NAME",
    "FREEZE_ENCODER", "HEAD_LR", "KERNEL_SIZES", "MAX_EPOCHS", "MAX_LEN",
    "MAX_SUBWORDS", "MODEL_DIR", "NER_FILTERS", "NER_KERNEL_SIZES",
    "NUM_FILTERS", "PATIENCE", "SEED", "TASKS", "WARMUP_RATIO",
]