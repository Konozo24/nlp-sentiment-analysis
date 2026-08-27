"""Settings for RobertaBase """

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "roberta_input.csv"  # same cleaned data as roberta
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "robertabase"

SEED = 42
MAX_LEN = 64

ENCODER_NAME = "cardiffnlp/twitter-roberta-base"

MAX_SUBWORDS = 128
DROPOUT = 0.3

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
    "FREEZE_ENCODER", "HEAD_LR", "MAX_EPOCHS", "MAX_LEN", "MAX_SUBWORDS",
    "MODEL_DIR", "PATIENCE", "SEED", "TASKS", "WARMUP_RATIO",
]