"""Settings for TRABSA. Change values here, not in the code.

TRABSA = pretrained Twitter-RoBERTa encoder -> BiLSTM -> attention pooling
-> one head per task (plus a CRF for NER).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "trabsa_input.csv"
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "trabsa"

SEED = 42  # fixed so train/evaluate rebuild the same split
MAX_LEN = 64  # tweets longer than 64 words get cut off

# TimeLMs checkpoint pretrained on 154M tweets up to end of 2021 -- chosen
# to cover 2022 World Cup (Qatar) vocabulary/slang, the dataset's largest
# chunk. Swap for "vinai/bertweet-base" (850M tweets, usually stronger) or
# "roberta-base" (generic) by editing this line.
ENCODER_NAME = "cardiffnlp/twitter-roberta-base-2022-154m"

MAX_SUBWORDS = 128  # a 64-word tweet becomes at most ~128 subword tokens
HIDDEN_SIZE = 256  # per LSTM direction, same as the BiLSTM baseline
DROPOUT = 0.3

# Set True to train only the BiLSTM/heads on top of a frozen encoder.
# Much lighter on GPU memory; use it if fine-tuning runs out of memory.
FREEZE_ENCODER = False

BATCH_SIZE = 16  # 16 fits ~5GB of GPU memory while fine-tuning
ENCODER_LR = 2e-5  # pretrained weights need a much smaller step than new layers
HEAD_LR = 1e-3
WARMUP_RATIO = 0.1  # standard transformer recipe: warm up, then decay linearly
MAX_EPOCHS = 6  # transformers converge in far fewer epochs than the BiLSTM
PATIENCE = 2

TASKS = ["sentiment", "emotion", "topic"]  # sentence-level tasks (NER handled separately)

__all__ = [
    "BATCH_SIZE",
    "DATA_PATH",
    "DROPOUT",
    "ENCODER_LR",
    "ENCODER_NAME",
    "FREEZE_ENCODER",
    "HEAD_LR",
    "HIDDEN_SIZE",
    "MAX_EPOCHS",
    "MAX_LEN",
    "MAX_SUBWORDS",
    "MODEL_DIR",
    "PATIENCE",
    "SEED",
    "TASKS",
    "WARMUP_RATIO",
]
