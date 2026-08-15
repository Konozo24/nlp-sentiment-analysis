"""Try the trained TRABSA model on any tweet you type.

Run:  python -m src.models.trabsa.predict                       (interactive)
      python -m src.models.trabsa.predict "Messi is on fire!"   (one-off)
"""

import sys

import sklearn  # noqa: F401 — must import before torch (Windows heap-corruption crash otherwise)
import torch
from transformers import AutoTokenizer

from src.data_cleaning.preprocess_trabsa import clean_for_trabsa

from .config import ENCODER_NAME, MAX_LEN, MODEL_DIR, TASKS
from .data import encode_batch, load_json
from .model import TRABSA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """Rebuild the trained model from the files train.py saved. Also used by evaluate.py."""
    labels = load_json(MODEL_DIR / "labels.json")
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    model = TRABSA.from_labels(labels)
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt", map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model, tokenizer, labels


def predict(tweet: str, model, tokenizer, labels):
    words = clean_for_trabsa(tweet).split()[:MAX_LEN]
    if not words:
        print("  (nothing left after cleaning)")
        return

    ner_ids = {tag: i for i, tag in enumerate(labels["ner"])}
    batch = encode_batch([words], [["O"] * len(words)], tokenizer, ner_ids)
    batch = {name: tensor.to(DEVICE) for name, tensor in batch.items()}
    with torch.no_grad():
        out = model(batch["input_ids"], batch["attention_mask"], batch["word_index"], batch["word_mask"])

    for task in TASKS:
        probs = torch.softmax(out[task][0], dim=-1)
        best = probs.argmax().item()
        print(f"  {task:9s}: {labels[task][best]}  ({probs[best]:.0%} sure)")

    tag_ids = model.crf.decode(out["ner"], mask=batch["word_mask"])[0]
    tagged = [f"{w}[{labels['ner'][t]}]" if labels["ner"][t] != "O" else w for w, t in zip(words, tag_ids)]
    print(f"  entities : {' '.join(tagged)}")


if __name__ == "__main__":
    model, tokenizer, labels = load_model()
    if len(sys.argv) > 1:
        predict(" ".join(sys.argv[1:]), model, tokenizer, labels)
    else:
        print("Type a tweet and press Enter (empty line to quit):")
        while True:
            tweet = input("> ").strip()
            if not tweet:
                break
            predict(tweet, model, tokenizer, labels)
