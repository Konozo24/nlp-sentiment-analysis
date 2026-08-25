"""Try the trained RobertaBase model on any tweet you type.

Run:  python -m src.models.robertabase.predict
      python -m src.models.robertabase.predict "Messi is on fire!"
"""

import sys

import sklearn  # noqa: F401 — must import before torch
import torch
from transformers import AutoTokenizer

from src.data_cleaning.preprocess_roberta import clean_for_roberta

from .config import ENCODER_NAME, MAX_LEN, MODEL_DIR, TASKS
from .data import encode_batch, load_json
from .model import RobertaBase

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model() -> tuple[RobertaBase, AutoTokenizer, dict[str, list[str]]]:
    labels = load_json(MODEL_DIR / "labels.json")
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    model = RobertaBase.from_labels(labels)
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt", map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model, tokenizer, labels


@torch.no_grad()
def predict_structured(
    tweet: str, model: RobertaBase, tokenizer, labels: dict[str, list[str]]
) -> dict | None:
    words = clean_for_roberta(tweet).split()[:MAX_LEN]
    if not words:
        return None

    ner_ids = {tag: i for i, tag in enumerate(labels["ner"])}
    batch = encode_batch([words], [["O"] * len(words)], tokenizer, ner_ids)
    batch = {name: tensor.to(DEVICE) for name, tensor in batch.items()}
    out = model(batch["input_ids"], batch["attention_mask"], batch["word_index"], batch["word_mask"])

    result = {"words": words, "oov": [], "tasks": {}}
    for task in TASKS:
        probs = torch.softmax(out[task][0], dim=-1)
        result["tasks"][task] = {
            "label": labels[task][probs.argmax().item()],
            "confidence": probs.max().item(),
            "distribution": {labels[task][i]: probs[i].item() for i in range(len(labels[task]))},
        }

    tag_ids = model.crf.decode(out["ner"], mask=batch["word_mask"])[0]
    result["ner"] = [(word, labels["ner"][tag]) for word, tag in zip(words, tag_ids, strict=True)]
    return result


def predict(tweet: str, model: RobertaBase, tokenizer, labels: dict[str, list[str]]) -> None:
    result = predict_structured(tweet, model, tokenizer, labels)
    if result is None:
        print("  (nothing left after cleaning)")
        return

    for task in TASKS:
        info = result["tasks"][task]
        print(f"  {task:9s}: {info['label']}  ({info['confidence']:.0%} sure)")

    tagged = [f"{w}[{tag}]" if tag != "O" else w for w, tag in result["ner"]]
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