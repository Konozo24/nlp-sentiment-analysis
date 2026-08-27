"""Train RobertaBase.

Performance differences trace to the pooling architecture rather than different training
regimes.

Run:  python -m src.models.robertabase.train [--epochs N]
"""

import argparse

import sklearn  # noqa: F401 — must import before torch (Windows heap-corruption crash)
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import (
    BATCH_SIZE, ENCODER_LR, ENCODER_NAME, HEAD_LR, MAX_EPOCHS,
    MODEL_DIR, PATIENCE, SEED, TASKS, WARMUP_RATIO,
)
from .data import build_labels, load_and_split, make_batches, save_json
from .model import RobertaBase

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_loss_functions(train_df, labels: dict[str, list[str]]) -> dict[str, nn.Module]:
    loss_fns = {}
    for task in TASKS:
        counts = train_df[task].astype(str).value_counts()
        weights = [min(len(train_df) / (len(labels[task]) * counts[name]), 10.0) for name in labels[task]]
        loss_fns[task] = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(DEVICE))
    return loss_fns


def batch_loss(model: nn.Module, batch: dict, loss_fns: dict[str, nn.Module]) -> torch.Tensor:
    batch = {name: tensor.to(DEVICE) for name, tensor in batch.items()}
    predictions = model(
        batch["input_ids"], batch["attention_mask"], batch["word_index"], batch["word_mask"]
    )
    loss = sum(loss_fns[task](predictions[task], batch[task]) for task in TASKS)
    loss = loss - model.crf(predictions["ner"], batch["ner"], mask=batch["word_mask"], reduction="mean")
    return loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print(f"Training on: {DEVICE} | encoder: {ENCODER_NAME}")

    train_df, val_df, _ = load_and_split()
    labels = build_labels(train_df)
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)

    model = RobertaBase.from_labels(labels).to(DEVICE)
    loss_fns = make_loss_functions(train_df, labels)

    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    head_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": ENCODER_LR},
            {"params": head_params, "lr": HEAD_LR},
        ]
    )

    n_train_batches = -(-len(train_df) // BATCH_SIZE)
    total_steps = n_train_batches * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_json(labels, MODEL_DIR / "labels.json")

    best_val_loss = float("inf")
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        batches = make_batches(
            train_df, tokenizer, labels, BATCH_SIZE, shuffle=True, shuffle_seed=SEED + epoch
        )
        for batch in tqdm(batches, total=n_train_batches, desc=f"epoch {epoch}", leave=False):
            loss = batch_loss(model, batch, loss_fns)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= n_train_batches

        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in make_batches(val_df, tokenizer, labels, BATCH_SIZE):
                val_loss += batch_loss(model, batch, loss_fns).item()
                n_val_batches += 1
        val_loss /= n_val_batches

        print(f"Epoch {epoch:2d}: train loss {train_loss:.4f} | val loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
            print("  -> best so far, saved")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"Val loss hasn't improved for {PATIENCE} epochs - stopping.")
                break

    print(f"Done. Best val loss: {best_val_loss:.4f}. Model saved in {MODEL_DIR}")


if __name__ == "__main__":
    main()