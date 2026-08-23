"""Train the BiLSTM.

Simpler than the transformer members' training loops: with the embedding
table frozen (see model.py), every trainable parameter starts randomly
initialised, so training needs only one flat learning rate and no warmup.

What is kept: class weights (neutral outnumbers negative roughly 3:1, so an
unweighted loss would learn to answer 'neutral'), gradient clipping, and early
stopping on validation loss.

Run:  python -m src.models.bilstm.train [--epochs N] [--resume]
"""

import argparse  # noqa: I001 - import order below is deliberate (see next comment)
import random

import sklearn  # noqa: F401 - must import before torch (Windows heap-corruption crash otherwise)
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import (
    BATCH_SIZE,
    LR,
    MAX_EPOCHS,
    MODEL_DIR,
    PATIENCE,
    SEED,
    TASKS,
    WEIGHT_DECAY,
)
from .data import (
    build_labels,
    load_and_split,
    load_embeddings,
    load_vocab,
    make_loader,
    save_json,
    unk_rate,
)
from .model import BiLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = MODEL_DIR / "best_model.pt"


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG that touches this run.

    torch alone is not enough: DataLoader shuffling draws from Python's random
    module, and anything numpy-side draws from numpy's. Miss one and two runs
    with 'the same seed' still diverge.

    cudnn.deterministic costs a little speed and buys runs that reproduce
    exactly - the right trade for a graded experiment where the reported
    numbers have to be defensible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op on CPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loss_functions(train_df, labels: dict[str, list[str]]) -> dict[str, nn.Module]:
    """One loss function per task, with class weights for the skewed tasks."""
    loss_fns = {}
    for task in TASKS:
        counts = train_df[task].astype(str).value_counts()
        weights = [
            min(len(train_df) / (len(labels[task]) * counts[name]), 10.0)
            for name in labels[task]
        ]
        weight_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
        loss_fns[task] = nn.CrossEntropyLoss(weight=weight_tensor)
    return loss_fns


def batch_loss(model: nn.Module, batch: dict, loss_fns: dict[str, nn.Module]) -> torch.Tensor:
    """Summed multi-task loss for one batch, already moved to DEVICE."""
    predictions = model(batch["input_ids"], batch["mask"])
    loss = sum(loss_fns[task](predictions[task], batch[task]) for task in TASKS)
    # the CRF returns log-likelihood, so negate it to get a loss to minimise
    return loss - model.crf(predictions["ner"], batch["ner"], mask=batch["mask"], reduction="mean")


def to_device(batch: dict) -> dict:
    return {name: tensor.to(DEVICE, non_blocking=True) for name, tensor in batch.items()}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fns: dict[str, nn.Module],
    epoch: int,
) -> float:
    model.train()  # dropout active
    total_loss = 0.0

    for batch in tqdm(loader, desc=f"epoch {epoch}", leave=False):
        batch = to_device(batch)
        loss = batch_loss(model, batch, loss_fns)

        optimizer.zero_grad(set_to_none=True)  # skips a memset vs zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()  # .item() only for logging, after backward

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, loss_fns: dict[str, nn.Module]) -> float:
    model.eval()  # dropout off - without this, val loss is noise
    total_loss = sum(batch_loss(model, to_device(batch), loss_fns).item() for batch in loader)
    return total_loss / len(loader)


def save_checkpoint(model, optimizer, epoch: int, val_loss: float, labels: dict) -> None:
    """Save enough to resume training, not just to run inference."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "labels": labels,
        },
        CHECKPOINT_PATH,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--resume", action="store_true", help="continue from the saved checkpoint")
    args = parser.parse_args()

    set_seed()

    train_df, val_df, _ = load_and_split()
    labels = build_labels(train_df)
    vocab = load_vocab()
    embeddings = load_embeddings()

    print(f"Training on: {DEVICE} | embeddings: {tuple(embeddings.shape)}")
    print(f"Out-of-vocabulary rate on the val split: {unk_rate(val_df, vocab):.2%}")

    pin = DEVICE.type == "cuda"
    train_loader = make_loader(train_df, vocab, labels, BATCH_SIZE, shuffle=True, pin_memory=pin)
    val_loader = make_loader(val_df, vocab, labels, BATCH_SIZE, pin_memory=pin)

    model = BiLSTM.from_labels(labels, embeddings).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,} (the embedding table is frozen)")

    loss_fns = make_loss_functions(train_df, labels)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_json(labels, MODEL_DIR / "labels.json")

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume and CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["val_loss"]
        print(f"Resumed from epoch {checkpoint['epoch']} (val loss {best_val_loss:.4f})")

    bad_epochs = 0
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fns, epoch)
        val_loss = evaluate_loss(model, val_loader, loss_fns)
        print(f"Epoch {epoch:2d}: train loss {train_loss:.4f} | val loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            save_checkpoint(model, optimizer, epoch, val_loss, labels)
            print("  -> best so far, saved")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"Val loss hasn't improved for {PATIENCE} epochs - stopping.")
                break

    print(f"Done. Best val loss: {best_val_loss:.4f}. Model saved in {MODEL_DIR}")


if __name__ == "__main__":
    main()
