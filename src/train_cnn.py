"""
Train ResNet-50 CNN detector with data augmentation and early stopping.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.cnn_detector import build_resnet50, save_model
from src.data_loader import create_dataloaders


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 1e-4
    best_loss: float = float("inf")
    counter: int = 0
    should_stop: bool = False

    def step(self, val_loss: float) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    train: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.train(train)
    losses = []
    all_probs = []
    all_labels = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for inputs, labels in tqdm(loader, desc="train" if train else "val"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            losses.append(loss.item())
            all_probs.append(probs)
            all_labels.append(labels.detach().cpu().numpy())

    all_probs_np = np.concatenate(all_probs)
    all_labels_np = np.concatenate(all_labels)
    return float(np.mean(losses)), all_probs_np, all_labels_np


def train_cnn(
    data_dir: str = "data",
    outputs_dir: str = "outputs",
    epochs: int = 15,
    batch_size: int = 32,
    img_size: int = 224,
    lr: float = 1e-4,
    patience: int = 5,
) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaders = create_dataloaders(data_dir, batch_size=batch_size, img_size=img_size)

    model = build_resnet50(num_classes=2, pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    early = EarlyStopping(patience=patience)

    history = []
    best_path = Path(outputs_dir) / "models" / "cnn_resnet50.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(epochs):
        train_loss, train_probs, train_labels = _run_epoch(
            model, loaders.train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_probs, val_labels = _run_epoch(
            model, loaders.val_loader, criterion, optimizer, device, train=False
        )

        train_metrics = _metrics(train_labels, train_probs)
        val_metrics = _metrics(val_labels, val_probs)

        scheduler.step(val_loss)
        early.step(val_loss)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            save_model(model, str(best_path))

        if early.should_stop:
            break

    history_df = pd.DataFrame(history)
    plots_dir = Path(outputs_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(plots_dir / "cnn_training_history.csv", index=False)
    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet-50 CNN detector")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_cnn(
        data_dir=args.data_dir,
        outputs_dir=args.outputs_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr=args.lr,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
