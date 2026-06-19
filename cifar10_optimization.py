import argparse
import os
import random
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@dataclass
class ExperimentConfig:
    name: str
    use_batch_norm: bool
    dropout: float
    optimizer_name: str
    lr: float
    momentum: float = 0.0


class CIFAR10CNN(nn.Module):
    def __init__(self, use_batch_norm: bool = False, dropout: float = 0.0):
        super().__init__()

        def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            ]
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.extend([
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            ])
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.extend([
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_loaders(batch_size: int, fast_dev_run: bool):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_full = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform_train)
    val_source = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform_test)

    indices = list(range(len(train_full)))
    train_idx, val_idx = random_split(indices, [45000, 5000], generator=torch.Generator().manual_seed(42))
    train_set = Subset(train_full, train_idx.indices)
    val_set = Subset(val_source, val_idx.indices)

    if fast_dev_run:
        train_set = Subset(train_set, range(10000))
        val_set = Subset(val_set, range(2000))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


def make_optimizer(model: nn.Module, cfg: ExperimentConfig):
    if cfg.optimizer_name == "SGD":
        return optim.SGD(model.parameters(), lr=cfg.lr)
    if cfg.optimizer_name == "SGD + Momentum":
        return optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    if cfg.optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=cfg.lr)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer_name}")


def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train_one_experiment(cfg: ExperimentConfig, train_loader, val_loader, epochs: int, device: torch.device):
    model = CIFAR10CNN(cfg.use_batch_norm, cfg.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, cfg)
    history = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        train_loss = total_loss / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history.append({
            "experiment": cfg.name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })
        print(f"{cfg.name:18s} | epoch {epoch:02d}/{epochs} | loss {train_loss:.4f} | val_acc {val_acc:.4f}")

    elapsed = time.time() - start
    hist_df = pd.DataFrame(history)
    best_val_acc = hist_df["val_accuracy"].max()
    convergence_threshold = 0.99 * best_val_acc
    convergence_epoch = int(hist_df.loc[hist_df["val_accuracy"] >= convergence_threshold, "epoch"].iloc[0])

    summary = {
        "experiment": cfg.name,
        "optimizer": cfg.optimizer_name,
        "batch_norm": cfg.use_batch_norm,
        "dropout": cfg.dropout,
        "final_train_loss": hist_df["train_loss"].iloc[-1],
        "best_val_accuracy": best_val_acc,
        "final_val_accuracy": hist_df["val_accuracy"].iloc[-1],
        "train_time_sec": elapsed,
        "convergence_epoch": convergence_epoch,
    }
    return summary, hist_df


def plot_curves(history_df: pd.DataFrame, output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, group in history_df.groupby("experiment"):
        axes[0].plot(group["epoch"], group["train_loss"], marker="o", label=name)
        axes[1].plot(group["epoch"], group["val_accuracy"], marker="o", label=name)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader = build_loaders(args.batch_size, args.fast_dev_run)
    configs = [
        ExperimentConfig("B1_Basic_SGD", False, 0.0, "SGD", 0.01),
        ExperimentConfig("B2_BN_DO02_SGDM", True, 0.2, "SGD + Momentum", 0.01, 0.9),
        ExperimentConfig("B2_BN_DO05_SGDM", True, 0.5, "SGD + Momentum", 0.01, 0.9),
        ExperimentConfig("B2_BN_DO02_Adam", True, 0.2, "Adam", 0.001),
        ExperimentConfig("B2_BN_DO05_Adam", True, 0.5, "Adam", 0.001),
    ]

    summaries = []
    histories = []
    for cfg in configs:
        summary, hist = train_one_experiment(cfg, train_loader, val_loader, args.epochs, device)
        summaries.append(summary)
        histories.append(hist)

    results_df = pd.DataFrame(summaries).sort_values("best_val_accuracy", ascending=False)
    history_df = pd.concat(histories, ignore_index=True)
    results_df.to_csv("outputs/optimization_results.csv", index=False)
    history_df.to_csv("outputs/training_history.csv", index=False)
    plot_curves(history_df, "outputs/training_curves.png")
    print(results_df)


if __name__ == "__main__":
    main()
