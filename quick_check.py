from __future__ import annotations

import shutil
from pathlib import Path

import torch
from torch import nn

from src.data import auto_split_dataset, build_dataloaders, create_synthetic_dataset
from src.engine import run_epoch
from src.model import create_model, get_trainable_parameters


def main() -> None:
    base_dir = Path("data/_quickcheck")
    raw_dir = base_dir / "raw"
    processed_dir = base_dir / "processed"

    if base_dir.exists():
        shutil.rmtree(base_dir)

    create_synthetic_dataset(raw_dir=raw_dir, samples_per_class=12)
    stats = auto_split_dataset(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42,
        overwrite=True,
    )

    train_loader, val_loader, _, class_names = build_dataloaders(
        processed_dir=processed_dir,
        image_size=96,
        batch_size=8,
        num_workers=0,
    )

    model = create_model("resnet18", num_classes=len(class_names), pretrained=False)
    device = torch.device("cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(get_trainable_parameters(model), lr=1e-3)

    train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
    val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)

    print("Quick check stats:", stats)
    print({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
    print("QUICK_CHECK_OK")


if __name__ == "__main__":
    main()

