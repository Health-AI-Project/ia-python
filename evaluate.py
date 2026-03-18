from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.config import EvalConfig
from src.data import build_dataloaders
from src.engine import load_checkpoint, run_epoch
from src.model import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on test split")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def evaluate_checkpoint(args: argparse.Namespace) -> dict:
    cfg = EvalConfig(
        processed_dir=args.processed_dir,
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = torch.device("cpu")
    checkpoint = load_checkpoint(cfg.checkpoint_path, device)

    _, _, test_loader, class_names = build_dataloaders(
        cfg.processed_dir, cfg.image_size, cfg.batch_size, cfg.num_workers
    )

    model = create_model(checkpoint["backbone"], num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device, optimizer=None)
    results = {"test_loss": test_loss, "test_acc": test_acc, "classes": class_names}
    return results


def main() -> None:
    args = parse_args()
    results = evaluate_checkpoint(args)

    print(results)


if __name__ == "__main__":
    main()

