from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn

from src.config import TrainConfig
from src.data import auto_split_dataset, build_dataloaders, create_synthetic_dataset, find_class_images
from src.engine import run_epoch, save_checkpoint
from src.model import create_model, get_trainable_parameters, unfreeze_for_finetune


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a local transfer-learning image classifier (CPU only)")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "mobilenet_v3_small"])
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained ImageNet weights")
    parser.add_argument("--unfreeze-epoch", type=int, default=1, help="Epoch to unfreeze part of backbone for fine-tuning")
    parser.add_argument("--skip-split", action="store_true", help="Use existing processed split folders")
    parser.add_argument(
        "--run-forever",
        action="store_true",
        help="Train indefinitely until manual stop (Ctrl+C)",
    )
    parser.add_argument(
        "--bootstrap-demo-data",
        action="store_true",
        help="If raw-dir is empty, generate a small synthetic dataset and continue",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _dataset_hint(raw_dir: Path) -> str:
    return (
        f"No images found in {raw_dir}.\n"
        "Expected structure:\n"
        "  data/raw/<class_name>/image1.jpg\n"
        "  data/raw/<class_name>/image2.png\n"
        "You can also run with --bootstrap-demo-data for a quick local demo."
    )


def train_model(args: argparse.Namespace) -> dict:
    cfg = TrainConfig(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        fine_tune_learning_rate=args.fine_tune_learning_rate,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        unfreeze_epoch=args.unfreeze_epoch,
    )

    set_seed(cfg.seed)
    device = torch.device("cpu")

    if not args.skip_split:
        if args.bootstrap_demo_data and not find_class_images(cfg.raw_dir):
            print(f"No images found in {cfg.raw_dir}. Generating synthetic demo data...")
            create_synthetic_dataset(raw_dir=cfg.raw_dir, image_size=96, samples_per_class=24)

        try:
            print(
                f"Preparing split from {cfg.raw_dir} to {cfg.processed_dir}... "
                "this can take several minutes on large datasets."
            )
            split_stats = auto_split_dataset(
                raw_dir=cfg.raw_dir,
                processed_dir=cfg.processed_dir,
                train_ratio=cfg.train_ratio,
                val_ratio=cfg.val_ratio,
                test_ratio=cfg.test_ratio,
                seed=cfg.seed,
                overwrite=True,
                verbose=True,
            )
        except ValueError as exc:
            if "No images found" in str(exc):
                raise ValueError(_dataset_hint(cfg.raw_dir)) from exc
            raise
        print("Split done:", json.dumps(split_stats, indent=2))

    train_loader, val_loader, _, class_names = build_dataloaders(
        cfg.processed_dir, cfg.image_size, cfg.batch_size, cfg.num_workers
    )

    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = create_model(cfg.backbone, num_classes=len(class_names), pretrained=cfg.pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(get_trainable_parameters(model), lr=cfg.learning_rate)

    best_val_acc = -1.0
    history = []
    run_forever = getattr(args, "run_forever", False)
    epoch_index = 0

    if run_forever:
        print("Infinite training mode enabled. Press Ctrl+C to stop.")

    try:
        while True:
            if (not run_forever) and (epoch_index >= cfg.epochs):
                break

            if epoch_index == cfg.unfreeze_epoch:
                unfreeze_for_finetune(model, cfg.backbone)
                optimizer = torch.optim.Adam(get_trainable_parameters(model), lr=cfg.fine_tune_learning_rate)

            train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)

            epoch_metrics = {
                "epoch": epoch_index + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            history.append(epoch_metrics)
            print(epoch_metrics)

            checkpoint_payload = {
                "model_state": model.state_dict(),
                "class_names": class_names,
                "backbone": cfg.backbone,
                "image_size": cfg.image_size,
                "epoch": epoch_index + 1,
                "metrics": epoch_metrics,
            }

            save_checkpoint(cfg.models_dir / "last.pt", checkpoint_payload)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(cfg.models_dir / "best.pt", checkpoint_payload)

            epoch_index += 1
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving history and exiting...")

    with (cfg.models_dir / "history.json").open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    completed_epochs = len(history)
    print(f"Training done. Completed epochs={completed_epochs} | Best val_acc={best_val_acc:.4f}")
    return {
        "best_val_acc": best_val_acc,
        "epochs": cfg.epochs,
        "run_forever": run_forever,
        "completed_epochs": completed_epochs,
        "backbone": cfg.backbone,
        "classes": class_names,
        "history": history,
        "best_checkpoint": str((cfg.models_dir / "best.pt").as_posix()),
        "last_checkpoint": str((cfg.models_dir / "last.pt").as_posix()),
    }


def main() -> None:
    args = parse_args()
    train_model(args)


if __name__ == "__main__":
    main()

