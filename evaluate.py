from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.config import EvalConfig
from src.data import build_dataloaders
from src.engine import load_checkpoint
from src.metrics import compute_classification_summary, export_evaluation_results
from src.model import create_model, load_model_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on test split")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("models/evaluation"))
    return parser.parse_args()


def evaluate_checkpoint(args: argparse.Namespace) -> dict:
    cfg = EvalConfig(
        processed_dir=args.processed_dir,
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )

    device = torch.device("cpu")
    checkpoint = load_checkpoint(cfg.checkpoint_path, device)

    _, _, test_loader, class_names = build_dataloaders(
        cfg.processed_dir,
        cfg.image_size,
        cfg.batch_size,
        cfg.num_workers,
        augmentations=False,
    )

    model = create_model(checkpoint["backbone"], num_classes=len(class_names), pretrained=False)
    load_model_state(model, checkpoint["model_state"])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * targets.size(0)
            total_correct += (preds == targets).sum().item()
            total_count += targets.size(0)

            all_targets.extend(targets.cpu().tolist())
            all_predictions.extend(preds.cpu().tolist())

    test_loss = total_loss / max(total_count, 1)
    test_acc = total_correct / max(total_count, 1)

    if total_count > 0:
        advanced_metrics = compute_classification_summary(all_targets, all_predictions, class_names)
    else:
        advanced_metrics = {
            "confusion_matrix": [],
            "classification_report_text": "No test samples available.",
            "classification_report_dict": {},
            "per_class": {},
            "macro_avg": {},
            "weighted_avg": {},
            "most_confused_pairs": [],
        }

    results = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "classes": class_names,
        **advanced_metrics,
    }

    export_paths = export_evaluation_results(results, cfg.output_dir)
    results["exported_files"] = {key: str(value) for key, value in export_paths.items()}
    return results


def main() -> None:
    args = parse_args()
    results = evaluate_checkpoint(args)

    print({"test_loss": results["test_loss"], "test_acc": results["test_acc"], "classes": results["classes"]})
    print(results["classification_report_text"])


if __name__ == "__main__":
    main()

