from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.config import PredictConfig
from src.data import build_transforms
from src.engine import load_checkpoint
from src.model import create_model, load_model_state, topk_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prediction on one image")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    return parser.parse_args()


def _load_image(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise ValueError(f"Invalid image file: {image_path}") from exc


def predict_image(args: argparse.Namespace) -> dict:
    cfg = PredictConfig(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        image_size=args.image_size,
        top_k=args.top_k,
        confidence_threshold=args.confidence_threshold,
    )

    device = torch.device("cpu")
    if not cfg.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}")

    checkpoint = load_checkpoint(cfg.checkpoint_path, device)
    class_names = checkpoint["class_names"]

    model = create_model(
        checkpoint["backbone"],
        num_classes=len(class_names),
        pretrained=False,
        dropout=float(checkpoint.get("config", {}).get("dropout", 0.2)),
    )
    load_model_state(model, checkpoint["model_state"])
    model.to(device)
    model.eval()

    _, transform = build_transforms(cfg.image_size, augmentations=False)
    image = _load_image(cfg.image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)

    raw_predictions = topk_predictions(logits, class_names, cfg.top_k)
    predictions = []
    for rank, (class_name, score) in enumerate(raw_predictions, start=1):
        predictions.append(
            {
                "rank": rank,
                "class_name": class_name,
                "score": score,
                "confidence_pct": round(score * 100.0, 1),
                "is_top": rank == 1,
                "above_threshold": score >= cfg.confidence_threshold,
            }
        )

    top_prediction = predictions[0]
    uncertain = top_prediction["score"] < cfg.confidence_threshold
    return {
        "image": str(cfg.image_path),
        "checkpoint": str(cfg.checkpoint_path),
        "predictions": predictions,
        "top_prediction": top_prediction,
        "confidence_threshold": cfg.confidence_threshold,
        "uncertain": uncertain,
        "status": "uncertain" if uncertain else "confident",
    }


def main() -> None:
    args = parse_args()
    results = predict_image(args)
    if results["uncertain"]:
        print(f"Warning: confidence below threshold ({results['confidence_threshold']:.2f})")
    for item in results["predictions"]:
        print(f"#{item['rank']} {item['class_name']}: {item['score']:.4f}")


if __name__ == "__main__":
    main()

