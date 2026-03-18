from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.config import PredictConfig
from src.engine import load_checkpoint
from src.model import create_model, topk_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prediction on one image")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def make_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict_image(args: argparse.Namespace) -> dict:
    cfg = PredictConfig(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        image_size=args.image_size,
        top_k=args.top_k,
    )

    device = torch.device("cpu")
    checkpoint = load_checkpoint(cfg.checkpoint_path, device)
    class_names = checkpoint["class_names"]

    model = create_model(checkpoint["backbone"], num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    transform = make_transform(cfg.image_size)
    image = Image.open(cfg.image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)

    predictions = [
        {"class_name": class_name, "score": score}
        for class_name, score in topk_predictions(logits, class_names, cfg.top_k)
    ]
    return {
        "image": str(cfg.image_path),
        "checkpoint": str(cfg.checkpoint_path),
        "predictions": predictions,
    }


def main() -> None:
    args = parse_args()
    results = predict_image(args)
    for item in results["predictions"]:
        print(f"{item['class_name']}: {item['score']:.4f}")


if __name__ == "__main__":
    main()

