from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from evaluate import evaluate_checkpoint
from predict import predict_image
from src.engine import load_checkpoint
from train import train_model

app = FastAPI(title="Local Image Classifier API", version="1.0.0")


class TrainRequest(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    models_dir: str = "models"
    image_size: int = Field(default=224, ge=32, le=1024)
    batch_size: int = Field(default=16, ge=1, le=256)
    epochs: int = Field(default=5, ge=1, le=200)
    learning_rate: float = Field(default=1e-3, gt=0)
    fine_tune_learning_rate: float = Field(default=1e-4, gt=0)
    train_ratio: float = Field(default=0.7, gt=0, lt=1)
    val_ratio: float = Field(default=0.2, gt=0, lt=1)
    test_ratio: float = Field(default=0.1, gt=0, lt=1)
    seed: int = 42
    num_workers: int = Field(default=0, ge=0, le=8)
    backbone: str = Field(default="resnet18", pattern="^(resnet18|mobilenet_v3_small)$")
    no_pretrained: bool = False
    unfreeze_epoch: int = Field(default=1, ge=0)
    skip_split: bool = False
    bootstrap_demo_data: bool = False


class EvaluateRequest(BaseModel):
    processed_dir: str = "data/processed"
    checkpoint: str = "models/best.pt"
    image_size: int = Field(default=224, ge=32, le=1024)
    batch_size: int = Field(default=16, ge=1, le=256)
    num_workers: int = Field(default=0, ge=0, le=8)


class PredictPathRequest(BaseModel):
    checkpoint: str = "models/best.pt"
    image: str
    image_size: int = Field(default=224, ge=32, le=1024)
    top_k: int = Field(default=3, ge=1, le=20)


def _to_namespace(payload: dict) -> argparse.Namespace:
    converted = {}
    for key, value in payload.items():
        if key in {"raw_dir", "processed_dir", "models_dir", "checkpoint", "image"}:
            converted[key] = Path(value)
        else:
            converted[key] = value
    return argparse.Namespace(**converted)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": str(torch.device("cpu"))}


@app.get("/model/status")
def model_status(checkpoint: str = "models/best.pt") -> dict:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        return {"ready": False, "checkpoint": str(checkpoint_path), "reason": "checkpoint_not_found"}

    data = load_checkpoint(checkpoint_path, torch.device("cpu"))
    classes = data.get("class_names", [])
    return {
        "ready": True,
        "checkpoint": str(checkpoint_path),
        "backbone": data.get("backbone"),
        "epoch": data.get("epoch"),
        "image_size": data.get("image_size"),
        "num_classes": len(classes),
        "class_names": classes,
    }


@app.post("/train")
def train_endpoint(request: TrainRequest) -> dict:
    if abs((request.train_ratio + request.val_ratio + request.test_ratio) - 1.0) > 1e-6:
        raise HTTPException(status_code=422, detail="train_ratio + val_ratio + test_ratio must equal 1.0")

    try:
        return train_model(_to_namespace(request.model_dump()))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/evaluate")
def evaluate_endpoint(request: EvaluateRequest) -> dict:
    checkpoint_path = Path(request.checkpoint)
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")

    try:
        return evaluate_checkpoint(_to_namespace(request.model_dump()))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/path")
def predict_path_endpoint(request: PredictPathRequest) -> dict:
    checkpoint_path = Path(request.checkpoint)
    image_path = Path(request.image)

    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")

    try:
        return predict_image(_to_namespace(request.model_dump()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict/upload")
async def predict_upload_endpoint(
    file: UploadFile = File(...),
    checkpoint: str = Form(default="models/best.pt"),
    image_size: int = Form(default=224),
    top_k: int = Form(default=3),
) -> dict:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")

    suffix = Path(file.filename or "upload.png").suffix or ".png"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = Path(tmp.name)
            content = await file.read()
            tmp.write(content)

        payload = {
            "checkpoint": str(checkpoint_path),
            "image": str(temp_path),
            "image_size": image_size,
            "top_k": top_k,
        }
        return predict_image(_to_namespace(payload))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)

