from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from psycopg import connect

from evaluate import evaluate_checkpoint
from predict import predict_image
from src.engine import load_checkpoint
from src.nutrition import estimate_calories_for_class, estimate_weighted_calories
from train import train_model

app = FastAPI(
    title="Local Image Classifier API",
    version="1.0.0",
    servers=[{"url": "https://python.medev-tech.fr", "description": "Production"}],
    docs_url="/docs",
    openapi_url="/openapi.json",
)
PREDICTION_CACHE: dict[str, dict] = {}
FEEDBACK_LOG_PATH = Path("models/feedback_log.jsonl")
DB_READY = False
DB_STATUS_MESSAGE = "database_not_initialized"
DEFAULT_DB_USER = "postgres"
DEFAULT_DB_PASSWORD = "uqCt0OqIlbjUcUhYPbZ3"
DEFAULT_DB_HOST = "137.74.113.34"
DEFAULT_DB_PORT = "5432"
DEFAULT_DB_NAME = "postgres"
DEFAULT_DATABASE_URL = (
    f"postgresql://{DEFAULT_DB_USER}:{DEFAULT_DB_PASSWORD}@{DEFAULT_DB_HOST}:{DEFAULT_DB_PORT}/{DEFAULT_DB_NAME}"
)


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


class FeedbackRequest(BaseModel):
    prediction_id: str
    is_correct: bool
    correct_class: str | None = None


def _build_database_url() -> str:
    explicit_url = os.getenv("DATABASE_URL")
    if explicit_url:
        return explicit_url

    user = os.getenv("DB_USER", DEFAULT_DB_USER)
    password = os.getenv("DB_PASSWORD", DEFAULT_DB_PASSWORD)
    host = os.getenv("DB_HOST", DEFAULT_DB_HOST)
    port = os.getenv("DB_PORT", DEFAULT_DB_PORT)
    name = os.getenv("DB_NAME", DEFAULT_DB_NAME)
    if all([user, password, host, port, name]):
        return f"postgresql://{user}:{password}@{host}:{port}/{name}"

    return DEFAULT_DATABASE_URL


def _ensure_database_schema() -> None:
    query = """
    CREATE TABLE IF NOT EXISTS api_predictions (
        id BIGSERIAL PRIMARY KEY,
        prediction_id TEXT UNIQUE NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        image TEXT,
        checkpoint TEXT,
        predicted_class TEXT,
        predictions_json JSONB NOT NULL,
        calories_json JSONB NOT NULL
    );

    CREATE TABLE IF NOT EXISTS api_feedback (
        id BIGSERIAL PRIMARY KEY,
        timestamp TIMESTAMPTZ NOT NULL,
        prediction_id TEXT NOT NULL,
        checkpoint TEXT,
        is_correct BOOLEAN NOT NULL,
        predicted_class TEXT NOT NULL,
        final_class TEXT NOT NULL,
        calories_json JSONB NOT NULL
    );
    """
    database_url = _build_database_url()
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
        conn.commit()


def _try_initialize_database() -> bool:
    global DB_READY, DB_STATUS_MESSAGE
    try:
        _ensure_database_schema()
        DB_READY = True
        DB_STATUS_MESSAGE = "ok"
        return True
    except Exception as exc:
        DB_READY = False
        DB_STATUS_MESSAGE = str(exc)
        return False


def _save_prediction_to_db(enriched: dict) -> bool:
    global DB_READY, DB_STATUS_MESSAGE
    if not DB_READY and not _try_initialize_database():
        return False

    database_url = _build_database_url()
    query = """
    INSERT INTO api_predictions (
        prediction_id,
        created_at,
        image,
        checkpoint,
        predicted_class,
        predictions_json,
        calories_json
    )
    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
    ON CONFLICT (prediction_id) DO UPDATE SET
        created_at = EXCLUDED.created_at,
        image = EXCLUDED.image,
        checkpoint = EXCLUDED.checkpoint,
        predicted_class = EXCLUDED.predicted_class,
        predictions_json = EXCLUDED.predictions_json,
        calories_json = EXCLUDED.calories_json;
    """
    try:
        with connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        enriched["prediction_id"],
                        enriched["created_at"],
                        enriched.get("image"),
                        enriched.get("checkpoint"),
                        enriched["top_prediction"]["class_name"],
                        json.dumps(enriched.get("predictions", []), ensure_ascii=True),
                        json.dumps(enriched.get("calories", {}), ensure_ascii=True),
                    ),
                )
            conn.commit()
        return True
    except Exception as exc:
        DB_READY = False
        DB_STATUS_MESSAGE = str(exc)
        return False


def _save_feedback_to_db(entry: dict) -> bool:
    global DB_READY, DB_STATUS_MESSAGE
    if not DB_READY and not _try_initialize_database():
        return False

    database_url = _build_database_url()
    query = """
    INSERT INTO api_feedback (
        timestamp,
        prediction_id,
        checkpoint,
        is_correct,
        predicted_class,
        final_class,
        calories_json
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb);
    """
    try:
        with connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        entry["timestamp"],
                        entry["prediction_id"],
                        entry.get("checkpoint"),
                        entry["is_correct"],
                        entry["predicted_class"],
                        entry["final_class"],
                        json.dumps(entry["calories"], ensure_ascii=True),
                    ),
                )
            conn.commit()
        return True
    except Exception as exc:
        DB_READY = False
        DB_STATUS_MESSAGE = str(exc)
        return False


def _append_feedback_log(entry: dict) -> None:
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=True) + "\n")


def _build_prediction_response(result: dict) -> dict:
    predictions = result.get("predictions", [])
    if not predictions:
        raise HTTPException(status_code=400, detail="Prediction failed: no predictions returned")

    best_prediction = predictions[0]
    top1_calories = estimate_calories_for_class(best_prediction["class_name"])
    weighted_calories = estimate_weighted_calories(predictions)
    prediction_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    enriched = {
        "prediction_id": prediction_id,
        "created_at": created_at,
        "image": result.get("image"),
        "checkpoint": result.get("checkpoint"),
        "predictions": predictions,
        "top_prediction": best_prediction,
        "calories": {
            "top1": top1_calories,
            "weighted_topk": weighted_calories,
        },
    }
    PREDICTION_CACHE[prediction_id] = {
        "created_at": created_at,
        "checkpoint": result.get("checkpoint"),
        "predicted_class": best_prediction["class_name"],
        "predictions": predictions,
    }
    enriched["database_saved"] = _save_prediction_to_db(enriched)
    return enriched


def _to_namespace(payload: dict) -> argparse.Namespace:
    converted = {}
    for key, value in payload.items():
        if key in {"raw_dir", "processed_dir", "models_dir", "checkpoint", "image"}:
            converted[key] = Path(value)
        else:
            converted[key] = value
    return argparse.Namespace(**converted)


@app.on_event("startup")
def startup() -> None:
    _try_initialize_database()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": str(torch.device("cpu")),
        "database": {
            "ready": DB_READY,
            "message": DB_STATUS_MESSAGE,
        },
    }


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
        result = predict_image(_to_namespace(request.model_dump()))
        return _build_prediction_response(result)
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
        result = predict_image(_to_namespace(payload))
        return _build_prediction_response(result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.post("/feedback")
def feedback_endpoint(request: FeedbackRequest) -> dict:
    prediction = PREDICTION_CACHE.get(request.prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="prediction_id not found")

    predicted_class = str(prediction["predicted_class"])
    if request.is_correct:
        final_class = predicted_class
    else:
        if not request.correct_class:
            raise HTTPException(status_code=422, detail="correct_class is required when is_correct is false")
        final_class = request.correct_class.strip().lower()

    calories = estimate_calories_for_class(final_class)
    feedback_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction_id": request.prediction_id,
        "checkpoint": prediction.get("checkpoint"),
        "is_correct": request.is_correct,
        "predicted_class": predicted_class,
        "final_class": final_class,
        "calories": calories,
    }
    database_saved = _save_feedback_to_db(feedback_entry)

    _append_feedback_log(feedback_entry)

    return {
        "message": "feedback_saved",
        "prediction_id": request.prediction_id,
        "predicted_class": predicted_class,
        "final_class": final_class,
        "calories": calories,
        "database_saved": database_saved,
        "feedback_log": str(FEEDBACK_LOG_PATH),
    }


