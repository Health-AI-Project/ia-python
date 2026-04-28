from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

import api


def test_to_namespace_converts_path_fields() -> None:
    payload = {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "models_dir": "models",
        "checkpoint": "models/best.pt",
        "image": "img.png",
        "epochs": 3,
    }

    ns = api._to_namespace(payload)

    assert isinstance(ns.raw_dir, Path)
    assert isinstance(ns.processed_dir, Path)
    assert isinstance(ns.models_dir, Path)
    assert isinstance(ns.checkpoint, Path)
    assert isinstance(ns.image, Path)
    assert ns.epochs == 3


def test_build_prediction_response_updates_cache(monkeypatch) -> None:
    monkeypatch.setattr(api, "_save_prediction_to_db", lambda _payload: False)
    api.PREDICTION_CACHE.clear()

    result = {
        "image": "img.png",
        "checkpoint": "models/best.pt",
        "predictions": [
            {"class_name": "pizza", "score": 0.9},
            {"class_name": "sushi", "score": 0.1},
        ],
    }

    enriched = api._build_prediction_response(result)

    assert enriched["top_prediction"]["class_name"] == "pizza"
    assert enriched["database_saved"] is False
    assert enriched["prediction_id"] in api.PREDICTION_CACHE


def test_build_prediction_response_rejects_empty_predictions() -> None:
    with pytest.raises(HTTPException, match="no predictions"):
        api._build_prediction_response({"predictions": []})

