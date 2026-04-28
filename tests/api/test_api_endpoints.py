from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import api


def test_health_endpoint_returns_status_ok(monkeypatch) -> None:
    monkeypatch.setattr(api, "_try_initialize_database", lambda: False)

    with TestClient(api.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "database" in payload


def test_train_endpoint_rejects_invalid_ratio_sum(monkeypatch) -> None:
    monkeypatch.setattr(api, "_try_initialize_database", lambda: False)

    with TestClient(api.app) as client:
        response = client.post(
            "/train",
            json={
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "models_dir": "models",
                "train_ratio": 0.7,
                "val_ratio": 0.2,
                "test_ratio": 0.2,
            },
        )

    assert response.status_code == 422
    assert "must equal 1.0" in response.json()["detail"]


def test_predict_path_returns_404_when_image_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(api, "_try_initialize_database", lambda: False)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"ok")

    with TestClient(api.app) as client:
        response = client.post(
            "/predict/path",
            json={
                "checkpoint": str(checkpoint),
                "image": str(tmp_path / "missing.png"),
                "image_size": 224,
                "top_k": 3,
            },
        )

    assert response.status_code == 404
    assert "Image not found" in response.json()["detail"]

