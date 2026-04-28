from __future__ import annotations

from src.nutrition import estimate_weighted_calories


def test_weighted_calories_fallback_when_empty_predictions() -> None:
    result = estimate_weighted_calories([])

    assert result["method"] == "fallback"
    assert result["source"] == "local_reference_v1"
    assert result["estimated_kcal"] > 0


def test_weighted_calories_returns_weighted_mean() -> None:
    predictions = [
        {"class_name": "pizza", "score": 0.75},
        {"class_name": "sushi", "score": 0.25},
    ]

    result = estimate_weighted_calories(predictions)

    # Pizza ~691.6 kcal, Sushi ~319.0 kcal => weighted ~598.45 => 598.5 rounded.
    assert result["method"] == "topk_weighted"
    assert result["estimated_kcal"] == 598.5

