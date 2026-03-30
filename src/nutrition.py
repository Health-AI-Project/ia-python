from __future__ import annotations

from typing import Dict, List


DEFAULT_KCAL_PER_100G = 220.0
DEFAULT_PORTION_G = 220.0
DEFAULT_VARIANCE_RATIO = 0.25


# Valeurs indicatives pour un MVP. Le fallback par defaut est applique pour les classes absentes.
_CALORIE_REFERENCE: Dict[str, Dict[str, float]] = {
    "apple_pie": {"kcal_per_100g": 237.0, "portion_g": 130.0},
    "baby_back_ribs": {"kcal_per_100g": 292.0, "portion_g": 250.0},
    "baklava": {"kcal_per_100g": 430.0, "portion_g": 90.0},
    "beef_carpaccio": {"kcal_per_100g": 170.0, "portion_g": 120.0},
    "beef_tartare": {"kcal_per_100g": 190.0, "portion_g": 150.0},
    "beet_salad": {"kcal_per_100g": 110.0, "portion_g": 180.0},
    "beignets": {"kcal_per_100g": 338.0, "portion_g": 90.0},
    "bibimbap": {"kcal_per_100g": 145.0, "portion_g": 350.0},
    "caesar_salad": {"kcal_per_100g": 180.0, "portion_g": 220.0},
    "cheesecake": {"kcal_per_100g": 321.0, "portion_g": 120.0},
    "chicken_curry": {"kcal_per_100g": 160.0, "portion_g": 300.0},
    "chicken_wings": {"kcal_per_100g": 254.0, "portion_g": 220.0},
    "chocolate_cake": {"kcal_per_100g": 371.0, "portion_g": 120.0},
    "churros": {"kcal_per_100g": 420.0, "portion_g": 100.0},
    "club_sandwich": {"kcal_per_100g": 250.0, "portion_g": 240.0},
    "croque_madame": {"kcal_per_100g": 260.0, "portion_g": 220.0},
    "cup_cakes": {"kcal_per_100g": 410.0, "portion_g": 75.0},
    "donuts": {"kcal_per_100g": 452.0, "portion_g": 70.0},
    "dumplings": {"kcal_per_100g": 190.0, "portion_g": 180.0},
    "eggs_benedict": {"kcal_per_100g": 225.0, "portion_g": 250.0},
    "french_fries": {"kcal_per_100g": 312.0, "portion_g": 150.0},
    "fried_rice": {"kcal_per_100g": 168.0, "portion_g": 280.0},
    "grilled_salmon": {"kcal_per_100g": 208.0, "portion_g": 190.0},
    "hamburger": {"kcal_per_100g": 295.0, "portion_g": 220.0},
    "hot_dog": {"kcal_per_100g": 290.0, "portion_g": 140.0},
    "ice_cream": {"kcal_per_100g": 207.0, "portion_g": 120.0},
    "lasagna": {"kcal_per_100g": 165.0, "portion_g": 280.0},
    "macaroni_and_cheese": {"kcal_per_100g": 164.0, "portion_g": 260.0},
    "omelette": {"kcal_per_100g": 154.0, "portion_g": 170.0},
    "pancakes": {"kcal_per_100g": 227.0, "portion_g": 160.0},
    "pizza": {"kcal_per_100g": 266.0, "portion_g": 260.0},
    "ramen": {"kcal_per_100g": 120.0, "portion_g": 450.0},
    "risotto": {"kcal_per_100g": 145.0, "portion_g": 300.0},
    "sashimi": {"kcal_per_100g": 130.0, "portion_g": 160.0},
    "spaghetti_bolognese": {"kcal_per_100g": 155.0, "portion_g": 320.0},
    "spaghetti_carbonara": {"kcal_per_100g": 192.0, "portion_g": 300.0},
    "steak": {"kcal_per_100g": 271.0, "portion_g": 220.0},
    "sushi": {"kcal_per_100g": 145.0, "portion_g": 220.0},
    "tacos": {"kcal_per_100g": 226.0, "portion_g": 180.0},
    "tiramisu": {"kcal_per_100g": 300.0, "portion_g": 120.0},
    "waffles": {"kcal_per_100g": 291.0, "portion_g": 130.0},
}


def estimate_calories_for_class(class_name: str) -> Dict[str, float | str]:
    key = class_name.strip().lower()
    values = _CALORIE_REFERENCE.get(
        key,
        {"kcal_per_100g": DEFAULT_KCAL_PER_100G, "portion_g": DEFAULT_PORTION_G},
    )
    estimated_kcal = values["kcal_per_100g"] * values["portion_g"] / 100.0
    spread = estimated_kcal * DEFAULT_VARIANCE_RATIO
    return {
        "class_name": key,
        "kcal_per_100g": round(values["kcal_per_100g"], 1),
        "portion_g": round(values["portion_g"], 1),
        "estimated_kcal": round(estimated_kcal, 1),
        "estimated_kcal_min": round(max(0.0, estimated_kcal - spread), 1),
        "estimated_kcal_max": round(estimated_kcal + spread, 1),
        "source": "local_reference_v1",
    }


def estimate_weighted_calories(predictions: List[Dict[str, float | str]]) -> Dict[str, float | str]:
    if not predictions:
        fallback = estimate_calories_for_class("unknown")
        fallback["method"] = "fallback"
        return fallback

    weighted_sum = 0.0
    total_score = 0.0
    for item in predictions:
        class_name = str(item.get("class_name", "unknown"))
        score = float(item.get("score", 0.0))
        estimate = estimate_calories_for_class(class_name)
        weighted_sum += estimate["estimated_kcal"] * score
        total_score += score

    weighted_kcal = weighted_sum / total_score if total_score > 0 else weighted_sum
    spread = weighted_kcal * DEFAULT_VARIANCE_RATIO
    return {
        "estimated_kcal": round(weighted_kcal, 1),
        "estimated_kcal_min": round(max(0.0, weighted_kcal - spread), 1),
        "estimated_kcal_max": round(weighted_kcal + spread, 1),
        "source": "local_reference_v1",
        "method": "topk_weighted",
    }
