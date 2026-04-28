from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence, cast

from sklearn.metrics import classification_report, confusion_matrix


def compute_classification_summary(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> dict:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report_dict: dict[str, Any] = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        zero_division=0,
    )

    most_confused = []
    for actual_index, actual_name in enumerate(class_names):
        for predicted_index, predicted_name in enumerate(class_names):
            if actual_index == predicted_index:
                continue
            count = int(cm[actual_index, predicted_index])
            if count > 0:
                most_confused.append(
                    {
                        "actual": actual_name,
                        "predicted": predicted_name,
                        "count": count,
                    }
                )
    most_confused.sort(key=lambda item: item["count"], reverse=True)

    per_class = {
        class_name: {
            "precision": float(cast(dict[str, Any], report_dict.get(class_name, {})).get("precision", 0.0)),
            "recall": float(cast(dict[str, Any], report_dict.get(class_name, {})).get("recall", 0.0)),
            "f1_score": float(cast(dict[str, Any], report_dict.get(class_name, {})).get("f1-score", 0.0)),
            "support": int(cast(dict[str, Any], report_dict.get(class_name, {})).get("support", 0)),
        }
        for class_name in class_names
        if class_name in report_dict
    }

    return {
        "confusion_matrix": cm.tolist(),
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
        "per_class": per_class,
        "macro_avg": report_dict.get("macro avg", {}),
        "weighted_avg": report_dict.get("weighted avg", {}),
        "most_confused_pairs": most_confused,
    }


def export_evaluation_results(results: dict, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    report_path = output_dir / "classification_report.txt"
    confusion_path = output_dir / "confusion_matrix.json"

    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)

    with report_path.open("w", encoding="utf-8") as fp:
        fp.write(results.get("classification_report_text", ""))

    with confusion_path.open("w", encoding="utf-8") as fp:
        json.dump(results.get("confusion_matrix", []), fp, indent=2)

    return {
        "results_json": json_path,
        "classification_report": report_path,
        "confusion_matrix": confusion_path,
    }

