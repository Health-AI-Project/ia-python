from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from src.data import _split_list, auto_split_dataset, compute_class_weights, find_class_images


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (120, 20, 20)).save(path)


def _make_corrupted_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-an-image")


def test_split_list_keeps_all_items_and_makes_3_splits_when_possible() -> None:
    items = [Path(f"img_{i}.png") for i in range(5)]

    train_items, val_items, test_items = _split_list(items, train_ratio=0.6, val_ratio=0.2, seed=42)

    assert len(train_items) >= 1
    assert len(val_items) >= 1
    assert len(test_items) >= 1
    assert sorted(train_items + val_items + test_items) == sorted(items)


def test_auto_split_dataset_creates_expected_counts(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    for class_name in ["pizza", "sushi"]:
        for i in range(8):
            _make_image(raw_dir / class_name / f"{class_name}_{i}.png")

    stats = auto_split_dataset(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=7,
        overwrite=True,
        verbose=False,
    )

    for class_name in ["pizza", "sushi"]:
        total = stats["train"][class_name] + stats["val"][class_name] + stats["test"][class_name]
        assert total == 8

    assert (processed_dir / "train" / "pizza").exists()
    assert (processed_dir / "val" / "pizza").exists()
    assert (processed_dir / "test" / "pizza").exists()


def test_auto_split_dataset_rejects_invalid_ratios(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _make_image(raw_dir / "pizza" / "a.png")

    with pytest.raises(ValueError, match="must equal 1.0"):
        auto_split_dataset(
            raw_dir=raw_dir,
            processed_dir=tmp_path / "processed",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.3,
            seed=1,
        )


def test_find_class_images_skips_corrupted_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _make_image(raw_dir / "pizza" / "good.png")
    _make_corrupted_file(raw_dir / "pizza" / "bad.png")

    class_images = find_class_images(raw_dir)

    assert len(class_images["pizza"]) == 1
    assert class_images["pizza"][0].name == "good.png"


def test_compute_class_weights_prefers_rare_classes() -> None:
    dummy_dataset = type(
        "DummyDataset",
        (),
        {"targets": [0, 0, 0, 1], "classes": ["pizza", "sushi"]},
    )()

    weights = compute_class_weights(dummy_dataset)

    assert weights[1] > weights[0]


