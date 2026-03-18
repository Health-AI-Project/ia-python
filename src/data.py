from __future__ import annotations

import random
import shutil
import stat
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def find_class_images(raw_dir: Path) -> Dict[str, List[Path]]:
    class_images: Dict[str, List[Path]] = defaultdict(list)
    if not raw_dir.exists():
        return {}

    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for file_path in class_dir.rglob("*"):
            if file_path.is_file() and _is_image(file_path):
                class_images[class_dir.name].append(file_path)

    # Keep deterministic order for reproducibility.
    for class_name in class_images:
        class_images[class_name] = sorted(class_images[class_name])

    return dict(class_images)


def _split_list(items: List[Path], train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[Path], List[Path], List[Path]]:
    if not items:
        return [], [], []

    shuffled = list(items)
    rnd = random.Random(seed)
    rnd.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # Ensure each split can receive examples when possible.
    if n >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)

    n_test_start = n_train + n_val
    train_items = shuffled[:n_train]
    val_items = shuffled[n_train:n_test_start]
    test_items = shuffled[n_test_start:]

    if n >= 3 and not test_items:
        test_items = [val_items.pop()] if val_items else [train_items.pop()]

    return train_items, val_items, test_items


def _on_rmtree_error(func, path, exc_info) -> None:
    # On Windows, read-only files can block rmtree; make them writable then retry.
    try:
        Path(path).chmod(stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def _robust_rmtree(path: Path, max_retries: int = 8, delay_seconds: float = 0.25) -> None:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        if not path.exists():
            return
        try:
            shutil.rmtree(path, onexc=_on_rmtree_error)
            return
        except OSError as exc:
            # WinError 145 (directory not empty) can happen transiently with OneDrive/AV scans.
            last_error = exc
            if attempt == max_retries - 1:
                break
            time.sleep(delay_seconds * (attempt + 1))

    if not path.exists():
        return

    quarantine = path.with_name(f"{path.name}_to_delete_{int(time.time())}")
    try:
        path.rename(quarantine)
        shutil.rmtree(quarantine, onexc=_on_rmtree_error)
        return
    except Exception:
        if last_error is not None:
            raise OSError(
                f"Unable to delete directory '{path}'. Close apps using it and retry. Original error: {last_error}"
            ) from last_error
        raise


def auto_split_dataset(
    raw_dir: Path,
    processed_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    overwrite: bool = True,
    verbose: bool = False,
) -> Dict[str, Dict[str, int]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    class_images = find_class_images(raw_dir)
    if not class_images:
        raise ValueError(f"No images found in {raw_dir}")

    if overwrite and processed_dir.exists():
        _robust_rmtree(processed_dir)

    stats: Dict[str, Dict[str, int]] = {split: {} for split in SPLITS}

    total_classes = len(class_images)
    for idx, (class_name, images) in enumerate(class_images.items(), start=1):
        if verbose and (idx == 1 or idx % 10 == 0 or idx == total_classes):
            print(f"[split] class {idx}/{total_classes}: {class_name} ({len(images)} images)")

        train_items, val_items, test_items = _split_list(images, train_ratio, val_ratio, seed)
        per_split = {
            "train": train_items,
            "val": val_items,
            "test": test_items,
        }

        for split, split_items in per_split.items():
            split_class_dir = processed_dir / split / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)
            for src_path in split_items:
                target_name = src_path.name
                target_path = split_class_dir / target_name
                if target_path.exists():
                    target_path = split_class_dir / f"{src_path.stem}_{abs(hash(src_path)) % 10000}{src_path.suffix}"
                shutil.copy2(src_path, target_path)
            stats[split][class_name] = len(split_items)

    return stats


def get_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, eval_transform


def build_dataloaders(processed_dir: Path, image_size: int, batch_size: int, num_workers: int = 0):
    train_transform, eval_transform = get_transforms(image_size)

    train_dataset = datasets.ImageFolder(processed_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(processed_dir / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(processed_dir / "test", transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_dataset.classes


def create_synthetic_dataset(raw_dir: Path, image_size: int = 96, samples_per_class: int = 20) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    classes = {
        "rouge": (220, 40, 40),
        "vert": (40, 180, 60),
    }

    for class_name, color in classes.items():
        class_dir = raw_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            image = Image.new("RGB", (image_size, image_size), color)
            if i % 2 == 0:
                image = ImageOps.expand(image, border=8, fill=(255, 255, 255))
            image.save(class_dir / f"{class_name}_{i:03d}.png")

