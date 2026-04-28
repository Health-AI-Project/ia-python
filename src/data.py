from __future__ import annotations

import random
import shutil
import stat
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def find_class_images(raw_dir: Path, verify_images: bool = True) -> Dict[str, List[Path]]:
    class_images: Dict[str, List[Path]] = defaultdict(list)
    if not raw_dir.exists():
        return {}

    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for file_path in class_dir.rglob("*"):
            if file_path.is_file() and _is_image(file_path):
                if verify_images and not _is_readable_image(file_path):
                    continue
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

    class_images = find_class_images(raw_dir, verify_images=True)
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


def build_transforms(image_size: int, augmentations: bool = True, augmentation_strength: float = 0.2):
    resize_size = max(image_size, int(image_size * 1.15))
    jitter = max(0.0, float(augmentation_strength))

    if augmentations:
        train_transforms = [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.75, 1.0),
                ratio=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(
                brightness=0.2 * jitter,
                contrast=0.2 * jitter,
                saturation=0.2 * jitter,
                hue=0.05 * jitter,
            ),
            transforms.RandomAutocontrast(p=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.05 * jitter, 0.05 * jitter), scale=(0.9, 1.1)),
        ]
    else:
        train_transforms = [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
        ]

    train_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    if augmentations:
        train_transforms.append(
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08), value="random")
        )

    train_transform = transforms.Compose(
        train_transforms
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform


def _validate_processed_split(split_dir: Path) -> list[Path]:
    invalid_files: list[Path] = []
    if not split_dir.exists():
        return invalid_files

    for file_path in split_dir.rglob("*"):
        if file_path.is_file() and _is_image(file_path) and not _is_readable_image(file_path):
            invalid_files.append(file_path)
    return invalid_files


def compute_class_weights(dataset) -> torch.Tensor:
    targets: Sequence[int] = getattr(dataset, "targets", [])
    class_names: Sequence[str] = getattr(dataset, "classes", [])
    if not class_names:
        raise ValueError("Dataset must expose class names to compute class weights")

    counts = Counter(targets)
    total = max(len(targets), 1)
    num_classes = len(class_names)
    weights = []
    for class_index in range(num_classes):
        class_count = counts.get(class_index, 0)
        if class_count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * class_count))
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    processed_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int = 0,
    augmentations: bool = True,
    augmentation_strength: float = 0.2,
):
    invalid_files: list[Path] = []
    for split in SPLITS:
        invalid_files.extend(_validate_processed_split(processed_dir / split))

    if invalid_files:
        sample = "\n".join(f"- {path}" for path in invalid_files[:10])
        raise ValueError(
            "Invalid or corrupted images were found in the processed dataset. "
            f"Remove or regenerate them before training. Examples:\n{sample}"
        )

    train_transform, eval_transform = build_transforms(
        image_size=image_size,
        augmentations=augmentations,
        augmentation_strength=augmentation_strength,
    )

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

