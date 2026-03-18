from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-4
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    seed: int = 42
    num_workers: int = 0
    backbone: str = "resnet18"
    pretrained: bool = True
    unfreeze_epoch: int = 1


@dataclass
class EvalConfig:
    processed_dir: Path
    checkpoint_path: Path
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 0


@dataclass
class PredictConfig:
    checkpoint_path: Path
    image_path: Path
    image_size: int = 224
    top_k: int = 3

