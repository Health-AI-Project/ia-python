from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from tqdm import tqdm


def run_epoch(model: nn.Module, loader, criterion: nn.Module, device: torch.device, optimizer=None) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, targets in tqdm(loader, disable=len(loader) < 2):
        images = images.to(device)
        targets = targets.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, targets)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * targets.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_count += targets.size(0)

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


def save_checkpoint(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device):
    return torch.load(path, map_location=device)

