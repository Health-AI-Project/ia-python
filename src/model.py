from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torchvision import models


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = trainable


def create_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    backbone = backbone.lower()
    if backbone not in {"resnet18", "mobilenet_v3_small"}:
        raise ValueError("Unsupported backbone. Use 'resnet18' or 'mobilenet_v3_small'.")

    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        try:
            model = models.resnet18(weights=weights)
        except Exception:
            model = models.resnet18(weights=None)

        _set_trainable(model, False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        _set_trainable(model.fc, True)
        return model

    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    try:
        model = models.mobilenet_v3_small(weights=weights)
    except Exception:
        model = models.mobilenet_v3_small(weights=None)

    _set_trainable(model, False)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    _set_trainable(model.classifier[-1], True)
    return model


def unfreeze_for_finetune(model: nn.Module, backbone: str) -> None:
    backbone = backbone.lower()
    if backbone == "resnet18":
        _set_trainable(model.layer4, True)
        _set_trainable(model.fc, True)
        return

    if backbone == "mobilenet_v3_small":
        for block in list(model.features.children())[-2:]:
            _set_trainable(block, True)
        _set_trainable(model.classifier, True)
        return

    raise ValueError("Unsupported backbone for unfreeze")


def get_trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def topk_predictions(logits: torch.Tensor, class_names: list[str], top_k: int = 3) -> list[Tuple[str, float]]:
    probs = torch.softmax(logits, dim=1)
    values, indices = torch.topk(probs, k=min(top_k, probs.shape[1]), dim=1)
    output = []
    for score, idx in zip(values[0].tolist(), indices[0].tolist()):
        output.append((class_names[idx], float(score)))
    return output

