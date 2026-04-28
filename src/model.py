from __future__ import annotations

from typing import Any, Tuple, cast

import torch
from torch import nn
from torchvision import models


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = trainable


def _load_state_dict_flexible(model: nn.Module, state_dict: dict[str, Any]) -> None:
    try:
        model.load_state_dict(state_dict)
        return
    except Exception:
        pass

    remapped = dict(state_dict)
    if "fc.weight" in remapped and "fc.1.weight" not in remapped:
        remapped["fc.1.weight"] = remapped.pop("fc.weight")
    if "fc.bias" in remapped and "fc.1.bias" not in remapped:
        remapped["fc.1.bias"] = remapped.pop("fc.bias")

    if "classifier.3.weight" in remapped and "classifier.3.1.weight" not in remapped:
        remapped["classifier.3.1.weight"] = remapped.pop("classifier.3.weight")
    if "classifier.3.bias" in remapped and "classifier.3.1.bias" not in remapped:
        remapped["classifier.3.1.bias"] = remapped.pop("classifier.3.bias")

    model.load_state_dict(remapped, strict=False)


def create_model(backbone: str, num_classes: int, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
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
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        _set_trainable(cast(nn.Module, model.fc), True)
        return model

    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    try:
        model = models.mobilenet_v3_small(weights=weights)
    except Exception:
        model = models.mobilenet_v3_small(weights=None)

    _set_trainable(model, False)
    in_features = model.classifier[-1].in_features
    classifier_layers = list(model.classifier.children())
    classifier_layers[-1] = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    model.classifier = nn.Sequential(*classifier_layers)
    _set_trainable(cast(nn.Module, model.classifier[-1]), True)
    return model


def unfreeze_for_finetune(model: nn.Module, backbone: str, trainable_layers: int = 1) -> None:
    backbone = backbone.lower()
    trainable_layers = max(1, int(trainable_layers))
    if backbone == "resnet18":
        layers = [model.layer4, model.layer3, model.layer2, model.layer1]
        for block in layers[:trainable_layers]:
            _set_trainable(cast(nn.Module, block), True)
        _set_trainable(cast(nn.Module, model.fc), True)
        return

    if backbone == "mobilenet_v3_small":
        for block in list(model.features.children())[-trainable_layers:]:
            _set_trainable(cast(nn.Module, block), True)
        _set_trainable(cast(nn.Module, model.classifier), True)
        return

    raise ValueError("Unsupported backbone for unfreeze")


def get_trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def load_model_state(model: nn.Module, state_dict: dict[str, Any]) -> None:
    _load_state_dict_flexible(model, state_dict)


def topk_predictions(logits: torch.Tensor, class_names: list[str], top_k: int = 3) -> list[Tuple[str, float]]:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    probs = torch.softmax(logits, dim=1)
    values, indices = torch.topk(probs, k=min(top_k, probs.shape[1]), dim=1)
    output = []
    for score, idx in zip(values[0].tolist(), indices[0].tolist()):
        output.append((class_names[idx], float(score)))
    return output

