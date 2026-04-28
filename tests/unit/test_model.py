from __future__ import annotations

import torch
from torch import nn

from src.model import create_model, topk_predictions, unfreeze_for_finetune


def test_topk_predictions_returns_ordered_pairs() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    classes = ["a", "b", "c"]

    result = topk_predictions(logits, classes, top_k=2)

    assert [name for name, _ in result] == ["b", "c"]
    assert result[0][1] >= result[1][1]


def test_topk_predictions_caps_top_k_to_num_classes() -> None:
    logits = torch.tensor([[0.1, 0.2]])

    result = topk_predictions(logits, ["x", "y"], top_k=5)

    assert len(result) == 2


def test_create_model_uses_dropout_head_for_resnet() -> None:
    model = create_model("resnet18", num_classes=3, pretrained=False, dropout=0.35)

    assert isinstance(model.fc, nn.Sequential)
    assert isinstance(model.fc[0], nn.Dropout)
    assert model.fc[0].p == 0.35


def test_unfreeze_for_finetune_unfreezes_last_resnet_block() -> None:
    model = create_model("resnet18", num_classes=3, pretrained=False)

    unfreeze_for_finetune(model, "resnet18", trainable_layers=1)

    assert any(param.requires_grad for param in model.layer4.parameters())
    assert all(param.requires_grad for param in model.fc.parameters())


