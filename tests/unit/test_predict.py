from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from PIL import Image
import torch

import predict


class _FakeModel:
    def load_state_dict(self, _state):
        return None

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, _tensor):
        return torch.tensor([[0.1, 2.0, 1.0]])


def test_predict_image_returns_serializable_payload(tmp_path: Path, monkeypatch) -> None:
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (20, 20), (0, 255, 0)).save(img_path)
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"fake-checkpoint")

    monkeypatch.setattr(
        predict,
        "load_checkpoint",
        lambda _path, _device: {
            "class_names": ["pizza", "salad", "sushi"],
            "backbone": "resnet18",
            "model_state": {},
        },
    )
    monkeypatch.setattr(predict, "create_model", lambda *_args, **_kwargs: _FakeModel())

    args = Namespace(checkpoint=checkpoint_path, image=img_path, image_size=64, top_k=2, confidence_threshold=0.9)
    result = predict.predict_image(args)

    assert result["image"] == str(img_path)
    assert result["checkpoint"] == str(tmp_path / "best.pt")
    assert len(result["predictions"]) == 2
    assert result["predictions"][0]["class_name"] == "salad"
    assert result["predictions"][0]["rank"] == 1
    assert result["uncertain"] is True
    assert result["status"] == "uncertain"

