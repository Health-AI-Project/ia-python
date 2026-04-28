# Guide detaille des tests PyTest

Ce document explique en detail la suite PyTest du projet: ce qui est teste, pourquoi c'est important, comment lancer les tests, et comment en ajouter proprement.

## 1) Objectif de la suite de tests

La suite actuelle vise un bon compromis entre:

- rapidite d'execution (retour rapide en dev),
- couverture des comportements critiques,
- isolation des dependances lourdes (entrainement, DB, I/O volumineux).

En clair, on teste surtout:

- la logique de split et de validation des ratios,
- la logique de prediction top-k,
- la logique nutritionnelle,
- les helpers API,
- les endpoints FastAPI sur cas normaux et erreurs.

## 2) Organisation des fichiers de test

```text
tests/
  conftest.py
  unit/
    test_data.py
    test_model.py
    test_nutrition.py
    test_predict.py
    test_api_helpers.py
  api/
    test_api_endpoints.py
```

Et la config PyTest globale est dans `pytest.ini`.

## 3) Outils PyTest utilises dans ce projet

- `tmp_path`: cree un espace disque temporaire par test.
- `monkeypatch`: remplace une fonction/objet au runtime.
- `pytest.raises`: verifie qu'une exception attendue est bien levee.
- `fastapi.testclient.TestClient`: teste les endpoints HTTP sans lancer uvicorn.

## 4) Exemples reels de tests (issus du projet)

### 4.1 Test unitaire: split de donnees

Fichier: `tests/unit/test_data.py`

```python
from pathlib import Path

from src.data import _split_list


def test_split_list_keeps_all_items_and_makes_3_splits_when_possible() -> None:
    items = [Path(f"img_{i}.png") for i in range(5)]

    train_items, val_items, test_items = _split_list(items, train_ratio=0.6, val_ratio=0.2, seed=42)

    assert len(train_items) >= 1
    assert len(val_items) >= 1
    assert len(test_items) >= 1
    assert sorted(train_items + val_items + test_items) == sorted(items)
```

Pourquoi ce test est important:

- valide qu'aucune image n'est perdue,
- force la presence des 3 splits quand c'est possible,
- protege contre les regressions de repartition.

### 4.2 Test unitaire avec fichiers temporaires (`tmp_path`)

Fichier: `tests/unit/test_data.py`

```python
from pathlib import Path

from PIL import Image
from src.data import auto_split_dataset


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (120, 20, 20)).save(path)


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
```

Pourquoi ce test est important:

- verifie un flux reel de creation de dataset,
- reste rapide grace a des mini images,
- ne depend pas des donnees `data/raw` du projet.

### 4.3 Test d'erreur attendue (`pytest.raises`)

Fichier: `tests/unit/test_data.py`

```python
import pytest

from src.data import auto_split_dataset


def test_auto_split_dataset_rejects_invalid_ratios(tmp_path):
    with pytest.raises(ValueError, match="must equal 1.0"):
        auto_split_dataset(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.3,
            seed=1,
        )
```

Ce pattern sert a verrouiller les gardes de validation metier.

### 4.4 Test unitaire tensoriel: `topk_predictions`

Fichier: `tests/unit/test_model.py`

```python
import torch

from src.model import topk_predictions


def test_topk_predictions_returns_ordered_pairs() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    classes = ["a", "b", "c"]

    result = topk_predictions(logits, classes, top_k=2)

    assert [name for name, _ in result] == ["b", "c"]
    assert result[0][1] >= result[1][1]
```

Ce test garantit:

- l'ordre decroissant des scores,
- la correspondance score -> classe.

### 4.5 Test unitaire logique metier nutrition

Fichier: `tests/unit/test_nutrition.py`

```python
from src.nutrition import estimate_weighted_calories


def test_weighted_calories_returns_weighted_mean() -> None:
    predictions = [
        {"class_name": "pizza", "score": 0.75},
        {"class_name": "sushi", "score": 0.25},
    ]

    result = estimate_weighted_calories(predictions)

    assert result["method"] == "topk_weighted"
    assert result["estimated_kcal"] == 598.5
```

Ce test protege une regle fonctionnelle tres visible pour l'utilisateur final.

### 4.6 Test unitaire avec `monkeypatch` pour eviter l'inference lourde

Fichier: `tests/unit/test_predict.py`

```python
from argparse import Namespace
from pathlib import Path

from PIL import Image
import predict


class _FakeModel:
    def load_state_dict(self, _state):
        return None

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, _tensor):
        import torch
        return torch.tensor([[0.1, 2.0, 1.0]])


def test_predict_image_returns_serializable_payload(tmp_path: Path, monkeypatch) -> None:
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (20, 20), (0, 255, 0)).save(img_path)

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

    args = Namespace(checkpoint=tmp_path / "best.pt", image=img_path, image_size=64, top_k=2)
    result = predict.predict_image(args)

    assert result["predictions"][0]["class_name"] == "salad"
```

Ce test est cle car il valide le format de sortie de prediction sans charger de vrai checkpoint.

### 4.7 Test helper API: conversion en `argparse.Namespace`

Fichier: `tests/unit/test_api_helpers.py`

```python
from pathlib import Path

import api


def test_to_namespace_converts_path_fields() -> None:
    payload = {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "models_dir": "models",
        "checkpoint": "models/best.pt",
        "image": "img.png",
        "epochs": 3,
    }

    ns = api._to_namespace(payload)

    assert isinstance(ns.raw_dir, Path)
    assert isinstance(ns.processed_dir, Path)
    assert isinstance(ns.models_dir, Path)
    assert isinstance(ns.checkpoint, Path)
    assert isinstance(ns.image, Path)
    assert ns.epochs == 3
```

Ce test evite les regressions silencieuses entre payload JSON et fonctions CLI internes.

### 4.8 Test API endpoint FastAPI

Fichier: `tests/api/test_api_endpoints.py`

```python
from fastapi.testclient import TestClient

import api


def test_predict_path_returns_404_when_image_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api, "_try_initialize_database", lambda: False)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"ok")

    with TestClient(api.app) as client:
        response = client.post(
            "/predict/path",
            json={
                "checkpoint": str(checkpoint),
                "image": str(tmp_path / "missing.png"),
                "image_size": 224,
                "top_k": 3,
            },
        )

    assert response.status_code == 404
    assert "Image not found" in response.json()["detail"]
```

Ce test verrouille le contrat HTTP en cas de fichier absent.

## 5) Commandes pour executer les tests

Depuis la racine du projet:

```powershell
python -m pytest
```

Uniquement les tests unitaires:

```powershell
python -m pytest tests/unit
```

Avec couverture:

```powershell
python -m pytest --cov=src --cov=api --cov=predict --cov-report=term-missing
```

Pipeline locale de verification rapide:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\ci_local.ps1
```

## 6) Lire les resultats PyTest

- `.` : test passe.
- `F` : echec d'assertion.
- `E` : erreur technique non attendue.
- `14 passed` : tous les tests de la suite actuelle sont OK.

Pour la couverture:

- la colonne `Miss` montre les lignes non executees,
- la colonne `Cover` donne le taux par fichier,
- `term-missing` liste les numeros de lignes manquantes.

## 7) Strategie pour ajouter de nouveaux tests

Checklist recommandee:

1. Choisir un comportement precis (une regle metier ou un contrat API).
2. Creer un test `test_<comportement>.py` dans `tests/unit` ou `tests/api`.
3. Structurer en Arrange / Act / Assert.
4. Mocker tout ce qui est lourd (DB, gros modeles, I/O externes).
5. Valider localement avec `python -m pytest`.

Exemple de squelette minimal:

```python
def test_nom_du_comportement(monkeypatch, tmp_path):
    # Arrange
    # Act
    # Assert
    assert True
```

## 8) Bonnes pratiques appliquees dans ce projet

- Tests deterministes (pas de hasard non controle).
- Donnees synthetiques minimales.
- Aucune dependance a un service distant pour passer la suite.
- Assertions sur le contrat fonctionnel (contenu et statut HTTP), pas seulement sur l'absence d'erreur.
- Priorite aux chemins d'erreur utiles utilisateur (`404`, `422`, validation ratio).

## 9) Limites actuelles de la suite

Actuellement, la suite ne couvre pas encore en profondeur:

- la boucle d'entrainement de `src/engine.py`,
- les chemins DB reels (`psycopg`) avec une vraie base,
- certains chemins de `api.py` comme `/predict/upload` et `/feedback` complets.

Ce n'est pas bloquant pour une base solide de rendu, mais c'est la prochaine priorite pour monter la couverture et la robustesse.

