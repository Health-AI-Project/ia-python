# Local image transfer learning

Documentation complète d’un projet de classification d’images entraîné en local, avec transfer learning, exécution sur CPU uniquement, scripts CLI, API FastAPI, interface Streamlit et estimation locale des calories.

## Sommaire

- [Vue d’ensemble](#vue-densemble)
- [Architecture du projet](#architecture-du-projet)
- [Organisation des fichiers](#organisation-des-fichiers)
- Données et format attendu
- [Comment fonctionne le pipeline](#comment-fonctionne-le-pipeline)
- [Installation](#installation)
- [Quick check](#quick-check)
- Entraînement
- Évaluation
- Prédiction en ligne de commande
- [API FastAPI](#api-fastapi)
- [Interface Streamlit](#interface-streamlit)
- Système de feedback
- [Estimation des calories](#estimation-des-calories)
- [Exemples de code Python](#exemples-de-code-python)
- Fichiers générés
- [Limites et points d’attention](#limites-et-points-dattention)

## Vue d’ensemble

Ce projet entraîne localement un classifieur d’images de nourriture à partir d’un dossier d’images classées par répertoire. Il utilise du **transfer learning** avec `resnet18` par défaut, et peut aussi fonctionner avec `mobilenet_v3_small`.

L’idée générale est simple :

1. vous placez vos images dans `data/raw/<classe>/...` ;
2. le projet crée automatiquement un split `train / val / test` dans `data/processed/` ;
3. le modèle est entraîné sur CPU ;
4. un checkpoint est sauvegardé dans `models/` ;
5. vous pouvez évaluer, prédire, exposer une API, ou utiliser l’interface Streamlit ;
6. les prédictions peuvent être associées à une estimation locale de calories ;
7. un feedback utilisateur peut être enregistré pour analyse ou réentraînement ultérieur.

Le projet ne dépend pas d’un service d’IA externe. Tout est exécuté localement.

---

## Architecture du projet

```text
data/raw/<classe>/image.jpg
        │
        ├─> split automatique
        │
data/processed/train/<classe>/...
data/processed/val/<classe>/...
data/processed/test/<classe>/...
        │
        ├─> entraînement (`train.py`)
        │
models/best.pt
models/last.pt
models/history.json
        │
        ├─> évaluation (`evaluate.py`)
        ├─> prédiction (`predict.py`)
        ├─> API FastAPI (`api.py`)
        └─> interface Streamlit (`app.py`)
```

### Rôle des composants

- `train.py` : orchestre la création du split, l’entraînement, le fine-tuning et la sauvegarde des checkpoints.
- `evaluate.py` : charge un checkpoint et calcule les métriques sur le split `test`.
- `predict.py` : prédit une image unique et renvoie le top-k de classes.
- `api.py` : expose les mêmes capacités via FastAPI, avec un système de `prediction_id`, de cache de prédictions et de feedback.
- `app.py` : interface Streamlit interactive pour tester le modèle et enregistrer des retours utilisateur.
- `src/data.py` : logique de split, chargement des datasets et transformations d’images.
- `src/model.py` : création du backbone, remplacement de la tête de classification, dégel partiel pour le fine-tuning.
- `src/engine.py` : boucle d’époque, sauvegarde/chargement des checkpoints.
- `src/nutrition.py` : estimation locale des calories par classe et pondération top-k.
- `src/config.py` : conteneurs de configuration typés pour l’entraînement, l’évaluation et la prédiction.

---

## Organisation des fichiers

### Racine du projet

- `train.py` : point d’entrée CLI pour l’entraînement.
- `evaluate.py` : point d’entrée CLI pour l’évaluation.
- `predict.py` : point d’entrée CLI pour la prédiction sur une image.
- `api.py` : serveur FastAPI.
- `app.py` : application Streamlit.
- `quick_check.py` : test rapide de bout en bout avec données synthétiques.
- `requirements.txt` : dépendances Python.
- `README.md` : cette documentation.

### Répertoires de données et modèles

- `data/raw/` : images brutes organisées par classe.
- `data/processed/` : split généré automatiquement.
- `models/` : checkpoints et historique d’entraînement.
- `feedback_log.json` : historique local généré par l’interface Streamlit.
- `models/feedback_log.jsonl` : journal de feedback généré par l’API.

### Répertoire `src/`

- `src/config.py` : dataclasses de configuration.
- `src/data.py` : lecture des images, split, transformations, dataloaders, génération de données de démonstration.
- `src/model.py` : modèles supportés et logique de fine-tuning.
- `src/engine.py` : entraînement, évaluation et I/O des checkpoints.
- `src/nutrition.py` : table de référence des calories et estimateurs.

---

<a id="donnees-et-format-attendu"></a>
## Données et format attendu

Le dataset doit être organisé par classe, directement dans `data/raw/` :

```text
data/raw/
  pizza/
    img_001.jpg
    img_002.png
  sushi/
    s1.jpg
    s2.jpeg
```

### Extensions d’images acceptées

Le projet détecte les fichiers ayant l’une de ces extensions :

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.webp`

### Règles importantes

- Chaque sous-dossier de `data/raw/` correspond à une classe.
- Le nom du dossier devient le nom de classe utilisé par le modèle.
- Le split est effectué classe par classe, avec un ordre déterministe des fichiers.
- Le projet prévoit un comportement robuste même avec peu d’images, mais chaque classe doit idéalement contenir au moins 3 images pour un split train/val/test stable.

### Structure générée

Après split, le projet crée automatiquement :

```text
data/processed/
  train/<classe>/...
  val/<classe>/...
  test/<classe>/...
```

Chaque image d’origine est copiée dans un seul split.

---

## Comment fonctionne le pipeline

### 1. Détection des classes et des images

`src/data.py` parcourt `data/raw/`, détecte les sous-dossiers de classe, puis collecte les images valides dans chaque dossier.

### 2. Split automatique

La fonction `auto_split_dataset()` :

- vérifie que la somme `train_ratio + val_ratio + test_ratio == 1.0` ;
- mélange les images de manière reproductible via une seed ;
- répartit les fichiers dans `train`, `val` et `test` ;
- recrée `data/processed/` si `overwrite=True`.

### 3. Chargement des datasets

`build_dataloaders()` utilise `torchvision.datasets.ImageFolder` :

- `train` applique des augmentations légères :
  - redimensionnement
  - flip horizontal aléatoire
  - `ColorJitter`
  - normalisation ImageNet
- `val` et `test` utilisent un pipeline plus simple, sans augmentation, pour des mesures propres.

### 4. Création du modèle

`src/model.py` instancie un backbone pré-entraîné ou non :

- `resnet18`
- `mobilenet_v3_small`

Les couches du backbone sont gelées au départ, puis la tête de classification est remplacée pour correspondre au nombre de classes du dataset.

### 5. Entraînement et fine-tuning

`train.py` entraîne d’abord la tête de classification. Ensuite, à partir de l’époque définie par `--unfreeze-epoch`, une partie plus profonde du backbone est dégelée pour du fine-tuning.

### 6. Sauvegarde

Le projet écrit :

- `models/last.pt` : dernier checkpoint
- `models/best.pt` : meilleur checkpoint selon l’accuracy validation
- `models/history.json` : historique des métriques par époque

### 7. Prédiction et calories

Lors d’une prédiction :

- `predict.py` renvoie les top-k probabilités ;
- `api.py` enrichit la réponse avec un identifiant de prédiction et deux estimations de calories ;
- `src/nutrition.py` calcule l’estimation top-1 et l’estimation pondérée top-k.

---

## Installation

### Pré-requis

- Windows PowerShell
- Python 3.11+ recommandé
- Un environnement virtuel Python est fortement conseillé

### Installation de base

```powershell
Set-Location "C:\Users\pierr\OneDrive\Bureau\cours epsi\SN3\MSPR\IA"
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Si vous voulez lancer l’interface Streamlit

`app.py` utilise Streamlit, installé via `requirements.streamlit.txt` :

```powershell
python -m pip install -r requirements.streamlit.txt
```

### Si vous voulez lancer l’API

Le serveur FastAPI repose sur `fastapi`, `uvicorn` et `python-multipart`, déjà présents dans `requirements.txt`.

### Déploiement Docker

Le projet fournit deux images Docker :

- `Dockerfile` : API FastAPI
- `Dockerfile.streamlit` : interface Streamlit

Les dépendances sont séparées pour accélérer les builds :

- `requirements.api.txt` : dépendances API (sans Streamlit)
- `requirements.streamlit.txt` : dépendances UI (inclut API + Streamlit)

Un `docker-compose.yml` est aussi fourni pour lancer les deux services.

#### Construire et lancer

```powershell
docker compose up --build
```

Si vous déployez seulement l’API (ex: Dokploy avec `Dockerfile`) :

```powershell
docker build -f Dockerfile -t ia-api .
docker run --rm -p 8000:8000 --env DATABASE_URL="postgresql://postgres:<mot_de_passe>@<hote>:5432/postgres" ia-api
```

#### Variables d’environnement PostgreSQL

L’API doit connaître la base PostgreSQL. Vous pouvez utiliser soit `DATABASE_URL`, soit les variables `DB_*`.

```powershell
$env:DATABASE_URL="postgresql://postgres:<mot_de_passe>@<hote>:5432/postgres"
```

ou :

```powershell
$env:DB_USER="postgres"
$env:DB_PASSWORD="<mot_de_passe>"
$env:DB_HOST="<hote>"
$env:DB_PORT="5432"
$env:DB_NAME="postgres"
```

#### Ports exposés

- API : `http://localhost:8000`
- Streamlit : `http://localhost:8501`

---

## Quick check

Le script `quick_check.py` sert de smoke test local. Il :

1. supprime `data/_quickcheck/` s’il existe ;
2. génère un petit dataset synthétique avec deux classes (`rouge`, `vert`) ;
3. crée un split ;
4. construit les dataloaders ;
5. lance une époque d’entraînement et une validation ;
6. affiche `QUICK_CHECK_OK` si tout s’est bien passé.

### Commande

```powershell
python quick_check.py
```

### Sortie attendue

```text
QUICK_CHECK_OK
```

Ce test est utile pour vérifier rapidement que PyTorch, torchvision, PIL et le pipeline local fonctionnent correctement.

---

<a id="entrainement"></a>
## Entraînement

### Commande de base

```powershell
python train.py --raw-dir data/raw --processed-dir data/processed --models-dir models --epochs 5 --batch-size 16 --backbone resnet18
```

### Ce que fait `train.py`

- lit `data/raw/` ;
- crée le split si nécessaire ;
- charge les dataloaders ;
- initialise le backbone ;
- entraîne le modèle sur CPU ;
- enregistre `best.pt`, `last.pt` et `history.json`.

### Arguments CLI disponibles

- `--raw-dir` : dossier source des images brutes.
- `--processed-dir` : dossier de split généré.
- `--models-dir` : dossier de sauvegarde des checkpoints.
- `--image-size` : taille d’entrée du modèle.
- `--batch-size` : taille de batch.
- `--epochs` : nombre d’époques.
- `--learning-rate` : taux d’apprentissage initial.
- `--fine-tune-learning-rate` : taux d’apprentissage après dégel.
- `--train-ratio`, `--val-ratio`, `--test-ratio` : ratios de split.
- `--seed` : seed de reproductibilité.
- `--num-workers` : nombre de workers DataLoader.
- `--backbone` : `resnet18` ou `mobilenet_v3_small`.
- `--no-pretrained` : désactive les poids ImageNet.
- `--unfreeze-epoch` : époque à partir de laquelle une partie du backbone est dégelée.
- `--skip-split` : réutilise un split déjà présent dans `data/processed/`.
- `--bootstrap-demo-data` : génère un mini dataset synthétique si `data/raw/` est vide.
- `--run-forever` : boucle d’entraînement infinie jusqu’à `Ctrl+C`.

### Exemple d’entraînement complet

```powershell
python train.py `
  --raw-dir data/raw `
  --processed-dir data/processed `
  --models-dir models `
  --epochs 10 `
  --batch-size 16 `
  --backbone mobilenet_v3_small `
  --unfreeze-epoch 2
```

### Avec génération de données de démonstration

```powershell
python train.py --bootstrap-demo-data --epochs 3 --batch-size 8
```

### Avec split déjà préparé

```powershell
python train.py --skip-split --processed-dir data/processed --models-dir models
```

### Détails techniques de l’entraînement

- Tout s’exécute sur `torch.device("cpu")`.
- Le critère utilisé est `CrossEntropyLoss`.
- L’optimiseur est `Adam`.
- Les paramètres entraînables au départ sont ceux de la tête ajoutée sur le backbone.
- Au moment du fine-tuning, `resnet18` dégage `layer4` et la couche finale, tandis que `mobilenet_v3_small` dégage les derniers blocs convolutifs et le classifieur.

### Historique de l’entraînement

`models/history.json` contient une liste d’objets de ce type :

```json
[
  {
    "epoch": 1,
    "train_loss": 1.234,
    "train_acc": 0.42,
    "val_loss": 1.101,
    "val_acc": 0.55
  }
]
```

---

<a id="evaluation"></a>
## Évaluation

### Commande

```powershell
python evaluate.py --processed-dir data/processed --checkpoint models/best.pt
```

### Ce que fait `evaluate.py`

- charge le checkpoint choisi ;
- reconstruit le modèle avec le bon backbone et le bon nombre de classes ;
- lit le split `test` ;
- calcule la loss et l’accuracy ;
- affiche un dictionnaire Python contenant :
  - `test_loss`
  - `test_acc`
  - `classes`

### Exemple de sortie

```text
{'test_loss': 0.8421, 'test_acc': 0.73, 'classes': ['pizza', 'sushi', 'salad']}
```

### Arguments CLI

- `--processed-dir`
- `--checkpoint`
- `--image-size`
- `--batch-size`
- `--num-workers`

---

<a id="prediction-en-ligne-de-commande"></a>
## Prédiction en ligne de commande

### Commande

```powershell
python predict.py --checkpoint models/best.pt --image data/raw/pizza/img1.jpg --top-k 3
```

### Ce que fait `predict.py`

- charge le checkpoint ;
- reconstruit le modèle ;
- charge l’image ;
- applique la transformation standard ;
- calcule les probabilités ;
- affiche le top-k.

### Format de sortie

La fonction renvoie un dictionnaire avec :

- `image` : chemin de l’image
- `checkpoint` : chemin du modèle
- `predictions` : liste ordonnée de prédictions

Chaque prédiction a la forme `{"class_name": "pizza", "score": 0.83}`.

### Exemple d’utilisation depuis Python

```python
from pathlib import Path
from argparse import Namespace
from predict import predict_image

args = Namespace(
    checkpoint=Path("models/best.pt"),
    image=Path("data/raw/pizza/img1.jpg"),
    image_size=224,
    top_k=3,
)

result = predict_image(args)
print(result["predictions"])
```

---

## API FastAPI

L’API expose les mêmes capacités que les scripts CLI, avec des endpoints HTTP simples.

### Lancement du serveur

```powershell
uvicorn api:app --host 127.0.0.1 --port 8000
```

### Vérifier l’état

```powershell
Invoke-RestMethod -Method GET -Uri "http://127.0.0.1:8000/health"
```

Réponse attendue :

```json
{
  "status": "ok",
  "device": "cpu"
}
```

### Vérifier un checkpoint

```powershell
Invoke-RestMethod -Method GET -Uri "http://127.0.0.1:8000/model/status?checkpoint=models/best.pt"
```

Réponse si le modèle est disponible :

```json
{
  "ready": true,
  "checkpoint": "models/best.pt",
  "backbone": "resnet18",
  "epoch": 5,
  "image_size": 224,
  "num_classes": 42,
  "class_names": ["pizza", "sushi"]
}
```

Si le checkpoint est absent, la réponse indique :

```json
{
  "ready": false,
  "checkpoint": "models/best.pt",
  "reason": "checkpoint_not_found"
}
```

### Endpoints disponibles

#### `GET /health`

Retourne l’état du serveur.

#### `GET /model/status`

Paramètre de requête : `checkpoint`.

Retourne les métadonnées du modèle sauvegardé.

#### `POST /train`

Déclenche un entraînement via JSON.

Exemple PowerShell :

```powershell
$train = @{
  raw_dir = "data/raw"
  processed_dir = "data/processed"
  models_dir = "models"
  image_size = 224
  batch_size = 16
  epochs = 5
  learning_rate = 0.001
  fine_tune_learning_rate = 0.0001
  train_ratio = 0.7
  val_ratio = 0.2
  test_ratio = 0.1
  seed = 42
  num_workers = 0
  backbone = "resnet18"
  no_pretrained = $false
  unfreeze_epoch = 1
  skip_split = $false
  bootstrap_demo_data = $false
} | ConvertTo-Json

Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/train" -ContentType "application/json" -Body $train
```

Règle importante : la somme des ratios doit valoir `1.0`.

#### `POST /evaluate`

Évalue un checkpoint sur le split `test`.

```powershell
$eval = @{
  processed_dir = "data/processed"
  checkpoint = "models/best.pt"
  image_size = 224
  batch_size = 16
  num_workers = 0
} | ConvertTo-Json

Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/evaluate" -ContentType "application/json" -Body $eval
```

#### `POST /predict/path`

Prédit une image locale à partir d’un chemin.

```powershell
$body = @{
  checkpoint = "models/best.pt"
  image = "data/raw/pizza/img1.jpg"
  image_size = 224
  top_k = 3
} | ConvertTo-Json

Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/predict/path" -ContentType "application/json" -Body $body
```

#### `POST /predict/upload`

Prédit une image envoyée en multipart/form-data.

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict/upload" `
  -F "file=@data/raw/pizza/img1.jpg" `
  -F "checkpoint=models/best.pt" `
  -F "image_size=224" `
  -F "top_k=3"
```

#### `POST /feedback`

Enregistre un retour utilisateur sur une prédiction déjà faite.

```powershell
$feedback = @{
  prediction_id = "<ID_REÇU_DU_ENDPOINT_PREDICT>"
  is_correct = $false
  correct_class = "pizza"
} | ConvertTo-Json

Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/feedback" -ContentType "application/json" -Body $feedback
```

### Format de réponse enrichie de prédiction

Les endpoints `/predict/path` et `/predict/upload` renvoient une réponse enrichie qui contient :

- `prediction_id` : identifiant unique UUID généré pour la prédiction ;
- `created_at` : horodatage UTC ISO 8601 ;
- `image` : chemin de l’image utilisée ;
- `checkpoint` : chemin du checkpoint ;
- `predictions` : liste top-k de prédictions ;
- `top_prediction` : meilleure classe ;
- `calories.top1` : estimation locale basée sur la classe la plus probable ;
- `calories.weighted_topk` : estimation pondérée sur le top-k.

Exemple de structure :

```json
{
  "prediction_id": "...",
  "created_at": "2026-04-07T12:34:56.000000+00:00",
  "image": "data/raw/pizza/img1.jpg",
  "checkpoint": "models/best.pt",
  "predictions": [
    {"class_name": "pizza", "score": 0.83},
    {"class_name": "lasagna", "score": 0.09}
  ],
  "top_prediction": {"class_name": "pizza", "score": 0.83},
  "calories": {
    "top1": {
      "class_name": "pizza",
      "kcal_per_100g": 266.0,
      "portion_g": 260.0,
      "estimated_kcal": 691.6,
      "estimated_kcal_min": 518.7,
      "estimated_kcal_max": 864.5,
      "source": "local_reference_v1"
    },
    "weighted_topk": {
      "estimated_kcal": 660.2,
      "estimated_kcal_min": 495.2,
      "estimated_kcal_max": 825.3,
      "source": "local_reference_v1",
      "method": "topk_weighted"
    }
  }
}
```

### Logique interne de l’API

- Les requêtes passent par des modèles Pydantic.
- Le CPU est utilisé systématiquement.
- Les images uploadées sont temporairement écrites sur disque puis supprimées après traitement.
- Les prédictions sont mises en cache en mémoire dans `PREDICTION_CACHE` pour permettre le feedback.
- Le feedback est ajouté à `models/feedback_log.jsonl` sous forme JSON Lines.

---

## Interface Streamlit

`app.py` fournit une interface locale interactive pour explorer le modèle.

### Lancement

```powershell
streamlit run app.py
```

### Ce que permet l’interface

- charger un checkpoint local ;
- choisir un top-k à afficher ;
- uploader une image ou choisir une image de démonstration depuis `data/raw/` ;
- visualiser les prédictions ;
- donner un feedback « correct » ou « faux » ;
- consulter un petit historique local de feedbacks.

### Différence avec l’API

L’interface Streamlit est pensée pour une utilisation humaine rapide. Elle stocke son feedback dans un fichier JSON local via le bouton de sauvegarde.

L’API, elle, garde un `prediction_id` et écrit le feedback dans `models/feedback_log.jsonl` pour un usage plus structuré et automatisable.

### Format de feedback Streamlit

Le feedback Streamlit garde des entrées du type :

```json
[
  {
    "type": "correct",
    "predicted_class": "pizza",
    "confidence": 0.83
  },
  {
    "type": "incorrect",
    "predicted_class": "burger",
    "correct_class": "pizza",
    "confidence": 0.61
  }
]
```

Le fichier est enregistré localement dans `feedback_log.json` à la racine du projet.

---

<a id="systeme-de-feedback"></a>
## Système de feedback

Le projet prévoit deux circuits distincts de feedback.

### 1. Feedback via l’API

Quand une prédiction est renvoyée par l’API :

- un `prediction_id` UUID est généré ;
- la prédiction est placée en cache mémoire ;
- l’utilisateur renvoie ce `prediction_id` sur `/feedback` ;
- l’API vérifie si la prédiction existe dans le cache ;
- le feedback est écrit dans `models/feedback_log.jsonl`.

#### Cas possibles

- `is_correct = true` : la classe finale est la classe prédite.
- `is_correct = false` : `correct_class` devient obligatoire.

### 2. Feedback via Streamlit

Dans `app.py`, le feedback est purement local à l’interface :

- la session garde une liste en mémoire ;
- l’utilisateur peut sauvegarder dans `feedback_log.json` ;
- il n’y a pas de `prediction_id`.

### Format du journal API

Chaque ligne de `models/feedback_log.jsonl` contient un objet JSON indépendant :

```json
{"timestamp":"2026-04-07T12:34:56+00:00","prediction_id":"...","checkpoint":"models/best.pt","is_correct":false,"predicted_class":"burger","final_class":"pizza","calories":{"...": "..."}}
```

---

## Estimation des calories

La logique de calories se trouve dans `src/nutrition.py`.

### Principe

Le projet n’utilise pas de base nutritionnelle distante. Il maintient une table locale de référence par classe, avec :

- `kcal_per_100g`
- `portion_g`

L’estimation renvoie ensuite :

- une estimation centrale `estimated_kcal` ;
- une fourchette `estimated_kcal_min` / `estimated_kcal_max` ;
- une source `local_reference_v1`.

### Estimation top-1

`estimate_calories_for_class(class_name)` calcule les calories pour une classe donnée.

### Estimation pondérée top-k

`estimate_weighted_calories(predictions)` combine les classes du top-k en pondérant chaque estimation par son score de probabilité.

### Valeurs par défaut

Si une classe n’est pas présente dans la table locale, le projet utilise :

- `DEFAULT_KCAL_PER_100G = 220.0`
- `DEFAULT_PORTION_G = 220.0`
- `DEFAULT_VARIANCE_RATIO = 0.25`

### Important

Ces valeurs sont des **estimations indicatives** pour un MVP, pas des données nutritionnelles médicalement validées.

---

## Exemples de code Python

### Charger un checkpoint et inspecter ses métadonnées

```python
from pathlib import Path
import torch
from src.engine import load_checkpoint

checkpoint = load_checkpoint(Path("models/best.pt"), torch.device("cpu"))
print(checkpoint["backbone"])
print(checkpoint["class_names"])
```

### Créer un modèle à partir du backbone sauvegardé

```python
from src.model import create_model

model = create_model("resnet18", num_classes=10, pretrained=False)
```

### Estimer les calories d’une classe

```python
from src.nutrition import estimate_calories_for_class

info = estimate_calories_for_class("pizza")
print(info["estimated_kcal"])
```

### Estimer les calories à partir d’un top-k

```python
from src.nutrition import estimate_weighted_calories

predictions = [
    {"class_name": "pizza", "score": 0.8},
    {"class_name": "lasagna", "score": 0.2},
]

print(estimate_weighted_calories(predictions))
```

### Lancer un entraînement depuis un script Python

```python
from argparse import Namespace
from pathlib import Path
from train import train_model

args = Namespace(
    raw_dir=Path("data/raw"),
    processed_dir=Path("data/processed"),
    models_dir=Path("models"),
    image_size=224,
    batch_size=16,
    epochs=5,
    learning_rate=1e-3,
    fine_tune_learning_rate=1e-4,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42,
    num_workers=0,
    backbone="resnet18",
    no_pretrained=False,
    unfreeze_epoch=1,
    skip_split=False,
    bootstrap_demo_data=False,
    run_forever=False,
)

result = train_model(args)
print(result["best_checkpoint"])
```

---

<a id="fichiers-generes"></a>
## Fichiers générés

### Après entraînement

- `models/best.pt` : meilleur checkpoint selon la validation.
- `models/last.pt` : dernier checkpoint enregistré.
- `models/history.json` : historique des métriques.

### Après usage de l’API

- `models/feedback_log.jsonl` : feedback enrichi par `prediction_id`.

### Après usage de Streamlit

- `feedback_log.json` : historique local du flux Streamlit.

### Après quick check

- `data/_quickcheck/raw/`
- `data/_quickcheck/processed/`

Ces dossiers sont recréés à chaque exécution du smoke test.

---

## Limites et points d’attention

- L’exécution est volontairement **CPU only**.
- Le chargement de poids pré-entraînés peut échouer si le téléchargement de torchvision n’est pas possible ; le code retombe alors sur une initialisation aléatoire.
- Les estimations de calories sont locales, heuristiques et indicatives.
- L’API maintient un cache mémoire des prédictions : si le processus redémarre, les anciens `prediction_id` ne sont plus valides.
- `app.py` nécessite Streamlit, mais la dépendance n’est pas listée dans `requirements.txt`.
- Le split des données est copié dans `data/processed/` ; si vous modifiez `data/raw/`, relancez le split ou l’entraînement.

---

## Raccourci des commandes utiles

```powershell
# Installer les dépendances
python -m pip install -r requirements.txt

# Lancer un entraînement
python train.py --raw-dir data/raw --processed-dir data/processed --models-dir models --epochs 5

# Évaluer le meilleur modèle
python evaluate.py --processed-dir data/processed --checkpoint models/best.pt

# Prédire une image
python predict.py --checkpoint models/best.pt --image data/raw/pizza/img1.jpg --top-k 3

# Tester le pipeline rapidement
python quick_check.py

cd "C:\Users\pierr\OneDrive\Bureau\cours epsi\SN3\MSPR\IA" ; .\.venv\Scripts\Activate.ps1 ; pip install python-multipart

# Démarrer l’API
uvicorn api:app --host 127.0.0.1 --port 8000

# Démarrer l’interface Streamlit
streamlit run app.py
```

---

## En résumé

Ce projet fournit un pipeline complet de classification d’images local : préparation des données, entraînement, évaluation, prédiction, API, interface web et feedback utilisateur. La logique est volontairement simple, robuste et facile à étendre, tout en restant entièrement exécutable sur une machine locale sans dépendance à un service distant.