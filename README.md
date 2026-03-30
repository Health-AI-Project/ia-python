# Local image transfer learning (CPU only)

This project trains an image classifier locally with transfer learning, no external API service.

## What it does

- Reads your images from `data/raw/<class_name>/*.jpg|png|...`
- Creates an automatic stratified split into:
  - `data/processed/train/<class_name>/...`
  - `data/processed/val/<class_name>/...`
  - `data/processed/test/<class_name>/...`
- Trains a transfer learning model (`resnet18` by default) on CPU only
- Saves checkpoints in `models/`
- Supports evaluation and single-image prediction

## Dataset format

Put your images here:

```text
data/raw/
  cats/
    img1.jpg
    img2.jpg
  dogs/
    dog1.png
    dog2.png
```

Each class must have at least 3 images for stable split behavior.

## Install

```powershell
Set-Location "C:\Users\pierr\OneDrive\Bureau\cours epsi\SN3\MSPR\IA"
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick smoke test (creates synthetic data)

```powershell
python quick_check.py
```

Expected final line:

```text
QUICK_CHECK_OK
```

## Train on your own images

```powershell
python train.py --raw-dir data/raw --processed-dir data/processed --models-dir models --epochs 5 --batch-size 16 --backbone resnet18
```

Useful options:

- `--backbone resnet18` or `--backbone mobilenet_v3_small`
- `--no-pretrained` to disable ImageNet weights
- `--skip-split` if `data/processed` already exists
- `--unfreeze-epoch 1` to start fine-tuning deeper layers after epoch 1
- `--bootstrap-demo-data` to auto-generate a tiny synthetic dataset when `data/raw` is empty
- `--run-forever` to train in an infinite loop until you stop with `Ctrl+C`

## Evaluate

```powershell
python evaluate.py --processed-dir data/processed --checkpoint models/best.pt
```

## Predict a single image

```powershell
python predict.py --checkpoint models/best.pt --image data/raw/cats/img1.jpg --top-k 3
```

## Run as API (FastAPI)

Start server:

```powershell
uvicorn api:app --host 127.0.0.1 --port 8000
```

Health check:

```powershell
Invoke-RestMethod -Method GET -Uri "http://127.0.0.1:8000/health"
```

Predict from local image path:

```powershell
$body = @{
  checkpoint = "models/best.pt"
  image = "data/raw/cats/img1.jpg"
  image_size = 224
  top_k = 3
} | ConvertTo-Json

Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/predict/path" -ContentType "application/json" -Body $body
```

Predict from uploaded file:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict/upload" -F "file=@data/raw/cats/img1.jpg" -F "checkpoint=models/best.pt" -F "top_k=3"
```

The prediction response now includes:

- `prediction_id` (to send user feedback)
- `top_prediction` and `predictions`
- `calories.top1` and `calories.weighted_topk` (estimated calories)

Send user feedback (correct/incorrect) and get adjusted calories:

```powershell
$feedback = @{
  prediction_id = "<ID_FROM_PREDICT_UPLOAD_OR_PATH>"
  is_correct = $false
  correct_class = "pizza"
} | ConvertTo-Json

Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/feedback" -ContentType "application/json" -Body $feedback
```

Feedback is appended to `models/feedback_log.jsonl` for later analysis/retraining.

Train via API:

```powershell
$train = @{
  raw_dir = "data/raw"
  processed_dir = "data/processed"
  models_dir = "models"
  epochs = 5
  batch_size = 16
  backbone = "resnet18"
  bootstrap_demo_data = $true
} | ConvertTo-Json

Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/train" -ContentType "application/json" -Body $train
```

Evaluate via API:

```powershell
$eval = @{
  processed_dir = "data/processed"
  checkpoint = "models/best.pt"
} | ConvertTo-Json

Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/evaluate" -ContentType "application/json" -Body $eval
```

## Notes

- CPU-only is enforced in code (`torch.device("cpu")`).
- If pretrained weights cannot be downloaded, code falls back to random initialization.
- Calories are estimates from a local reference table and default portions (MVP behavior).

# Le modèle doit d'abord être entraîné
python train.py --raw-dir data/raw --processed-dir data/processed --models-dir models --epochs 5 --batch-size 8

# Ensuite, lancez l'interface
cd "C:\Users\pierr\OneDrive\Bureau\cours epsi\SN3\MSPR\IA" ; .\.venv\Scripts\streamlit run app.py --logger.level=error