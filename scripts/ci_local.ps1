$ErrorActionPreference = "Stop"

Write-Host "[ci] Running unit and API tests..."
python -m pytest

Write-Host "[ci] Running quick pipeline smoke check..."
python quick_check.py

Write-Host "[ci] OK"

