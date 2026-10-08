$ErrorActionPreference = "SilentlyContinue"
Write-Host "Verifying ports..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe scripts/kill_ports.py
Write-Host "Launching Telco Churn FastAPI backend on http://127.0.0.1:8000..." -ForegroundColor Green
Write-Host "OpenAPI docs available at: http://127.0.0.1:8000/docs" -ForegroundColor Yellow
& .\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
