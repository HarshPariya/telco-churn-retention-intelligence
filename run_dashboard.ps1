$ErrorActionPreference = "SilentlyContinue"
Write-Host "Launching Telco Churn Streamlit Dashboard on http://localhost:8501..." -ForegroundColor Green
& .\.venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.port 8501
