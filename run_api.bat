@echo off
.\.venv\Scripts\python.exe scripts\kill_ports.py
echo Starting Telco Churn FastAPI backend on http://127.0.0.1:8000...
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
