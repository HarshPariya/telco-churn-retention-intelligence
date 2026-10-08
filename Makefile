.PHONY: help install lint format test data train evaluate api dashboard docker-build docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  make install       Install all project dependencies"
	@echo "  make data          Download and prepare Telco dataset splits"
	@echo "  make train         Run full multi-model benchmark, tuning, and artifact registration"
	@echo "  make evaluate      Evaluate production model on holdout test set"
	@echo "  make test          Run pytest unit and integration test suite"
	@echo "  make lint          Run Ruff linter and static checks"
	@echo "  make format        Format codebase with Ruff"
	@echo "  make api           Launch FastAPI service on port 8000"
	@echo "  make dashboard     Launch Streamlit dashboard on port 8501"
	@echo "  make docker-build  Build Docker images for API and Dashboard"
	@echo "  make docker-up     Start multi-container platform with Docker Compose"
	@echo "  make docker-down   Stop and remove Docker containers"
	@echo "  make clean         Remove build artifacts, caches, and temp files"

install:
	pip install -r requirements-dev.txt

data:
	python scripts/download_data.py
	python scripts/validate_data.py
	python scripts/prepare_data.py

train:
	python scripts/train_model.py
	python scripts/generate_reports.py

evaluate:
	python scripts/evaluate_model.py

test:
	pytest tests/ --durations=10

lint:
	ruff check .

format:
	ruff format .

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run dashboard/app.py --server.port 8501

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
