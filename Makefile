# ML Drift Monitor — Makefile
# Run from project root. All commands assume venv is activated.

.PHONY: help setup download train simulate monitor api dashboard test lint typecheck all docker-up docker-down docker-build clean

PYTHON  := python
VENV    := .venv

##@ Setup

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

setup: ## Create virtualenv and install dependencies
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	@echo "Setup complete. Activate with: source $(VENV)/bin/activate"

download: ## Download Credit Card Fraud dataset from Kaggle
	mkdir -p data/raw
	kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
	@echo "Dataset ready at data/raw/creditcard.csv"

##@ Pipeline

train: ## Train XGBoost model and save artifacts
	$(PYTHON) -m src.train

simulate: ## Generate 12 weekly batches with gradual drift
	$(PYTHON) -m src.simulate_drift

monitor: ## Run Evidently drift analysis on all batches
	$(PYTHON) -m src.monitor

##@ Services

api: ## Start FastAPI server (http://localhost:8000)
	uvicorn src.infrastructure.api:app --host 0.0.0.0 --port 8000 --reload

dashboard: ## Start Streamlit dashboard (served via http://localhost:8000/dashboard)
	$(PYTHON) -m streamlit run dashboard/app.py --server.baseUrlPath=dashboard --server.port=8501 --server.headless=true

##@ Docker

docker-build: ## Build all Docker images
	docker compose build

docker-up: ## Start all services via Docker Compose
	docker compose up -d
	@echo "Hub:       http://localhost:8000"
	@echo "Dashboard: http://localhost:8501"
	@echo "API Docs:  http://localhost:8000/docs"

docker-down: ## Stop all Docker services
	docker compose down

##@ Quality

test: ## Run unit tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing

lint: ## Run ruff linter
	ruff check src/ dashboard/ tests/

typecheck: ## Run mypy type checker
	mypy src/ dashboard/ --ignore-missing-imports

##@ Orchestration

all: ## Run full pipeline: train -> simulate -> monitor
	$(MAKE) train
	$(MAKE) simulate
	$(MAKE) monitor
	@echo ""
	@echo "Pipeline complete."
	@echo "  Start API:        make api"
	@echo "  Start Dashboard:  make dashboard"
	@echo "  Start both:       make docker-up"

##@ Maintenance

clean: ## Remove generated artifacts
	rm -rf reports/* metrics/* data/simulated/* model/*
	@echo "Artifacts cleaned."
