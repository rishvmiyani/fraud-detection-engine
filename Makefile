# =================================================================
# FRAUD DETECTION ENGINE - MAKEFILE
# =================================================================

.PHONY: help install install-dev test test-coverage lint format security docker-build docker-run clean setup

# Default Python interpreter
PYTHON := python
PIP := pip
DOCKER := docker
DOCKER_COMPOSE := docker-compose

# Project name
PROJECT_NAME := fraud-detection-engine
VERSION := 1.0.0

# Docker configuration
DOCKER_IMAGE := $(PROJECT_NAME):$(VERSION)
DOCKER_REGISTRY := ghcr.io
DOCKER_REPO := $(DOCKER_REGISTRY)/$(PROJECT_NAME)

# =================================================================
# HELP
# =================================================================
help: ## Show this help message
	@echo "Fraud Detection Engine - Makefile Commands"
	@echo "=========================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $1, $2}' $(MAKEFILE_LIST)

# =================================================================
# SETUP AND INSTALLATION
# =================================================================
setup: ## Initial project setup
	@echo "🚀 Setting up Fraud Detection Engine..."
	$(PYTHON) -m venv venv
	@echo "✅ Virtual environment created"
	@echo "Activate with: source venv/bin/activate (Linux/Mac) or venv\Scripts\activate (Windows)"

install: ## Install production dependencies
	@echo "📦 Installing production dependencies..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	@echo "📦 Installing development dependencies..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -r requirements-test.txt
	pre-commit install

install-all: install-dev ## Install all dependencies

# =================================================================
# DEVELOPMENT
# =================================================================
run: ## Run the development server
	@echo "🚀 Starting development server..."
	$(PYTHON) -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

run-prod: ## Run the production server
	@echo "🚀 Starting production server..."
	$(PYTHON) -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4

# =================================================================
# TESTING
# =================================================================
test: ## Run all tests
	@echo "🧪 Running tests..."
	$(PYTHON) -m pytest tests/ -v

test-unit: ## Run unit tests only
	@echo "🧪 Running unit tests..."
	$(PYTHON) -m pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "🧪 Running integration tests..."
	$(PYTHON) -m pytest tests/integration/ -v

test-performance: ## Run performance tests
	@echo "🧪 Running performance tests..."
	$(PYTHON) -m pytest tests/performance/ -v --benchmark-only

test-coverage: ## Run tests with coverage report
	@echo "🧪 Running tests with coverage..."
	$(PYTHON) -m pytest tests/ -v \
		--cov=src \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-report=xml

test-watch: ## Run tests in watch mode
	@echo "🧪 Running tests in watch mode..."
	$(PYTHON) -m pytest-watch tests/ src/

# =================================================================
# CODE QUALITY
# =================================================================
lint: ## Run linting
	@echo "🔍 Running linting..."
	black --check src/ tests/
	flake8 src/ tests/
	mypy src/

format: ## Format code
	@echo "🎨 Formatting code..."
	black src/ tests/
	isort src/ tests/

format-check: ## Check code formatting
	@echo "🔍 Checking code formatting..."
	black --check --diff src/ tests/
	isort --check-only --diff src/ tests/

# =================================================================
# SECURITY
# =================================================================
security: ## Run security checks
	@echo "🔒 Running security checks..."
	bandit -r src/
	safety check

security-full: ## Run comprehensive security scan
	@echo "🔒 Running comprehensive security scan..."
	bandit -r src/ -f json -o reports/bandit-report.json
	safety check --json --output reports/safety-report.json
	semgrep --config=auto src/

# =================================================================
# DATABASE
# =================================================================
db-init: ## Initialize database
	@echo "🗃️ Initializing database..."
	$(PYTHON) scripts/database/init_database.py

db-migrate: ## Run database migrations
	@echo "🗃️ Running database migrations..."
	alembic upgrade head

db-migration: ## Create new database migration
	@echo "🗃️ Creating new migration..."
	@read -p "Migration message: " message; \
	alembic revision --autogenerate -m "$$message"

db-reset: ## Reset database
	@echo "🗃️ Resetting database..."
	$(PYTHON) scripts/database/reset_database.py

# =================================================================
# DATA PROCESSING
# =================================================================
process-data: ## Process transaction data
	@echo "📊 Processing transaction data..."
	$(PYTHON) scripts/data_processing/process_transactions.py \
		--input data/raw/ \
		--output data/processed/

generate-synthetic-data: ## Generate synthetic test data
	@echo "🎲 Generating synthetic data..."
	$(PYTHON) scripts/data_processing/generate_synthetic_data.py \
		--output data/synthetic/ \
		--size 10000

# =================================================================
# MODEL TRAINING
# =================================================================
train-model: ## Train fraud detection model
	@echo "🤖 Training fraud detection model..."
	$(PYTHON) scripts/model_training/train_model.py \
		--data-path data/processed/ \
		--output-path models/

evaluate-model: ## Evaluate trained model
	@echo "📊 Evaluating model..."
	$(PYTHON) scripts/model_training/evaluate_model.py \
		--model-path models/best_model.pkl \
		--test-data data/processed/test_data.parquet

# =================================================================
# DOCKER
# =================================================================
docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	$(DOCKER) build -t $(DOCKER_IMAGE) -f infrastructure/docker/Dockerfile .

docker-build-prod: ## Build production Docker image
	@echo "🐳 Building production Docker image..."
	$(DOCKER) build --target production -t $(DOCKER_IMAGE) -f infrastructure/docker/Dockerfile .

docker-run: ## Run Docker container
	@echo "🐳 Running Docker container..."
	$(DOCKER) run -d -p 8000:8000 --name $(PROJECT_NAME) $(DOCKER_IMAGE)

docker-stop: ## Stop Docker container
	@echo "🐳 Stopping Docker container..."
	$(DOCKER) stop $(PROJECT_NAME) || true
	$(DOCKER) rm $(PROJECT_NAME) || true

docker-push: ## Push Docker image to registry
	@echo "🐳 Pushing Docker image..."
	$(DOCKER) tag $(DOCKER_IMAGE) $(DOCKER_REPO):$(VERSION)
	$(DOCKER) tag $(DOCKER_IMAGE) $(DOCKER_REPO):latest
	$(DOCKER) push $(DOCKER_REPO):$(VERSION)
	$(DOCKER) push $(DOCKER_REPO):latest

# =================================================================
# DOCKER COMPOSE
# =================================================================
up: ## Start all services with docker-compose
	@echo "🚀 Starting all services..."
	$(DOCKER_COMPOSE) up -d

down: ## Stop all services
	@echo "🛑 Stopping all services..."
	$(DOCKER_COMPOSE) down

restart: ## Restart all services
	@echo "🔄 Restarting all services..."
	$(DOCKER_COMPOSE) restart

logs: ## Show logs from all services
	@echo "📋 Showing logs..."
	$(DOCKER_COMPOSE) logs -f

logs-api: ## Show logs from API service only
	@echo "📋 Showing API logs..."
	$(DOCKER_COMPOSE) logs -f fraud-detection-api

status: ## Show service status
	@echo "📊 Service status..."
	$(DOCKER_COMPOSE) ps

# =================================================================
# MONITORING
# =================================================================
monitoring-up: ## Start monitoring stack
	@echo "📊 Starting monitoring stack..."
	$(DOCKER_COMPOSE) -f monitoring/docker-compose.monitoring.yml up -d

monitoring-down: ## Stop monitoring stack
	@echo "📊 Stopping monitoring stack..."
	$(DOCKER_COMPOSE) -f monitoring/docker-compose.monitoring.yml down

# =================================================================
# CLEANUP
# =================================================================
clean: ## Clean up temporary files
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.log" -delete
	rm -rf .coverage htmlcov/ .mypy_cache/ .tox/

clean-docker: ## Clean up Docker resources
	@echo "🧹 Cleaning up Docker resources..."
	$(DOCKER) system prune -f
	$(DOCKER) volume prune -f
	$(DOCKER) network prune -f

clean-all: clean clean-docker ## Clean up everything

# =================================================================
# DEPLOYMENT
# =================================================================
deploy-dev: ## Deploy to development environment
	@echo "🚀 Deploying to development..."
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.dev.yml up -d

deploy-staging: ## Deploy to staging environment
	@echo "🚀 Deploying to staging..."
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.staging.yml up -d

deploy-prod: ## Deploy to production environment
	@echo "🚀 Deploying to production..."
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.prod.yml up -d

# =================================================================
# UTILITIES
# =================================================================
shell: ## Open Python shell with project context
	@echo "🐍 Opening Python shell..."
	$(PYTHON) -c "import sys; sys.path.append('src'); from src.main import *"

notebook: ## Start Jupyter notebook server
	@echo "📓 Starting Jupyter notebook server..."
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	sphinx-build -b html docs/ docs/_build/html/

version: ## Show version information
	@echo "Fraud Detection Engine v$(VERSION)"
	@echo "Python: $(shell python --version)"
	@echo "Docker: $(shell docker --version)"

health-check: ## Check system health
	@echo "🏥 Checking system health..."
	curl -f http://localhost:8000/health || echo "API not running"
	curl -f http://localhost:9090 || echo "Prometheus not running"
	curl -f http://localhost:3000 || echo "Grafana not running"
