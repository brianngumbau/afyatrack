# AfyaTrack developer entry points.
#
# Every target is also what CI runs, so a green `make lint test` locally means
# the same gates pass in the pipeline.

PYTHON ?= python
PIP    ?= $(PYTHON) -m pip
PORT   ?= 8501
COMPOSE ?= docker compose

.DEFAULT_GOAL := help
.PHONY: help install test lint run docker-up docker-down docker-test clean

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install pinned runtime and developer dependencies
	$(PIP) install --upgrade pip
	$(PIP) install --requirement requirements.txt

test: ## Run the unit test suite with verbose output
	$(PYTHON) -m pytest --verbose

lint: ## Run flake8 across the package, app, and tests
	$(PYTHON) -m flake8 src tests app.py

run: ## Serve the dashboard locally on $(PORT)
	$(PYTHON) -m streamlit run app.py --server.port=$(PORT)

docker-up: ## Build and start the containerized dashboard
	$(COMPOSE) up --build

docker-down: ## Stop the stack and remove its containers
	$(COMPOSE) down --remove-orphans

docker-test: ## Run the test suite inside the runtime image
	$(COMPOSE) run --rm --no-deps afyatrack pytest --verbose

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ __pycache__ src/__pycache__ tests/__pycache__
