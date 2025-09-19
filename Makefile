.PHONY: help install lint format test check run-app run-docker build-docker

help:
	@echo "Commands:"
	@echo "  install      : Install Python dependencies from requirements.txt"
	@echo "  lint         : Run ruff linter"
	@echo "  format       : Run black code formatter"
	@echo "  check        : Run lint, format check, mypy, and tests"
	@echo "  test         : Run pytest"
	@echo "  run-app      : Run the Streamlit application locally"
	@echo "  build-docker : Build the Docker image for the application"
	@echo "  run-docker   : Run the application inside a Docker container"


install:
		. .venv/bin/activate && \
		pip install --upgrade pip --break-system-packages && \
		pip install --break-system-packages -r requirements.txt

lint:
	ruff check .

format:
	black .

test:
	PYTHONPATH=. pytest

check: lint
	black --check .
	mypy . --ignore-missing-imports
	pytest

run-app:
		. .venv/bin/activate && \
		streamlit run app/main.py

build-docker:
	docker build -t portfolio-optimizer -f docker/Dockerfile .

run-docker:
	docker run -p 8501:8501 portfolio-optimizer