# Makefile for Airbnb Market Insights - Seattle

.PHONY: help install test lint format run clean

help:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║     Airbnb Market Insights - Seattle - Makefile Commands       ║"
	@echo "╠════════════════════════════════════════════════════════════════╣"
	@echo "║  make install      Install dependencies                        ║"
	@echo "║  make test         Run pytest test suite                       ║"
	@echo "║  make lint         Run flake8 linting                          ║"
	@echo "║  make format       Format code with black                      ║"
	@echo "║  make run          Run the analysis pipeline                   ║"
	@echo "║  make charts       Generate all visualizations                 ║"
	@echo "║  make clean        Clean build artifacts                       ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black isort flake8

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ tests/ --max-line-length=100

format:
	black src/ tests/ main.py
	isort src/ tests/ main.py

run:
	python main.py

charts:
	python main.py --export-charts

run-full:
	python main.py --export-charts --top 20

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ htmlcov/ .coverage 2>/dev/null || true

download-data:
	@echo "Download data from: https://www.kaggle.com/datasets/airbnb/seattle"
	@echo "Place CSV files in: data/raw/"
