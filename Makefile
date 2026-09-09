# SentinelScreen
#
# Determinism is the contract: `make reproduce` runs the whole pipeline twice
# from a cold start and fails if a single artefact byte differs. CI runs it.

PYTHON ?= python

.PHONY: help install run test reproduce readme api lint typecheck clean

help:
	@echo "install    install pinned dependencies"
	@echo "run        regenerate all data and artefacts from a cold start"
	@echo "test       run the pytest suite"
	@echo "reproduce  run the pipeline twice and diff every artefact"
	@echo "readme     regenerate README.md from outputs/"
	@echo "api        serve the FastAPI app on :8000"
	@echo "typecheck  mypy --strict over src/"
	@echo "clean      remove generated data, outputs and caches"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) scripts/run_pipeline.py
	$(PYTHON) scripts/render_readme.py

test:
	$(PYTHON) -m pytest

reproduce:
	$(PYTHON) scripts/reproduce.py

readme:
	$(PYTHON) scripts/render_readme.py

api:
	$(PYTHON) -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

typecheck:
	$(PYTHON) -m mypy

lint: typecheck

clean:
	rm -rf data outputs/*.csv outputs/*.json outputs/*.jsonl outputs/*.log outputs/plots
	rm -rf outputs/_repro_a outputs/_repro_b
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
