.PHONY: install check test lint format evaluate serve

install:
	uv sync --extra dev

check: lint test evaluate

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

evaluate:
	uv run voxadapt evaluate data/sample_eval.jsonl --output artifacts/baseline.json

serve:
	uv run voxadapt serve
