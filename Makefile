.PHONY: install test lint format

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	mypy ttsbench

format:
	ruff format .
	ruff check --fix .
