.PHONY: check test lint format clean help

PYTHON ?= python

help:
	@echo "Available commands:"
	@echo "  make check   Run full suite: ruff format check, ruff lint, mypy, pytest"
	@echo "  make test    Run pytest unit and integration tests"
	@echo "  make lint    Run ruff and mypy"
	@echo "  make format  Run ruff format and fix"
	@echo "  make clean   Clean build and test caches"

check: lint test

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check packages apps tests
	$(PYTHON) -m ruff format --check packages apps tests
	$(PYTHON) -m mypy packages apps tests

format:
	$(PYTHON) -m ruff format packages apps tests
	$(PYTHON) -m ruff check --fix packages apps tests

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
