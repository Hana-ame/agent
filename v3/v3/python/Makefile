.PHONY: help install test lint format clean

help:
	@echo "Available targets:"
	@echo "  install   Install package and dependencies"
	@echo "  test      Run tests"
	@echo "  lint      Run linters (mypy, flake8)"
	@echo "  format    Format code with black and isort"
	@echo "  clean     Remove build artifacts"

install:
	pip install -e .[dev]

test:
	pytest

lint:
	mypy src
	flake8 src tests

format:
	black src tests
	isort src tests

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
