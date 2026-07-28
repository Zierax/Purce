.PHONY: install test lint clean benchmark

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

lint:
	ruff check purce/ tests/
	ruff format --check purce/ tests/

format:
	ruff format purce/ tests/

clean:
	rm -rf out/ build/ dist/ *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

benchmark:
	python benchmarks/benchmark_matmul.py
