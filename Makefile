PATH := $(CURDIR)/.venv/bin:$(PATH)

.PHONY: lint test

lint:
	ruff check . && mypy

# ponytail: bare `pytest` is direct-exec'd by make with the original PATH (no .venv); `uv run` resolves everywhere — upgrade: force shell via .ONESHELL on make >= 4
test:
	uv run pytest -q
