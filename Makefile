UV_RUN ?= uv run --frozen
PYTEST_ARGS ?=

.PHONY: sync format lint lint-fix test test-coverage check

sync:
	uv sync

format:
	$(UV_RUN) ruff format

lint:
	$(UV_RUN) ruff check

lint-fix:
	$(UV_RUN) ruff check --fix

# The world-model unit tests run in their own process: several of them patch
# module-level state in ejepa_wm.backends, which leaks into the harness tests
# when both roots share one pytest session.
test:
	$(UV_RUN) pytest $(PYTEST_ARGS)
	$(UV_RUN) pytest src/ejepa_wm/tests

test-coverage:
	$(UV_RUN) pytest -q --cov --cov-report=term-missing:skip-covered --cov-report=xml:coverage.xml

check: lint test
