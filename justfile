# Astrojax Development Recipes
# Usage: just <recipe> [args...]

# ───── Installation ─────

# Install/sync all dev dependencies (including notebook extras) and set up pre-commit hooks
install:
    uv sync --dev --extra extras
    uv run pre-commit install

# ───── Testing ─────

# Run tests (verbose by default)
test *flags:
    uv run pytest tests/ -v {{flags}}

# Run tests with coverage report
test-cov *flags:
    uv run pytest --cov=astrojax --cov-report=term-missing {{flags}}

# ───── Code Quality ─────

# Auto-format code
fmt:
    uv run ruff format

# Lint with auto-fix
lint:
    uv run ruff check --fix

# Type check with ty
typecheck:
    uvx ty check

# Run all quality checks: format, lint, typecheck
check: fmt lint typecheck
    @echo "All quality checks passed."

# ───── Documentation ─────

# Build documentation (clean)
docs-build:
    uv run zensical build --clean

# Serve documentation locally (clean pycache + build)
docs-serve: docs-clean
    uv run zensical serve --open

# Clear __pycache__ dirs (useful before docs build)
docs-clean:
    find . -path ./.venv -prune -o -type d -name __pycache__ -print -exec rm -rf {} +

# ───── Pre-Commit Hooks ─────

# Install pre-commit hooks into .git/hooks
hooks-install:
    uv run pre-commit install

# Run all pre-commit hooks on all files
hooks-run:
    uv run pre-commit run --all-files
