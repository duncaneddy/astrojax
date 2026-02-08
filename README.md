# astrojax

Astrodynamics written in JAX for massively parallel simulation.

## Install

```bash
pip install astrojax
# or
uv add astrojax
```

## Development

This project uses [`just`](https://github.com/casey/just) as a command runner.
Every recipe has an equivalent raw command you can run directly.

| Task | `just` recipe | Raw command |
|------|---------------|-------------|
| Install dev deps | `just install` | `uv sync --dev` |
| Run tests | `just test` | `uv run pytest tests/ -v` |
| Test with coverage | `just test-cov` | `uv run pytest --cov=astrojax --cov-report=term-missing` |
| Format code | `just fmt` | `uv run ruff format` |
| Lint (auto-fix) | `just lint` | `uv run ruff check --fix` |
| Type check | `just typecheck` | `uvx ty check` |
| All quality checks | `just check` | Runs fmt + lint + typecheck |
| Build docs | `just docs-build` | `uv run zensical build --clean` |
| Serve docs | `just docs-serve` | `uv run zensical serve --clean` |

## License

The code in this repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.
