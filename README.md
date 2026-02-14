<p align="center">
  <a href="https://github.com/duncaneddy/astrojax/"><img src="https://raw.githubusercontent.com/duncaneddy/astrojax/main/docs/images/astrojax_logo.png" alt="Astrojax"></a>
</p>
<p align="center">
    <em>Astrojax - Blazing fast astrodynamics in JAX</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/astrojax/"><img src="https://img.shields.io/pypi/v/astrojax" alt="PyPI"></a>
  <a href="https://duncaneddy.github.io/astrojax/"><img src="https://img.shields.io/badge/docs-latest-blue" alt="Documentation"></a>
  <a href="https://coveralls.io/github/duncaneddy/astrojax?branch=main"><img src="https://coveralls.io/repos/github/duncaneddy/astrojax/badge.svg?branch=main" alt="Coverage Status"></a>
</p>

---

Astrodynamics written in JAX for massively parallel simulation.

The goal of this project is to provide a high-performance astrodynamics library that can be used for research and education. The library is built on top of JAX, which allows for automatic differentiation and GPU/TPU acceleration. The goal is to provide a proof-of-concept implementation of common astrodynamics algorithms that can be used as a starting point for further development. It is _not_ intended to be a full-featured, high-accuracy astrodynamics library (at least not yet).

## Install

```bash
pip install astrojax
# or
uv add astrojax
```

If you have a GPU and want to take advantage of JAX's acceleration can then install the package with the appropriate JAX extras:

```bash
pip install astrojax[cuda13]
# or
uv add astrojax[cuda13]
```

Astrojax supports CUDA 12 and CUDA 13.

## Quickstart

This project uses [`just`](https://github.com/casey/just) as a command runner and [`uv`](https://docs.astral.sh/uv/) for Python package management.

```bash
# Install just (macOS)
brew install just

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install everything (dev deps + extras)
just install
```

## Development

Every recipe has an equivalent raw command you can run directly.

| Task | `just` recipe | Raw command |
|------|---------------|-------------|
| Install dev deps + extras | `just install` | `uv sync --dev --extra extras` |
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
