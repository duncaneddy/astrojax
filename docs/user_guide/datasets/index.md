# Datasets

The `astrojax.datasets` module provides access to external catalogs of
celestial objects.  Each dataset is downloaded, cached locally, and
exposed as a Polars DataFrame with convenience functions for lookups and
JAX-compatible state computation.

| Dataset | Description |
|---------|-------------|
| [MPC Asteroids](mpc_asteroids.md) | Minor Planet Center asteroid orbit catalog: loading, querying, and heliocentric state vectors |
