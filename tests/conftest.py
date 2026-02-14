import jax.numpy as jnp
import pytest

from astrojax.config import set_dtype


@pytest.fixture(autouse=True)
def _ensure_float64():
    """Set float64 precision before every test.

    With pytest-xdist, each worker process starts with the default float32.
    This fixture ensures all tests get float64 unless they explicitly override
    it (e.g. test_config.py has its own autouse fixture that sets float32).
    """
    set_dtype(jnp.float64)
