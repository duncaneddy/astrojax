"""Tests for the astrojax.config module."""

import jax
import jax.numpy as jnp
import pytest

from astrojax.config import get_dtype, get_epoch_eq_tolerance, set_dtype
from astrojax.constants import R_EARTH
from astrojax.coordinates import (
    position_ecef_to_geodetic,
    position_geocentric_to_ecef,
    state_koe_to_eci,
)
from astrojax.epoch import Epoch
from astrojax.orbits import orbital_period

pytestmark = pytest.mark.order("first")


@pytest.fixture(autouse=True)
def reset_dtype():
    """Reset dtype to float32 before and after each test."""
    set_dtype(jnp.float32)
    yield
    set_dtype(jnp.float32)


class TestGetSetDtype:
    def test_default_dtype(self):
        assert get_dtype() == jnp.float32

    def test_set_float64(self):
        set_dtype(jnp.float64)
        assert get_dtype() == jnp.float64

    def test_set_float16(self):
        set_dtype(jnp.float16)
        assert get_dtype() == jnp.float16

    def test_set_bfloat16(self):
        set_dtype(jnp.bfloat16)
        assert get_dtype() == jnp.bfloat16

    def test_set_float32(self):
        set_dtype(jnp.float64)
        set_dtype(jnp.float32)
        assert get_dtype() == jnp.float32

    def test_roundtrip(self):
        for dtype in (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64):
            set_dtype(dtype)
            assert get_dtype() == dtype

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            set_dtype(jnp.int32)

    def test_invalid_dtype_string_raises(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            set_dtype("float32")

    def test_float64_enables_x64(self):
        set_dtype(jnp.float64)
        assert jax.config.jax_enable_x64 is True


class TestEpochEqTolerance:
    def test_float32_tolerance(self):
        set_dtype(jnp.float32)
        assert get_epoch_eq_tolerance() == 1e-3

    def test_float64_tolerance(self):
        set_dtype(jnp.float64)
        assert get_epoch_eq_tolerance() == 1e-9

    def test_float16_tolerance(self):
        set_dtype(jnp.float16)
        assert get_epoch_eq_tolerance() == 0.1

    def test_bfloat16_tolerance(self):
        set_dtype(jnp.bfloat16)
        assert get_epoch_eq_tolerance() == 0.1


class TestDtypeSwitchingOutputs:
    """Verify that output dtypes match the configured dtype."""

    def test_epoch_seconds_dtype_float64(self):
        set_dtype(jnp.float64)
        epc = Epoch(2024, 1, 1, 12, 0, 0.0)
        assert epc._seconds.dtype == jnp.float64

    def test_epoch_kahan_dtype_float64(self):
        set_dtype(jnp.float64)
        epc = Epoch(2024, 1, 1, 12, 0, 0.0)
        assert epc._kahan_c.dtype == jnp.float64

    def test_epoch_jd_stays_int32(self):
        set_dtype(jnp.float64)
        epc = Epoch(2024, 1, 1, 12, 0, 0.0)
        assert epc._jd.dtype == jnp.int32

    def test_orbital_period_dtype_float64(self):
        set_dtype(jnp.float64)
        a = jnp.float64(R_EARTH + 500e3)
        T = orbital_period(a)
        assert T.dtype == jnp.float64

    def test_geocentric_dtype_float64(self):
        set_dtype(jnp.float64)
        x_geoc = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64)
        x_ecef = position_geocentric_to_ecef(x_geoc)
        assert x_ecef.dtype == jnp.float64

    def test_koe_to_eci_dtype_float64(self):
        set_dtype(jnp.float64)
        oe = jnp.array(
            [R_EARTH + 500e3, 0.001, 0.9, 0.5, 0.3, 0.1],
            dtype=jnp.float64,
        )
        state = state_koe_to_eci(oe)
        assert state.dtype == jnp.float64


class TestFloat64Precision:
    """Verify that float64 achieves tight absolute precision."""

    def test_epoch_kahan_precision_float64(self):
        """Float64 should accumulate negligible error over many small additions."""
        set_dtype(jnp.float64)
        epc = Epoch(2024, 1, 1)
        n_steps = 10000
        dt = 0.001  # 1 ms

        for _ in range(n_steps):
            epc = epc + dt

        expected = n_steps * dt  # 10.0 seconds
        actual = float(epc._seconds) - float(Epoch(2024, 1, 1)._seconds)
        error = abs(actual - expected)

        # Float64 with Kahan summation should have sub-microsecond error
        # over 10000 additions of 1ms
        assert error < 1e-6

    def test_jd_precision_float64(self):
        """Float64 JD should have sub-millisecond sub-day precision."""
        set_dtype(jnp.float64)
        epc = Epoch(2024, 6, 15, 6, 30, 0.0)
        jd_f64 = float(epc.jd())

        # At float64, the JD should faithfully represent the sub-day
        # fractional component. 6h30m = 0.270833... day.
        fractional = jd_f64 - int(jd_f64)
        expected_frac = (6.0 * 3600 + 30.0 * 60 + 43200.0) / 86400.0
        error = abs(fractional - expected_frac)

        # Sub-millisecond precision in fractional day
        assert error < 1e-8

    def test_geodetic_convergence_float64(self):
        """Float64 Bowring iteration should converge to sub-metre accuracy."""
        set_dtype(jnp.float64)
        x_ecef = jnp.array([R_EARTH + 500e3, 0.0, 0.0], dtype=jnp.float64)
        geod_f64 = position_ecef_to_geodetic(x_ecef)
        alt_f64 = float(geod_f64[2])

        # At float64, the altitude on the equator should be very close
        # to the expected value (geodetic altitude ≈ geocentric altitude
        # on the equator)
        assert abs(alt_f64 - 500e3) < 1.0  # Sub-metre accuracy


class TestJITRetrace:
    """Verify JIT retraces when dtype changes."""

    def test_jit_retrace_on_dtype_change(self):
        """JIT should retrace when input dtypes change."""

        @jax.jit
        def compute_period(a):
            return orbital_period(a)

        set_dtype(jnp.float32)
        a_f32 = jnp.float32(R_EARTH + 500e3)
        result_f32 = compute_period(a_f32)
        assert result_f32.dtype == jnp.float32

        set_dtype(jnp.float64)
        a_f64 = jnp.float64(R_EARTH + 500e3)
        result_f64 = compute_period(a_f64)
        assert result_f64.dtype == jnp.float64
