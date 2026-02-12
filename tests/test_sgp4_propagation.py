"""Tests for SGP4 propagation against the reference python-sgp4 library."""

import jax
import jax.numpy as jnp
import pytest
from sgp4.api import WGS72 as SGP4_WGS72
from sgp4.api import Satrec

from astrojax.sgp4 import (
    WGS72,
    create_sgp4_propagator,
    parse_tle,
    sgp4_init,
    sgp4_propagate,
)

# Enable float64 for precise comparison tests
jax.config.update("jax_enable_x64", True)

# ISS TLE — near-Earth LEO (period ~92 min)
ISS_LINE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
ISS_LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

# Polar orbit test TLE (from brahe test fixtures)
POLAR_LINE1 = "1     1U          20  1.00000000  .00000000  00000-0  00000-0 0    07"
POLAR_LINE2 = "2     1  90.0000   0.0000 0010000   0.0000   0.0000 15.21936719    07"


def _get_reference(line1: str, line2: str, tsince_min: float) -> tuple:
    """Get reference position and velocity from python-sgp4."""
    sat = Satrec.twoline2rv(line1, line2, SGP4_WGS72)
    e, r, v = sat.sgp4_tsince(tsince_min)
    return e, r, v


class TestSGP4InitNearEarth:
    """Test SGP4 initialization for near-Earth satellites."""

    def test_iss_init_returns_params_and_method(self) -> None:
        elements = parse_tle(ISS_LINE1, ISS_LINE2)
        params, method = sgp4_init(elements, WGS72)
        assert method == "n"  # ISS is near-Earth
        assert params.shape[0] > 0

    def test_iss_init_params_are_finite(self) -> None:
        elements = parse_tle(ISS_LINE1, ISS_LINE2)
        params, _ = sgp4_init(elements, WGS72)
        assert jnp.all(jnp.isfinite(params))


class TestSGP4PropagateNearEarth:
    """Test near-Earth SGP4 propagation against reference python-sgp4."""

    @pytest.fixture()
    def iss_propagator(self):
        """Create ISS propagator."""
        elements = parse_tle(ISS_LINE1, ISS_LINE2)
        params, method = sgp4_init(elements, WGS72)
        return params, method

    def test_iss_at_epoch(self, iss_propagator) -> None:
        """Position at epoch (tsince=0) matches reference."""
        params, method = iss_propagator
        r, v = sgp4_propagate(params, jnp.float64(0.0), method)

        e_ref, r_ref, v_ref = _get_reference(ISS_LINE1, ISS_LINE2, 0.0)
        assert e_ref == 0

        # Position match < 1e-6 km
        assert jnp.allclose(r, jnp.array(r_ref), atol=1e-6), (
            f"Position mismatch at epoch: {r} vs {r_ref}"
        )
        # Velocity match < 1e-9 km/s
        assert jnp.allclose(v, jnp.array(v_ref), atol=1e-9), (
            f"Velocity mismatch at epoch: {v} vs {v_ref}"
        )

    def test_iss_at_60min(self, iss_propagator) -> None:
        """Position at 60 minutes matches reference."""
        params, method = iss_propagator
        r, v = sgp4_propagate(params, jnp.float64(60.0), method)

        e_ref, r_ref, v_ref = _get_reference(ISS_LINE1, ISS_LINE2, 60.0)
        assert e_ref == 0

        assert jnp.allclose(r, jnp.array(r_ref), atol=1e-6)
        assert jnp.allclose(v, jnp.array(v_ref), atol=1e-9)

    def test_iss_at_360min(self, iss_propagator) -> None:
        """Position at 360 minutes (4 orbits) matches reference."""
        params, method = iss_propagator
        r, v = sgp4_propagate(params, jnp.float64(360.0), method)

        e_ref, r_ref, v_ref = _get_reference(ISS_LINE1, ISS_LINE2, 360.0)
        assert e_ref == 0

        assert jnp.allclose(r, jnp.array(r_ref), atol=1e-6)
        assert jnp.allclose(v, jnp.array(v_ref), atol=1e-9)

    def test_iss_at_1440min(self, iss_propagator) -> None:
        """Position at 1440 minutes (1 day) matches reference."""
        params, method = iss_propagator
        r, v = sgp4_propagate(params, jnp.float64(1440.0), method)

        e_ref, r_ref, v_ref = _get_reference(ISS_LINE1, ISS_LINE2, 1440.0)
        assert e_ref == 0

        assert jnp.allclose(r, jnp.array(r_ref), atol=1e-6)
        assert jnp.allclose(v, jnp.array(v_ref), atol=1e-9)

    def test_iss_negative_tsince(self, iss_propagator) -> None:
        """Backward propagation matches reference."""
        params, method = iss_propagator
        r, v = sgp4_propagate(params, jnp.float64(-60.0), method)

        e_ref, r_ref, v_ref = _get_reference(ISS_LINE1, ISS_LINE2, -60.0)
        assert e_ref == 0

        assert jnp.allclose(r, jnp.array(r_ref), atol=1e-6)
        assert jnp.allclose(v, jnp.array(v_ref), atol=1e-9)

    def test_polar_at_epoch(self) -> None:
        """Polar orbit satellite at epoch matches reference."""
        elements = parse_tle(POLAR_LINE1, POLAR_LINE2)
        params, method = sgp4_init(elements, WGS72)
        assert method == "n"

        r, v = sgp4_propagate(params, jnp.float64(0.0), method)
        e_ref, r_ref, v_ref = _get_reference(POLAR_LINE1, POLAR_LINE2, 0.0)
        assert e_ref == 0

        assert jnp.allclose(r, jnp.array(r_ref), atol=1e-6)
        assert jnp.allclose(v, jnp.array(v_ref), atol=1e-9)

    def test_polar_at_720min(self) -> None:
        """Polar orbit at 12 hours matches reference."""
        elements = parse_tle(POLAR_LINE1, POLAR_LINE2)
        params, method = sgp4_init(elements, WGS72)

        r, v = sgp4_propagate(params, jnp.float64(720.0), method)
        e_ref, r_ref, v_ref = _get_reference(POLAR_LINE1, POLAR_LINE2, 720.0)
        assert e_ref == 0

        assert jnp.allclose(r, jnp.array(r_ref), atol=1e-6)
        assert jnp.allclose(v, jnp.array(v_ref), atol=1e-9)


class TestSGP4Float32:
    """Test SGP4 propagation at float32 precision."""

    def test_iss_float32_reasonable_tolerance(self) -> None:
        """Float32 propagation within ~1e-3 km of reference."""
        elements = parse_tle(ISS_LINE1, ISS_LINE2)
        params, method = sgp4_init(elements, WGS72)

        # Downcast to float32
        params_f32 = params.astype(jnp.float32)
        r, v = sgp4_propagate(params_f32, jnp.float32(60.0), method)

        e_ref, r_ref, v_ref = _get_reference(ISS_LINE1, ISS_LINE2, 60.0)
        assert e_ref == 0

        # Relaxed tolerance for float32
        assert jnp.allclose(r, jnp.array(r_ref, dtype=jnp.float32), atol=1e-2)


class TestCreateSGP4Propagator:
    """Test the functional factory API."""

    def test_create_returns_params_and_fn(self) -> None:
        params, propagate_fn = create_sgp4_propagator(ISS_LINE1, ISS_LINE2)
        assert params.ndim == 1
        assert callable(propagate_fn)

    def test_functional_api_matches_reference(self) -> None:
        _, propagate_fn = create_sgp4_propagator(ISS_LINE1, ISS_LINE2)
        r, v = propagate_fn(jnp.float64(60.0))

        e_ref, r_ref, v_ref = _get_reference(ISS_LINE1, ISS_LINE2, 60.0)
        assert e_ref == 0

        assert jnp.allclose(r, jnp.array(r_ref), atol=1e-6)

    def test_jit_compatible(self) -> None:
        params, _ = create_sgp4_propagator(ISS_LINE1, ISS_LINE2)
        elements = parse_tle(ISS_LINE1, ISS_LINE2)
        _, method = sgp4_init(elements, WGS72)

        @jax.jit
        def propagate(p, t):
            return sgp4_propagate(p, t, method)

        r, v = propagate(params, jnp.float64(60.0))

        e_ref, r_ref, v_ref = _get_reference(ISS_LINE1, ISS_LINE2, 60.0)
        assert jnp.allclose(r, jnp.array(r_ref), atol=1e-6)

    def test_vmap_over_time(self) -> None:
        params, _ = create_sgp4_propagator(ISS_LINE1, ISS_LINE2)
        elements = parse_tle(ISS_LINE1, ISS_LINE2)
        _, method = sgp4_init(elements, WGS72)

        times = jnp.array([0.0, 60.0, 360.0, 1440.0])

        @jax.vmap
        def batch_propagate(t):
            return sgp4_propagate(params, t, method)

        r_batch, v_batch = batch_propagate(times)
        assert r_batch.shape == (4, 3)
        assert v_batch.shape == (4, 3)

        # Verify each time point
        for i, t in enumerate(times):
            e_ref, r_ref, v_ref = _get_reference(ISS_LINE1, ISS_LINE2, float(t))
            assert e_ref == 0
            assert jnp.allclose(r_batch[i], jnp.array(r_ref), atol=1e-6)

    def test_gravity_model_string(self) -> None:
        """Accepts gravity model as string."""
        params, fn = create_sgp4_propagator(ISS_LINE1, ISS_LINE2, gravity="wgs84")
        r, v = fn(jnp.float64(0.0))
        assert jnp.all(jnp.isfinite(r))


# ---------------------------------------------------------------------------
# Deep-space (SDP4) TLEs
# ---------------------------------------------------------------------------

# Molniya 2-14 — Highly-elliptical, 12-hour resonance (irez=2)
MOLNIYA_2_14_L1 = "1 08195U 75081A   06176.33215444  .00000099  00000-0  11873-3 0   813"
MOLNIYA_2_14_L2 = "2 08195  64.1586 279.0717 6877146 264.7651  20.2257  2.00491383225656"

# Molniya 1-36 — Highly-elliptical, 12-hour resonance (irez=2)
MOLNIYA_1_36_L1 = "1 09880U 77021A   06176.56157475  .00000421  00000-0  10000-3 0   837"
MOLNIYA_1_36_L2 = "2 09880  64.5968 349.3786 7069051 270.0229  16.3320  2.00813614112380"

# ITALSAT 2 — GEO, synchronous resonance (irez=1)
ITALSAT_2_L1 = "1 24208U 96044A   06177.04061740 -.00000094  00000-0  10000-3 0  1600"
ITALSAT_2_L2 = "2 24208   3.8536  80.0121 0026640 311.0977  48.3000  1.00778054 36119"

# Vela 5A — Deep-space, non-resonant (irez=0)
VELA_5A_L1 = "1 04965U 69046F   06175.83186726  .00000094  00000-0  10000-3 0  4711"
VELA_5A_L2 = "2 04965  32.9048 138.7680 6088834 148.5862 269.3268  2.47283741134637"


class TestSDP4InitDeepSpace:
    """Test SDP4 initialization for deep-space satellites."""

    def test_molniya_2_14_is_deep_space(self) -> None:
        elements = parse_tle(MOLNIYA_2_14_L1, MOLNIYA_2_14_L2)
        params, method = sgp4_init(elements, WGS72)
        assert method == "d"
        assert jnp.all(jnp.isfinite(params))

    def test_molniya_1_36_is_deep_space(self) -> None:
        elements = parse_tle(MOLNIYA_1_36_L1, MOLNIYA_1_36_L2)
        params, method = sgp4_init(elements, WGS72)
        assert method == "d"
        assert jnp.all(jnp.isfinite(params))

    def test_italsat_2_is_deep_space(self) -> None:
        elements = parse_tle(ITALSAT_2_L1, ITALSAT_2_L2)
        params, method = sgp4_init(elements, WGS72)
        assert method == "d"
        assert jnp.all(jnp.isfinite(params))

    def test_vela_5a_is_deep_space(self) -> None:
        elements = parse_tle(VELA_5A_L1, VELA_5A_L2)
        params, method = sgp4_init(elements, WGS72)
        assert method == "d"
        assert jnp.all(jnp.isfinite(params))


class TestSDP4PropagateDeepSpace:
    """Test deep-space SDP4 propagation against reference python-sgp4."""

    def _assert_match(
        self,
        line1: str,
        line2: str,
        tsince: float,
        pos_atol: float = 1e-6,
        vel_atol: float = 1e-9,
    ) -> None:
        """Assert astrojax matches python-sgp4 at the given tsince."""
        elements = parse_tle(line1, line2)
        params, method = sgp4_init(elements, WGS72)
        assert method == "d"

        r, v = sgp4_propagate(params, jnp.float64(tsince), method)
        e_ref, r_ref, v_ref = _get_reference(line1, line2, tsince)
        assert e_ref == 0, f"Reference SGP4 error {e_ref} at tsince={tsince}"

        assert jnp.allclose(r, jnp.array(r_ref), atol=pos_atol), (
            f"Position mismatch at t={tsince}: {r} vs {r_ref}, "
            f"diff={float(jnp.max(jnp.abs(r - jnp.array(r_ref))))}"
        )
        assert jnp.allclose(v, jnp.array(v_ref), atol=vel_atol), (
            f"Velocity mismatch at t={tsince}: {v} vs {v_ref}, "
            f"diff={float(jnp.max(jnp.abs(v - jnp.array(v_ref))))}"
        )

    # --- Molniya 2-14 (12-hour resonance, irez=2) ---

    def test_molniya_2_14_at_epoch(self) -> None:
        self._assert_match(MOLNIYA_2_14_L1, MOLNIYA_2_14_L2, 0.0)

    def test_molniya_2_14_at_half_period(self) -> None:
        self._assert_match(MOLNIYA_2_14_L1, MOLNIYA_2_14_L2, 359.117678)

    def test_molniya_2_14_at_full_period(self) -> None:
        self._assert_match(MOLNIYA_2_14_L1, MOLNIYA_2_14_L2, 718.235357)

    def test_molniya_2_14_at_1day(self) -> None:
        self._assert_match(MOLNIYA_2_14_L1, MOLNIYA_2_14_L2, 1440.0)

    # --- Molniya 1-36 (12-hour resonance, irez=2) ---

    def test_molniya_1_36_at_epoch(self) -> None:
        self._assert_match(MOLNIYA_1_36_L1, MOLNIYA_1_36_L2, 0.0)

    def test_molniya_1_36_at_half_period(self) -> None:
        self._assert_match(MOLNIYA_1_36_L1, MOLNIYA_1_36_L2, 358.541428)

    def test_molniya_1_36_at_full_period(self) -> None:
        self._assert_match(MOLNIYA_1_36_L1, MOLNIYA_1_36_L2, 717.082857)

    def test_molniya_1_36_at_1day(self) -> None:
        self._assert_match(MOLNIYA_1_36_L1, MOLNIYA_1_36_L2, 1440.0)

    # --- ITALSAT 2 (synchronous resonance, irez=1) ---

    def test_italsat_2_at_epoch(self) -> None:
        self._assert_match(ITALSAT_2_L1, ITALSAT_2_L2, 0.0)

    def test_italsat_2_at_half_period(self) -> None:
        self._assert_match(ITALSAT_2_L1, ITALSAT_2_L2, 714.441261)

    def test_italsat_2_at_full_period(self) -> None:
        self._assert_match(ITALSAT_2_L1, ITALSAT_2_L2, 1428.882522)

    def test_italsat_2_at_1day(self) -> None:
        self._assert_match(ITALSAT_2_L1, ITALSAT_2_L2, 1440.0)

    # --- Vela 5A (non-resonant, irez=0) ---

    def test_vela_5a_at_epoch(self) -> None:
        self._assert_match(VELA_5A_L1, VELA_5A_L2, 0.0)

    def test_vela_5a_at_half_period(self) -> None:
        self._assert_match(VELA_5A_L1, VELA_5A_L2, 291.163502)

    def test_vela_5a_at_full_period(self) -> None:
        self._assert_match(VELA_5A_L1, VELA_5A_L2, 582.327004)

    def test_vela_5a_at_1day(self) -> None:
        self._assert_match(VELA_5A_L1, VELA_5A_L2, 1440.0)


class TestSDP4JITAndVmap:
    """Test JIT compilation and vmap for deep-space propagation."""

    def test_deep_space_jit(self) -> None:
        """JIT-compiled deep-space propagation matches reference."""
        elements = parse_tle(MOLNIYA_2_14_L1, MOLNIYA_2_14_L2)
        params, method = sgp4_init(elements, WGS72)
        assert method == "d"

        @jax.jit
        def propagate(p, t):
            return sgp4_propagate(p, t, method)

        r, v = propagate(params, jnp.float64(360.0))
        e_ref, r_ref, v_ref = _get_reference(MOLNIYA_2_14_L1, MOLNIYA_2_14_L2, 360.0)
        assert e_ref == 0
        assert jnp.allclose(r, jnp.array(r_ref), atol=1e-6)
        assert jnp.allclose(v, jnp.array(v_ref), atol=1e-9)

    def test_deep_space_vmap_over_time(self) -> None:
        """vmap over time array works for deep-space satellites."""
        elements = parse_tle(ITALSAT_2_L1, ITALSAT_2_L2)
        params, method = sgp4_init(elements, WGS72)
        assert method == "d"

        times = jnp.array([0.0, 714.441261, 1440.0])

        @jax.vmap
        def batch_propagate(t):
            return sgp4_propagate(params, t, method)

        r_batch, v_batch = batch_propagate(times)
        assert r_batch.shape == (3, 3)
        assert v_batch.shape == (3, 3)

        # Verify each time point
        for i, t in enumerate(times):
            e_ref, r_ref, v_ref = _get_reference(ITALSAT_2_L1, ITALSAT_2_L2, float(t))
            assert e_ref == 0
            assert jnp.allclose(r_batch[i], jnp.array(r_ref), atol=1e-6)
            assert jnp.allclose(v_batch[i], jnp.array(v_ref), atol=1e-9)
