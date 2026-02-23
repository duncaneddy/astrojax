"""Tests for SGP4 propagation against the reference python-sgp4 library."""

import jax
import jax.numpy as jnp
import pytest
from sgp4.api import WGS72 as SGP4_WGS72
from sgp4.api import Satrec

from astrojax._gp_record import GPRecord
from astrojax.sgp4 import (
    WGS72,
    create_sgp4_propagator,
    create_sgp4_propagator_from_elements,
    create_sgp4_propagator_from_gp_record,
    create_sgp4_propagator_from_omm,
    elements_to_array,
    gp_record_to_array,
    omm_to_array,
    parse_tle,
    sgp4_init,
    sgp4_init_jax,
    sgp4_propagate,
    sgp4_propagate_unified,
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


# ---------------------------------------------------------------------------
# ISS OMM fields matching the ISS TLE used above
# ---------------------------------------------------------------------------

ISS_OMM_FIELDS: dict[str, str] = {
    "EPOCH": "2008-09-20T12:25:40.104192",
    "MEAN_MOTION": "15.72125391",
    "ECCENTRICITY": "0.0006703",
    "INCLINATION": "51.6416",
    "RA_OF_ASC_NODE": "247.4627",
    "ARG_OF_PERICENTER": "130.5360",
    "MEAN_ANOMALY": "325.0288",
    "NORAD_CAT_ID": "25544",
    "BSTAR": "-0.11606e-4",
    "MEAN_MOTION_DOT": "-0.00002182",
    "MEAN_MOTION_DDOT": "0",
    "CLASSIFICATION_TYPE": "U",
    "OBJECT_ID": "1998-067A",
    "EPHEMERIS_TYPE": "0",
    "ELEMENT_SET_NO": "292",
    "REV_AT_EPOCH": "56353",
}


class TestCreateSGP4PropagatorFromElements:
    """Test create_sgp4_propagator_from_elements."""

    def test_returns_params_and_fn(self) -> None:
        """Returns a params array and callable propagation function."""
        elements = parse_tle(ISS_LINE1, ISS_LINE2)
        params, propagate_fn = create_sgp4_propagator_from_elements(elements)
        assert params.ndim == 1
        assert callable(propagate_fn)

    def test_matches_tle_factory(self) -> None:
        """Output matches create_sgp4_propagator exactly."""
        elements = parse_tle(ISS_LINE1, ISS_LINE2)
        params_elem, fn_elem = create_sgp4_propagator_from_elements(elements)
        params_tle, fn_tle = create_sgp4_propagator(ISS_LINE1, ISS_LINE2)

        # Params should be identical
        assert jnp.allclose(params_elem, params_tle, atol=1e-12)

        # Propagation should match
        r_elem, v_elem = fn_elem(jnp.float64(60.0))
        r_tle, v_tle = fn_tle(jnp.float64(60.0))
        assert jnp.allclose(r_elem, r_tle, atol=1e-12)
        assert jnp.allclose(v_elem, v_tle, atol=1e-12)

    def test_gravity_model_string(self) -> None:
        """Accepts gravity model as string."""
        elements = parse_tle(ISS_LINE1, ISS_LINE2)
        params, fn = create_sgp4_propagator_from_elements(elements, gravity="wgs84")
        r, v = fn(jnp.float64(0.0))
        assert jnp.all(jnp.isfinite(r))


class TestCreateSGP4PropagatorFromOMM:
    """Test create_sgp4_propagator_from_omm."""

    def test_returns_params_and_fn(self) -> None:
        """Returns a params array and callable propagation function."""
        params, propagate_fn = create_sgp4_propagator_from_omm(ISS_OMM_FIELDS)
        assert params.ndim == 1
        assert callable(propagate_fn)

    def test_results_are_finite(self) -> None:
        """Propagation produces finite results."""
        _, fn = create_sgp4_propagator_from_omm(ISS_OMM_FIELDS)
        r, v = fn(jnp.float64(60.0))
        assert jnp.all(jnp.isfinite(r))
        assert jnp.all(jnp.isfinite(v))

    def test_leo_magnitude(self) -> None:
        """Position magnitude is in LEO range (~6500-7200 km)."""
        _, fn = create_sgp4_propagator_from_omm(ISS_OMM_FIELDS)
        r, _ = fn(jnp.float64(0.0))
        r_mag = float(jnp.linalg.norm(r))
        assert 6000.0 < r_mag < 7200.0

    def test_jit_compatible(self) -> None:
        """Propagation closure works under jax.jit."""
        _, fn = create_sgp4_propagator_from_omm(ISS_OMM_FIELDS)
        jit_fn = jax.jit(fn)
        r, v = jit_fn(jnp.float64(60.0))
        assert jnp.all(jnp.isfinite(r))

    def test_gravity_model_string(self) -> None:
        """Accepts gravity model as string."""
        params, fn = create_sgp4_propagator_from_omm(ISS_OMM_FIELDS, gravity="wgs84")
        r, v = fn(jnp.float64(0.0))
        assert jnp.all(jnp.isfinite(r))

    def test_missing_field_raises(self) -> None:
        """Missing required field raises KeyError."""
        incomplete = {k: v for k, v in ISS_OMM_FIELDS.items() if k != "EPOCH"}
        with pytest.raises(KeyError):
            create_sgp4_propagator_from_omm(incomplete)


class TestCreateSGP4PropagatorFromGPRecord:
    """Test create_sgp4_propagator_from_gp_record."""

    def test_full_record(self) -> None:
        """Full GPRecord produces valid propagator."""
        record = GPRecord.from_json_dict(ISS_OMM_FIELDS)
        params, fn = create_sgp4_propagator_from_gp_record(record)
        r, v = fn(jnp.float64(0.0))
        assert jnp.all(jnp.isfinite(r))
        r_mag = float(jnp.linalg.norm(r))
        assert 6000.0 < r_mag < 7200.0

    def test_minimal_record(self) -> None:
        """GPRecord with only required fields works."""
        minimal = {
            "EPOCH": ISS_OMM_FIELDS["EPOCH"],
            "MEAN_MOTION": ISS_OMM_FIELDS["MEAN_MOTION"],
            "ECCENTRICITY": ISS_OMM_FIELDS["ECCENTRICITY"],
            "INCLINATION": ISS_OMM_FIELDS["INCLINATION"],
            "RA_OF_ASC_NODE": ISS_OMM_FIELDS["RA_OF_ASC_NODE"],
            "ARG_OF_PERICENTER": ISS_OMM_FIELDS["ARG_OF_PERICENTER"],
            "MEAN_ANOMALY": ISS_OMM_FIELDS["MEAN_ANOMALY"],
            "NORAD_CAT_ID": ISS_OMM_FIELDS["NORAD_CAT_ID"],
            "BSTAR": ISS_OMM_FIELDS["BSTAR"],
        }
        record = GPRecord.from_json_dict(minimal)
        params, fn = create_sgp4_propagator_from_gp_record(record)
        r, _ = fn(jnp.float64(0.0))
        assert jnp.all(jnp.isfinite(r))

    def test_missing_epoch_raises(self) -> None:
        """GPRecord without epoch raises KeyError."""
        no_epoch = {k: v for k, v in ISS_OMM_FIELDS.items() if k != "EPOCH"}
        record = GPRecord.from_json_dict(no_epoch)
        with pytest.raises(KeyError):
            create_sgp4_propagator_from_gp_record(record)

    def test_gravity_model_string(self) -> None:
        """Accepts gravity model as string."""
        record = GPRecord.from_json_dict(ISS_OMM_FIELDS)
        params, fn = create_sgp4_propagator_from_gp_record(record, gravity="wgs84")
        r, _ = fn(jnp.float64(0.0))
        assert jnp.all(jnp.isfinite(r))


# ---------------------------------------------------------------------------
# JIT-compilable init tests
# ---------------------------------------------------------------------------


class TestElementsToArray:
    """Test elements_to_array conversion."""

    def test_shape(self) -> None:
        """Returns array of shape (11,)."""
        elements = parse_tle(ISS_LINE1, ISS_LINE2)
        arr = elements_to_array(elements)
        assert arr.shape == (11,)

    def test_round_trip(self) -> None:
        """Values match the original elements."""
        elements = parse_tle(ISS_LINE1, ISS_LINE2)
        arr = elements_to_array(elements)
        assert float(arr[0]) == pytest.approx(elements.jdsatepoch)
        assert float(arr[1]) == pytest.approx(elements.jdsatepochF)
        assert float(arr[2]) == pytest.approx(elements.no_kozai)
        assert float(arr[3]) == pytest.approx(elements.ecco)
        assert float(arr[4]) == pytest.approx(elements.inclo)
        assert float(arr[5]) == pytest.approx(elements.nodeo)
        assert float(arr[6]) == pytest.approx(elements.argpo)
        assert float(arr[7]) == pytest.approx(elements.mo)
        assert float(arr[8]) == pytest.approx(elements.bstar)


class TestOMMToArray:
    """Test omm_to_array conversion."""

    def test_shape(self) -> None:
        """Returns array of shape (11,)."""
        arr = omm_to_array(ISS_OMM_FIELDS)
        assert arr.shape == (11,)

    def test_matches_elements_to_array(self) -> None:
        """omm_to_array matches manual parse_omm + elements_to_array."""
        from astrojax.sgp4 import parse_omm

        elements = parse_omm(ISS_OMM_FIELDS)
        arr_manual = elements_to_array(elements)
        arr_omm = omm_to_array(ISS_OMM_FIELDS)
        assert jnp.allclose(arr_manual, arr_omm, atol=1e-15)


class TestGPRecordToArray:
    """Test gp_record_to_array conversion."""

    def test_shape(self) -> None:
        """Returns array of shape (11,)."""
        record = GPRecord.from_json_dict(ISS_OMM_FIELDS)
        arr = gp_record_to_array(record)
        assert arr.shape == (11,)


class TestSGP4InitJax:
    """Test JIT-compilable SGP4 initialization."""

    def test_iss_near_earth_method_flag(self) -> None:
        """ISS should have method=0.0 (near-earth)."""
        elements = parse_tle(ISS_LINE1, ISS_LINE2)
        arr = elements_to_array(elements)
        params = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")
        from astrojax.sgp4._propagation import _IDX

        assert float(params[_IDX["method"]]) == 0.0

    def test_molniya_deep_space_method_flag(self) -> None:
        """Molniya should have method=1.0 (deep-space)."""
        elements = parse_tle(MOLNIYA_2_14_L1, MOLNIYA_2_14_L2)
        arr = elements_to_array(elements)
        params = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")
        from astrojax.sgp4._propagation import _IDX

        assert float(params[_IDX["method"]]) == 1.0

    def test_italsat_deep_space_method_flag(self) -> None:
        """ITALSAT (synchronous resonance) should have method=1.0."""
        elements = parse_tle(ITALSAT_2_L1, ITALSAT_2_L2)
        arr = elements_to_array(elements)
        params = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")
        from astrojax.sgp4._propagation import _IDX

        assert float(params[_IDX["method"]]) == 1.0

    def test_vela_deep_space_method_flag(self) -> None:
        """Vela (non-resonant deep-space) should have method=1.0."""
        elements = parse_tle(VELA_5A_L1, VELA_5A_L2)
        arr = elements_to_array(elements)
        params = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")
        from astrojax.sgp4._propagation import _IDX

        assert float(params[_IDX["method"]]) == 1.0

    def test_iss_init_parity(self) -> None:
        """sgp4_init_jax near-earth params match sgp4_init within tolerance."""
        elements = parse_tle(ISS_LINE1, ISS_LINE2)
        params_py, method = sgp4_init(elements, WGS72)
        assert method == "n"

        arr = elements_to_array(elements)
        params_jax = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")

        # Compare all params except the method flag (last element)
        assert jnp.allclose(params_jax[:-1], params_py[:-1], atol=1e-12), (
            f"Max diff: {float(jnp.max(jnp.abs(params_jax[:-1] - params_py[:-1])))}"
        )

    def test_polar_init_parity(self) -> None:
        """sgp4_init_jax near-earth params match sgp4_init for polar orbit."""
        elements = parse_tle(POLAR_LINE1, POLAR_LINE2)
        params_py, method = sgp4_init(elements, WGS72)
        assert method == "n"

        arr = elements_to_array(elements)
        params_jax = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")

        assert jnp.allclose(params_jax[:-1], params_py[:-1], atol=1e-12), (
            f"Max diff: {float(jnp.max(jnp.abs(params_jax[:-1] - params_py[:-1])))}"
        )

    def test_molniya_init_parity(self) -> None:
        """sgp4_init_jax deep-space params match sgp4_init for Molniya."""
        elements = parse_tle(MOLNIYA_2_14_L1, MOLNIYA_2_14_L2)
        params_py, method = sgp4_init(elements, WGS72)
        assert method == "d"

        arr = elements_to_array(elements)
        params_jax = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")

        assert jnp.allclose(params_jax[:-1], params_py[:-1], atol=1e-12), (
            f"Max diff: {float(jnp.max(jnp.abs(params_jax[:-1] - params_py[:-1])))}"
        )

    def test_italsat_init_parity(self) -> None:
        """sgp4_init_jax deep-space params match sgp4_init for ITALSAT."""
        elements = parse_tle(ITALSAT_2_L1, ITALSAT_2_L2)
        params_py, method = sgp4_init(elements, WGS72)
        assert method == "d"

        arr = elements_to_array(elements)
        params_jax = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")

        assert jnp.allclose(params_jax[:-1], params_py[:-1], atol=1e-12), (
            f"Max diff: {float(jnp.max(jnp.abs(params_jax[:-1] - params_py[:-1])))}"
        )

    def test_vela_init_parity(self) -> None:
        """sgp4_init_jax deep-space params match sgp4_init for Vela."""
        elements = parse_tle(VELA_5A_L1, VELA_5A_L2)
        params_py, method = sgp4_init(elements, WGS72)
        assert method == "d"

        arr = elements_to_array(elements)
        params_jax = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")

        assert jnp.allclose(params_jax[:-1], params_py[:-1], atol=1e-12), (
            f"Max diff: {float(jnp.max(jnp.abs(params_jax[:-1] - params_py[:-1])))}"
        )

    def test_params_are_finite(self) -> None:
        """All parameter values are finite."""
        elements = parse_tle(ISS_LINE1, ISS_LINE2)
        arr = elements_to_array(elements)
        params = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")
        assert jnp.all(jnp.isfinite(params))

    def test_jit_compilation(self) -> None:
        """sgp4_init_jax compiles and produces correct results under JIT."""
        elements = parse_tle(ISS_LINE1, ISS_LINE2)
        arr = elements_to_array(elements)

        init_jit = jax.jit(sgp4_init_jax, static_argnames=("gravity", "opsmode"))
        params_jit = init_jit(arr, gravity=WGS72, opsmode="i")
        params_eager = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")

        assert jnp.allclose(params_jit, params_eager, atol=1e-15)

    def test_jit_compilation_deep_space(self) -> None:
        """sgp4_init_jax compiles for deep-space satellites under JIT."""
        elements = parse_tle(MOLNIYA_2_14_L1, MOLNIYA_2_14_L2)
        arr = elements_to_array(elements)

        init_jit = jax.jit(sgp4_init_jax, static_argnames=("gravity", "opsmode"))
        params_jit = init_jit(arr, gravity=WGS72, opsmode="i")
        params_eager = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")

        assert jnp.allclose(params_jit, params_eager, atol=1e-15)

    def test_omm_init(self) -> None:
        """sgp4_init_jax works from OMM fields via omm_to_array."""
        arr = omm_to_array(ISS_OMM_FIELDS)
        params = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")
        assert jnp.all(jnp.isfinite(params))

    def test_gp_record_init(self) -> None:
        """sgp4_init_jax works from GPRecord via gp_record_to_array."""
        record = GPRecord.from_json_dict(ISS_OMM_FIELDS)
        arr = gp_record_to_array(record)
        params = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")
        assert jnp.all(jnp.isfinite(params))


class TestSGP4PropagateUnified:
    """Test unified propagation with auto-detection."""

    def test_iss_matches_reference(self) -> None:
        """Unified propagation for ISS matches reference python-sgp4."""
        elements = parse_tle(ISS_LINE1, ISS_LINE2)
        arr = elements_to_array(elements)
        params = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")

        r, v = sgp4_propagate_unified(params, jnp.float64(60.0))
        e_ref, r_ref, v_ref = _get_reference(ISS_LINE1, ISS_LINE2, 60.0)
        assert e_ref == 0
        assert jnp.allclose(r, jnp.array(r_ref), atol=1e-6)
        assert jnp.allclose(v, jnp.array(v_ref), atol=1e-9)

    def test_molniya_matches_reference(self) -> None:
        """Unified propagation for Molniya matches reference."""
        elements = parse_tle(MOLNIYA_2_14_L1, MOLNIYA_2_14_L2)
        arr = elements_to_array(elements)
        params = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")

        r, v = sgp4_propagate_unified(params, jnp.float64(360.0))
        e_ref, r_ref, v_ref = _get_reference(MOLNIYA_2_14_L1, MOLNIYA_2_14_L2, 360.0)
        assert e_ref == 0
        assert jnp.allclose(r, jnp.array(r_ref), atol=1e-6)
        assert jnp.allclose(v, jnp.array(v_ref), atol=1e-9)

    def test_italsat_matches_reference(self) -> None:
        """Unified propagation for ITALSAT matches reference."""
        elements = parse_tle(ITALSAT_2_L1, ITALSAT_2_L2)
        arr = elements_to_array(elements)
        params = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")

        r, v = sgp4_propagate_unified(params, jnp.float64(1440.0))
        e_ref, r_ref, v_ref = _get_reference(ITALSAT_2_L1, ITALSAT_2_L2, 1440.0)
        assert e_ref == 0
        assert jnp.allclose(r, jnp.array(r_ref), atol=1e-6)
        assert jnp.allclose(v, jnp.array(v_ref), atol=1e-9)

    def test_vela_matches_reference(self) -> None:
        """Unified propagation for Vela matches reference."""
        elements = parse_tle(VELA_5A_L1, VELA_5A_L2)
        arr = elements_to_array(elements)
        params = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")

        r, v = sgp4_propagate_unified(params, jnp.float64(1440.0))
        e_ref, r_ref, v_ref = _get_reference(VELA_5A_L1, VELA_5A_L2, 1440.0)
        assert e_ref == 0
        assert jnp.allclose(r, jnp.array(r_ref), atol=1e-6)
        assert jnp.allclose(v, jnp.array(v_ref), atol=1e-9)

    def test_backward_compat_with_sgp4_init(self) -> None:
        """sgp4_propagate_unified works with params from sgp4_init."""
        elements = parse_tle(ISS_LINE1, ISS_LINE2)
        params, method = sgp4_init(elements, WGS72)
        assert method == "n"

        r, v = sgp4_propagate_unified(params, jnp.float64(60.0))
        e_ref, r_ref, v_ref = _get_reference(ISS_LINE1, ISS_LINE2, 60.0)
        assert e_ref == 0
        assert jnp.allclose(r, jnp.array(r_ref), atol=1e-6)

    def test_backward_compat_deep_space(self) -> None:
        """sgp4_propagate_unified works with deep-space params from sgp4_init."""
        elements = parse_tle(MOLNIYA_2_14_L1, MOLNIYA_2_14_L2)
        params, method = sgp4_init(elements, WGS72)
        assert method == "d"

        r, v = sgp4_propagate_unified(params, jnp.float64(360.0))
        e_ref, r_ref, v_ref = _get_reference(MOLNIYA_2_14_L1, MOLNIYA_2_14_L2, 360.0)
        assert e_ref == 0
        assert jnp.allclose(r, jnp.array(r_ref), atol=1e-6)

    def test_jit_compiled(self) -> None:
        """Unified propagation works under JIT."""
        elements = parse_tle(ISS_LINE1, ISS_LINE2)
        arr = elements_to_array(elements)
        params = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")

        prop_jit = jax.jit(sgp4_propagate_unified)
        r, v = prop_jit(params, jnp.float64(60.0))
        e_ref, r_ref, v_ref = _get_reference(ISS_LINE1, ISS_LINE2, 60.0)
        assert e_ref == 0
        assert jnp.allclose(r, jnp.array(r_ref), atol=1e-6)

    def test_vmap_over_time(self) -> None:
        """vmap over time works with unified propagation."""
        elements = parse_tle(ISS_LINE1, ISS_LINE2)
        arr = elements_to_array(elements)
        params = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")

        times = jnp.array([0.0, 60.0, 360.0, 1440.0])
        r_batch, v_batch = jax.vmap(lambda t: sgp4_propagate_unified(params, t))(times)
        assert r_batch.shape == (4, 3)
        assert v_batch.shape == (4, 3)

        for i, t in enumerate(times):
            e_ref, r_ref, v_ref = _get_reference(ISS_LINE1, ISS_LINE2, float(t))
            assert e_ref == 0
            assert jnp.allclose(r_batch[i], jnp.array(r_ref), atol=1e-6)

    def test_omm_end_to_end(self) -> None:
        """End-to-end: OMM -> init_jax -> propagate_unified."""
        arr = omm_to_array(ISS_OMM_FIELDS)
        params = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")
        r, v = sgp4_propagate_unified(params, jnp.float64(60.0))
        assert jnp.all(jnp.isfinite(r))
        r_mag = float(jnp.linalg.norm(r))
        assert 6000.0 < r_mag < 7200.0

    def test_gp_record_end_to_end(self) -> None:
        """End-to-end: GPRecord -> init_jax -> propagate_unified."""
        record = GPRecord.from_json_dict(ISS_OMM_FIELDS)
        arr = gp_record_to_array(record)
        params = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")
        r, v = sgp4_propagate_unified(params, jnp.float64(60.0))
        assert jnp.all(jnp.isfinite(r))
        r_mag = float(jnp.linalg.norm(r))
        assert 6000.0 < r_mag < 7200.0


class TestSGP4InitJaxVmap:
    """Test vmap over satellite initialization."""

    def test_vmap_over_satellites(self) -> None:
        """Batch-init multiple satellites and verify results."""
        tle_pairs = [
            (ISS_LINE1, ISS_LINE2),
            (POLAR_LINE1, POLAR_LINE2),
            (MOLNIYA_2_14_L1, MOLNIYA_2_14_L2),
            (ITALSAT_2_L1, ITALSAT_2_L2),
        ]

        arrays = []
        for l1, l2 in tle_pairs:
            elements = parse_tle(l1, l2)
            arrays.append(elements_to_array(elements))

        batch = jnp.stack(arrays)
        assert batch.shape == (4, 11)

        init_vmap = jax.vmap(
            lambda e: sgp4_init_jax(e, gravity=WGS72, opsmode="i"),
        )
        params_batch = init_vmap(batch)
        assert params_batch.shape[0] == 4

        # Verify each matches individual init
        for i, (l1, l2) in enumerate(tle_pairs):
            elements = parse_tle(l1, l2)
            arr = elements_to_array(elements)
            params_single = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")
            assert jnp.allclose(params_batch[i], params_single, atol=1e-12)

    def test_vmap_init_then_propagate(self) -> None:
        """vmap init + propagate produces valid results for mixed sat types."""
        tle_pairs = [
            (ISS_LINE1, ISS_LINE2),
            (MOLNIYA_2_14_L1, MOLNIYA_2_14_L2),
        ]

        arrays = []
        for l1, l2 in tle_pairs:
            elements = parse_tle(l1, l2)
            arrays.append(elements_to_array(elements))
        batch = jnp.stack(arrays)

        init_vmap = jax.vmap(
            lambda e: sgp4_init_jax(e, gravity=WGS72, opsmode="i"),
        )
        params_batch = init_vmap(batch)

        # Propagate all at t=0
        prop_vmap = jax.vmap(lambda p: sgp4_propagate_unified(p, jnp.float64(0.0)))
        r_batch, v_batch = prop_vmap(params_batch)
        assert r_batch.shape == (2, 3)
        assert jnp.all(jnp.isfinite(r_batch))


class TestSGP4InitJaxEndToEnd:
    """End-to-end tests: init_jax + propagate_unified vs reference."""

    def _assert_match(
        self,
        line1: str,
        line2: str,
        tsince: float,
        pos_atol: float = 1e-6,
        vel_atol: float = 1e-9,
    ) -> None:
        """Assert sgp4_init_jax + sgp4_propagate_unified matches reference."""
        elements = parse_tle(line1, line2)
        arr = elements_to_array(elements)
        params = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")

        r, v = sgp4_propagate_unified(params, jnp.float64(tsince))
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

    def test_iss_multiple_times(self) -> None:
        """ISS at multiple time points."""
        for t in [0.0, 60.0, 360.0, 1440.0, -60.0]:
            self._assert_match(ISS_LINE1, ISS_LINE2, t)

    def test_molniya_multiple_times(self) -> None:
        """Molniya 2-14 at multiple time points."""
        for t in [0.0, 359.117678, 718.235357, 1440.0]:
            self._assert_match(MOLNIYA_2_14_L1, MOLNIYA_2_14_L2, t)

    def test_italsat_multiple_times(self) -> None:
        """ITALSAT 2 at multiple time points."""
        for t in [0.0, 714.441261, 1428.882522, 1440.0]:
            self._assert_match(ITALSAT_2_L1, ITALSAT_2_L2, t)

    def test_vela_multiple_times(self) -> None:
        """Vela 5A at multiple time points."""
        for t in [0.0, 291.163502, 582.327004, 1440.0]:
            self._assert_match(VELA_5A_L1, VELA_5A_L2, t)

    def test_molniya_1_36_multiple_times(self) -> None:
        """Molniya 1-36 at multiple time points."""
        for t in [0.0, 358.541428, 717.082857, 1440.0]:
            self._assert_match(MOLNIYA_1_36_L1, MOLNIYA_1_36_L2, t)


class TestSGP4Differentiability:
    """Test gradient computation through init + propagate.

    Note: sgp4_propagate_unified uses jax.lax.cond which traces both
    branches. The deep-space branch contains a while_loop that is not
    compatible with reverse-mode AD. Use sgp4_propagate with method='n'
    for differentiable near-earth propagation.
    """

    def test_grad_through_near_earth_propagate(self) -> None:
        """jax.grad through near-earth propagation produces finite gradients."""
        elements = parse_tle(ISS_LINE1, ISS_LINE2)
        arr = elements_to_array(elements)
        params = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")

        def loss(p):
            r, v = sgp4_propagate(p, jnp.float64(60.0), "n")
            return jnp.sum(r**2)

        grad_fn = jax.grad(loss)
        grads = grad_fn(params)
        assert jnp.all(jnp.isfinite(grads)), "Gradients contain NaN or Inf"

    def test_grad_through_init_and_near_earth_propagate(self) -> None:
        """jax.grad through init + near-earth propagate produces finite gradients."""
        elements = parse_tle(ISS_LINE1, ISS_LINE2)
        arr = elements_to_array(elements)

        @jax.jit
        def loss(elems):
            p = sgp4_init_jax(elems, gravity=WGS72, opsmode="i")
            r, v = sgp4_propagate(p, jnp.float64(60.0), "n")
            return jnp.sum(r**2)

        grad_fn = jax.grad(loss)
        grads = grad_fn(arr)
        assert grads.shape == (11,)
        assert jnp.all(jnp.isfinite(grads)), f"Gradients contain NaN/Inf: {grads}"
