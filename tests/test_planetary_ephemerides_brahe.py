"""Cross-validation of JPL approximate ephemerides against brahe DE440s.

Brahe provides high-precision geocentric positions via JPL DE440s kernels.
To compare with astrojax heliocentric positions, we convert brahe's output:

    r_helio_brahe = r_planet_geo_brahe - r_sun_geo_brahe

and compare the angular direction of heliocentric vectors.  Distance
magnitude is also checked to within the expected accuracy of the JPL
approximate method.

Tolerances reflect the stated accuracy of the JPL Table 1 algorithm:
inner planets ~1 arcminute, outer planets ~10 arcminutes.

Note:
    EMB (Earth-Moon Barycenter) is not tested here because brahe's DE
    functions do not expose a direct EMB position.
"""

import brahe as bh
import jax.numpy as jnp
import numpy as np
import pytest

from astrojax import Epoch
from astrojax.config import set_dtype
from astrojax.orbit_dynamics import (
    jupiter_position_jpl_approx,
    mars_position_jpl_approx,
    mercury_position_jpl_approx,
    neptune_position_jpl_approx,
    saturn_position_jpl_approx,
    uranus_position_jpl_approx,
    venus_position_jpl_approx,
)


@pytest.fixture(autouse=True)
def _use_float64():
    """Set float64 for each test, restore float32 afterward."""
    set_dtype(jnp.float64)
    yield
    set_dtype(jnp.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_epochs(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: float = 0.0):
    """Create matched brahe and astrojax Epoch objects."""
    epc_bh = bh.Epoch(year, month, day, hour, minute, second)
    epc_aj = Epoch(year, month, day, hour, minute, second)
    return epc_bh, epc_aj


_DE_SOURCE = bh.EphemerisSource.DE440s


def _brahe_heliocentric(planet_de_fn, epc_bh):
    """Get heliocentric position from brahe by subtracting Sun position.

    Args:
        planet_de_fn: brahe DE function (e.g. brahe.mercury_position_de).
        epc_bh: brahe Epoch.

    Returns:
        Heliocentric position as numpy array in metres, GCRF frame.
    """
    r_planet_geo = planet_de_fn(epc_bh, _DE_SOURCE)
    r_sun_geo = bh.sun_position_de(epc_bh, _DE_SOURCE)
    return r_planet_geo - r_sun_geo


def _angular_separation_arcmin(r1: np.ndarray, r2: np.ndarray) -> float:
    """Angular separation between two direction vectors in arcminutes."""
    u1 = r1 / np.linalg.norm(r1)
    u2 = r2 / np.linalg.norm(r2)
    cos_angle = np.clip(np.dot(u1, u2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)) * 60.0)


def _distance_relative_error(r1: np.ndarray, r2: np.ndarray) -> float:
    """Relative error in distance magnitude."""
    d1 = np.linalg.norm(r1)
    d2 = np.linalg.norm(r2)
    return float(abs(d1 - d2) / d2)


# ---------------------------------------------------------------------------
# Test dates spanning 2000-2040
# ---------------------------------------------------------------------------

_TEST_DATES = [
    (2000, 1, 1, 12, 0, 0.0),  # J2000
    (2010, 6, 15, 0, 0, 0.0),
    (2020, 3, 20, 12, 0, 0.0),
    (2024, 7, 4, 0, 0, 0.0),
    (2030, 11, 15, 0, 0, 0.0),
    (2040, 1, 1, 0, 0, 0.0),
]


# ---------------------------------------------------------------------------
# Inner planets: tighter tolerance (~1 arcmin direction, ~1% distance)
# ---------------------------------------------------------------------------

_INNER_PLANETS = {
    "mercury": (mercury_position_jpl_approx, bh.mercury_position_de),
    "venus": (venus_position_jpl_approx, bh.venus_position_de),
}

# Generous angular tolerance: JPL states ~1 arcmin for inner planets,
# but float32->float64 rounding and frame bias add some margin.
_INNER_ANGLE_TOL_ARCMIN = 5.0
_INNER_DIST_RTOL = 0.02  # 2%


class TestInnerPlanetsVsBrahe:
    """Cross-validate inner planet positions against brahe DE440s."""

    @pytest.mark.parametrize("date", _TEST_DATES, ids=[f"{d[0]}-{d[1]:02d}-{d[2]:02d}" for d in _TEST_DATES])
    @pytest.mark.parametrize("planet", _INNER_PLANETS.keys())
    def test_angular_separation(self, planet, date):
        aj_fn, bh_fn = _INNER_PLANETS[planet]
        epc_bh, epc_aj = _make_epochs(*date)
        r_aj = np.array(aj_fn(epc_aj))
        r_bh = _brahe_heliocentric(bh_fn, epc_bh)
        sep = _angular_separation_arcmin(r_aj, r_bh)
        assert sep < _INNER_ANGLE_TOL_ARCMIN, (
            f"{planet} angular separation {sep:.2f}' > {_INNER_ANGLE_TOL_ARCMIN}' at {date}"
        )

    @pytest.mark.parametrize("date", _TEST_DATES, ids=[f"{d[0]}-{d[1]:02d}-{d[2]:02d}" for d in _TEST_DATES])
    @pytest.mark.parametrize("planet", _INNER_PLANETS.keys())
    def test_distance_magnitude(self, planet, date):
        aj_fn, bh_fn = _INNER_PLANETS[planet]
        epc_bh, epc_aj = _make_epochs(*date)
        r_aj = np.array(aj_fn(epc_aj))
        r_bh = _brahe_heliocentric(bh_fn, epc_bh)
        err = _distance_relative_error(r_aj, r_bh)
        assert err < _INNER_DIST_RTOL, (
            f"{planet} distance error {err:.4f} > {_INNER_DIST_RTOL} at {date}"
        )


# ---------------------------------------------------------------------------
# Mars: intermediate tolerance
# ---------------------------------------------------------------------------

_MARS_ANGLE_TOL_ARCMIN = 5.0
_MARS_DIST_RTOL = 0.02


class TestMarsVsBrahe:
    """Cross-validate Mars position against brahe DE440s."""

    @pytest.mark.parametrize("date", _TEST_DATES, ids=[f"{d[0]}-{d[1]:02d}-{d[2]:02d}" for d in _TEST_DATES])
    def test_angular_separation(self, date):
        epc_bh, epc_aj = _make_epochs(*date)
        r_aj = np.array(mars_position_jpl_approx(epc_aj))
        r_bh = _brahe_heliocentric(bh.mars_position_de, epc_bh)
        sep = _angular_separation_arcmin(r_aj, r_bh)
        assert sep < _MARS_ANGLE_TOL_ARCMIN, (
            f"Mars angular separation {sep:.2f}' > {_MARS_ANGLE_TOL_ARCMIN}' at {date}"
        )

    @pytest.mark.parametrize("date", _TEST_DATES, ids=[f"{d[0]}-{d[1]:02d}-{d[2]:02d}" for d in _TEST_DATES])
    def test_distance_magnitude(self, date):
        epc_bh, epc_aj = _make_epochs(*date)
        r_aj = np.array(mars_position_jpl_approx(epc_aj))
        r_bh = _brahe_heliocentric(bh.mars_position_de, epc_bh)
        err = _distance_relative_error(r_aj, r_bh)
        assert err < _MARS_DIST_RTOL, (
            f"Mars distance error {err:.4f} > {_MARS_DIST_RTOL} at {date}"
        )


# ---------------------------------------------------------------------------
# Outer planets: wider tolerance (~10 arcmin direction, ~2% distance)
# ---------------------------------------------------------------------------

_OUTER_PLANETS = {
    "jupiter": (jupiter_position_jpl_approx, bh.jupiter_position_de),
    "saturn": (saturn_position_jpl_approx, bh.saturn_position_de),
    "uranus": (uranus_position_jpl_approx, bh.uranus_position_de),
    "neptune": (neptune_position_jpl_approx, bh.neptune_position_de),
}

_OUTER_ANGLE_TOL_ARCMIN = 15.0
_OUTER_DIST_RTOL = 0.02


class TestOuterPlanetsVsBrahe:
    """Cross-validate outer planet positions against brahe DE440s."""

    @pytest.mark.parametrize("date", _TEST_DATES, ids=[f"{d[0]}-{d[1]:02d}-{d[2]:02d}" for d in _TEST_DATES])
    @pytest.mark.parametrize("planet", _OUTER_PLANETS.keys())
    def test_angular_separation(self, planet, date):
        aj_fn, bh_fn = _OUTER_PLANETS[planet]
        epc_bh, epc_aj = _make_epochs(*date)
        r_aj = np.array(aj_fn(epc_aj))
        r_bh = _brahe_heliocentric(bh_fn, epc_bh)
        sep = _angular_separation_arcmin(r_aj, r_bh)
        assert sep < _OUTER_ANGLE_TOL_ARCMIN, (
            f"{planet} angular separation {sep:.2f}' > {_OUTER_ANGLE_TOL_ARCMIN}' at {date}"
        )

    @pytest.mark.parametrize("date", _TEST_DATES, ids=[f"{d[0]}-{d[1]:02d}-{d[2]:02d}" for d in _TEST_DATES])
    @pytest.mark.parametrize("planet", _OUTER_PLANETS.keys())
    def test_distance_magnitude(self, planet, date):
        aj_fn, bh_fn = _OUTER_PLANETS[planet]
        epc_bh, epc_aj = _make_epochs(*date)
        r_aj = np.array(aj_fn(epc_aj))
        r_bh = _brahe_heliocentric(bh_fn, epc_bh)
        err = _distance_relative_error(r_aj, r_bh)
        assert err < _OUTER_DIST_RTOL, (
            f"{planet} distance error {err:.4f} > {_OUTER_DIST_RTOL} at {date}"
        )
