"""Cross-validation of NRLMSISE-00 atmospheric density against brahe 1.0+.

Both libraries implement the NRLMSISE-00 empirical atmosphere model.  The
core GTD7D algorithm is identical, but differences arise from:

  - Space weather data files (download timestamps, caching strategies).
  - Space weather interpolation/lookup (F10.7, Ap array indexing).
  - Epoch-to-calendar conversions (fractional second handling).

Because these upstream inputs differ slightly, the computed densities can
diverge by up to ~15 % in the worst case (epochs where F10.7 or Ap is
changing rapidly).  The tolerance chosen (``_DENSITY_RTOL = 0.15``) covers
the observed spread while still catching gross implementation errors.

For epochs in well-sampled regions (2024+) the agreement is typically
better than 1 %.
"""

from __future__ import annotations

import brahe as bh
import jax.numpy as jnp
import numpy as np
import pytest

from astrojax import Epoch
from astrojax.coordinates import position_geodetic_to_ecef
from astrojax.orbit_dynamics.nrlmsise00 import (
    density_nrlmsise00,
    density_nrlmsise00_geod,
)
from astrojax.space_weather import load_default_sw

# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------
# The NRLMSISE-00 core algorithm is the same in both libraries, but
# different space-weather data sources / interpolation schemes cause
# density offsets of ~2-15 %.  We use 15 % relative tolerance to
# accommodate the worst observed cases while still detecting any
# algorithmic regression.
_DENSITY_RTOL = 0.15


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sw():
    """Load astrojax space weather data (module-scoped for speed)."""
    return load_default_sw()


@pytest.fixture(scope="module", autouse=True)
def _init_brahe():
    """Initialize brahe EOP and space weather providers."""
    bh.initialize_eop()
    bh.initialize_sw()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_epochs(year, month, day, hour=0, minute=0, second=0.0):
    """Create matched brahe and astrojax Epoch objects."""
    epc_bh = bh.Epoch(year, month, day, hour, minute, second)
    epc_aj = Epoch(year, month, day, hour, minute, second)
    return epc_bh, epc_aj


# ---------------------------------------------------------------------------
# Test epochs (summer solstice, winter solstice, equinox)
# ---------------------------------------------------------------------------

_EPOCHS = [
    # (year, month, day, hour, minute, second) -- label
    (2024, 6, 15, 12, 0, 0),  # near summer solstice
    (2024, 12, 21, 0, 0, 0),  # winter solstice
    (2024, 3, 20, 12, 0, 0),  # vernal equinox
]


# ---------------------------------------------------------------------------
# Test: density_nrlmsise00_geod  (geodetic interface)
# ---------------------------------------------------------------------------


class TestDensityNrlmsise00GeodBrahe:
    """Cross-validate density_nrlmsise00_geod against brahe.density_nrlmsise00_geod."""

    @pytest.mark.parametrize("alt_km", [200, 300, 400, 600])
    @pytest.mark.parametrize("lat_deg", [0.0, 45.0, -60.0])
    @pytest.mark.parametrize(
        "year,month,day,hour,minute,second",
        _EPOCHS,
    )
    def test_density_geod_matches_brahe(
        self,
        sw,
        year,
        month,
        day,
        hour,
        minute,
        second,
        lat_deg,
        alt_km,
    ):
        """Geodetic density should match brahe within space-weather tolerance.

        Args:
            sw: Space weather data fixture.
            year: Epoch year.
            month: Epoch month.
            day: Epoch day.
            hour: Epoch hour.
            minute: Epoch minute.
            second: Epoch second.
            lat_deg: Geodetic latitude in degrees.
            alt_km: Altitude in kilometres.
        """
        lon_deg = -70.0
        alt_m = alt_km * 1000.0

        epc_bh, epc_aj = _make_epochs(year, month, day, hour, minute, second)

        # brahe
        geod_bh = np.array([lon_deg, lat_deg, alt_m])
        rho_bh = bh.density_nrlmsise00_geod(epc_bh, geod_bh)

        # astrojax
        geod_aj = jnp.array([lon_deg, lat_deg, alt_m])
        rho_aj = float(density_nrlmsise00_geod(sw, epc_aj, geod_aj))

        assert rho_aj > 0.0, "Density must be positive"
        assert rho_bh > 0.0, "Brahe density must be positive"
        np.testing.assert_allclose(
            rho_aj,
            rho_bh,
            rtol=_DENSITY_RTOL,
            err_msg=(
                f"NRLMSISE-00 geod mismatch at "
                f"{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:04.1f}, "
                f"lon={lon_deg}, lat={lat_deg}, alt={alt_km} km"
            ),
        )


# ---------------------------------------------------------------------------
# Test: density_nrlmsise00  (ECEF interface)
# ---------------------------------------------------------------------------


class TestDensityNrlmsise00EcefBrahe:
    """Cross-validate density_nrlmsise00 (ECEF) against brahe.density_nrlmsise00."""

    @pytest.mark.parametrize("alt_km", [200, 400, 600])
    @pytest.mark.parametrize("lat_deg", [0.0, 45.0])
    def test_density_ecef_matches_brahe(self, sw, alt_km, lat_deg):
        """ECEF density should match brahe within space-weather tolerance.

        Args:
            sw: Space weather data fixture.
            alt_km: Altitude in kilometres.
            lat_deg: Geodetic latitude in degrees.
        """
        lon_deg = 0.0
        alt_m = alt_km * 1000.0

        epc_bh, epc_aj = _make_epochs(2024, 6, 15, 12, 0, 0)

        # Build ECEF from geodetic -- each library converts internally
        geod_bh = np.array([lon_deg, lat_deg, alt_m])
        x_ecef_bh = bh.position_geodetic_to_ecef(geod_bh, bh.AngleFormat.DEGREES)
        rho_bh = bh.density_nrlmsise00(epc_bh, x_ecef_bh)

        geod_aj = jnp.array([lon_deg, lat_deg, alt_m])
        x_ecef_aj = position_geodetic_to_ecef(geod_aj, use_degrees=True)
        rho_aj = float(density_nrlmsise00(sw, epc_aj, x_ecef_aj))

        assert rho_aj > 0.0, "Density must be positive"
        assert rho_bh > 0.0, "Brahe density must be positive"
        np.testing.assert_allclose(
            rho_aj,
            rho_bh,
            rtol=_DENSITY_RTOL,
            err_msg=(f"NRLMSISE-00 ECEF mismatch at alt={alt_km} km, lat={lat_deg}"),
        )


# ---------------------------------------------------------------------------
# Test: altitude monotonicity (both libraries agree on ordering)
# ---------------------------------------------------------------------------


class TestDensityMonotonicityBrahe:
    """Verify both libraries agree that density decreases with altitude."""

    def test_density_decreases_with_altitude(self, sw):
        """Density at lower altitude should exceed density at higher altitude.

        Args:
            sw: Space weather data fixture.
        """
        epc_bh, epc_aj = _make_epochs(2024, 6, 15, 12, 0, 0)
        lon_deg, lat_deg = 0.0, 0.0
        altitudes_km = [200, 300, 400, 500, 600, 700, 800]

        densities_bh = []
        densities_aj = []
        for alt_km in altitudes_km:
            alt_m = alt_km * 1000.0
            geod_bh = np.array([lon_deg, lat_deg, alt_m])
            geod_aj = jnp.array([lon_deg, lat_deg, alt_m])

            densities_bh.append(bh.density_nrlmsise00_geod(epc_bh, geod_bh))
            densities_aj.append(float(density_nrlmsise00_geod(sw, epc_aj, geod_aj)))

        # Both should be strictly decreasing
        for i in range(len(altitudes_km) - 1):
            assert densities_bh[i] > densities_bh[i + 1], (
                f"brahe: density at {altitudes_km[i]} km should exceed "
                f"density at {altitudes_km[i + 1]} km"
            )
            assert densities_aj[i] > densities_aj[i + 1], (
                f"astrojax: density at {altitudes_km[i]} km should exceed "
                f"density at {altitudes_km[i + 1]} km"
            )


# ---------------------------------------------------------------------------
# Test: seasonal variation (both libraries see different densities)
# ---------------------------------------------------------------------------


class TestDensitySeasonalVariationBrahe:
    """Verify both libraries produce distinct densities for different epochs."""

    def test_seasonal_variation(self, sw):
        """Density at a fixed location should differ across seasons.

        Args:
            sw: Space weather data fixture.
        """
        geod_bh = np.array([0.0, 0.0, 400e3])
        geod_aj = jnp.array([0.0, 0.0, 400e3])

        densities_bh = []
        densities_aj = []
        for year, month, day, hour, minute, second in _EPOCHS:
            epc_bh, epc_aj = _make_epochs(year, month, day, hour, minute, second)
            densities_bh.append(bh.density_nrlmsise00_geod(epc_bh, geod_bh))
            densities_aj.append(float(density_nrlmsise00_geod(sw, epc_aj, geod_aj)))

        # At least two of the three seasons should produce different densities
        assert len(set(densities_bh)) > 1, "brahe should show seasonal variation"
        assert len(set(densities_aj)) > 1, "astrojax should show seasonal variation"


# ---------------------------------------------------------------------------
# Test: ECEF and geodetic consistency within astrojax
# ---------------------------------------------------------------------------


class TestDensityConsistencyBrahe:
    """Verify that ECEF and geodetic paths produce consistent results."""

    @pytest.mark.parametrize("lat_deg", [0.0, 45.0, -60.0])
    @pytest.mark.parametrize("alt_km", [200, 300, 400])
    def test_ecef_geod_consistency(self, sw, lat_deg, alt_km):
        """ECEF-based and geod-based densities should agree within astrojax.

        Args:
            sw: Space weather data fixture.
            lat_deg: Geodetic latitude in degrees.
            alt_km: Altitude in kilometres.
        """
        lon_deg = -70.0
        alt_m = alt_km * 1000.0

        epc_aj = Epoch(2024, 6, 15, 12, 0, 0)
        geod_aj = jnp.array([lon_deg, lat_deg, alt_m])

        rho_geod = float(density_nrlmsise00_geod(sw, epc_aj, geod_aj))

        x_ecef_aj = position_geodetic_to_ecef(geod_aj, use_degrees=True)
        rho_ecef = float(density_nrlmsise00(sw, epc_aj, x_ecef_aj))

        np.testing.assert_allclose(
            rho_ecef,
            rho_geod,
            rtol=1e-10,
            err_msg=(
                f"ECEF vs geod density mismatch in astrojax at lat={lat_deg}, alt={alt_km} km"
            ),
        )
