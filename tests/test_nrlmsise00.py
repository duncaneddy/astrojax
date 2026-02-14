"""Tests for the NRLMSISE-00 atmospheric density model.

Validates the JAX implementation of NRLMSISE-00 against the Brodowski
C reference test cases and verifies the high-level API (density lookups
from geodetic coordinates and ECEF positions).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from astrojax import Epoch
from astrojax.orbit_dynamics.nrlmsise00 import (
    density_nrlmsise00,
    density_nrlmsise00_geod,
    gtd7,
    gtd7d,
)
from astrojax.space_weather import static_space_weather

# ===========================================================================
# Brodowski C Reference Cases
# ===========================================================================

# Reference values from the Brodowski C implementation of NRLMSISE-00.
# CGS units: number densities in cm^-3, total mass density in g/cm^3.
# Our implementation returns SI: m^-3 and kg/m^3.
#
# Conversion factors applied in-test:
#   number density: multiply by 1e6 (cm^-3 -> m^-3)
#   mass density:   multiply by 1e3 (g/cm^3 -> kg/m^3)
#
# Format: (case_id, doy, sec, alt, g_lat, g_lon, lst, f107a, f107, ap,
#           expected_t, expected_d)
BRODOWSKI_CASES = [
    (
        1,
        172,
        29000.0,
        400.0,
        60.0,
        -70.0,
        16.0,
        150.0,
        150.0,
        4.0,
        [1250.54, 1241.42],
        [
            6.6651769e05,
            1.1388056e08,
            1.9982109e07,
            4.0227636e05,
            3.5574650e03,
            4.0747135e-15,
            3.4753124e04,
            4.0959133e06,
            2.6672732e04,
        ],
    ),
    (
        2,
        81,
        29000.0,
        400.0,
        60.0,
        -70.0,
        16.0,
        150.0,
        150.0,
        4.0,
        [1166.75, 1161.71],
        [
            3.4072932e06,
            1.5863334e08,
            1.3911174e07,
            3.2625595e05,
            1.5596182e03,
            5.0018457e-15,
            4.8542085e04,
            4.3809667e06,
            6.9566820e03,
        ],
    ),
    (
        3,
        172,
        75000.0,
        1000.0,
        60.0,
        -70.0,
        16.0,
        150.0,
        150.0,
        4.0,
        [1239.89, 1239.89],
        [
            1.1237672e05,
            6.9341301e04,
            4.2471052e01,
            1.3227501e-01,
            2.6188484e-05,
            2.7567723e-18,
            2.0167499e04,
            5.7412559e03,
            2.3743942e04,
        ],
    ),
    (
        4,
        172,
        29000.0,
        100.0,
        60.0,
        -70.0,
        16.0,
        150.0,
        150.0,
        4.0,
        [1027.32, 206.89],
        [
            5.4115544e07,
            1.9188934e11,
            6.1158256e12,
            1.2252011e12,
            6.0232120e10,
            3.5844263e-10,
            1.0598797e07,
            2.6157367e05,
            2.8198794e-42,
        ],
    ),
    (
        5,
        172,
        29000.0,
        400.0,
        0.0,
        -70.0,
        16.0,
        150.0,
        150.0,
        4.0,
        [1212.40, 1208.14],
        [
            1.8511225e06,
            1.4765548e08,
            1.5793562e07,
            2.6337950e05,
            1.5887814e03,
            4.8096302e-15,
            5.8161668e04,
            5.4789845e06,
            1.2644459e03,
        ],
    ),
    (
        6,
        172,
        29000.0,
        400.0,
        60.0,
        0.0,
        16.0,
        150.0,
        150.0,
        4.0,
        [1220.15, 1212.71],
        [
            8.6730952e05,
            1.2788618e08,
            1.8225766e07,
            2.9222142e05,
            2.4029624e03,
            4.3558656e-15,
            3.6863892e04,
            3.8972755e06,
            2.6672732e04,
        ],
    ),
    (
        7,
        172,
        29000.0,
        400.0,
        60.0,
        -70.0,
        4.0,
        150.0,
        150.0,
        4.0,
        [1116.39, 1113.00],
        [
            5.7762512e05,
            6.9791387e07,
            1.2368136e07,
            2.4928677e05,
            1.4057387e03,
            2.4706514e-15,
            5.2919856e04,
            1.0698141e06,
            2.6672732e04,
        ],
    ),
    (
        8,
        172,
        29000.0,
        400.0,
        60.0,
        -70.0,
        16.0,
        70.0,
        150.0,
        4.0,
        [1031.25, 1024.85],
        [
            3.7403041e05,
            4.7827201e07,
            5.2403800e06,
            1.7598746e05,
            5.5016488e02,
            1.5718887e-15,
            8.8967757e04,
            1.9797408e06,
            9.1218149e03,
        ],
    ),
    (
        9,
        172,
        29000.0,
        400.0,
        60.0,
        -70.0,
        16.0,
        150.0,
        180.0,
        4.0,
        [1306.05, 1293.37],
        [
            6.7483388e05,
            1.2453153e08,
            2.3690095e07,
            4.9115832e05,
            4.5787811e03,
            4.5644202e-15,
            3.2445948e04,
            5.3708331e06,
            2.6672732e04,
        ],
    ),
    (
        10,
        172,
        29000.0,
        400.0,
        60.0,
        -70.0,
        16.0,
        150.0,
        150.0,
        40.0,
        [1361.87, 1347.39],
        [
            5.5286008e05,
            1.1980413e08,
            3.4957978e07,
            9.3396184e05,
            1.0962548e04,
            4.9745431e-15,
            2.6864279e04,
            4.8899742e06,
            2.8054448e04,
        ],
    ),
    (
        11,
        172,
        29000.0,
        0.0,
        60.0,
        -70.0,
        16.0,
        150.0,
        150.0,
        4.0,
        [1027.32, 281.46],
        [1.3754876e14, 0.0, 2.0496870e19, 5.4986954e18, 2.4517332e17, 1.2610657e-03, 0.0, 0.0, 0.0],
    ),
    (
        12,
        172,
        29000.0,
        10.0,
        60.0,
        -70.0,
        16.0,
        150.0,
        150.0,
        4.0,
        [1027.32, 227.42],
        [4.4274426e13, 0.0, 6.5975672e18, 1.7699293e18, 7.8916800e16, 4.0591394e-04, 0.0, 0.0, 0.0],
    ),
    (
        13,
        172,
        29000.0,
        30.0,
        60.0,
        -70.0,
        16.0,
        150.0,
        150.0,
        4.0,
        [1027.32, 237.44],
        [2.1278288e12, 0.0, 3.1707906e17, 8.5062798e16, 3.7927411e15, 1.9508222e-05, 0.0, 0.0, 0.0],
    ),
    (
        14,
        172,
        29000.0,
        50.0,
        60.0,
        -70.0,
        16.0,
        150.0,
        150.0,
        4.0,
        [1027.32, 279.56],
        [1.4121835e11, 0.0, 2.1043696e16, 5.6453924e15, 2.5171417e14, 1.2947090e-06, 0.0, 0.0, 0.0],
    ),
    (
        15,
        172,
        29000.0,
        70.0,
        60.0,
        -70.0,
        16.0,
        150.0,
        150.0,
        4.0,
        [1027.32, 219.07],
        [1.2548844e10, 0.0, 1.8745328e15, 4.9230510e14, 2.2396854e13, 1.1476677e-07, 0.0, 0.0, 0.0],
    ),
]


@pytest.mark.parametrize(
    "case_id,doy,sec,alt,g_lat,g_lon,lst,f107a,f107,ap,expected_t,expected_d",
    BRODOWSKI_CASES,
    ids=[f"case_{c[0]}" for c in BRODOWSKI_CASES],
)
def test_gtd7_brodowski(
    case_id: int,
    doy: int,
    sec: float,
    alt: float,
    g_lat: float,
    g_lon: float,
    lst: float,
    f107a: float,
    f107: float,
    ap: float,
    expected_t: list[float],
    expected_d: list[float],
) -> None:
    """Validate gtd7 against Brodowski C reference (case {case_id}).

    Brodowski reference values are in CGS; our implementation returns SI.
    Number densities are converted cm^-3 -> m^-3 (x1e6), mass density
    g/cm^3 -> kg/m^3 (x1e3).
    """
    ap_array = jnp.array([4.0] * 7)

    t, d = gtd7(
        doy,
        sec,
        alt,
        g_lat,
        g_lon,
        lst,
        f107a,
        f107,
        ap,
        ap_array,
        use_ap_array=False,
    )

    # -- Temperature checks (absolute tolerance 0.05 K) --
    for i in range(2):
        assert float(t[i]) == pytest.approx(expected_t[i], abs=0.05), (
            f"Case {case_id}: t[{i}] = {float(t[i])}, expected {expected_t[i]}"
        )

    # -- Density checks --
    # Convert CGS reference values to SI
    for i in range(9):
        actual = float(d[i])
        if i == 5:
            # Total mass density: g/cm^3 -> kg/m^3
            expected_si = expected_d[i] * 1e3
        else:
            # Number densities: cm^-3 -> m^-3
            expected_si = expected_d[i] * 1e6

        if expected_si == 0.0:
            # Zero values: absolute tolerance
            assert actual == pytest.approx(0.0, abs=1e-10), (
                f"Case {case_id}: d[{i}] = {actual}, expected 0.0"
            )
        elif abs(expected_si) < 1e-30:
            # Subnormal / extremely small values (e.g. case 4 d[8]).
            # Use absolute tolerance since relative tolerance is
            # meaningless for values near float precision limits.
            assert actual == pytest.approx(expected_si, abs=abs(expected_si) * 0.1), (
                f"Case {case_id}: d[{i}] = {actual}, expected ~{expected_si}"
            )
        else:
            # Normal values: relative tolerance 1e-5
            assert actual == pytest.approx(expected_si, rel=1e-5), (
                f"Case {case_id}: d[{i}] = {actual}, expected {expected_si}"
            )


# ===========================================================================
# gtd7d: Anomalous oxygen in total density
# ===========================================================================
class TestGtd7d:
    """Tests for gtd7d (total density includes anomalous oxygen)."""

    def test_anomalous_oxygen_included_in_total_density(self) -> None:
        """gtd7d total density d[5] should exceed gtd7 d[5] when anomalous O is non-zero.

        At 400 km altitude the anomalous oxygen contribution (d[8]) is
        non-negligible, so gtd7d should produce a higher total mass density
        than gtd7.
        """
        ap_array = jnp.array([4.0] * 7)

        t7, d7 = gtd7(
            172,
            29000.0,
            400.0,
            60.0,
            -70.0,
            16.0,
            150.0,
            150.0,
            4.0,
            ap_array,
            use_ap_array=False,
        )

        t7d, d7d = gtd7d(
            172,
            29000.0,
            400.0,
            60.0,
            -70.0,
            16.0,
            150.0,
            150.0,
            4.0,
            ap_array,
            use_ap_array=False,
        )

        # Anomalous oxygen should be non-zero at 400 km
        assert float(d7[8]) > 0.0, "Anomalous O should be non-zero at 400 km"

        # Both should return the same species densities (d[0]-d[4], d[6]-d[8])
        for i in [0, 1, 2, 3, 4, 6, 7, 8]:
            assert float(d7d[i]) == pytest.approx(float(d7[i]), rel=1e-12), (
                f"d[{i}] should be identical between gtd7 and gtd7d"
            )

        # Temperatures should be identical
        for i in range(2):
            assert float(t7d[i]) == pytest.approx(float(t7[i]), rel=1e-12)

        # gtd7d total density should be larger because it includes anomalous O
        assert float(d7d[5]) > float(d7[5]), (
            "gtd7d total density should exceed gtd7 when anomalous O is present"
        )

    def test_gtd7d_matches_manual_recomputation(self) -> None:
        """gtd7d d[5] should equal the manual mass density sum including d[8].

        Verify that the total mass density returned by gtd7d equals the
        weighted sum of all species including anomalous oxygen.
        """
        ap_array = jnp.array([4.0] * 7)

        _, d7d = gtd7d(
            172,
            29000.0,
            400.0,
            60.0,
            -70.0,
            16.0,
            150.0,
            150.0,
            4.0,
            ap_array,
            use_ap_array=False,
        )

        # Manual recomputation: 1.66e-24 * weighted sum / 1000.0
        d5_manual = 1.66e-24 * (
            4.0 * float(d7d[0])
            + 16.0 * float(d7d[1])
            + 28.0 * float(d7d[2])
            + 32.0 * float(d7d[3])
            + 40.0 * float(d7d[4])
            + 1.0 * float(d7d[6])
            + 14.0 * float(d7d[7])
            + 16.0 * float(d7d[8])
        )
        d5_manual = d5_manual / 1000.0  # kg/m^3

        assert float(d7d[5]) == pytest.approx(d5_manual, rel=1e-10)


# ===========================================================================
# High-level API: density_nrlmsise00_geod
# ===========================================================================
class TestDensityNrlmsise00Geod:
    """Tests for density_nrlmsise00_geod (geodetic coordinate interface)."""

    def test_positive_density_at_leo(self) -> None:
        """Density should be positive at typical LEO altitudes."""
        sw = static_space_weather()
        epc = Epoch(2020, 6, 1, 12, 0, 0.0)
        geod = jnp.array([-74.0, 40.7, 400e3])  # lon, lat, alt_m

        rho = density_nrlmsise00_geod(sw, epc, geod)

        assert float(rho) > 0.0, "Density at 400 km LEO should be positive"
        # Typical density at 400 km is ~1e-12 to 1e-11 kg/m^3
        assert 1e-14 < float(rho) < 1e-9, f"Density {float(rho)} kg/m^3 outside expected LEO range"

    def test_density_decreases_with_altitude(self) -> None:
        """Density should decrease monotonically with increasing altitude."""
        sw = static_space_weather()
        epc = Epoch(2020, 6, 1, 12, 0, 0.0)

        rho_values = []
        for alt_km in [200, 400, 600, 800]:
            geod = jnp.array([0.0, 45.0, alt_km * 1e3])
            rho = density_nrlmsise00_geod(sw, epc, geod)
            rho_values.append(float(rho))

        for i in range(len(rho_values) - 1):
            assert rho_values[i] > rho_values[i + 1], (
                f"Density at {[200, 400, 600, 800][i]} km "
                f"({rho_values[i]:.3e}) should exceed density at "
                f"{[200, 400, 600, 800][i + 1]} km ({rho_values[i + 1]:.3e})"
            )

    def test_jit_compatible(self) -> None:
        """density_nrlmsise00_geod should work under jax.jit."""
        sw = static_space_weather()
        epc = Epoch(2020, 6, 1, 12, 0, 0.0)
        geod = jnp.array([-74.0, 40.7, 400e3])

        rho_eager = density_nrlmsise00_geod(sw, epc, geod)
        rho_jit = jax.jit(density_nrlmsise00_geod)(sw, epc, geod)

        assert jnp.allclose(rho_eager, rho_jit, rtol=1e-10, atol=1e-20)

    def test_vmap_compatible(self) -> None:
        """density_nrlmsise00_geod should work under jax.vmap over positions."""
        sw = static_space_weather()
        epc = Epoch(2020, 6, 1, 12, 0, 0.0)

        geods = jnp.array(
            [
                [-74.0, 40.7, 300e3],
                [0.0, 0.0, 400e3],
                [120.0, -30.0, 500e3],
                [45.0, 60.0, 600e3],
            ]
        )

        rho_vmap = jax.vmap(density_nrlmsise00_geod, in_axes=(None, None, 0))(
            sw,
            epc,
            geods,
        )

        assert rho_vmap.shape == (4,)

        # All densities should be positive
        for i in range(4):
            assert float(rho_vmap[i]) > 0.0, f"Density for geod[{i}] should be positive"

        # Verify against eager evaluation
        for i in range(4):
            rho_i = density_nrlmsise00_geod(sw, epc, geods[i])
            assert float(rho_vmap[i]) == pytest.approx(float(rho_i), rel=1e-10)


# ===========================================================================
# High-level API: density_nrlmsise00 (ECEF interface)
# ===========================================================================
class TestDensityNrlmsise00:
    """Tests for density_nrlmsise00 (ECEF position interface)."""

    def test_positive_density_at_leo(self) -> None:
        """Density should be positive at typical LEO altitudes."""
        sw = static_space_weather()
        epc = Epoch(2020, 6, 1, 12, 0, 0.0)
        r_ecef = jnp.array([6778137.0, 0.0, 0.0])  # ~400 km above equator

        rho = density_nrlmsise00(sw, epc, r_ecef)

        assert float(rho) > 0.0
        assert 1e-14 < float(rho) < 1e-9

    def test_consistent_with_geod_interface(self) -> None:
        """density_nrlmsise00 should match density_nrlmsise00_geod for the same point."""
        from astrojax.coordinates import position_geodetic_to_ecef

        sw = static_space_weather()
        epc = Epoch(2020, 6, 1, 12, 0, 0.0)
        geod = jnp.array([10.0, 45.0, 400e3])  # lon, lat, alt_m

        r_ecef = position_geodetic_to_ecef(geod, use_degrees=True)

        rho_ecef = density_nrlmsise00(sw, epc, r_ecef)
        rho_geod = density_nrlmsise00_geod(sw, epc, geod)

        assert float(rho_ecef) == pytest.approx(float(rho_geod), rel=1e-6)

    def test_jit_compatible(self) -> None:
        """density_nrlmsise00 should work under jax.jit."""
        sw = static_space_weather()
        epc = Epoch(2020, 6, 1, 12, 0, 0.0)
        r_ecef = jnp.array([6778137.0, 0.0, 0.0])

        rho_eager = density_nrlmsise00(sw, epc, r_ecef)
        rho_jit = jax.jit(density_nrlmsise00)(sw, epc, r_ecef)

        assert jnp.allclose(rho_eager, rho_jit, rtol=1e-10, atol=1e-20)
