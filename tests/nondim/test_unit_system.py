"""UnitSystem construction, validation, and derived properties."""

from __future__ import annotations

import math

import pytest

from astrojax.constants import GM_EARTH, GM_MOON, GM_SUN, R_EARTH
from astrojax.nondim import UnitSystem


class TestConstruction:
    def test_basic_construction(self):
        u = UnitSystem(LU=1000.0, TU=10.0, MU=2.0)
        assert u.LU == 1000.0
        assert u.TU == 10.0
        assert u.MU == 2.0

    def test_is_frozen(self):
        u = UnitSystem(LU=1.0, TU=1.0, MU=1.0)
        with pytest.raises(AttributeError):
            u.LU = 2.0  # type: ignore[misc]

    @pytest.mark.parametrize(
        "LU,TU,MU",
        [
            (0.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 1.0, -1.0),
        ],
    )
    def test_non_positive_raises(self, LU, TU, MU):
        with pytest.raises(ValueError, match="must be positive"):
            UnitSystem(LU=LU, TU=TU, MU=MU)


class TestDerivedProperties:
    def test_derived_scales(self):
        u = UnitSystem(LU=2.0, TU=3.0, MU=5.0)
        assert u.VU == pytest.approx(2.0 / 3.0)
        assert u.accel == pytest.approx(2.0 / 9.0)
        assert u.mu == pytest.approx(8.0 / 9.0)
        assert u.density == pytest.approx(5.0 / 8.0)
        assert u.force == pytest.approx(5.0 * 2.0 / 9.0)

    def test_dimensional_consistency(self):
        # accel * MU == force; mu == LU^3 / TU^2; density * LU^3 == MU.
        u = UnitSystem(LU=7e6, TU=806.0, MU=1.0)
        assert u.force == pytest.approx(u.accel * u.MU)
        assert u.mu == pytest.approx(u.LU**3 / u.TU**2)
        assert u.density * u.LU**3 == pytest.approx(u.MU)


class TestConstructors:
    def test_from_scales_basic(self):
        u = UnitSystem.from_scales(LU=10.0, TU=2.0, MU=3.0)
        assert u.LU == 10.0 and u.TU == 2.0 and u.MU == 3.0

    def test_from_scales_default_mass(self):
        u = UnitSystem.from_scales(LU=10.0, TU=2.0)
        assert u.MU == 1.0

    def test_from_orbit_makes_mu_nondim_one(self):
        # μ_nd = μ_si / unit_mu = μ_si / (LU^3 / TU^2) = 1 by construction
        a = 7000e3
        u = UnitSystem.from_orbit(a, GM_EARTH)
        assert u.LU == pytest.approx(a)
        expected_TU = math.sqrt(a**3 / GM_EARTH)
        assert u.TU == pytest.approx(expected_TU)
        assert GM_EARTH / u.mu == pytest.approx(1.0)

    def test_from_orbit_makes_mean_motion_nondim_one(self):
        # n_si = sqrt(μ/a^3); n_nd = n_si * TU = 1 exactly.
        a = 7000e3
        u = UnitSystem.from_orbit(a, GM_EARTH)
        n_si = math.sqrt(GM_EARTH / a**3)
        assert n_si * u.TU == pytest.approx(1.0)

    def test_from_orbit_relative_keeps_n_nondim_one_but_position_scale_differs(self):
        a = 7000e3
        LU_rel = 1e3  # 1 km
        u = UnitSystem.from_orbit_relative(a, GM_EARTH, LU_rel=LU_rel)
        assert u.LU == pytest.approx(LU_rel)
        # TU is still the chief's orbital time scale → n_nd = 1.
        n_si = math.sqrt(GM_EARTH / a**3)
        assert n_si * u.TU == pytest.approx(1.0)
        # μ_nd = μ_SI / units.mu = (a / LU_rel)^3 in this construction.
        assert GM_EARTH / u.mu == pytest.approx((a / LU_rel) ** 3)


class TestPresets:
    def test_earth_canonical_makes_mu_one(self):
        u = UnitSystem.earth_canonical()
        assert u.LU == pytest.approx(R_EARTH)
        assert GM_EARTH / u.mu == pytest.approx(1.0)

    def test_lunar_canonical_makes_mu_moon_one(self):
        u = UnitSystem.lunar_canonical()
        assert GM_MOON / u.mu == pytest.approx(1.0)

    def test_solar_canonical_makes_mu_sun_one(self):
        u = UnitSystem.solar_canonical()
        assert GM_SUN / u.mu == pytest.approx(1.0)
