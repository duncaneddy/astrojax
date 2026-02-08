import math

import jax
import jax.numpy as jnp
import pytest

from astrojax.epoch import Epoch

# float32 seconds precision: ~8ms at worst (near 86400), ~1us at small values.
# Use 0.01 tolerance for caldate seconds that go through float32 storage.
_F32_SEC_TOL = 0.01


# ──────────────────────────────────────────────
# Construction: date components
# ──────────────────────────────────────────────


class TestEpochDateConstruction:
    def test_epoch_from_date(self):
        epc = Epoch(2000, 1, 1, 12, 0, 0.0)
        year, month, day, hour, minute, second = epc.caldate()
        assert year == 2000
        assert month == 1
        assert day == 1
        assert hour == 12
        assert minute == 0
        assert second == pytest.approx(0.0, abs=_F32_SEC_TOL)

    def test_epoch_from_date_defaults(self):
        epc = Epoch(2000, 1, 1)
        year, month, day, hour, minute, second = epc.caldate()
        assert year == 2000
        assert month == 1
        assert day == 1
        assert hour == 0
        assert minute == 0
        assert second == pytest.approx(0.0, abs=_F32_SEC_TOL)

    def test_epoch_from_date_with_time(self):
        epc = Epoch(2024, 3, 15, 6, 30, 45.0)
        year, month, day, hour, minute, second = epc.caldate()
        assert year == 2024
        assert month == 3
        assert day == 15
        assert hour == 6
        assert minute == 30
        assert second == pytest.approx(45.0, abs=_F32_SEC_TOL)

    def test_epoch_from_date_fractional_second(self):
        epc = Epoch(2020, 6, 15, 10, 30, 15.123)
        year, month, day, hour, minute, second = epc.caldate()
        assert year == 2020
        assert month == 6
        assert day == 15
        assert hour == 10
        assert minute == 30
        assert second == pytest.approx(15.123, abs=_F32_SEC_TOL)

    def test_epoch_from_date_partial_args(self):
        epc = Epoch(2000, 1, 1, 12)
        year, month, day, hour, minute, second = epc.caldate()
        assert year == 2000
        assert month == 1
        assert day == 1
        assert hour == 12
        assert minute == 0
        assert second == pytest.approx(0.0, abs=_F32_SEC_TOL)


# ──────────────────────────────────────────────
# Construction: string
# ──────────────────────────────────────────────


class TestEpochStringConstruction:
    def test_epoch_from_string_date_only(self):
        epc = Epoch("2000-01-01")
        year, month, day, hour, minute, second = epc.caldate()
        assert year == 2000
        assert month == 1
        assert day == 1
        assert hour == 0
        assert minute == 0
        assert second == pytest.approx(0.0, abs=_F32_SEC_TOL)

    def test_epoch_from_string_iso(self):
        epc = Epoch("2018-01-01T12:00:00Z")
        year, month, day, hour, minute, second = epc.caldate()
        assert year == 2018
        assert month == 1
        assert day == 1
        assert hour == 12
        assert minute == 0
        assert second == pytest.approx(0.0, abs=_F32_SEC_TOL)

    def test_epoch_from_string_fractional_seconds(self):
        epc = Epoch("2020-06-15T10:30:15.500Z")
        year, month, day, hour, minute, second = epc.caldate()
        assert year == 2020
        assert month == 6
        assert day == 15
        assert hour == 10
        assert minute == 30
        assert second == pytest.approx(15.5, abs=_F32_SEC_TOL)

    def test_epoch_from_string_invalid(self):
        with pytest.raises(ValueError, match="not ISO 8601"):
            Epoch("not-a-date")

    def test_epoch_from_string_matches_date_construction(self):
        epc_str = Epoch("2024-03-15T06:30:45Z")
        epc_date = Epoch(2024, 3, 15, 6, 30, 45.0)
        assert epc_str == epc_date


# ──────────────────────────────────────────────
# Construction: copy
# ──────────────────────────────────────────────


class TestEpochCopyConstruction:
    def test_epoch_copy(self):
        original = Epoch(2000, 1, 1, 12, 0, 0.0)
        copy = Epoch(original)
        assert copy == original
        assert copy is not original

    def test_epoch_copy_independence(self):
        original = Epoch(2000, 1, 1)
        copy = Epoch(original)
        copy += 100.0
        assert copy != original


# ──────────────────────────────────────────────
# Construction: invalid
# ──────────────────────────────────────────────


class TestEpochInvalidConstruction:
    def test_epoch_no_args(self):
        with pytest.raises(ValueError):
            Epoch()

    def test_epoch_invalid_type(self):
        with pytest.raises(ValueError):
            Epoch(12345)

    def test_epoch_too_many_args(self):
        with pytest.raises(ValueError):
            Epoch(2000, 1, 1, 12, 0, 0.0, 0.0)


# ──────────────────────────────────────────────
# Julian Date and MJD
# ──────────────────────────────────────────────


class TestEpochJulianDate:
    def test_epoch_jd_j2000(self):
        epc = Epoch(2000, 1, 1, 12, 0, 0.0)
        # 2451545.0 is exactly representable in float32
        assert epc.jd() == pytest.approx(2451545.0, abs=0.5)

    def test_epoch_jd_midnight(self):
        epc = Epoch(2000, 1, 1)
        # 2451544.5 is exactly representable in float32
        assert epc.jd() == pytest.approx(2451544.5, abs=0.5)

    def test_epoch_mjd(self):
        epc = Epoch(2000, 1, 1, 12, 0, 0.0)
        assert epc.mjd() == pytest.approx(51544.5, abs=0.01)

    def test_epoch_mjd_midnight(self):
        epc = Epoch(2000, 1, 1)
        assert epc.mjd() == pytest.approx(51544.0, abs=0.01)


# ──────────────────────────────────────────────
# Arithmetic: addition
# ──────────────────────────────────────────────


class TestEpochAddition:
    def test_epoch_add_seconds(self):
        epc = Epoch(2000, 1, 1) + 60.0
        year, month, day, hour, minute, second = epc.caldate()
        assert hour == 0
        assert minute == 1
        assert second == pytest.approx(0.0, abs=_F32_SEC_TOL)

    def test_epoch_add_hours(self):
        epc = Epoch(2000, 1, 1) + 3600.0
        year, month, day, hour, minute, second = epc.caldate()
        assert hour == 1

    def test_epoch_add_day_rollover(self):
        epc = Epoch(2000, 1, 1) + 86400.0
        year, month, day, hour, minute, second = epc.caldate()
        assert year == 2000
        assert month == 1
        assert day == 2
        assert hour == 0

    def test_epoch_add_negative(self):
        epc = Epoch(2000, 1, 2) + (-86400.0)
        assert epc == Epoch(2000, 1, 1)

    def test_epoch_iadd(self):
        epc = Epoch(2000, 1, 1)
        epc += 3600.0
        year, month, day, hour, minute, second = epc.caldate()
        assert hour == 1

    def test_epoch_add_fractional(self):
        epc = Epoch(2000, 1, 1) + 0.5
        year, month, day, hour, minute, second = epc.caldate()
        assert second == pytest.approx(0.5, abs=_F32_SEC_TOL)


# ──────────────────────────────────────────────
# Arithmetic: subtraction
# ──────────────────────────────────────────────


class TestEpochSubtraction:
    def test_epoch_subtract_seconds(self):
        epc = Epoch(2000, 1, 1, 1, 0, 0.0) - 3600.0
        assert epc == Epoch(2000, 1, 1)

    def test_epoch_subtract_epoch(self):
        epc1 = Epoch(2000, 1, 2)
        epc2 = Epoch(2000, 1, 1)
        diff = epc1 - epc2
        assert diff == pytest.approx(86400.0, abs=_F32_SEC_TOL)

    def test_epoch_subtract_epoch_negative(self):
        epc1 = Epoch(2000, 1, 1)
        epc2 = Epoch(2000, 1, 2)
        diff = epc1 - epc2
        assert diff == pytest.approx(-86400.0, abs=_F32_SEC_TOL)

    def test_epoch_subtract_epoch_fractional(self):
        epc1 = Epoch(2000, 1, 1, 0, 0, 30.5)
        epc2 = Epoch(2000, 1, 1)
        diff = epc1 - epc2
        assert diff == pytest.approx(30.5, abs=_F32_SEC_TOL)

    def test_epoch_isub(self):
        epc = Epoch(2000, 1, 1, 1, 0, 0.0)
        epc -= 3600.0
        assert epc == Epoch(2000, 1, 1)


# ──────────────────────────────────────────────
# Arithmetic: Kahan precision
# ──────────────────────────────────────────────


class TestEpochKahanPrecision:
    def test_kahan_many_small_additions(self):
        """Kahan summation limits error to O(eps) rather than O(N*eps)."""
        epc = Epoch(2000, 1, 1)
        original = Epoch(2000, 1, 1)
        n = 10000
        dt = 0.001
        for _ in range(n):
            epc += dt
        expected = n * dt
        actual = epc - original
        # float32 Kahan: error ~ O(eps * value) ~ 6e-8 * 43210 ~ 0.003
        assert actual == pytest.approx(expected, abs=0.01)

    def test_kahan_add_subtract_roundtrip(self):
        epc = Epoch(2000, 1, 1)
        n = 10000
        dt = 0.001
        for _ in range(n):
            epc += dt
        for _ in range(n):
            epc -= dt
        # float32 Kahan roundtrip: error should be small but not sub-nanosecond
        assert abs(epc - Epoch(2000, 1, 1)) < 0.01


# ──────────────────────────────────────────────
# Comparison operators
# ──────────────────────────────────────────────


class TestEpochComparison:
    def test_epoch_equality(self):
        assert Epoch(2000, 1, 1) == Epoch(2000, 1, 1)

    def test_epoch_inequality(self):
        assert Epoch(2000, 1, 1) != Epoch(2000, 1, 2)

    def test_epoch_less_than(self):
        assert Epoch(2000, 1, 1) < Epoch(2000, 1, 2)

    def test_epoch_less_than_same_day(self):
        assert Epoch(2000, 1, 1, 0, 0, 0.0) < Epoch(2000, 1, 1, 0, 0, 1.0)

    def test_epoch_less_equal(self):
        assert Epoch(2000, 1, 1) <= Epoch(2000, 1, 1)
        assert Epoch(2000, 1, 1) <= Epoch(2000, 1, 2)

    def test_epoch_greater_than(self):
        assert Epoch(2000, 1, 2) > Epoch(2000, 1, 1)

    def test_epoch_greater_equal(self):
        assert Epoch(2000, 1, 1) >= Epoch(2000, 1, 1)
        assert Epoch(2000, 1, 2) >= Epoch(2000, 1, 1)

    def test_epoch_not_equal_to_non_epoch(self):
        assert (Epoch(2000, 1, 1) == 42) is False

    def test_epoch_ordering_consistency(self):
        e1 = Epoch(2000, 1, 1)
        e2 = Epoch(2000, 6, 15)
        e3 = Epoch(2001, 1, 1)
        assert e1 < e2 < e3
        assert e3 > e2 > e1


# ──────────────────────────────────────────────
# GMST
# ──────────────────────────────────────────────


class TestEpochGMST:
    def test_gmst_j2000_midnight(self):
        epc = Epoch(2000, 1, 1)
        gmst_deg = epc.gmst(use_degrees=True)
        # Vallado GMST82 at 2000-01-01 00:00:00 UTC ~ 99.97 degrees
        assert gmst_deg == pytest.approx(99.97, abs=0.1)

    def test_gmst_j2000_noon(self):
        epc = Epoch(2000, 1, 1, 12, 0, 0.0)
        gmst_deg = epc.gmst(use_degrees=True)
        # At J2000.0 noon, T=0, GMST_sec = 67310.54841
        # GMST_deg = 67310.54841 / 240 = 280.461 degrees
        assert gmst_deg == pytest.approx(280.461, abs=0.1)

    def test_gmst_radians(self):
        epc = Epoch(2000, 1, 1)
        gmst_rad = epc.gmst(use_degrees=False)
        gmst_deg = epc.gmst(use_degrees=True)
        assert gmst_rad == pytest.approx(float(gmst_deg) * math.pi / 180.0, abs=1e-4)

    def test_gmst_positive(self):
        epc = Epoch(2000, 1, 1)
        assert epc.gmst() >= 0.0
        assert epc.gmst() < 2.0 * math.pi

    def test_gmst_changes_with_time(self):
        epc1 = Epoch(2000, 1, 1)
        epc2 = Epoch(2000, 1, 2)
        # GMST advances ~361 degrees per day (sidereal rate)
        diff = (float(epc2.gmst(use_degrees=True)) - float(epc1.gmst(use_degrees=True))) % 360.0
        # Should be approximately 0.985 degrees (360.985 - 360)
        assert diff == pytest.approx(0.986, abs=0.2)


# ──────────────────────────────────────────────
# String representation
# ──────────────────────────────────────────────


class TestEpochString:
    def test_epoch_str_format(self):
        epc = Epoch(2000, 1, 1, 12, 0, 0.0)
        s = str(epc)
        assert s == "2000-01-01T12:00:00.000Z"

    def test_epoch_str_midnight(self):
        epc = Epoch(2000, 1, 1)
        s = str(epc)
        assert s == "2000-01-01T00:00:00.000Z"

    def test_epoch_repr(self):
        epc = Epoch(2000, 1, 1, 12, 0, 0.0)
        r = repr(epc)
        assert r.startswith("Epoch(_jd=")


# ──────────────────────────────────────────────
# Hash
# ──────────────────────────────────────────────


class TestEpochHash:
    def test_epoch_hash_equal_epochs(self):
        e1 = Epoch(2000, 1, 1)
        e2 = Epoch(2000, 1, 1)
        assert hash(e1) == hash(e2)

    def test_epoch_in_set(self):
        s = {Epoch(2000, 1, 1), Epoch(2000, 1, 1)}
        assert len(s) == 1

    def test_epoch_as_dict_key(self):
        d = {Epoch(2000, 1, 1): "J2000"}
        assert d[Epoch(2000, 1, 1)] == "J2000"


# ──────────────────────────────────────────────
# JAX compatibility
# ──────────────────────────────────────────────


class TestEpochJAXCompatibility:
    def test_epoch_jit_add(self):
        """Verify addition works under jax.jit."""
        epc = Epoch(2000, 1, 1)
        result = jax.jit(lambda e: e + 60.0)(epc)
        year, month, day, hour, minute, second = result.caldate()
        assert year == 2000
        assert month == 1
        assert day == 1
        assert hour == 0
        assert minute == 1
        assert second == pytest.approx(0.0, abs=_F32_SEC_TOL)

    def test_epoch_jit_sub_epochs(self):
        """Verify epoch difference works under jax.jit."""
        epc1 = Epoch(2000, 1, 2)
        epc2 = Epoch(2000, 1, 1)
        diff = jax.jit(lambda a, b: a - b)(epc1, epc2)
        assert float(diff) == pytest.approx(86400.0, abs=_F32_SEC_TOL)

    def test_epoch_jit_gmst(self):
        """Verify GMST computation works under jax.jit."""
        epc = Epoch(2000, 1, 1, 12, 0, 0.0)
        gmst_deg = jax.jit(lambda e: e.gmst(use_degrees=True))(epc)
        assert float(gmst_deg) == pytest.approx(280.461, abs=0.1)

    def test_epoch_jit_jd(self):
        """Verify jd() works under jax.jit."""
        epc = Epoch(2000, 1, 1, 12, 0, 0.0)
        jd_val = jax.jit(lambda e: e.jd())(epc)
        assert float(jd_val) == pytest.approx(2451545.0, abs=0.5)

    def test_epoch_vmap_add(self):
        """Verify vectorized addition over a batch of time deltas."""
        epc = Epoch(2000, 1, 1)
        deltas = jnp.array([0.0, 60.0, 3600.0, 86400.0])
        # Compare using epoch subtraction (preserves split-representation
        # precision) instead of lossy jd() comparison
        results = jax.vmap(lambda dt: (epc + dt) - epc)(deltas)
        assert jnp.allclose(results, deltas, atol=_F32_SEC_TOL)

    def test_epoch_pytree_roundtrip(self):
        """Verify flatten/unflatten preserves Epoch values."""
        epc = Epoch(2000, 1, 1, 6, 30, 15.5)
        leaves, treedef = jax.tree_util.tree_flatten(epc)
        reconstructed = treedef.unflatten(leaves)
        assert reconstructed == epc

    def test_epoch_lax_scan(self):
        """Verify time stepping with jax.lax.scan."""
        epc0 = Epoch(2000, 1, 1)
        dt = 60.0  # 1 minute steps
        n_steps = 60  # 1 hour total

        def step(epc, _):
            epc_next = epc + dt
            return epc_next, epc_next - epc0

        final_epc, elapsed_history = jax.lax.scan(step, epc0, None, length=n_steps)

        # After 60 steps of 60s = 3600s = 1 hour
        year, month, day, hour, minute, second = final_epc.caldate()
        assert year == 2000
        assert month == 1
        assert day == 1
        assert hour == 1
        assert minute == 0
        assert second == pytest.approx(0.0, abs=_F32_SEC_TOL)

        # History should have 60 elapsed-second values, monotonically increasing
        assert elapsed_history.shape == (60,)
        assert jnp.all(jnp.diff(elapsed_history) > 0)
