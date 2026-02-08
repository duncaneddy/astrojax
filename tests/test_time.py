import jax
import jax.numpy as jnp
import pytest

from astrojax.config import get_dtype, set_dtype
from astrojax.time import (
    caldate_to_jd,
    caldate_to_mjd,
    jd_to_caldate,
    jd_to_mjd,
    mjd_to_caldate,
    mjd_to_jd,
)


def test_caldate_to_mjd():
    assert caldate_to_mjd(2000, 1, 1, 12, 0, 0) == pytest.approx(51544.5, abs=1e-9)


def test_caldate_to_jd():
    assert caldate_to_jd(2000, 1, 1, 12, 0, 0) == pytest.approx(2451545.0, abs=1e-9)


def test_jd_to_mjd():
    assert jd_to_mjd(2451545.0) == pytest.approx(51544.5, abs=1e-9)


def test_mjd_to_jd():
    assert mjd_to_jd(51544.5) == pytest.approx(2451545.0, abs=1e-9)


def test_jd_to_caldate_j2000():
    year, month, day, hour, minute, second = jd_to_caldate(2451545.0)
    assert year == 2000
    assert month == 1
    assert day == 1
    assert hour == 12
    assert minute == 0
    assert second == pytest.approx(0.0, abs=1e-6)


def test_jd_to_caldate_midnight():
    year, month, day, hour, minute, second = jd_to_caldate(2451544.5)
    assert year == 2000
    assert month == 1
    assert day == 1
    assert hour == 0
    assert minute == 0
    assert second == pytest.approx(0.0, abs=1e-6)


def test_jd_to_caldate_with_time():
    # JD for 2024-03-15 12:00:00 (noon). Noon aligns with the JD integer
    # boundary so is exactly representable even in float32.
    year, month, day, hour, minute, second = jd_to_caldate(2460385.0)
    assert year == 2024
    assert month == 3
    assert day == 15
    assert hour == 12
    assert minute == 0
    assert second == pytest.approx(0.0, abs=1e-3)


def test_mjd_to_caldate_j2000():
    year, month, day, hour, minute, second = mjd_to_caldate(51544.5)
    assert year == 2000
    assert month == 1
    assert day == 1
    assert hour == 12
    assert minute == 0
    assert second == pytest.approx(0.0, abs=1e-6)


def test_jd_caldate_roundtrip():
    """Verify that caldate -> JD -> caldate round-trips for date components.

    caldate_to_jd returns float32, which has ~0.25 day ULP near typical JD
    values (~2.45M). Sub-day time components are lost. This test verifies
    that the date (year/month/day) round-trips correctly and the time
    components are within float32 JD precision (~6 hours).
    """
    jd = caldate_to_jd(2024, 7, 4, 12, 0, 0.0)
    result = jd_to_caldate(jd)
    assert result[0] == 2024
    assert result[1] == 7
    assert result[2] == 4
    assert result[3] == 12
    assert result[4] == 0
    assert result[5] == pytest.approx(0.0, abs=1e-3)


# --- JAX tracing tests ---


def test_caldate_to_mjd_jit():
    """Verify caldate_to_mjd works under jax.jit."""
    mjd_eager = caldate_to_mjd(2000, 1, 1, 12, 0, 0.0)
    mjd_jit = jax.jit(caldate_to_mjd, static_argnums=())(2000, 1, 1, 12, 0, 0.0)
    assert float(mjd_jit) == pytest.approx(float(mjd_eager), abs=1e-6)


def test_caldate_to_jd_jit():
    """Verify caldate_to_jd works under jax.jit."""
    jd_eager = caldate_to_jd(2000, 1, 1, 12, 0, 0.0)
    jd_jit = jax.jit(caldate_to_jd, static_argnums=())(2000, 1, 1, 12, 0, 0.0)
    assert float(jd_jit) == pytest.approx(float(jd_eager), abs=1e-6)


def test_jd_to_caldate_jit():
    """Verify jd_to_caldate works under jax.jit."""
    jd = jnp.float32(2451545.0)
    year, month, day, hour, minute, second = jax.jit(jd_to_caldate)(jd)
    assert int(year) == 2000
    assert int(month) == 1
    assert int(day) == 1
    assert int(hour) == 12
    assert int(minute) == 0
    assert float(second) == pytest.approx(0.0, abs=1e-3)


def test_mjd_to_caldate_jit():
    """Verify mjd_to_caldate works under jax.jit."""
    mjd = jnp.float32(51544.5)
    year, month, day, hour, minute, second = jax.jit(mjd_to_caldate)(mjd)
    assert int(year) == 2000
    assert int(month) == 1
    assert int(day) == 1
    assert int(hour) == 12
    assert int(minute) == 0
    assert float(second) == pytest.approx(0.0, abs=1e-3)


def test_caldate_to_mjd_vmap():
    """Verify caldate_to_mjd works with vmap over batched inputs."""
    set_dtype(jnp.float32)
    years = jnp.array([2000, 2024, 1999], dtype=jnp.int32)
    months = jnp.array([1, 3, 12], dtype=jnp.int32)
    days = jnp.array([1, 15, 31], dtype=jnp.int32)
    hours = jnp.array([12, 6, 0], dtype=jnp.int32)
    minutes = jnp.array([0, 30, 0], dtype=jnp.int32)
    seconds = jnp.array([0.0, 45.0, 0.0], dtype=get_dtype())

    vmapped = jax.vmap(caldate_to_mjd)(years, months, days, hours, minutes, seconds)

    for i in range(3):
        expected = caldate_to_mjd(
            int(years[i]), int(months[i]), int(days[i]),
            int(hours[i]), int(minutes[i]), float(seconds[i]),
        )
        assert float(vmapped[i]) == pytest.approx(float(expected), abs=1e-6)


def test_jd_to_caldate_vmap():
    """Verify jd_to_caldate works with vmap over batched JDs."""
    jds = jnp.array([2451545.0, 2460385.0, 2451543.5])

    years, months, days, hours, minutes, seconds = jax.vmap(jd_to_caldate)(jds)

    for i in range(3):
        ey, em, ed, eh, emi, es = jd_to_caldate(float(jds[i]))
        assert int(years[i]) == int(ey)
        assert int(months[i]) == int(em)
        assert int(days[i]) == int(ed)
        assert int(hours[i]) == int(eh)
        assert int(minutes[i]) == int(emi)
        assert float(seconds[i]) == pytest.approx(float(es), abs=1e-3)


def test_caldate_to_mjd_february():
    """Verify the jnp.where branch for January/February dates."""
    # January (month <= 2 branch)
    mjd_jan = caldate_to_mjd(2024, 1, 15)
    year_jan, month_jan, day_jan, _, _, _ = jd_to_caldate(mjd_jan + 2400000.5)
    assert int(year_jan) == 2024
    assert int(month_jan) == 1
    assert int(day_jan) == 15

    # February (month <= 2 branch)
    mjd_feb = caldate_to_mjd(2024, 2, 29)
    year_feb, month_feb, day_feb, _, _, _ = jd_to_caldate(mjd_feb + 2400000.5)
    assert int(year_feb) == 2024
    assert int(month_feb) == 2
    assert int(day_feb) == 29

    # March (month > 2 branch)
    mjd_mar = caldate_to_mjd(2024, 3, 1)
    year_mar, month_mar, day_mar, _, _, _ = jd_to_caldate(mjd_mar + 2400000.5)
    assert int(year_mar) == 2024
    assert int(month_mar) == 3
    assert int(day_mar) == 1
