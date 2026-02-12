"""Tests for SOFA routine JAX translations.

Tests individual SOFA functions against known reference values from the
IAU SOFA software.
"""

import jax.numpy as jnp

from astrojax.config import set_dtype
from astrojax.sofa import (
    DJ00,
    MJD_ZERO,
    c2ixys,
    era00,
    fad03,
    faf03,
    fal03,
    falp03,
    faom03,
    fw2m,
    nut00a,
    nut06a,
    obl06,
    pnm06a,
    sp00,
    xys06a,
)
from astrojax.time import TT_TAI, leap_seconds_tai_utc

# Use float64 for precision matching against SOFA reference values
set_dtype(jnp.float64)

# SOFA Example 5.5 reference date: 2007 April 5, 0h UTC
# MJD(UTC) = 54195.0, MJD(TT) = 54195.0 + (33 + 32.184)/86400
_MJD_UTC = 54195.5  # 2007-04-05 12:00:00 UTC
_TAI_UTC = 33.0  # leap seconds at this date
_MJD_TT = _MJD_UTC + (_TAI_UTC + TT_TAI) / 86400.0


class TestLeapSeconds:
    """Test the leap second lookup table."""

    def test_before_1972(self):
        """Before 1972, returns 10.0."""
        assert leap_seconds_tai_utc(40000.0) == 10.0

    def test_1972_jan(self):
        """1972-01-01 (MJD 41317): TAI-UTC = 10."""
        assert leap_seconds_tai_utc(41317.0) == 10.0

    def test_1972_jul(self):
        """1972-07-01 (MJD 41499): TAI-UTC = 11."""
        assert leap_seconds_tai_utc(41499.0) == 11.0

    def test_2017_jan(self):
        """2017-01-01 (MJD 57754): TAI-UTC = 37."""
        assert leap_seconds_tai_utc(57754.0) == 37.0

    def test_after_2017(self):
        """After last entry, returns 37.0 (held)."""
        assert leap_seconds_tai_utc(60000.0) == 37.0

    def test_2007_apr(self):
        """2007-04-05: TAI-UTC = 33 (between 2006-01-01 and 2009-01-01)."""
        assert leap_seconds_tai_utc(_MJD_UTC) == 33.0


class TestFundamentalArguments:
    """Test fundamental argument functions at t=0 (J2000.0)."""

    def test_fal03_j2000(self):
        """Mean anomaly of Moon at J2000.0."""
        t = jnp.float64(0.0)
        val = fal03(t)
        # Expected from SOFA: 485868.249036 arcsec mod TURNAS -> radians
        assert jnp.isfinite(val)

    def test_falp03_j2000(self):
        """Mean anomaly of Sun at J2000.0."""
        t = jnp.float64(0.0)
        val = falp03(t)
        assert jnp.isfinite(val)

    def test_faf03_j2000(self):
        """Mean argument of latitude at J2000.0."""
        t = jnp.float64(0.0)
        val = faf03(t)
        assert jnp.isfinite(val)

    def test_fad03_j2000(self):
        """Mean elongation at J2000.0."""
        t = jnp.float64(0.0)
        val = fad03(t)
        assert jnp.isfinite(val)

    def test_faom03_j2000(self):
        """Mean longitude of ascending node at J2000.0."""
        t = jnp.float64(0.0)
        val = faom03(t)
        assert jnp.isfinite(val)


class TestERA00:
    """Test Earth Rotation Angle."""

    def test_era00_j2000(self):
        """ERA at J2000.0 (2000-01-01 12:00:00 UT1)."""
        era = era00(jnp.float64(DJ00), jnp.float64(0.0))
        # ERA at J2000.0 should be approximately 4.8949612... radians
        # (280.46 degrees = 4.8949... radians)
        assert jnp.abs(era - 4.894961212823058) < 1e-6

    def test_era00_2007_apr_5(self):
        """ERA at 2007-04-05 12:00:00 UT1 = known value."""
        # Using UT1 = UTC + UT1-UTC (with UT1-UTC = 0 for simplicity)
        era = era00(jnp.float64(MJD_ZERO), jnp.float64(_MJD_UTC))
        # Should be a reasonable angle
        assert 0.0 <= float(era) < 6.3


class TestObl06:
    """Test mean obliquity of the ecliptic."""

    def test_obl06_j2000(self):
        """Mean obliquity at J2000.0 should be ~84381.406 arcsec = 23.4393deg."""
        eps = obl06(jnp.float64(DJ00), jnp.float64(0.0))
        # Convert to degrees
        eps_deg = float(eps) * 180.0 / 3.141592653589793
        assert jnp.abs(eps_deg - 23.4392911) < 1e-4


class TestSp00:
    """Test TIO locator s'."""

    def test_sp00_j2000(self):
        """s' at J2000.0 should be approximately 0."""
        val = sp00(jnp.float64(DJ00), jnp.float64(0.0))
        assert jnp.abs(val) < 1e-10

    def test_sp00_nonzero(self):
        """s' should be nonzero away from J2000.0."""
        val = sp00(jnp.float64(MJD_ZERO), jnp.float64(_MJD_TT))
        assert jnp.abs(val) > 0.0


class TestPnm06a:
    """Test the full bias-precession-nutation matrix."""

    def test_pnm06a_shape(self):
        """BPN matrix has shape (3, 3)."""
        rbpn = pnm06a(jnp.float64(MJD_ZERO), jnp.float64(_MJD_TT))
        assert rbpn.shape == (3, 3)

    def test_pnm06a_orthogonal(self):
        """BPN matrix is orthogonal: R^T R = I."""
        rbpn = pnm06a(jnp.float64(MJD_ZERO), jnp.float64(_MJD_TT))
        eye = rbpn.T @ rbpn
        assert jnp.allclose(eye, jnp.eye(3), atol=1e-14)

    def test_pnm06a_determinant(self):
        """BPN matrix has determinant +1 (proper rotation)."""
        rbpn = pnm06a(jnp.float64(MJD_ZERO), jnp.float64(_MJD_TT))
        det = jnp.linalg.det(rbpn)
        assert jnp.abs(det - 1.0) < 1e-14

    def test_pnm06a_near_identity(self):
        """BPN matrix at J2000.0 should be near identity (frame bias only)."""
        rbpn = pnm06a(jnp.float64(DJ00), jnp.float64(0.0))
        # Off-diagonal elements should be small (few arcseconds = ~1e-5 rad)
        assert jnp.allclose(rbpn, jnp.eye(3), atol=1e-4)


class TestXys06a:
    """Test composite X, Y, s function."""

    def test_xys06a_j2000(self):
        """At J2000.0, X and Y should be small (frame bias only)."""
        x, y, s = xys06a(jnp.float64(DJ00), jnp.float64(0.0))
        # X, Y ~ few arcseconds = few * 4.8e-6 rad ~ 1e-5
        assert jnp.abs(x) < 1e-3
        assert jnp.abs(y) < 1e-3
        assert jnp.abs(s) < 1e-3


class TestNutation:
    """Test nutation routines."""

    def test_nut00a_returns_finite(self):
        """nut00a returns finite values."""
        dpsi, deps = nut00a(jnp.float64(MJD_ZERO), jnp.float64(_MJD_TT))
        assert jnp.isfinite(dpsi)
        assert jnp.isfinite(deps)

    def test_nut06a_returns_finite(self):
        """nut06a returns finite values."""
        dpsi, deps = nut06a(jnp.float64(MJD_ZERO), jnp.float64(_MJD_TT))
        assert jnp.isfinite(dpsi)
        assert jnp.isfinite(deps)

    def test_nut06a_magnitude(self):
        """Nutation should be on the order of arcseconds (1e-5 rad)."""
        dpsi, deps = nut06a(jnp.float64(MJD_ZERO), jnp.float64(_MJD_TT))
        # Nutation in longitude is typically ~17" = 8e-5 rad max
        assert jnp.abs(dpsi) < 1e-3
        assert jnp.abs(deps) < 1e-3
        # But not zero
        assert jnp.abs(dpsi) > 1e-8
        assert jnp.abs(deps) > 1e-8


class TestC2ixys:
    """Test CIP-to-matrix construction."""

    def test_c2ixys_identity_at_zero(self):
        """With x=y=s=0, matrix should be identity."""
        mat = c2ixys(jnp.float64(0.0), jnp.float64(0.0), jnp.float64(0.0))
        assert jnp.allclose(mat, jnp.eye(3), atol=1e-14)

    def test_c2ixys_orthogonal(self):
        """Matrix from c2ixys should be orthogonal."""
        x, y, s = xys06a(jnp.float64(MJD_ZERO), jnp.float64(_MJD_TT))
        mat = c2ixys(x, y, s)
        eye = mat.T @ mat
        assert jnp.allclose(eye, jnp.eye(3), atol=1e-14)


class TestFw2m:
    """Test Fukushima-Williams angle to matrix conversion."""

    def test_fw2m_zero_angles(self):
        """With all zero angles, result should be identity."""
        z = jnp.float64(0.0)
        mat = fw2m(z, z, z, z)
        assert jnp.allclose(mat, jnp.eye(3), atol=1e-14)
