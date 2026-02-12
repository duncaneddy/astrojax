"""High-level TLE satellite class for SGP4/SDP4 propagation.

Provides :class:`TLE`, a convenience wrapper that combines TLE parsing,
SGP4 initialization, propagation, and frame transformations into a single
object with user-friendly properties and methods.
"""

from __future__ import annotations

from math import pi as _py_pi

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.eop._types import EOPData
from astrojax.epoch import Epoch
from astrojax.frames.teme import (
    state_teme_to_gcrf,
    state_teme_to_itrf,
    state_teme_to_pef,
)
from astrojax.sgp4._constants import GRAVITY_MODELS, EarthGravity
from astrojax.sgp4._propagation import sgp4_init, sgp4_propagate
from astrojax.sgp4._tle import parse_tle
from astrojax.sgp4._types import SGP4Elements

_RAD2DEG = 180.0 / _py_pi


class TLE:
    """A parsed TLE with SGP4/SDP4 propagation and frame transforms.

    Wraps TLE parsing, SGP4 initialization, and propagation into a single
    object. Provides orbital element properties in user-friendly units and
    state vector methods in various reference frames.

    The ``propagate`` method returns raw SGP4 output in km/km/s (for
    comparison testing). The ``state_*`` methods return SI units (m, m/s).

    Examples:
        ```python
        from astrojax.sgp4 import TLE

        line1 = "1 25544U 98067A   08264.51782528 ..."
        line2 = "2 25544  51.6416 247.4627 ..."
        sat = TLE(line1, line2)

        # Properties
        sat.epoch       # Epoch object
        sat.n           # mean motion [rev/day]
        sat.e           # eccentricity
        sat.i           # inclination [deg]

        # Raw SGP4 output (km, km/s)
        r_km, v_kms = sat.propagate(60.0)

        # State in TEME (m, m/s) at epoch + 3600 seconds
        x_teme = sat.state_teme(3600.0)

        # State in GCRF at a specific Epoch
        x_gcrf = sat.state_gcrf(some_epoch, eop=eop_data)
        ```

    Args:
        line1: First TLE line.
        line2: Second TLE line.
        gravity: Gravity model name or :class:`EarthGravity` instance.
    """

    def __init__(
        self,
        line1: str,
        line2: str,
        gravity: str | EarthGravity = "wgs72",
    ) -> None:
        if isinstance(gravity, str):
            gravity = GRAVITY_MODELS[gravity.lower()]

        self._elements: SGP4Elements = parse_tle(line1, line2)
        self._gravity: EarthGravity = gravity
        self._params: Array
        self._method: str
        self._params, self._method = sgp4_init(self._elements, gravity)

        # Pre-compute epoch from split JD
        jd_int = int(self._elements.jdsatepoch)
        frac_seconds = ((self._elements.jdsatepoch - jd_int) + self._elements.jdsatepochF) * 86400.0
        self._epoch = Epoch._from_internal(
            jnp.int32(jd_int),
            jnp.float64(frac_seconds),
            jnp.float64(0.0),
        )
        self._epoch._normalize()

    # ------------------------------------------------------------------
    # Properties (user-friendly units)
    # ------------------------------------------------------------------

    @property
    def epoch(self) -> Epoch:
        """TLE epoch as an :class:`Epoch` object."""
        return self._epoch

    @property
    def satnum(self) -> str:
        """NORAD catalog number (string, e.g. ``'25544'``)."""
        return self._elements.satnum_str

    @property
    def n(self) -> float:
        """Mean motion [rev/day]."""
        xpdotp = 1440.0 / (2.0 * _py_pi)
        return self._elements.no_kozai * xpdotp

    @property
    def e(self) -> float:
        """Eccentricity [dimensionless]."""
        return self._elements.ecco

    @property
    def i(self) -> float:
        """Inclination [degrees]."""
        return self._elements.inclo * _RAD2DEG

    @property
    def raan(self) -> float:
        """Right ascension of ascending node [degrees]."""
        return self._elements.nodeo * _RAD2DEG

    @property
    def argp(self) -> float:
        """Argument of perigee [degrees]."""
        return self._elements.argpo * _RAD2DEG

    @property
    def M(self) -> float:
        """Mean anomaly [degrees]."""
        return self._elements.mo * _RAD2DEG

    @property
    def bstar(self) -> float:
        """B* drag coefficient [1/earth_radii]."""
        return self._elements.bstar

    @property
    def method(self) -> str:
        """Propagation method: ``'n'`` (near-earth) or ``'d'`` (deep-space)."""
        return self._method

    @property
    def params(self) -> Array:
        """Raw SGP4 parameter array (for advanced use)."""
        return self._params

    # ------------------------------------------------------------------
    # Time conversion
    # ------------------------------------------------------------------

    def _compute_tsince(self, t: Epoch | float | ArrayLike) -> ArrayLike:
        """Convert a time argument to minutes since epoch.

        Args:
            t: Either an :class:`Epoch` (absolute time) or a float/array
               representing seconds since the TLE epoch.

        Returns:
            Time since epoch in minutes.
        """
        if isinstance(t, Epoch):
            delta_seconds = t - self._epoch
            return delta_seconds / 60.0
        return jnp.asarray(t) / 60.0

    # ------------------------------------------------------------------
    # Raw SGP4 output (km, km/s)
    # ------------------------------------------------------------------

    def propagate(self, tsince_min: float | ArrayLike) -> tuple[Array, Array]:
        """Propagate using SGP4 and return raw output.

        Args:
            tsince_min: Time since TLE epoch in **minutes**.

        Returns:
            Tuple ``(r_km, v_kms)`` — position [km] and velocity [km/s]
            in the TEME frame.
        """
        return sgp4_propagate(self._params, jnp.asarray(tsince_min), self._method)

    # ------------------------------------------------------------------
    # State methods (SI: m, m/s)
    # ------------------------------------------------------------------

    def state(self, t: Epoch | float | ArrayLike) -> Array:
        """Compute state in the TEME frame (alias for :meth:`state_teme`).

        Args:
            t: :class:`Epoch` (absolute time) or seconds since TLE epoch.

        Returns:
            6-element TEME state ``[x, y, z, vx, vy, vz]`` in m and m/s.
        """
        return self.state_teme(t)

    def state_teme(self, t: Epoch | float | ArrayLike) -> Array:
        """Compute state in the TEME frame.

        Args:
            t: :class:`Epoch` (absolute time) or seconds since TLE epoch.

        Returns:
            6-element TEME state ``[x, y, z, vx, vy, vz]`` in m and m/s.
        """
        tsince = self._compute_tsince(t)
        r_km, v_kms = sgp4_propagate(self._params, tsince, self._method)
        return jnp.concatenate([r_km * 1e3, v_kms * 1e3])

    def state_pef(self, t: Epoch | float | ArrayLike) -> Array:
        """Compute state in the PEF frame.

        Args:
            t: :class:`Epoch` (absolute time) or seconds since TLE epoch.

        Returns:
            6-element PEF state ``[x, y, z, vx, vy, vz]`` in m and m/s.
        """
        x_teme = self.state_teme(t)
        epc = self._epoch_at(t)
        return state_teme_to_pef(epc, x_teme)

    def state_itrf(
        self,
        t: Epoch | float | ArrayLike,
        eop: EOPData | None = None,
    ) -> Array:
        """Compute state in the ITRF frame.

        Args:
            t: :class:`Epoch` (absolute time) or seconds since TLE epoch.
            eop: EOP data. If ``None``, uses zero EOP (no polar motion).

        Returns:
            6-element ITRF state ``[x, y, z, vx, vy, vz]`` in m and m/s.
        """
        eop = self._resolve_eop(eop)
        x_teme = self.state_teme(t)
        epc = self._epoch_at(t)
        return state_teme_to_itrf(eop, epc, x_teme)

    def state_gcrf(
        self,
        t: Epoch | float | ArrayLike,
        eop: EOPData | None = None,
    ) -> Array:
        """Compute state in the GCRF frame.

        Args:
            t: :class:`Epoch` (absolute time) or seconds since TLE epoch.
            eop: EOP data. If ``None``, uses zero EOP (no polar motion).

        Returns:
            6-element GCRF state ``[x, y, z, vx, vy, vz]`` in m and m/s.
        """
        eop = self._resolve_eop(eop)
        x_teme = self.state_teme(t)
        epc = self._epoch_at(t)
        return state_teme_to_gcrf(eop, epc, x_teme)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _epoch_at(self, t: Epoch | float | ArrayLike) -> Epoch:
        """Return the Epoch corresponding to time argument ``t``.

        Args:
            t: :class:`Epoch` (returned as-is) or seconds since TLE epoch.

        Returns:
            Epoch at the requested time.
        """
        if isinstance(t, Epoch):
            return t
        return self._epoch + jnp.asarray(t, dtype=jnp.float64)

    @staticmethod
    def _resolve_eop(eop: EOPData | None) -> EOPData:
        """Return the given EOP data or zero EOP if ``None``."""
        if eop is None:
            from astrojax.eop import zero_eop

            return zero_eop()
        return eop

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TLE(satnum={self.satnum!r}, epoch={self._epoch}, "
            f"n={self.n:.8f} rev/day, method={self._method!r})"
        )
