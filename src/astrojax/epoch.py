"""The epoch module provides the ``Epoch`` class for representing instants in time.

The Epoch class uses an internal representation of integer Julian Day number,
seconds within the day, and a Kahan summation compensator for maintaining
precision during arithmetic operations.

The Kahan compensator tracks floating-point rounding errors that accumulate
during repeated additions (e.g., time stepping in numerical integration),
preventing error growth from O(N) to O(1) machine epsilon.

The Epoch class is registered as a JAX pytree, making it compatible with
``jax.jit``, ``jax.vmap``, and ``jax.lax.scan``. All arithmetic, comparison,
and time-computation methods use JAX operations and are fully traceable.

Internal floats use float32 for GPU/TPU compatibility. The split
representation (int32 days + float32 seconds) provides ~8ms time precision,
which is sufficient for most training and simulation workloads.
"""

from __future__ import annotations

import math
import re

import jax
import jax.numpy as jnp

from .time import caldate_to_jd, jd_to_caldate
from .constants import JD_MJD_OFFSET

# J2000.0 epoch Julian Date
_JD_J2000 = 2451545

# Seconds in a day
_SECONDS_PER_DAY = 86400.0

# Valid ISO 8601 epoch string patterns
_EPOCH_PATTERNS = [
    # YYYY-MM-DD
    re.compile(r'^(\d{4})-(\d{2})-(\d{2})$'),
    # YYYY-MM-DDTHH:MM:SSZ
    re.compile(r'^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z$'),
    # YYYY-MM-DDTHH:MM:SS.fffZ
    re.compile(r'^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})\.(\d+)Z$'),
]


class Epoch:
    """Represents a single instant in time with high-precision arithmetic.

    The internal representation uses three private components:
        ``_jd`` (jnp.int32), ``_seconds`` (jnp.float32),
        ``_kahan_c`` (jnp.float32).
    Use ``jd()`` and ``mjd()`` to access the absolute time as Julian Date
    or Modified Julian Date.

    The Kahan compensator accumulates floating-point rounding errors during
    arithmetic operations, preventing error growth when many small time steps
    are added (e.g., during numerical integration).

    This class is registered as a JAX pytree and is compatible with
    ``jax.jit``, ``jax.vmap``, and ``jax.lax.scan``.

    Precision note:
        The split (int32 + float32) representation gives ~8ms time precision.
        The ``jd()`` and ``mjd()`` accessors return a single float32, which
        has ~0.25 day precision near typical JD values. For high-precision
        time differences, use epoch subtraction (``epc1 - epc2``) which
        preserves the full split-representation precision.

    Constructors:
        Epoch(2018, 1, 1)
        Epoch(2018, 1, 1, 12, 0, 0.0)
        Epoch("2018-01-01T12:00:00Z")
        Epoch(other_epoch)
    """

    __slots__ = ('_jd', '_seconds', '_kahan_c')

    def __init__(self, *args: int | float | str | Epoch) -> None:
        """Initialize Epoch. Supports multiple constructor forms.

        Args:
            *args: Either (year, month, day[, hour, minute, second]),
                a string in ISO 8601 format, or another Epoch instance.
        """
        self._jd = jnp.int32(0)
        self._seconds = jnp.float32(0.0)
        self._kahan_c = jnp.float32(0.0)

        if len(args) == 1:
            if isinstance(args[0], str):
                self._init_string(args[0])
            elif isinstance(args[0], Epoch):
                self._init_epoch(args[0])
            else:
                raise ValueError(f"Cannot construct Epoch from {type(args[0])}")
        elif 3 <= len(args) <= 6:
            self._init_date(*args)
        else:
            raise ValueError(
                "Epoch requires date components (3-6 args), a string, or an Epoch"
            )

    @classmethod
    def _from_internal(cls, jd, seconds, kahan_c):
        """Create an Epoch from raw JAX arrays without Python-side processing.

        Used by pytree unflatten and arithmetic operators. No normalization
        is performed — the caller must ensure values are already normalized.

        Args:
            jd (jnp.int32): Julian Day number.
            seconds (jnp.float32): Seconds within the day [0, 86400).
            kahan_c (jnp.float32): Kahan summation compensator.

        Returns:
            Epoch: New Epoch instance.
        """
        obj = object.__new__(cls)
        obj._jd = jd
        obj._seconds = seconds
        obj._kahan_c = kahan_c
        return obj

    def _init_date(self, year, month, day, hour=0, minute=0, second=0.0):
        """Initialize from calendar date components.

        Args:
            year (int): Year.
            month (int): Month.
            day (int): Day.
            hour (int): Hour. Default: 0
            minute (int): Minute. Default: 0
            second (float): Second, may include fractional part. Default: 0.0
        """
        # Compute JD for the date only (no time component)
        jd_full = float(caldate_to_jd(year, month, day))

        jd_int = int(math.floor(jd_full))
        frac_day = jd_full - jd_int

        # Convert fractional day to seconds and add time components
        seconds = (frac_day * _SECONDS_PER_DAY
                   + hour * 3600.0 + minute * 60.0 + second)

        self._jd = jnp.int32(jd_int)
        self._seconds = jnp.float32(seconds)
        self._kahan_c = jnp.float32(0.0)

        self._normalize()

    def _init_string(self, string):
        """Initialize from an ISO 8601 string.

        Supported formats:
            - ``YYYY-MM-DD``
            - ``YYYY-MM-DDTHH:MM:SSZ``
            - ``YYYY-MM-DDTHH:MM:SS.fffZ``

        Args:
            string (str): ISO 8601 date/time string.
        """
        for pattern in _EPOCH_PATTERNS:
            m = pattern.match(string)
            if m:
                groups = m.groups()
                year = int(groups[0])
                month = int(groups[1])
                day = int(groups[2])

                hour = 0
                minute = 0
                second = 0.0

                if len(groups) >= 6:
                    hour = int(groups[3])
                    minute = int(groups[4])
                    second = float(groups[5])

                if len(groups) == 7:
                    frac_str = groups[6]
                    second += float(f"0.{frac_str}")

                self._init_date(year, month, day, hour, minute, second)
                return

        raise ValueError(
            f'Invalid Epoch string: "{string}" is not ISO 8601 compliant'
        )

    def _init_epoch(self, other):
        """Initialize as a copy of another Epoch.

        Args:
            other (Epoch): Epoch to copy.
        """
        self._jd = other._jd
        self._seconds = other._seconds
        self._kahan_c = other._kahan_c

    def _normalize(self):
        """Normalize seconds to [0, 86400) by adjusting the Julian day number.

        Uses ``jnp.floor`` division instead of while loops so the operation
        is traceable under ``jax.jit``.
        """
        day_offset = jnp.int32(jnp.floor(self._seconds / _SECONDS_PER_DAY))
        self._seconds = self._seconds - jnp.float32(day_offset) * jnp.float32(_SECONDS_PER_DAY)
        self._jd = self._jd + day_offset

    def _compensated_seconds(self):
        """Return the compensated seconds value.

        Returns:
            jnp.float32: Seconds with Kahan compensation applied.
        """
        return self._seconds - self._kahan_c

    # Arithmetic operators

    def __iadd__(self, delta: float) -> Epoch:
        """Add seconds to this epoch using Kahan compensated summation.

        Returns a new Epoch instance (Python rebinds the name on ``+=``).
        This functional style is required for JAX traceability — JAX pytree
        leaves are immutable during tracing.

        Args:
            delta (float): Seconds to add.

        Returns:
            Epoch: New Epoch with delta seconds added.
        """
        delta = jnp.float32(delta)
        y = delta - self._kahan_c
        t = self._seconds + y
        new_kahan_c = (t - self._seconds) - y
        new_seconds = t

        # Normalize: single floor-division handles any magnitude of overflow
        day_offset = jnp.int32(jnp.floor(new_seconds / _SECONDS_PER_DAY))
        new_seconds = new_seconds - jnp.float32(day_offset) * jnp.float32(_SECONDS_PER_DAY)
        new_jd = self._jd + day_offset

        return Epoch._from_internal(new_jd, new_seconds, new_kahan_c)

    def __isub__(self, delta: float) -> Epoch:
        """Subtract seconds from this epoch.

        Args:
            delta (float): Seconds to subtract.

        Returns:
            Epoch: New Epoch with delta seconds subtracted.
        """
        return self.__iadd__(-jnp.float32(delta))

    def __add__(self, delta: float) -> Epoch:
        """Return a new Epoch with seconds added.

        Args:
            delta (float): Seconds to add.

        Returns:
            Epoch: New Epoch advanced by delta seconds.
        """
        return self.__iadd__(delta)

    def __sub__(self, other: Epoch | float) -> Epoch | jax.Array:
        """Subtract seconds or compute difference between Epochs.

        Args:
            other: If Epoch, returns the time difference in seconds.
                If numeric, returns a new Epoch with seconds subtracted.

        Returns:
            float or Epoch: Time difference in seconds, or new Epoch.
        """
        if isinstance(other, Epoch):
            return ((self._jd - other._jd) * _SECONDS_PER_DAY
                    + (self._compensated_seconds()
                       - other._compensated_seconds()))
        return self.__iadd__(-jnp.float32(other))

    # Comparison operators

    def __eq__(self, other):
        if not isinstance(other, Epoch):
            return NotImplemented
        return (self._jd == other._jd) & (
            jnp.abs(self._compensated_seconds()
                    - other._compensated_seconds()) < 1e-3
        )

    def __ne__(self, other):
        if not isinstance(other, Epoch):
            return NotImplemented
        return ~self.__eq__(other)

    def __lt__(self, other):
        if not isinstance(other, Epoch):
            return NotImplemented
        return jnp.where(
            self._jd != other._jd,
            self._jd < other._jd,
            self._compensated_seconds() < other._compensated_seconds(),
        )

    def __le__(self, other):
        if not isinstance(other, Epoch):
            return NotImplemented
        return self.__lt__(other) | self.__eq__(other)

    def __gt__(self, other):
        if not isinstance(other, Epoch):
            return NotImplemented
        return jnp.where(
            self._jd != other._jd,
            self._jd > other._jd,
            self._compensated_seconds() > other._compensated_seconds(),
        )

    def __ge__(self, other):
        if not isinstance(other, Epoch):
            return NotImplemented
        return self.__gt__(other) | self.__eq__(other)

    # Time properties

    def caldate(self) -> tuple[jax.Array, jax.Array, jax.Array, int, int, float]:
        """Return the calendar date components.

        This method extracts concrete Python values from JAX arrays and
        is not traceable under ``jax.jit``. It should only be called
        outside of JIT-compiled functions.

        Returns:
            tuple: (year, month, day, hour, minute, second) where second
                includes fractional part.
        """
        comp_seconds = float(self._compensated_seconds())
        jd_full = int(self._jd) + comp_seconds / _SECONDS_PER_DAY

        # Get date (year, month, day) from the JD — always accurate
        year, month, day, _, _, _ = jd_to_caldate(jd_full)

        # Compute h/m/s directly from seconds to preserve full precision.
        # JD day starts at noon, so shift by 43200s to get civil time of day.
        civil_time = (comp_seconds + 43200.0) % _SECONDS_PER_DAY

        hour = int(civil_time // 3600)
        civil_time -= hour * 3600
        minute = int(civil_time // 60)
        second = civil_time - minute * 60

        return year, month, day, hour, minute, second

    def jd(self) -> jax.Array:
        """Return the Julian Date as a single float32.

        Note:
            A single float32 near typical JD values (~2.45M) has ~0.25 day
            precision. For time-of-day sensitive computations, use the Epoch
            object directly or ``caldate()``.

        Returns:
            jnp.float32: Julian Date (lossy for sub-day precision).
        """
        return jnp.float32(self._jd) + self._compensated_seconds() / jnp.float32(_SECONDS_PER_DAY)

    def mjd(self) -> jax.Array:
        """Return the Modified Julian Date as a single float32.

        Note:
            MJD values (~51544) are smaller than JD, giving ~0.004 day
            (~6 min) precision in float32 — better than ``jd()`` but still
            lossy for sub-minute precision.

        Returns:
            jnp.float32: Modified Julian Date.
        """
        return self.jd() - jnp.float32(JD_MJD_OFFSET)

    # Sidereal time

    def gmst(self, use_degrees: bool = False) -> jax.Array:
        """Compute Greenwich Mean Sidereal Time using the IAU 1982 model.

        Uses the Vallado GMST82 polynomial approximation. This implementation
        assumes UTC approximates UT1, which introduces at most ~1 second of
        error for most applications.

        Computes T_UT1 directly from the split (int32 + float32) internal
        representation to avoid the precision loss of a single-float JD.

        All operations use ``jnp`` and are fully traceable under ``jax.jit``.

        Args:
            use_degrees (bool): If True, return in degrees. Default: False
                (radians).

        Returns:
            jnp.float32: Greenwich Mean Sidereal Time. Units: rad (or deg if
                use_degrees=True)

        References:

            1. D. Vallado, *Fundamentals of Astrodynamics and Applications
               (4th Ed.)*, 2010.
        """
        # Compute T_UT1 (Julian centuries from J2000) using split
        # representation to preserve precision. The int32 day difference
        # is exact, and the float32 fractional day has ~1e-4 day precision.
        days_from_j2000 = jnp.float32(self._jd - jnp.int32(_JD_J2000))
        frac_day = self._compensated_seconds() / jnp.float32(_SECONDS_PER_DAY)
        t_ut1 = (days_from_j2000 + frac_day) / jnp.float32(36525.0)

        # GMST in seconds of time (polynomial in Julian centuries from J2000)
        gmst_sec = (jnp.float32(67310.54841)
                    + jnp.float32(876600.0 * 3600.0 + 8640184.812866) * t_ut1
                    + jnp.float32(0.093104) * t_ut1 * t_ut1
                    - jnp.float32(6.2e-6) * t_ut1 * t_ut1 * t_ut1)

        # Convert seconds of time to radians (1 second = 1/240 degree)
        # and normalize to [0, 2*pi)
        gmst_rad = (gmst_sec / jnp.float32(240.0) * jnp.pi / jnp.float32(180.0)) % (jnp.float32(2.0) * jnp.pi)

        gmst_rad = jnp.where(gmst_rad < 0, gmst_rad + jnp.float32(2.0) * jnp.pi, gmst_rad)

        return jnp.where(use_degrees, gmst_rad * jnp.float32(180.0) / jnp.pi, gmst_rad)

    # String representations

    def __str__(self):
        year, month, day, hour, minute, second = self.caldate()
        return (f'{year:04d}-{month:02d}-{day:02d}T'
                f'{hour:02d}:{minute:02d}:{second:06.3f}Z')

    def __repr__(self):
        return (f'Epoch(_jd={int(self._jd)}, _seconds={float(self._seconds)}, '
                f'_kahan_c={float(self._kahan_c)})')

    def __hash__(self):
        return hash((int(self._jd), round(float(self._seconds), 3)))


# Register Epoch as a JAX pytree so it can be used with jit, vmap, scan, etc.
jax.tree_util.register_pytree_node(
    Epoch,
    lambda e: ((e._jd, e._seconds, e._kahan_c), None),
    lambda _, children: Epoch._from_internal(*children),
)
