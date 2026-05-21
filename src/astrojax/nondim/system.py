"""UnitSystem: scales for nondimensional astrodynamics computation.

A UnitSystem is three independent SI scales (length, time, mass).  All
other physical scales (velocity, acceleration, mu, density, force)
follow by dimensional analysis.

UnitSystem instances are immutable Python value objects, intentionally
*not* JAX pytrees — they pass through JIT as static configuration so
changes do not trigger retracing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class UnitSystem:
    """Three independent SI scales defining a nondimensional unit system.

    Args:
        LU: Length unit [m].  Must be positive.
        TU: Time unit [s].   Must be positive.
        MU: Mass unit [kg].  Must be positive.
    """

    LU: float
    TU: float
    MU: float

    def __post_init__(self) -> None:
        for name, value in (("LU", self.LU), ("TU", self.TU), ("MU", self.MU)):
            if value <= 0.0 or not math.isfinite(value):
                raise ValueError(f"UnitSystem.{name} must be positive and finite, got {value!r}")

    @property
    def VU(self) -> float:
        """Velocity scale [m/s]."""
        return self.LU / self.TU

    @property
    def accel(self) -> float:
        """Acceleration scale [m/s^2]."""
        return self.LU / (self.TU * self.TU)

    @property
    def mu(self) -> float:
        """Gravitational-parameter scale [m^3/s^2]."""
        return (self.LU**3) / (self.TU * self.TU)

    @property
    def density(self) -> float:
        """Mass-density scale [kg/m^3]."""
        return self.MU / (self.LU**3)

    @property
    def force(self) -> float:
        """Force scale [N]."""
        return self.MU * self.LU / (self.TU * self.TU)

    # ─── constructors ──────────────────────────────────────────────────

    @classmethod
    def from_scales(cls, LU: float, TU: float, MU: float = 1.0) -> UnitSystem:
        """Construct a UnitSystem from explicit SI scales.

        Args:
            LU: Length scale [m].
            TU: Time scale [s].
            MU: Mass scale [kg]. Defaults to 1.0.
        """
        return cls(LU=float(LU), TU=float(TU), MU=float(MU))

    @classmethod
    def from_orbit(cls, a: float, mu: float, mass: float = 1.0) -> UnitSystem:
        """Construct canonical units for an orbit with semi-major axis ``a``.

        Sets LU = a and TU = sqrt(a^3 / mu). As a consequence the
        nondimensional gravitational parameter mu_nd = mu_SI / system.mu
        equals 1, and the nondimensional mean motion n_nd = n_SI * TU
        also equals 1.
        """
        a = float(a)
        mu = float(mu)
        return cls(LU=a, TU=math.sqrt(a**3 / mu), MU=float(mass))

    @classmethod
    def from_orbit_relative(
        cls, a: float, mu: float, LU_rel: float, mass: float = 1.0
    ) -> UnitSystem:
        """Construct nondim units for relative motion about an orbit.

        Time scale matches the chief orbit (so n_nd = 1) but length scale
        is the user-chosen relative separation. The nondimensional
        gravitational parameter is mu_nd = (a / LU_rel)^3, *not* 1 —
        that is the value to use when invoking force functions that
        take an explicit mu argument in this unit system.
        """
        a = float(a)
        mu = float(mu)
        return cls(LU=float(LU_rel), TU=math.sqrt(a**3 / mu), MU=float(mass))

    # ─── presets ───────────────────────────────────────────────────────

    @classmethod
    def earth_canonical(cls) -> UnitSystem:
        """LU = R_EARTH, TU chosen so μ_⊕ = 1."""
        from astrojax.constants import GM_EARTH, R_EARTH

        return cls.from_orbit(a=R_EARTH, mu=GM_EARTH)

    @classmethod
    def lunar_canonical(cls) -> UnitSystem:
        """LU = R_MOON, TU chosen so μ_moon = 1."""
        from astrojax.constants import GM_MOON, R_MOON

        return cls.from_orbit(a=R_MOON, mu=GM_MOON)

    @classmethod
    def solar_canonical(cls) -> UnitSystem:
        """LU = R_SUN, TU chosen so μ_sun = 1."""
        from astrojax.constants import GM_SUN, R_SUN

        return cls.from_orbit(a=R_SUN, mu=GM_SUN)
