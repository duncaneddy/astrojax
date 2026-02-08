"""Keplerian orbital mechanics functions.

This sub-module provides functions for:

- **Orbital period and semi-major axis**: computing period from semi-major
  axis, mean motion, or state vectors, and the inverse conversions.
- **Velocities at apsides**: perigee and apogee velocity magnitudes.
- **Distances and altitudes**: periapsis/apoapsis distances and altitudes
  above Earth's surface.
- **Anomaly conversions**: converting between mean, eccentric, and true
  anomalies, including a JAX-traceable Kepler equation solver.
- **Special orbits**: sun-synchronous inclination and geostationary
  semi-major axis.
- **Mean-osculating conversions**: first-order J2 mapping between mean
  and osculating Keplerian elements (Brouwer-Lyddane theory).
"""

from .keplerian import (
    anomaly_eccentric_to_mean,
    anomaly_eccentric_to_true,
    anomaly_mean_to_eccentric,
    anomaly_mean_to_true,
    anomaly_true_to_eccentric,
    anomaly_true_to_mean,
    apoapsis_distance,
    apogee_altitude,
    apogee_velocity,
    geo_sma,
    mean_motion,
    orbital_period,
    orbital_period_from_state,
    perigee_altitude,
    perigee_velocity,
    periapsis_distance,
    semimajor_axis,
    semimajor_axis_from_orbital_period,
    sun_synchronous_inclination,
)
from .mean_elements import (
    state_koe_mean_to_osc,
    state_koe_osc_to_mean,
)

__all__ = [
    "orbital_period",
    "orbital_period_from_state",
    "semimajor_axis_from_orbital_period",
    "semimajor_axis",
    "mean_motion",
    "perigee_velocity",
    "apogee_velocity",
    "periapsis_distance",
    "apoapsis_distance",
    "perigee_altitude",
    "apogee_altitude",
    "sun_synchronous_inclination",
    "geo_sma",
    "anomaly_eccentric_to_mean",
    "anomaly_mean_to_eccentric",
    "anomaly_true_to_eccentric",
    "anomaly_eccentric_to_true",
    "anomaly_true_to_mean",
    "anomaly_mean_to_true",
    "state_koe_osc_to_mean",
    "state_koe_mean_to_osc",
]
