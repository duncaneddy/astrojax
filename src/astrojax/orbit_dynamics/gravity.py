"""Gravity force models: point-mass and spherical harmonics.

Provides gravitational acceleration due to point-mass central bodies and
spherical harmonic gravity field models (e.g. EGM2008, GGM05S, JGM3).
All inputs and outputs use SI base units (metres, metres/second squared).

References:
    1. O. Montenbruck and E. Gill, *Satellite Orbits: Models, Methods
       and Applications*, 2012, p. 56-68.
"""

from __future__ import annotations

import importlib.resources
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype
from astrojax.constants import GM_EARTH


# ---------------------------------------------------------------------------
# Point-mass gravity (existing)
# ---------------------------------------------------------------------------


def accel_point_mass(
    r_object: ArrayLike,
    r_body: ArrayLike,
    gm: float,
) -> Array:
    """Acceleration due to point-mass gravity.

    Computes the gravitational acceleration on *r_object* due to a body
    at *r_body* with gravitational parameter *gm*.  When the central body
    is at the origin (``r_body = [0, 0, 0]``), the standard two-body
    expression ``-gm * r / |r|^3`` is used.  Otherwise the indirect
    (third-body) form is applied.

    Args:
        r_object: Position of the object [m].  Shape ``(3,)`` or ``(6,)``
            (only first 3 elements used).
        r_body: Position of the attracting body [m].  Shape ``(3,)``.
        gm: Gravitational parameter of the attracting body [m^3/s^2].

    Returns:
        Acceleration vector [m/s^2], shape ``(3,)``.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.constants import R_EARTH, GM_EARTH
        from astrojax.orbit_dynamics import accel_point_mass
        r = jnp.array([R_EARTH, 0.0, 0.0])
        a = accel_point_mass(r, jnp.zeros(3), GM_EARTH)
        ```
    """
    _float = get_dtype()
    r_obj = jnp.asarray(r_object, dtype=_float)[:3]
    r_cb = jnp.asarray(r_body, dtype=_float)

    d = r_obj - r_cb
    d_norm = jnp.linalg.norm(d)
    r_cb_norm = jnp.linalg.norm(r_cb)

    # Third-body form (r_body != 0): -gm * (d/|d|^3 + r_body/|r_body|^3)
    # Central-body form (r_body = 0): -gm * d/|d|^3
    a_third = -gm * (d / d_norm**3 + r_cb / r_cb_norm**3)
    a_central = -gm * d / d_norm**3

    return jnp.where(r_cb_norm > _float(0.0), a_third, a_central)


def accel_gravity(r_object: ArrayLike) -> Array:
    """Acceleration due to Earth's point-mass gravity.

    Convenience wrapper for :func:`accel_point_mass` with Earth's
    gravitational parameter and the central body at the origin.

    Args:
        r_object: Position of the object in ECI [m].  Shape ``(3,)`` or
            ``(6,)`` (only first 3 elements used).

    Returns:
        Acceleration vector [m/s^2], shape ``(3,)``.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.constants import R_EARTH
        from astrojax.orbit_dynamics import accel_gravity
        r = jnp.array([R_EARTH, 0.0, 0.0])
        a = accel_gravity(r)
        ```
    """
    return accel_point_mass(r_object, jnp.zeros(3, dtype=get_dtype()), GM_EARTH)


# ---------------------------------------------------------------------------
# Gravity model data types
# ---------------------------------------------------------------------------

_PACKAGED_MODELS = {
    "EGM2008_360": "EGM2008_360.gfc",
    "GGM05S": "GGM05S.gfc",
    "JGM3": "JGM3.gfc",
}


class GravityModel:
    """Spherical harmonic gravity field model.

    Stores Stokes coefficients (C_nm, S_nm) parsed from ICGEM GFC format
    files.  The coefficient matrix layout follows the Montenbruck & Gill
    convention:

    - ``data[n, m]`` stores the C coefficient for degree *n*, order *m*
    - ``data[m-1, n]`` stores the S coefficient for *m* > 0

    This is a plain Python class (not a JAX pytree) since it holds static
    configuration data that does not participate in differentiation.

    Args:
        model_name: Human-readable name of the gravity model.
        gm: Gravitational parameter [m^3/s^2].
        radius: Reference radius [m].
        n_max: Maximum degree of the model.
        m_max: Maximum order of the model.
        data: Coefficient matrix, shape ``(n_max+1, m_max+1)``.
        tide_system: Tide system convention (e.g. ``"tide_free"``).
        normalization: Normalization convention (e.g. ``"fully_normalized"``).

    Examples:
        ```python
        from astrojax.orbit_dynamics.gravity import GravityModel
        model = GravityModel.from_type("JGM3")
        c20, s20 = model.get(2, 0)
        ```
    """

    def __init__(
        self,
        model_name: str,
        gm: float,
        radius: float,
        n_max: int,
        m_max: int,
        data: np.ndarray,
        tide_system: str = "unknown",
        normalization: str = "fully_normalized",
    ):
        self.model_name = model_name
        self.gm = gm
        self.radius = radius
        self.n_max = n_max
        self.m_max = m_max
        self.data = data
        self.tide_system = tide_system
        self.normalization = normalization

    @property
    def is_normalized(self) -> bool:
        """Whether the coefficients are fully normalized."""
        return self.normalization == "fully_normalized"

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, filepath: str | Path) -> GravityModel:
        """Load a gravity model from a GFC format file.

        Args:
            filepath: Path to the ``.gfc`` file.

        Returns:
            GravityModel: Loaded gravity model.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If required header fields are missing.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Gravity model file not found: {filepath}")
        with open(filepath) as f:
            return cls._parse_gfc(f)

    @classmethod
    def from_type(cls, model_type: str) -> GravityModel:
        """Load a packaged gravity model by name.

        Available models:

        - ``"EGM2008_360"`` — truncated 360x360 EGM2008
        - ``"GGM05S"`` — full 180x180 GGM05S
        - ``"JGM3"`` — full 70x70 JGM3

        Args:
            model_type: One of the packaged model names.

        Returns:
            GravityModel: Loaded gravity model.

        Raises:
            ValueError: If the model type is not recognized.
        """
        if model_type not in _PACKAGED_MODELS:
            raise ValueError(
                f"Unknown gravity model type: {model_type!r}. "
                f"Available: {list(_PACKAGED_MODELS.keys())}"
            )
        filename = _PACKAGED_MODELS[model_type]
        data_pkg = importlib.resources.files("astrojax.data.gravity_models")
        resource = data_pkg.joinpath(filename)
        with importlib.resources.as_file(resource) as path:
            return cls.from_file(path)

    # ------------------------------------------------------------------
    # Coefficient access
    # ------------------------------------------------------------------

    def get(self, n: int, m: int) -> tuple[float, float]:
        """Retrieve the (C_nm, S_nm) coefficients for degree *n*, order *m*.

        Args:
            n: Degree of the harmonic.
            m: Order of the harmonic.

        Returns:
            tuple[float, float]: (C_nm, S_nm) coefficient pair.

        Raises:
            ValueError: If (n, m) exceeds the model bounds.
        """
        if n > self.n_max or m > self.m_max:
            raise ValueError(
                f"Requested (n={n}, m={m}) exceeds model bounds "
                f"(n_max={self.n_max}, m_max={self.m_max})."
            )
        if m == 0:
            return float(self.data[n, m]), 0.0
        return float(self.data[n, m]), float(self.data[m - 1, n])

    # ------------------------------------------------------------------
    # Model truncation
    # ------------------------------------------------------------------

    def set_max_degree_order(self, n: int, m: int) -> None:
        """Truncate the model to a smaller degree and order.

        Coefficients beyond the new limits are discarded.  This is
        irreversible.

        Args:
            n: New maximum degree (must be <= current ``n_max``).
            m: New maximum order (must be <= *n* and <= current ``m_max``).

        Raises:
            ValueError: If validation fails.
        """
        if m > n:
            raise ValueError(
                f"Maximum order (m={m}) cannot exceed maximum degree (n={n})."
            )
        if n > self.n_max:
            raise ValueError(
                f"Requested degree (n={n}) exceeds model's n_max={self.n_max}."
            )
        if m > self.m_max:
            raise ValueError(
                f"Requested order (m={m}) exceeds model's m_max={self.m_max}."
            )
        if n == self.n_max and m == self.m_max:
            return

        new_size = n + 1
        self.data = self.data[:new_size, :new_size].copy()
        self.n_max = n
        self.m_max = m

    # ------------------------------------------------------------------
    # GFC parser
    # ------------------------------------------------------------------

    @classmethod
    def _parse_gfc(cls, fileobj) -> GravityModel:
        """Parse an ICGEM GFC format file.

        Args:
            fileobj: File-like object with GFC content.

        Returns:
            GravityModel: Parsed gravity model.
        """
        model_name = "Unknown"
        gm = 0.0
        radius = 0.0
        n_max = 0
        m_max = 0
        tide_system = "unknown"
        normalization = "fully_normalized"

        # Read header
        in_header = True
        lines = iter(fileobj)
        for line in lines:
            line = line.strip()
            if line.startswith("end_of_head"):
                in_header = False
                break

            parts = line.split()
            if len(parts) < 2:
                continue

            key = parts[0].lower()
            value = parts[-1]

            if key == "modelname":
                model_name = value
            elif key == "earth_gravity_constant":
                gm = float(value.replace("D", "e").replace("d", "e"))
            elif key == "radius":
                radius = float(value.replace("D", "e").replace("d", "e"))
            elif key == "max_degree":
                n_max = int(value)
                m_max = n_max
            elif key == "tide_system":
                tide_system = value
            elif key in ("errors",):
                pass  # Stored but not used
            elif key in ("norm", "normalization"):
                normalization = value

        if in_header:
            raise ValueError("GFC file missing 'end_of_head' marker.")
        if gm == 0.0:
            raise ValueError("GFC header missing 'earth_gravity_constant'.")
        if radius == 0.0:
            raise ValueError("GFC header missing 'radius'.")
        if n_max == 0:
            raise ValueError("GFC header missing 'max_degree'.")

        # Read coefficient data
        data = np.zeros((n_max + 1, m_max + 1), dtype=np.float64)

        for line in lines:
            line = line.strip()
            if not line or not line.startswith("gfc"):
                continue

            # Replace Fortran-style D/d exponent notation
            line = line.replace("D", "e").replace("d", "e")
            parts = line.split()

            # gfc  n  m  C  S  [sig_C  sig_S]
            n = int(parts[1])
            m = int(parts[2])
            c = float(parts[3])
            s = float(parts[4])

            if n <= n_max and m <= m_max:
                data[n, m] = c
                if m > 0:
                    data[m - 1, n] = s

        return cls(
            model_name=model_name,
            gm=gm,
            radius=radius,
            n_max=n_max,
            m_max=m_max,
            data=data,
            tide_system=tide_system,
            normalization=normalization,
        )

    def __repr__(self) -> str:
        return (
            f"GravityModel(name={self.model_name!r}, "
            f"n_max={self.n_max}, m_max={self.m_max}, "
            f"gm={self.gm:.6e}, radius={self.radius:.1f})"
        )


# ---------------------------------------------------------------------------
# Spherical harmonic gravity acceleration
# ---------------------------------------------------------------------------


def _factorial_product(n: int, m: int) -> float:
    """Compute (n-m)!/(n+m)! efficiently without full factorials.

    Args:
        n: Degree.
        m: Order.

    Returns:
        float: The factorial ratio.
    """
    p = 1.0
    for i in range(n - m + 1, n + m + 1):
        p /= i
    return p


def accel_gravity_spherical_harmonics(
    r_eci: ArrayLike,
    R_eci_to_ecef: ArrayLike,
    gravity_model: GravityModel,
    n_max: int,
    m_max: int,
) -> Array:
    """Acceleration from spherical harmonic gravity field expansion.

    Computes the gravitational acceleration using recursively-computed
    associated Legendre functions (V/W matrix method).  The position is
    transformed to the body-fixed frame, the acceleration is computed
    there, and transformed back to ECI.

    Args:
        r_eci: Position in ECI frame [m].  Shape ``(3,)`` or ``(6,)``
            (only first 3 elements used).
        R_eci_to_ecef: Rotation matrix from ECI to ECEF, shape ``(3, 3)``.
        gravity_model: Loaded gravity model with Stokes coefficients.
        n_max: Maximum degree for evaluation (must be <= model's n_max).
        m_max: Maximum order for evaluation (must be <= n_max and
            <= model's m_max).

    Returns:
        Acceleration in ECI frame [m/s^2], shape ``(3,)``.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.orbit_dynamics.gravity import (
            GravityModel, accel_gravity_spherical_harmonics,
        )
        model = GravityModel.from_type("JGM3")
        r = jnp.array([6878e3, 0.0, 0.0])
        R = jnp.eye(3)
        a = accel_gravity_spherical_harmonics(r, R, model, 20, 20)
        ```
    """
    _float = get_dtype()
    r = jnp.asarray(r_eci, dtype=_float)[:3]
    R = jnp.asarray(R_eci_to_ecef, dtype=_float)

    # Transform to body-fixed frame
    r_bf = R @ r

    # Convert model data to JAX array
    CS = jnp.asarray(gravity_model.data, dtype=_float)

    # Compute acceleration in body-fixed frame
    a_bf = _compute_spherical_harmonics(
        r_bf, CS, n_max, m_max,
        gravity_model.radius, gravity_model.gm,
        gravity_model.is_normalized,
    )

    # Transform back to ECI
    return R.T @ a_bf


def _compute_spherical_harmonics(
    r_bf: Array,
    CS: Array,
    n_max: int,
    m_max: int,
    r_ref: float,
    gm: float,
    is_normalized: bool,
) -> Array:
    """Core V/W recursion for spherical harmonic gravity.

    Implements the algorithm from Montenbruck & Gill (2012), p. 56-68.
    Uses Python loops (traced by JAX) rather than ``jax.lax.fori_loop``
    for clarity.  The ``n_max`` and ``m_max`` parameters are static
    Python ints, so changing them triggers recompilation.

    Args:
        r_bf: Position in body-fixed frame [m], shape ``(3,)``.
        CS: Coefficient matrix from GravityModel, shape ``(N+1, M+1)``.
        n_max: Maximum degree for evaluation.
        m_max: Maximum order for evaluation.
        r_ref: Reference radius [m].
        gm: Gravitational parameter [m^3/s^2].
        is_normalized: Whether coefficients are fully normalized.

    Returns:
        Acceleration in body-fixed frame [m/s^2], shape ``(3,)``.
    """
    # Auxiliary quantities
    r_sqr = jnp.dot(r_bf, r_bf)
    rho = r_ref * r_ref / r_sqr

    # Normalized coordinates
    x0 = r_ref * r_bf[0] / r_sqr
    y0 = r_ref * r_bf[1] / r_sqr
    z0 = r_ref * r_bf[2] / r_sqr

    # V and W intermediary matrices
    size = n_max + 2
    V = jnp.zeros((size, size), dtype=r_bf.dtype)
    W = jnp.zeros((size, size), dtype=r_bf.dtype)

    # Zonal terms V(n,0); W(n,0) = 0
    V = V.at[0, 0].set(r_ref / jnp.sqrt(r_sqr))
    V = V.at[1, 0].set(z0 * V[0, 0])

    for n in range(2, n_max + 2):
        nf = float(n)
        V = V.at[n, 0].set(
            ((2.0 * nf - 1.0) * z0 * V[n - 1, 0]
             - (nf - 1.0) * rho * V[n - 2, 0]) / nf
        )

    # Tesseral and sectorial terms
    for m in range(1, m_max + 2):
        mf = float(m)
        V = V.at[m, m].set(
            (2.0 * mf - 1.0) * (x0 * V[m - 1, m - 1] - y0 * W[m - 1, m - 1])
        )
        W = W.at[m, m].set(
            (2.0 * mf - 1.0) * (x0 * W[m - 1, m - 1] + y0 * V[m - 1, m - 1])
        )

        if m <= n_max:
            V = V.at[m + 1, m].set((2.0 * mf + 1.0) * z0 * V[m, m])
            W = W.at[m + 1, m].set((2.0 * mf + 1.0) * z0 * W[m, m])

        for n in range(m + 2, n_max + 2):
            nf = float(n)
            V = V.at[n, m].set(
                ((2.0 * nf - 1.0) * z0 * V[n - 1, m]
                 - (nf + mf - 1.0) * rho * V[n - 2, m]) / (nf - mf)
            )
            W = W.at[n, m].set(
                ((2.0 * nf - 1.0) * z0 * W[n - 1, m]
                 - (nf + mf - 1.0) * rho * W[n - 2, m]) / (nf - mf)
            )

    # Accumulate accelerations
    ax = jnp.float64(0.0) if r_bf.dtype == jnp.float64 else jnp.float32(0.0)
    ay = ax
    az = ax

    for m in range(m_max + 1):
        mf = float(m)
        for n in range(m, n_max + 1):
            nf = float(n)
            if m == 0:
                # Denormalize if needed
                if is_normalized:
                    N = math.sqrt(2.0 * nf + 1.0)
                    C = N * CS[n, 0]
                else:
                    C = CS[n, 0]

                ax = ax - C * V[n + 1, 1]
                ay = ay - C * W[n + 1, 1]
                az = az - (nf + 1.0) * C * V[n + 1, 0]
            else:
                # Denormalize if needed
                if is_normalized:
                    kron = 0.0 if m != 0 else 1.0
                    N = math.sqrt(
                        (2.0 - kron) * (2.0 * nf + 1.0)
                        * _factorial_product(n, m)
                    )
                    C = N * CS[n, m]
                    S = N * CS[m - 1, n]
                else:
                    C = CS[n, m]
                    S = CS[m - 1, n]

                Fac = 0.5 * (nf - mf + 1.0) * (nf - mf + 2.0)
                ax = ax + (
                    0.5 * (-C * V[n + 1, m + 1] - S * W[n + 1, m + 1])
                    + Fac * (C * V[n + 1, m - 1] + S * W[n + 1, m - 1])
                )
                ay = ay + (
                    0.5 * (-C * W[n + 1, m + 1] + S * V[n + 1, m + 1])
                    + Fac * (-C * W[n + 1, m - 1] + S * V[n + 1, m - 1])
                )
                az = az + (nf - mf + 1.0) * (-C * V[n + 1, m] - S * W[n + 1, m])

    # Scale by GM/R_ref^2
    scale = gm / (r_ref * r_ref)
    return scale * jnp.array([ax, ay, az])
