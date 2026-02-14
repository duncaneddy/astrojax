"""Configuration dataclasses for composable orbit dynamics.

Provides :class:`SpacecraftParams` for physical spacecraft properties and
:class:`ForceModelConfig` for selecting which perturbation forces to
include in an orbit propagation.  Configuration is static — Python ``if``
branches on boolean toggles are resolved at JAX trace time, producing a
single optimized computation graph with no runtime branching.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from astrojax.orbit_dynamics.gravity import GravityModel


@dataclass(frozen=True)
class SpacecraftParams:
    """Physical properties of the spacecraft.

    All values are SI.  Defaults represent a generic small satellite.

    Args:
        mass: Spacecraft mass [kg].
        drag_area: Wind-facing cross-sectional area [m^2].
        srp_area: Sun-facing cross-sectional area [m^2].
        cd: Coefficient of drag [dimensionless].
        cr: Coefficient of reflectivity [dimensionless].
    """

    mass: float = 1000.0
    drag_area: float = 10.0
    srp_area: float = 10.0
    cd: float = 2.2
    cr: float = 1.3


@dataclass(frozen=True)
class ForceModelConfig:
    """Configuration for composable orbit dynamics.

    Selects which perturbation forces to include in the dynamics closure
    returned by :func:`~astrojax.orbit_dynamics.factory.create_orbit_dynamics`.

    Args:
        gravity_type: ``"point_mass"`` or ``"spherical_harmonics"``.
        gravity_model: A :class:`GravityModel` instance.  Required when
            *gravity_type* is ``"spherical_harmonics"``.
        gravity_degree: Maximum degree for spherical harmonic evaluation.
        gravity_order: Maximum order for spherical harmonic evaluation.
        drag: Enable atmospheric drag.
        density_model: Atmospheric density model — ``"harris_priester"``
            or ``"nrlmsise00"``.  Only used when *drag* is ``True``.
        srp: Enable solar radiation pressure.
        third_body_sun: Enable Sun gravitational perturbation.
        third_body_moon: Enable Moon gravitational perturbation.
        eclipse_model: Shadow model for SRP — ``"conical"``,
            ``"cylindrical"``, or ``"none"``.
        spacecraft: Spacecraft physical properties.

    Examples:
        ```python
        from astrojax.orbit_dynamics.config import ForceModelConfig
        config = ForceModelConfig()  # point-mass only
        config.gravity_type
        ```
    """

    # Gravity
    gravity_type: str = "point_mass"
    gravity_model: GravityModel | None = None
    gravity_degree: int = 20
    gravity_order: int = 20

    # Perturbation toggles
    drag: bool = False
    density_model: str = "harris_priester"
    srp: bool = False
    third_body_sun: bool = False
    third_body_moon: bool = False

    # SRP eclipse model
    eclipse_model: str = "conical"

    # Spacecraft
    spacecraft: SpacecraftParams = field(default_factory=SpacecraftParams)

    def __post_init__(self) -> None:
        if self.gravity_type not in ("point_mass", "spherical_harmonics"):
            raise ValueError(
                f"gravity_type must be 'point_mass' or 'spherical_harmonics', "
                f"got '{self.gravity_type}'"
            )
        if self.density_model not in ("harris_priester", "nrlmsise00"):
            raise ValueError(
                f"density_model must be 'harris_priester' or 'nrlmsise00', "
                f"got '{self.density_model}'"
            )
        if self.eclipse_model not in ("conical", "cylindrical", "none"):
            raise ValueError(
                f"eclipse_model must be 'conical', 'cylindrical', or 'none', "
                f"got '{self.eclipse_model}'"
            )

    @staticmethod
    def two_body() -> ForceModelConfig:
        """Preset: point-mass gravity only (Keplerian two-body).

        Returns:
            ForceModelConfig: Configuration with only point-mass gravity.

        Examples:
            ```python
            config = ForceModelConfig.two_body()
            config.gravity_type
            ```
        """
        return ForceModelConfig()

    @staticmethod
    def leo_default(
        gravity_model: GravityModel | None = None,
        density_model: str = "harris_priester",
    ) -> ForceModelConfig:
        """Preset: typical LEO force model.

        Includes 20x20 spherical harmonic gravity, atmospheric drag, SRP,
        and Sun/Moon third-body perturbations.  If no gravity model is
        provided, ``JGM3`` is loaded automatically.

        Args:
            gravity_model: Optional pre-loaded gravity model.
            density_model: Atmospheric density model — ``"harris_priester"``
                or ``"nrlmsise00"``.

        Returns:
            ForceModelConfig: LEO-appropriate configuration.

        Examples:
            ```python
            config = ForceModelConfig.leo_default()
            config.drag
            ```
        """
        if gravity_model is None:
            gravity_model = GravityModel.from_type("JGM3")
        return ForceModelConfig(
            gravity_type="spherical_harmonics",
            gravity_model=gravity_model,
            gravity_degree=20,
            gravity_order=20,
            drag=True,
            density_model=density_model,
            srp=True,
            third_body_sun=True,
            third_body_moon=True,
        )

    @staticmethod
    def geo_default(
        gravity_model: GravityModel | None = None,
    ) -> ForceModelConfig:
        """Preset: typical GEO force model.

        Includes 8x8 spherical harmonic gravity, SRP, and Sun/Moon
        third-body perturbations.  No atmospheric drag (altitude too high
        for Harris-Priester).

        Args:
            gravity_model: Optional pre-loaded gravity model.

        Returns:
            ForceModelConfig: GEO-appropriate configuration.

        Examples:
            ```python
            config = ForceModelConfig.geo_default()
            config.drag
            ```
        """
        if gravity_model is None:
            gravity_model = GravityModel.from_type("JGM3")
        return ForceModelConfig(
            gravity_type="spherical_harmonics",
            gravity_model=gravity_model,
            gravity_degree=8,
            gravity_order=8,
            drag=False,
            srp=True,
            third_body_sun=True,
            third_body_moon=True,
        )
