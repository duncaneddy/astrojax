"""Configuration dataclasses for composable attitude dynamics.

Provides :class:`SpacecraftInertia` for rigid-body inertia properties and
:class:`AttitudeDynamicsConfig` for selecting which torque models to include
in an attitude propagation.  Configuration is static — Python ``if`` branches
on boolean toggles are resolved at JAX trace time, producing a single
optimized computation graph with no runtime branching.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array

from astrojax.config import get_dtype
from astrojax.constants import GM_EARTH


@dataclass(frozen=True)
class SpacecraftInertia:
    """Rigid-body inertia tensor of the spacecraft.

    Stores the 3x3 inertia tensor in the body frame.  For most
    spacecraft the tensor is diagonal (principal axes aligned with body
    axes) and can be constructed with :meth:`from_principal`.

    Args:
        I: 3x3 inertia tensor [kg m^2].

    Examples:
        ```python
        from astrojax.attitude_dynamics.config import SpacecraftInertia
        inertia = SpacecraftInertia.from_principal(100.0, 200.0, 300.0)
        inertia.I.shape
        ```
    """

    I: Array = field(default_factory=lambda: jnp.eye(3, dtype=get_dtype()))  # noqa: E741

    @staticmethod
    def from_principal(Ixx: float, Iyy: float, Izz: float) -> SpacecraftInertia:
        """Create an inertia tensor from principal moments.

        Args:
            Ixx: Moment of inertia about the body x-axis [kg m^2].
            Iyy: Moment of inertia about the body y-axis [kg m^2].
            Izz: Moment of inertia about the body z-axis [kg m^2].

        Returns:
            SpacecraftInertia: Diagonal inertia tensor.

        Examples:
            ```python
            inertia = SpacecraftInertia.from_principal(10.0, 20.0, 30.0)
            float(inertia.I[0, 0])
            ```
        """
        _float = get_dtype()
        I = jnp.diag(jnp.array([Ixx, Iyy, Izz], dtype=_float))  # noqa: E741
        return SpacecraftInertia(I=I)


@dataclass(frozen=True)
class AttitudeDynamicsConfig:
    """Configuration for composable attitude dynamics.

    Selects which torque models to include in the dynamics closure
    returned by
    :func:`~astrojax.attitude_dynamics.factory.create_attitude_dynamics`.

    Args:
        inertia: Spacecraft inertia tensor.
        gravity_gradient: Enable gravity gradient torque.
        mu: Gravitational parameter [m^3/s^2] for gravity gradient.

    Examples:
        ```python
        from astrojax.attitude_dynamics.config import AttitudeDynamicsConfig
        config = AttitudeDynamicsConfig.torque_free(
            SpacecraftInertia.from_principal(10.0, 20.0, 30.0)
        )
        config.gravity_gradient
        ```
    """

    inertia: SpacecraftInertia = field(default_factory=SpacecraftInertia)
    gravity_gradient: bool = False
    mu: float = GM_EARTH

    @staticmethod
    def torque_free(inertia: SpacecraftInertia) -> AttitudeDynamicsConfig:
        """Preset: torque-free rigid body rotation.

        Args:
            inertia: Spacecraft inertia tensor.

        Returns:
            AttitudeDynamicsConfig: Configuration with no external torques.

        Examples:
            ```python
            config = AttitudeDynamicsConfig.torque_free(
                SpacecraftInertia.from_principal(10.0, 20.0, 30.0)
            )
            config.gravity_gradient
            ```
        """
        return AttitudeDynamicsConfig(inertia=inertia)

    @staticmethod
    def with_gravity_gradient(
        inertia: SpacecraftInertia,
        mu: float = GM_EARTH,
    ) -> AttitudeDynamicsConfig:
        """Preset: rigid body with gravity gradient torque.

        Args:
            inertia: Spacecraft inertia tensor.
            mu: Gravitational parameter [m^3/s^2].

        Returns:
            AttitudeDynamicsConfig: Configuration with gravity gradient enabled.

        Examples:
            ```python
            config = AttitudeDynamicsConfig.with_gravity_gradient(
                SpacecraftInertia.from_principal(10.0, 20.0, 30.0)
            )
            config.gravity_gradient
            ```
        """
        return AttitudeDynamicsConfig(
            inertia=inertia,
            gravity_gradient=True,
            mu=mu,
        )
