"""Euler axis (axis-angle) attitude representation.

Provides the ``EulerAxis`` class representing a rotation as a unit
axis vector and a rotation angle.  Internal storage is always in
radians.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from astrojax.config import get_dtype
from astrojax.utils import to_radians
from astrojax.attitude_representations._tolerance import get_attitude_epsilon

if TYPE_CHECKING:
    from astrojax.attitude_representations.euler_angle import EulerAngle, EulerAngleOrder
    from astrojax.attitude_representations.quaternion import Quaternion
    from astrojax.attitude_representations.rotation_matrix import RotationMatrix


class EulerAxis:
    """Rotation defined by a unit axis and angle.

    The axis is stored as a shape ``(3,)`` array and the angle as a
    scalar, both in the configured float dtype.  Angles are always
    stored in radians.

    This class is registered as a JAX pytree with ``(axis, angle)``
    as leaves and no auxiliary data.

    Args:
        axis (jax.Array): Rotation axis (need not be unit length; will be stored as-is).
        angle (float): Rotation angle.
        use_degrees (bool): If ``True``, interpret ``angle`` as degrees. Default: ``False``.
    """

    __slots__ = ('_axis', '_angle')

    def __init__(self, axis: jax.Array, angle: float, use_degrees: bool = False) -> None:
        _float = get_dtype()
        self._axis = jnp.asarray(axis, dtype=_float)
        self._angle = _float(to_radians(angle, use_degrees))

    @classmethod
    def _from_internal(cls, axis: jax.Array, angle: jax.Array) -> EulerAxis:
        """Create from raw JAX arrays without conversion.

        Used by pytree unflatten and conversion outputs.

        Args:
            axis (jax.Array): Unit axis array of shape ``(3,)``.
            angle (jax.Array): Angle scalar in radians.

        Returns:
            EulerAxis: New instance.
        """
        obj = object.__new__(cls)
        obj._axis = axis
        obj._angle = angle
        return obj

    # Properties

    @property
    def axis(self) -> jax.Array:
        """Unit rotation axis of shape ``(3,)``."""
        return self._axis

    @property
    def angle(self) -> jax.Array:
        """Rotation angle in radians."""
        return self._angle

    # Factory methods

    @classmethod
    def from_values(cls, x: float, y: float, z: float, angle: float, use_degrees: bool = False) -> EulerAxis:
        """Create from individual axis components and angle.

        Args:
            x (float): Axis x-component.
            y (float): Axis y-component.
            z (float): Axis z-component.
            angle (float): Rotation angle.
            use_degrees (bool): If ``True``, interpret as degrees.

        Returns:
            EulerAxis: New instance.
        """
        return cls(jnp.array([x, y, z]), angle, use_degrees=use_degrees)

    @classmethod
    def from_vector(cls, v: jax.Array, use_degrees: bool = False, vector_first: bool = True) -> EulerAxis:
        """Create from a 4-element vector.

        Args:
            v (jax.Array): Array-like of shape ``(4,)``.
            use_degrees (bool): If ``True``, the angle component is in degrees.
            vector_first (bool): If ``True``, ``v = [x, y, z, angle]``.
                If ``False``, ``v = [angle, x, y, z]``.

        Returns:
            EulerAxis: New instance.
        """
        if vector_first:
            return cls(jnp.array([v[0], v[1], v[2]]), v[3], use_degrees=use_degrees)
        else:
            return cls(jnp.array([v[1], v[2], v[3]]), v[0], use_degrees=use_degrees)

    def to_vector(self, use_degrees: bool = False, vector_first: bool = True) -> jax.Array:
        """Return as a 4-element vector.

        Args:
            use_degrees (bool): If ``True``, output angle in degrees.
            vector_first (bool): If ``True``, return ``[x, y, z, angle]``.
                If ``False``, return ``[angle, x, y, z]``.

        Returns:
            jnp.ndarray: Array of shape ``(4,)``.
        """
        angle = jnp.where(use_degrees, jnp.rad2deg(self._angle), self._angle)
        if vector_first:
            return jnp.array([self._axis[0], self._axis[1], self._axis[2], angle])
        else:
            return jnp.array([angle, self._axis[0], self._axis[1], self._axis[2]])

    # Conversion methods

    def to_quaternion(self) -> Quaternion:
        """Convert to ``Quaternion``.

        Returns:
            Quaternion: Equivalent quaternion.
        """
        from astrojax.attitude_representations.conversions import euler_axis_to_quaternion
        from astrojax.attitude_representations.quaternion import Quaternion

        q = euler_axis_to_quaternion(self._axis, self._angle)
        return Quaternion._from_internal(q)

    def to_rotation_matrix(self) -> RotationMatrix:
        """Convert to ``RotationMatrix``.

        Routes through quaternion.

        Returns:
            RotationMatrix: Equivalent rotation matrix.
        """
        return self.to_quaternion().to_rotation_matrix()

    def to_euler_angle(self, order: EulerAngleOrder) -> EulerAngle:
        """Convert to ``EulerAngle``.

        Args:
            order (EulerAngleOrder): ``EulerAngleOrder`` specifying the rotation sequence.

        Returns:
            EulerAngle: Equivalent euler angle.
        """
        return self.to_quaternion().to_euler_angle(order)

    def to_euler_axis(self) -> EulerAxis:
        """Return a copy.

        Returns:
            EulerAxis: New instance with same data.
        """
        return EulerAxis._from_internal(self._axis, self._angle)

    @classmethod
    def from_quaternion(cls, q: Quaternion) -> EulerAxis:
        """Create from a ``Quaternion``.

        Args:
            q (Quaternion): Source quaternion.

        Returns:
            EulerAxis: Equivalent euler axis.
        """
        return q.to_euler_axis()

    @classmethod
    def from_rotation_matrix(cls, r: RotationMatrix) -> EulerAxis:
        """Create from a ``RotationMatrix``.

        Args:
            r (RotationMatrix): Source rotation matrix.

        Returns:
            EulerAxis: Equivalent euler axis.
        """
        return r.to_quaternion().to_euler_axis()

    @classmethod
    def from_euler_angle(cls, e: EulerAngle) -> EulerAxis:
        """Create from an ``EulerAngle``.

        Args:
            e (EulerAngle): Source euler angle.

        Returns:
            EulerAxis: Equivalent euler axis.
        """
        return e.to_quaternion().to_euler_axis()

    @classmethod
    def from_euler_axis(cls, ea: EulerAxis) -> EulerAxis:
        """Create from another ``EulerAxis``.

        Args:
            ea (EulerAxis): Source euler axis.

        Returns:
            EulerAxis: Copy.
        """
        return ea.to_euler_axis()

    # Comparison

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EulerAxis):
            return NotImplemented
        eps = get_attitude_epsilon()
        return (
            jnp.all(jnp.abs(self._axis - other._axis) < eps)
            and jnp.abs(self._angle - other._angle) < eps
        )

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, EulerAxis):
            return NotImplemented
        return not self.__eq__(other)

    def __getitem__(self, idx: int) -> jax.Array:
        return self._axis[idx]

    # String representations

    def __str__(self) -> str:
        return (
            f"EulerAxis(axis=[{float(self._axis[0]):.6f}, "
            f"{float(self._axis[1]):.6f}, "
            f"{float(self._axis[2]):.6f}], "
            f"angle={float(self._angle):.6f})"
        )

    def __repr__(self) -> str:
        return (
            f"EulerAxis(axis=[{float(self._axis[0])}, "
            f"{float(self._axis[1])}, "
            f"{float(self._axis[2])}], "
            f"angle={float(self._angle)})"
        )


# Register as JAX pytree
jax.tree_util.register_pytree_node(
    EulerAxis,
    lambda e: ((e._axis, e._angle), None),
    lambda _, children: EulerAxis._from_internal(children[0], children[1]),
)
