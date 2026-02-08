"""Quaternion attitude representation.

Provides the ``Quaternion`` class representing a rotation as a unit
quaternion in scalar-first convention ``[w, x, y, z]``.

The quaternion is normalized on construction.  This class serves as the
central hub for attitude conversions -- most cross-type conversions
route through ``Quaternion`` (matching the Rust brahe architecture).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from astrojax.attitude_representations._tolerance import get_attitude_epsilon
from astrojax.config import get_dtype

if TYPE_CHECKING:
    from astrojax.attitude_representations.euler_angle import EulerAngle, EulerAngleOrder
    from astrojax.attitude_representations.euler_axis import EulerAxis
    from astrojax.attitude_representations.rotation_matrix import RotationMatrix


class Quaternion:
    """Unit quaternion representing a 3D rotation.

    Internal storage is a shape ``(4,)`` array in scalar-first order
    ``[w, x, y, z]``.  The quaternion is normalized on construction.

    This class is registered as a JAX pytree with the data array as
    the sole leaf and no auxiliary data.

    Args:
        s (float): Scalar (real) component.
        v1 (float): First vector (imaginary) component.
        v2 (float): Second vector (imaginary) component.
        v3 (float): Third vector (imaginary) component.
    """

    __slots__ = ('_data',)

    def __init__(self, s: float, v1: float, v2: float, v3: float) -> None:
        _float = get_dtype()
        q = jnp.array([_float(s), _float(v1), _float(v2), _float(v3)])
        self._data = q / jnp.linalg.norm(q)

    @classmethod
    def _from_internal(cls, data: jax.Array) -> Quaternion:
        """Create from a raw JAX array without normalization.

        Used by pytree unflatten and conversion outputs.

        Args:
            data (jax.Array): Array of shape ``(4,)`` in scalar-first order.

        Returns:
            Quaternion: New instance.
        """
        obj = object.__new__(cls)
        obj._data = data
        return obj

    # Properties

    @property
    def w(self) -> jax.Array:
        """Scalar component."""
        return self._data[0]

    @property
    def x(self) -> jax.Array:
        """First vector component."""
        return self._data[1]

    @property
    def y(self) -> jax.Array:
        """Second vector component."""
        return self._data[2]

    @property
    def z(self) -> jax.Array:
        """Third vector component."""
        return self._data[3]

    # Factory methods

    @classmethod
    def from_vector(cls, v: jax.Array, scalar_first: bool = True) -> Quaternion:
        """Create from a 4-element vector.

        Args:
            v (jax.Array): Array-like of shape ``(4,)``.
            scalar_first (bool): If ``True``, ``v = [w, x, y, z]``.
                If ``False``, ``v = [x, y, z, w]``.

        Returns:
            Quaternion: New normalized quaternion.
        """
        if scalar_first:
            return cls(v[0], v[1], v[2], v[3])
        else:
            return cls(v[3], v[0], v[1], v[2])

    def to_vector(self, scalar_first: bool = True) -> jax.Array:
        """Return the quaternion as a 4-element vector.

        Args:
            scalar_first (bool): If ``True``, return ``[w, x, y, z]``.
                If ``False``, return ``[x, y, z, w]``.

        Returns:
            jnp.ndarray: Array of shape ``(4,)``.
        """
        if scalar_first:
            return self._data
        else:
            return jnp.array([self._data[1], self._data[2], self._data[3], self._data[0]])

    # Methods

    def normalize(self) -> Quaternion:
        """Return a new normalized quaternion.

        Returns:
            Quaternion: Unit quaternion.
        """
        return Quaternion._from_internal(self._data / jnp.linalg.norm(self._data))

    def norm(self) -> jax.Array:
        """Return the Euclidean norm.

        Returns:
            jax.Array: Scalar norm.
        """
        return jnp.linalg.norm(self._data)

    def conjugate(self) -> Quaternion:
        """Return the conjugate (adjoint) quaternion.

        For a unit quaternion, the conjugate equals the inverse.

        Returns:
            Quaternion: Conjugate quaternion ``[w, -x, -y, -z]``.
        """
        return Quaternion(self._data[0], -self._data[1], -self._data[2], -self._data[3])

    def inverse(self) -> Quaternion:
        """Return the multiplicative inverse.

        For a unit quaternion this is the conjugate; for non-unit
        quaternions the conjugate is divided by the squared norm.

        Returns:
            Quaternion: Inverse quaternion.
        """
        n = self.norm()
        return Quaternion._from_internal(
            jnp.array([self._data[0], -self._data[1], -self._data[2], -self._data[3]]) / n
        )

    def slerp(self, other: Quaternion, t: float) -> Quaternion:
        """Spherical linear interpolation.

        Args:
            other (Quaternion): Target quaternion.
            t (float): Interpolation parameter in ``[0, 1]``.  ``t=0`` returns
                ``self``, ``t=1`` returns ``other``.

        Returns:
            Quaternion: Interpolated quaternion.
        """
        from astrojax.attitude_representations.conversions import quaternion_slerp

        q = quaternion_slerp(self._data, other._data, t)
        return Quaternion._from_internal(q)

    # Operators

    def __add__(self, other: Quaternion) -> Quaternion:
        if not isinstance(other, Quaternion):
            return NotImplemented
        q = self._data + other._data
        return Quaternion.from_vector(q, scalar_first=True)

    def __sub__(self, other: Quaternion) -> Quaternion:
        if not isinstance(other, Quaternion):
            return NotImplemented
        q = self._data - other._data
        return Quaternion.from_vector(q, scalar_first=True)

    def __mul__(self, other: Quaternion) -> Quaternion:
        """Hamilton product (quaternion * quaternion)."""
        if not isinstance(other, Quaternion):
            return NotImplemented
        from astrojax.attitude_representations.conversions import quaternion_multiply

        q = quaternion_multiply(self._data, other._data)
        return Quaternion._from_internal(q)

    def __neg__(self) -> Quaternion:
        return Quaternion(self._data[0], -self._data[1], -self._data[2], -self._data[3])

    def __getitem__(self, idx: int) -> jax.Array:
        return self._data[idx]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quaternion):
            return NotImplemented
        eps = get_attitude_epsilon()
        # Normalize before comparing to handle non-unit quaternions
        d1 = self._data / jnp.linalg.norm(self._data)
        d2 = other._data / jnp.linalg.norm(other._data)
        return jnp.all(jnp.abs(d1 - d2) < eps)

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Quaternion):
            return NotImplemented
        return not self.__eq__(other)

    # Conversion methods

    def to_rotation_matrix(self) -> RotationMatrix:
        """Convert to ``RotationMatrix``.

        Returns:
            RotationMatrix: Equivalent rotation matrix.
        """
        from astrojax.attitude_representations.conversions import quaternion_to_rotation_matrix
        from astrojax.attitude_representations.rotation_matrix import RotationMatrix

        R = quaternion_to_rotation_matrix(self._data)
        return RotationMatrix._from_internal(R)

    def to_euler_axis(self) -> EulerAxis:
        """Convert to ``EulerAxis``.

        Returns:
            EulerAxis: Equivalent euler axis.
        """
        from astrojax.attitude_representations.conversions import quaternion_to_euler_axis
        from astrojax.attitude_representations.euler_axis import EulerAxis

        axis, angle = quaternion_to_euler_axis(self._data)
        return EulerAxis._from_internal(axis, angle)

    def to_euler_angle(self, order: EulerAngleOrder) -> EulerAngle:
        """Convert to ``EulerAngle`` with specified rotation sequence.

        Goes through the rotation matrix representation (matching Rust).

        Args:
            order (EulerAngleOrder): ``EulerAngleOrder`` specifying the rotation sequence.

        Returns:
            EulerAngle: Equivalent euler angle.
        """
        from astrojax.attitude_representations.conversions import (
            quaternion_to_rotation_matrix,
            rotation_matrix_to_euler_angle,
        )
        from astrojax.attitude_representations.euler_angle import EulerAngle

        R = quaternion_to_rotation_matrix(self._data)
        angles = rotation_matrix_to_euler_angle(jnp.int32(order.value), R)
        return EulerAngle._from_internal(order, angles[0], angles[1], angles[2])

    def to_quaternion(self) -> Quaternion:
        """Return a copy.

        Returns:
            Quaternion: New quaternion with same data.
        """
        return Quaternion._from_internal(self._data)

    @classmethod
    def from_quaternion(cls, q: Quaternion) -> Quaternion:
        """Create from another ``Quaternion``.

        Args:
            q (Quaternion): Source quaternion.

        Returns:
            Quaternion: Copy.
        """
        return q.to_quaternion()

    @classmethod
    def from_euler_axis(cls, ea: EulerAxis) -> Quaternion:
        """Create from an ``EulerAxis``.

        Args:
            ea (EulerAxis): Source euler axis.

        Returns:
            Quaternion: Equivalent quaternion.
        """
        from astrojax.attitude_representations.conversions import euler_axis_to_quaternion

        q = euler_axis_to_quaternion(ea.axis, ea.angle)
        return Quaternion._from_internal(q)

    @classmethod
    def from_euler_angle(cls, e: EulerAngle) -> Quaternion:
        """Create from an ``EulerAngle``.

        Args:
            e (EulerAngle): Source euler angle.

        Returns:
            Quaternion: Equivalent quaternion.
        """
        return e.to_quaternion()

    @classmethod
    def from_rotation_matrix(cls, r: RotationMatrix) -> Quaternion:
        """Create from a ``RotationMatrix``.

        Args:
            r (RotationMatrix): Source rotation matrix.

        Returns:
            Quaternion: Equivalent quaternion.
        """
        from astrojax.attitude_representations.conversions import rotation_matrix_to_quaternion

        q = rotation_matrix_to_quaternion(r.to_matrix())
        return Quaternion._from_internal(q)

    # String representations

    def __str__(self) -> str:
        return (
            f"Quaternion(w={float(self._data[0]):.6f}, "
            f"x={float(self._data[1]):.6f}, "
            f"y={float(self._data[2]):.6f}, "
            f"z={float(self._data[3]):.6f})"
        )

    def __repr__(self) -> str:
        return (
            f"Quaternion(w={float(self._data[0])}, "
            f"x={float(self._data[1])}, "
            f"y={float(self._data[2])}, "
            f"z={float(self._data[3])})"
        )


# Register as JAX pytree
jax.tree_util.register_pytree_node(
    Quaternion,
    lambda q: ((q._data,), None),
    lambda _, children: Quaternion._from_internal(children[0]),
)
