"""Euler angle attitude representation.

Provides the ``EulerAngleOrder`` enum defining the 12 standard Euler angle
rotation sequences, and the ``EulerAngle`` class representing an attitude
as three successive rotations.

The ``EulerAngleOrder`` values are contiguous integers 0--11, suitable for
use as branch indices in ``jax.lax.switch`` for JIT-compatible dispatch.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from astrojax.attitude_representations._tolerance import get_attitude_epsilon
from astrojax.config import get_dtype
from astrojax.utils import to_radians

if TYPE_CHECKING:
    from astrojax.attitude_representations.euler_axis import EulerAxis
    from astrojax.attitude_representations.quaternion import Quaternion
    from astrojax.attitude_representations.rotation_matrix import RotationMatrix


class EulerAngleOrder(enum.IntEnum):
    """The 12 standard Euler angle rotation sequences.

    Each member specifies the axes for three successive rotations.  For
    example, ``XYZ`` means rotate first about X, then Y, then Z.

    Values are contiguous integers 0--11 for use as ``jax.lax.switch``
    branch indices.

    Attributes:
        XYX: X-Y-X symmetric sequence (index 0).
        XYZ: X-Y-Z Tait-Bryan sequence, also known as Roll-Pitch-Yaw (index 1).
        XZX: X-Z-X symmetric sequence (index 2).
        XZY: X-Z-Y Tait-Bryan sequence (index 3).
        YXY: Y-X-Y symmetric sequence (index 4).
        YXZ: Y-X-Z Tait-Bryan sequence (index 5).
        YZX: Y-Z-X Tait-Bryan sequence (index 6).
        YZY: Y-Z-Y symmetric sequence (index 7).
        ZXY: Z-X-Y Tait-Bryan sequence (index 8).
        ZXZ: Z-X-Z symmetric sequence (index 9).
        ZYX: Z-Y-X Tait-Bryan sequence, also known as Yaw-Pitch-Roll (index 10).
        ZYZ: Z-Y-Z symmetric sequence (index 11).
    """

    XYX = 0
    XYZ = 1
    XZX = 2
    XZY = 3
    YXY = 4
    YXZ = 5
    YZX = 6
    YZY = 7
    ZXY = 8
    ZXZ = 9
    ZYX = 10
    ZYZ = 11


class EulerAngle:
    """Attitude represented as three successive rotations about specified axes.

    Internal storage is always in radians.  The ``use_degrees`` parameter on
    the constructor converts degree inputs to radians on construction.

    This class is registered as a JAX pytree.  The three angle scalars are
    leaves; the ``order`` (an ``EulerAngleOrder``) is auxiliary data.

    Args:
        order (EulerAngleOrder): Rotation sequence (e.g. ``EulerAngleOrder.XYZ``).
        phi (float): First rotation angle.
        theta (float): Second rotation angle.
        psi (float): Third rotation angle.
        use_degrees (bool): If ``True``, interpret angles as degrees. Default: ``False``.
    """

    __slots__ = ('_order', '_phi', '_theta', '_psi')

    def __init__(
        self,
        order: EulerAngleOrder,
        phi: float,
        theta: float,
        psi: float,
        use_degrees: bool = False,
    ) -> None:
        _float = get_dtype()
        self._order = EulerAngleOrder(order)
        self._phi = _float(to_radians(phi, use_degrees))
        self._theta = _float(to_radians(theta, use_degrees))
        self._psi = _float(to_radians(psi, use_degrees))

    @classmethod
    def _from_internal(cls, order: EulerAngleOrder, phi: jax.Array, theta: jax.Array, psi: jax.Array) -> EulerAngle:
        """Create from raw JAX arrays without conversion.

        Used by pytree unflatten and conversion outputs.

        Args:
            order (EulerAngleOrder): Rotation sequence.
            phi (jax.Array): First angle in radians.
            theta (jax.Array): Second angle in radians.
            psi (jax.Array): Third angle in radians.

        Returns:
            EulerAngle: New instance.
        """
        obj = object.__new__(cls)
        obj._order = EulerAngleOrder(order)
        obj._phi = phi
        obj._theta = theta
        obj._psi = psi
        return obj

    # Properties

    @property
    def order(self) -> EulerAngleOrder:
        """Rotation sequence."""
        return self._order

    @property
    def phi(self) -> jax.Array:
        """First rotation angle in radians."""
        return self._phi

    @property
    def theta(self) -> jax.Array:
        """Second rotation angle in radians."""
        return self._theta

    @property
    def psi(self) -> jax.Array:
        """Third rotation angle in radians."""
        return self._psi

    # Factory methods

    @classmethod
    def from_vector(cls, vec: jax.Array, order: EulerAngleOrder, use_degrees: bool = False) -> EulerAngle:
        """Create from a 3-element vector [phi, theta, psi].

        Args:
            vec (jax.Array): Array-like of shape (3,).
            order (EulerAngleOrder): Rotation sequence.
            use_degrees (bool): If ``True``, interpret as degrees.

        Returns:
            EulerAngle: New instance.
        """
        return cls(order, vec[0], vec[1], vec[2], use_degrees=use_degrees)

    # Conversion methods

    def to_quaternion(self) -> Quaternion:
        """Convert to ``Quaternion``.

        Returns:
            Quaternion: Equivalent quaternion.
        """
        from astrojax.attitude_representations.conversions import euler_angle_to_quaternion
        from astrojax.attitude_representations.quaternion import Quaternion

        q = euler_angle_to_quaternion(
            jnp.int32(self._order.value), self._phi, self._theta, self._psi
        )
        return Quaternion._from_internal(q)

    def to_rotation_matrix(self) -> RotationMatrix:
        """Convert to ``RotationMatrix``.

        Returns:
            RotationMatrix: Equivalent rotation matrix.
        """
        return self.to_quaternion().to_rotation_matrix()

    def to_euler_axis(self) -> EulerAxis:
        """Convert to ``EulerAxis``.

        Returns:
            EulerAxis: Equivalent euler axis.
        """
        return self.to_quaternion().to_euler_axis()

    def to_euler_angle(self, order: EulerAngleOrder) -> EulerAngle:
        """Convert to ``EulerAngle`` with a (possibly different) order.

        Args:
            order (EulerAngleOrder): Target rotation sequence.

        Returns:
            EulerAngle: Equivalent euler angle in the target order.
        """
        return self.to_quaternion().to_euler_angle(order)

    @classmethod
    def from_quaternion(cls, q: Quaternion, order: EulerAngleOrder) -> EulerAngle:
        """Create from a ``Quaternion``.

        Args:
            q (Quaternion): Source quaternion.
            order (EulerAngleOrder): Target rotation sequence.

        Returns:
            EulerAngle: Equivalent euler angle.
        """
        return q.to_euler_angle(order)

    @classmethod
    def from_rotation_matrix(cls, r: RotationMatrix, order: EulerAngleOrder) -> EulerAngle:
        """Create from a ``RotationMatrix``.

        Args:
            r (RotationMatrix): Source rotation matrix.
            order (EulerAngleOrder): Target rotation sequence.

        Returns:
            EulerAngle: Equivalent euler angle.
        """
        return r.to_euler_angle(order)

    @classmethod
    def from_euler_axis(cls, ea: EulerAxis, order: EulerAngleOrder) -> EulerAngle:
        """Create from an ``EulerAxis``.

        Args:
            ea (EulerAxis): Source euler axis.
            order (EulerAngleOrder): Target rotation sequence.

        Returns:
            EulerAngle: Equivalent euler angle.
        """
        return ea.to_euler_angle(order)

    @classmethod
    def from_euler_angle(cls, e: EulerAngle, order: EulerAngleOrder) -> EulerAngle:
        """Create from another ``EulerAngle`` with a different order.

        Args:
            e (EulerAngle): Source euler angle.
            order (EulerAngleOrder): Target rotation sequence.

        Returns:
            EulerAngle: Equivalent euler angle in the target order.
        """
        return e.to_euler_angle(order)

    # Comparison

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EulerAngle):
            return NotImplemented
        eps = get_attitude_epsilon()
        return (
            self._order == other._order
            and jnp.abs(self._phi - other._phi) < eps
            and jnp.abs(self._theta - other._theta) < eps
            and jnp.abs(self._psi - other._psi) < eps
        )

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, EulerAngle):
            return NotImplemented
        return not self.__eq__(other)

    # String representations

    def __str__(self) -> str:
        return (
            f"EulerAngle(order={self._order.name}, "
            f"phi={float(self._phi):.6f}, "
            f"theta={float(self._theta):.6f}, "
            f"psi={float(self._psi):.6f})"
        )

    def __repr__(self) -> str:
        return (
            f"EulerAngle(order={self._order.name}, "
            f"phi={float(self._phi)}, "
            f"theta={float(self._theta)}, "
            f"psi={float(self._psi)})"
        )


# Register as JAX pytree: angles are leaves, order is auxiliary data
jax.tree_util.register_pytree_node(
    EulerAngle,
    lambda e: ((e._phi, e._theta, e._psi), e._order),
    lambda order, children: EulerAngle._from_internal(order, *children),
)
