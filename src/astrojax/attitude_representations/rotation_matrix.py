"""Rotation matrix (DCM) attitude representation.

Provides the ``RotationMatrix`` class representing a rotation as a
3x3 orthogonal matrix with determinant +1 (SO(3)).

The constructor validates that the matrix is a proper rotation matrix
(orthogonal with det +1). The ``_from_internal`` classmethod bypasses
validation for use in pytree unflatten and conversion outputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from astrojax.attitude_representations._tolerance import get_attitude_epsilon
from astrojax.attitude_representations.rotation_matrices import Rx as _Rx
from astrojax.attitude_representations.rotation_matrices import Ry as _Ry
from astrojax.attitude_representations.rotation_matrices import Rz as _Rz
from astrojax.config import get_dtype

if TYPE_CHECKING:
    from astrojax.attitude_representations.euler_angle import EulerAngle, EulerAngleOrder
    from astrojax.attitude_representations.euler_axis import EulerAxis
    from astrojax.attitude_representations.quaternion import Quaternion


def _is_so3(matrix: jax.Array, tol: float = 1e-6) -> bool:
    """Check if a matrix is in SO(3).

    Tests orthogonality (R^T R ≈ I) and positive determinant (det ≈ +1).

    Args:
        matrix (jax.Array): Array of shape ``(3, 3)``.
        tol (float): Tolerance for the checks.

    Returns:
        bool: ``True`` if the matrix is a proper rotation matrix.
    """
    rtrt = matrix.T @ matrix
    orth_err = jnp.max(jnp.abs(rtrt - jnp.eye(3)))
    det = jnp.linalg.det(matrix)
    return bool(orth_err < tol and det > 0.0)


class RotationMatrix:
    """3x3 rotation matrix (Direction Cosine Matrix).

    Internal storage is a shape ``(3, 3)`` JAX array.  The constructor
    validates SO(3) membership.  Use ``_from_internal`` or ``from_matrix``
    with ``validate=False`` to skip validation when the matrix is already
    known to be valid.

    This class is registered as a JAX pytree with the data matrix as
    the sole leaf.

    Args:
        r11 (float): Element (0, 0).
        r12 (float): Element (0, 1).
        r13 (float): Element (0, 2).
        r21 (float): Element (1, 0).
        r22 (float): Element (1, 1).
        r23 (float): Element (1, 2).
        r31 (float): Element (2, 0).
        r32 (float): Element (2, 1).
        r33 (float): Element (2, 2).
    """

    __slots__ = ("_data",)

    def __init__(
        self,
        r11: float,
        r12: float,
        r13: float,
        r21: float,
        r22: float,
        r23: float,
        r31: float,
        r32: float,
        r33: float,
    ) -> None:
        _float = get_dtype()
        data = jnp.array(
            [
                [_float(r11), _float(r12), _float(r13)],
                [_float(r21), _float(r22), _float(r23)],
                [_float(r31), _float(r32), _float(r33)],
            ]
        )
        if not _is_so3(data):
            raise ValueError(
                f"Matrix is not a proper rotation matrix. det={float(jnp.linalg.det(data)):.6f}"
            )
        self._data = data

    @classmethod
    def _from_internal(cls, data: jax.Array) -> RotationMatrix:
        """Create from a raw JAX array without validation.

        Used by pytree unflatten and conversion outputs.

        Args:
            data (jax.Array): Array of shape ``(3, 3)``.

        Returns:
            RotationMatrix: New instance.
        """
        obj = object.__new__(cls)
        obj._data = data
        return obj

    # Properties

    @property
    def r11(self) -> jax.Array:
        """Element (0, 0)."""
        return self._data[0, 0]

    @property
    def r12(self) -> jax.Array:
        """Element (0, 1)."""
        return self._data[0, 1]

    @property
    def r13(self) -> jax.Array:
        """Element (0, 2)."""
        return self._data[0, 2]

    @property
    def r21(self) -> jax.Array:
        """Element (1, 0)."""
        return self._data[1, 0]

    @property
    def r22(self) -> jax.Array:
        """Element (1, 1)."""
        return self._data[1, 1]

    @property
    def r23(self) -> jax.Array:
        """Element (1, 2)."""
        return self._data[1, 2]

    @property
    def r31(self) -> jax.Array:
        """Element (2, 0)."""
        return self._data[2, 0]

    @property
    def r32(self) -> jax.Array:
        """Element (2, 1)."""
        return self._data[2, 1]

    @property
    def r33(self) -> jax.Array:
        """Element (2, 2)."""
        return self._data[2, 2]

    # Factory methods

    @classmethod
    def from_matrix(cls, matrix: jax.Array, validate: bool = True) -> RotationMatrix:
        """Create from a 3x3 array.

        Args:
            matrix (jax.Array): Array-like of shape ``(3, 3)``.
            validate (bool): If ``True``, check SO(3) membership. Default: ``True``.

        Returns:
            RotationMatrix: New instance.

        Raises:
            ValueError: If ``validate=True`` and matrix is not SO(3).
        """
        data = jnp.asarray(matrix, dtype=get_dtype())
        if validate and not _is_so3(data):
            raise ValueError(
                f"Matrix is not a proper rotation matrix. det={float(jnp.linalg.det(data)):.6f}"
            )
        return cls._from_internal(data)

    def to_matrix(self) -> jax.Array:
        """Return the underlying 3x3 array.

        Returns:
            jnp.ndarray: Array of shape ``(3, 3)``.
        """
        return self._data

    @classmethod
    def rotation_x(cls, angle: float, use_degrees: bool = False) -> RotationMatrix:
        """Rotation about the x-axis.

        Delegates to the existing ``Rx`` function.

        Args:
            angle (float): Rotation angle.
            use_degrees (bool): If ``True``, interpret as degrees.

        Returns:
            RotationMatrix: Rotation about x.
        """
        return cls._from_internal(_Rx(angle, use_degrees))

    @classmethod
    def rotation_y(cls, angle: float, use_degrees: bool = False) -> RotationMatrix:
        """Rotation about the y-axis.

        Delegates to the existing ``Ry`` function.

        Args:
            angle (float): Rotation angle.
            use_degrees (bool): If ``True``, interpret as degrees.

        Returns:
            RotationMatrix: Rotation about y.
        """
        return cls._from_internal(_Ry(angle, use_degrees))

    @classmethod
    def rotation_z(cls, angle: float, use_degrees: bool = False) -> RotationMatrix:
        """Rotation about the z-axis.

        Delegates to the existing ``Rz`` function.

        Args:
            angle (float): Rotation angle.
            use_degrees (bool): If ``True``, interpret as degrees.

        Returns:
            RotationMatrix: Rotation about z.
        """
        return cls._from_internal(_Rz(angle, use_degrees))

    # Operators

    def __mul__(self, other: RotationMatrix | jax.Array) -> RotationMatrix | jax.Array:
        """Matrix-matrix or matrix-vector multiplication.

        Args:
            other (RotationMatrix | jax.Array): ``RotationMatrix`` or 3-element array.

        Returns:
            RotationMatrix | jax.Array: Result of multiplication.
        """
        if isinstance(other, RotationMatrix):
            return RotationMatrix._from_internal(self._data @ other._data)
        # Assume vector
        v = jnp.asarray(other)
        if v.shape == (3,):
            return self._data @ v
        return NotImplemented

    def __getitem__(self, idx: int | tuple[int, int]) -> jax.Array:
        return self._data[idx]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RotationMatrix):
            return NotImplemented
        eps = get_attitude_epsilon()
        return jnp.all(jnp.abs(self._data - other._data) < eps)

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, RotationMatrix):
            return NotImplemented
        return not self.__eq__(other)

    # Conversion methods

    def to_quaternion(self) -> Quaternion:
        """Convert to ``Quaternion``.

        Returns:
            Quaternion: Equivalent quaternion.
        """
        from astrojax.attitude_representations.conversions import rotation_matrix_to_quaternion
        from astrojax.attitude_representations.quaternion import Quaternion

        q = rotation_matrix_to_quaternion(self._data)
        return Quaternion._from_internal(q)

    def to_euler_axis(self) -> EulerAxis:
        """Convert to ``EulerAxis``.

        Routes through quaternion.

        Returns:
            EulerAxis: Equivalent euler axis.
        """
        return self.to_quaternion().to_euler_axis()

    def to_euler_angle(self, order: EulerAngleOrder) -> EulerAngle:
        """Convert to ``EulerAngle``.

        Args:
            order (EulerAngleOrder): ``EulerAngleOrder`` specifying the rotation sequence.

        Returns:
            EulerAngle: Equivalent euler angle.
        """
        from astrojax.attitude_representations.conversions import rotation_matrix_to_euler_angle
        from astrojax.attitude_representations.euler_angle import EulerAngle

        angles = rotation_matrix_to_euler_angle(jnp.int32(order.value), self._data)
        return EulerAngle._from_internal(order, angles[0], angles[1], angles[2])

    def to_rotation_matrix(self) -> RotationMatrix:
        """Return a copy.

        Returns:
            RotationMatrix: New instance with same data.
        """
        return RotationMatrix._from_internal(self._data)

    @classmethod
    def from_quaternion(cls, q: Quaternion) -> RotationMatrix:
        """Create from a ``Quaternion``.

        Args:
            q (Quaternion): Source quaternion.

        Returns:
            RotationMatrix: Equivalent rotation matrix.
        """
        return q.to_rotation_matrix()

    @classmethod
    def from_euler_axis(cls, ea: EulerAxis) -> RotationMatrix:
        """Create from an ``EulerAxis``.

        Args:
            ea (EulerAxis): Source euler axis.

        Returns:
            RotationMatrix: Equivalent rotation matrix.
        """
        return ea.to_rotation_matrix()

    @classmethod
    def from_euler_angle(cls, e: EulerAngle) -> RotationMatrix:
        """Create from an ``EulerAngle``.

        Args:
            e (EulerAngle): Source euler angle.

        Returns:
            RotationMatrix: Equivalent rotation matrix.
        """
        return e.to_rotation_matrix()

    @classmethod
    def from_rotation_matrix(cls, r: RotationMatrix) -> RotationMatrix:
        """Create from another ``RotationMatrix``.

        Args:
            r (RotationMatrix): Source rotation matrix.

        Returns:
            RotationMatrix: Copy.
        """
        return r.to_rotation_matrix()

    # String representations

    def __str__(self) -> str:
        d = self._data
        return (
            f"RotationMatrix(\n"
            f"  [{float(d[0, 0]):10.6f} {float(d[0, 1]):10.6f} {float(d[0, 2]):10.6f}]\n"
            f"  [{float(d[1, 0]):10.6f} {float(d[1, 1]):10.6f} {float(d[1, 2]):10.6f}]\n"
            f"  [{float(d[2, 0]):10.6f} {float(d[2, 1]):10.6f} {float(d[2, 2]):10.6f}])"
        )

    def __repr__(self) -> str:
        return self.__str__()


# Register as JAX pytree
jax.tree_util.register_pytree_node(
    RotationMatrix,
    lambda r: ((r._data,), None),
    lambda _, children: RotationMatrix._from_internal(children[0]),
)
