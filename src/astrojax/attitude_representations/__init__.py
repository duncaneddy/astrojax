"""Attitude representations for 3D rotations.

Provides four interconvertible attitude representation types:

- :class:`Quaternion` -- unit quaternion (scalar-first ``[w, x, y, z]``)
- :class:`RotationMatrix` -- 3x3 direction cosine matrix (SO(3))
- :class:`EulerAngle` -- three successive rotations with 12 possible orders
- :class:`EulerAxis` -- rotation axis + angle (axis-angle)

Also re-exports the elementary rotation functions :func:`Rx`, :func:`Ry`,
:func:`Rz` and the :class:`EulerAngleOrder` enum.
"""

from .rotation_matrices import (
    Rx,
    Ry,
    Rz,
)

from .quaternion import Quaternion
from .rotation_matrix import RotationMatrix
from .euler_angle import EulerAngle, EulerAngleOrder
from .euler_axis import EulerAxis

__all__ = [
    # Elementary rotations
    "Rx",
    "Ry",
    "Rz",
    # Attitude representations
    "Quaternion",
    "RotationMatrix",
    "EulerAngle",
    "EulerAngleOrder",
    "EulerAxis",
]
