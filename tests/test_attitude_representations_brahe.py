"""Cross-validation tests comparing astrojax attitude outputs against brahe 1.0+.

These tests ensure that astrojax is a faithful reimplementation of the
core attitude algorithms in brahe.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from astrojax.config import set_dtype  # noqa: E402
set_dtype(jnp.float64)

import brahe as bh  # noqa: E402

from astrojax import (  # noqa: E402
    Quaternion,
    RotationMatrix,
    EulerAngle,
    EulerAngleOrder,
    EulerAxis,
)

ATOL = 1e-12


@pytest.fixture(autouse=True)
def _ensure_float64():
    """Ensure float64 is active for brahe comparison tests."""
    set_dtype(jnp.float64)
    yield
    set_dtype(jnp.float64)

# Map astrojax EulerAngleOrder to brahe EulerAngleOrder
_ORDER_MAP = {
    EulerAngleOrder.XYX: bh.EulerAngleOrder.XYX,
    EulerAngleOrder.XYZ: bh.EulerAngleOrder.XYZ,
    EulerAngleOrder.XZX: bh.EulerAngleOrder.XZX,
    EulerAngleOrder.XZY: bh.EulerAngleOrder.XZY,
    EulerAngleOrder.YXY: bh.EulerAngleOrder.YXY,
    EulerAngleOrder.YXZ: bh.EulerAngleOrder.YXZ,
    EulerAngleOrder.YZX: bh.EulerAngleOrder.YZX,
    EulerAngleOrder.YZY: bh.EulerAngleOrder.YZY,
    EulerAngleOrder.ZXY: bh.EulerAngleOrder.ZXY,
    EulerAngleOrder.ZXZ: bh.EulerAngleOrder.ZXZ,
    EulerAngleOrder.ZYX: bh.EulerAngleOrder.ZYX,
    EulerAngleOrder.ZYZ: bh.EulerAngleOrder.ZYZ,
}


def _brahe_q_to_array(q):
    """Extract brahe quaternion as [w, x, y, z] numpy array."""
    v = q.to_vector(True)
    return np.array([v[0], v[1], v[2], v[3]])


def _brahe_r_to_array(r):
    """Extract brahe rotation matrix as 3x3 numpy array."""
    m = r.to_matrix()
    return np.array(m).reshape(3, 3)


# ===========================================================================
# Quaternion conversions vs brahe
# ===========================================================================

class TestQuaternionVsBrahe:

    @pytest.mark.parametrize("s,v1,v2,v3", [
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0),
        (0.675, 0.42, 0.5, 0.71),
        (0.5, 0.5, 0.5, 0.5),
    ])
    def test_quaternion_to_rotation_matrix(self, s, v1, v2, v3):
        aj_q = Quaternion(s, v1, v2, v3)
        bh_q = bh.Quaternion(s, v1, v2, v3)

        aj_r = aj_q.to_rotation_matrix().to_matrix()
        bh_r = _brahe_r_to_array(bh_q.to_rotation_matrix())

        np.testing.assert_allclose(np.array(aj_r), bh_r, atol=ATOL)

    @pytest.mark.parametrize("s,v1,v2,v3", [
        (1.0, 0.0, 0.0, 0.0),
        (0.675, 0.42, 0.5, 0.71),
    ])
    def test_quaternion_to_euler_axis(self, s, v1, v2, v3):
        aj_q = Quaternion(s, v1, v2, v3)
        bh_q = bh.Quaternion(s, v1, v2, v3)

        aj_ea = aj_q.to_euler_axis()
        bh_ea = bh_q.to_euler_axis()

        np.testing.assert_allclose(
            np.array(aj_ea.axis), np.array(bh_ea.axis), atol=ATOL
        )
        np.testing.assert_allclose(
            float(aj_ea.angle), float(bh_ea.angle), atol=ATOL
        )

    @pytest.mark.parametrize("order", list(EulerAngleOrder))
    def test_quaternion_to_euler_angle(self, order):
        s, v1, v2, v3 = 0.675, 0.42, 0.5, 0.71
        aj_q = Quaternion(s, v1, v2, v3)
        bh_q = bh.Quaternion(s, v1, v2, v3)

        bh_order = _ORDER_MAP[order]
        aj_e = aj_q.to_euler_angle(order)
        bh_e = bh_q.to_euler_angle(bh_order)

        np.testing.assert_allclose(float(aj_e.phi), float(bh_e.phi), atol=ATOL)
        np.testing.assert_allclose(float(aj_e.theta), float(bh_e.theta), atol=ATOL)
        np.testing.assert_allclose(float(aj_e.psi), float(bh_e.psi), atol=ATOL)

    def test_quaternion_multiply(self):
        aj_q1 = Quaternion(1.0, 1.0, 0.0, 0.0)
        aj_q2 = Quaternion(1.0, 0.0, 1.0, 0.0)
        bh_q1 = bh.Quaternion(1.0, 1.0, 0.0, 0.0)
        bh_q2 = bh.Quaternion(1.0, 0.0, 1.0, 0.0)

        aj_result = aj_q1 * aj_q2
        bh_result = bh_q1 * bh_q2

        aj_v = np.array(aj_result.to_vector())
        bh_v = _brahe_q_to_array(bh_result)

        np.testing.assert_allclose(aj_v, bh_v, atol=ATOL)

    def test_quaternion_conjugate(self):
        aj_q = Quaternion(0.675, 0.42, 0.5, 0.71)
        bh_q = bh.Quaternion(0.675, 0.42, 0.5, 0.71)

        aj_c = aj_q.conjugate()
        bh_c = bh_q.conjugate()

        np.testing.assert_allclose(
            np.array(aj_c.to_vector()), _brahe_q_to_array(bh_c), atol=ATOL
        )

    def test_quaternion_inverse(self):
        aj_q = Quaternion(1.0, 2.0, 3.0, 4.0)
        bh_q = bh.Quaternion(1.0, 2.0, 3.0, 4.0)

        aj_inv = aj_q.inverse()
        bh_inv = bh_q.inverse()

        np.testing.assert_allclose(
            np.array(aj_inv.to_vector()), _brahe_q_to_array(bh_inv), atol=ATOL
        )


# ===========================================================================
# Euler Angle conversions vs brahe
# ===========================================================================

class TestEulerAngleVsBrahe:

    @pytest.mark.parametrize("order", list(EulerAngleOrder))
    def test_euler_angle_to_quaternion(self, order):
        bh_order = _ORDER_MAP[order]

        aj_e = EulerAngle(order, 30.0, 45.0, 60.0, use_degrees=True)
        bh_e = bh.EulerAngle(bh_order, 30.0, 45.0, 60.0, bh.AngleFormat.DEGREES)

        aj_q = aj_e.to_quaternion()
        bh_q = bh_e.to_quaternion()

        np.testing.assert_allclose(
            np.array(aj_q.to_vector()), _brahe_q_to_array(bh_q), atol=ATOL
        )

    @pytest.mark.parametrize("order", list(EulerAngleOrder))
    def test_euler_angle_to_rotation_matrix(self, order):
        bh_order = _ORDER_MAP[order]

        aj_e = EulerAngle(order, 30.0, 45.0, 60.0, use_degrees=True)
        bh_e = bh.EulerAngle(bh_order, 30.0, 45.0, 60.0, bh.AngleFormat.DEGREES)

        aj_r = aj_e.to_rotation_matrix().to_matrix()
        bh_r = _brahe_r_to_array(bh_e.to_rotation_matrix())

        np.testing.assert_allclose(np.array(aj_r), bh_r, atol=ATOL)


# ===========================================================================
# Euler Axis conversions vs brahe
# ===========================================================================

class TestEulerAxisVsBrahe:

    @pytest.mark.parametrize("axis,angle_deg", [
        ([1.0, 0.0, 0.0], 45.0),
        ([0.0, 1.0, 0.0], 45.0),
        ([0.0, 0.0, 1.0], 45.0),
        ([1.0, 0.0, 0.0], 90.0),
        ([1.0, 1.0, 1.0], 60.0),
    ])
    def test_euler_axis_to_quaternion(self, axis, angle_deg):
        aj_ea = EulerAxis(jnp.array(axis), angle_deg, use_degrees=True)
        bh_ea = bh.EulerAxis(axis, angle_deg, bh.AngleFormat.DEGREES)

        aj_q = aj_ea.to_quaternion()
        bh_q = bh_ea.to_quaternion()

        np.testing.assert_allclose(
            np.array(aj_q.to_vector()), _brahe_q_to_array(bh_q), atol=ATOL
        )

    @pytest.mark.parametrize("axis,angle_deg", [
        ([1.0, 0.0, 0.0], 45.0),
        ([0.0, 1.0, 0.0], 45.0),
        ([0.0, 0.0, 1.0], 45.0),
    ])
    def test_euler_axis_to_rotation_matrix(self, axis, angle_deg):
        aj_ea = EulerAxis(jnp.array(axis), angle_deg, use_degrees=True)
        bh_ea = bh.EulerAxis(axis, angle_deg, bh.AngleFormat.DEGREES)

        aj_r = aj_ea.to_rotation_matrix().to_matrix()
        bh_r = _brahe_r_to_array(bh_ea.to_rotation_matrix())

        np.testing.assert_allclose(np.array(aj_r), bh_r, atol=ATOL)


# ===========================================================================
# Rotation Matrix conversions vs brahe
# ===========================================================================

class TestRotationMatrixVsBrahe:

    def test_rotation_matrix_to_quaternion(self):
        s2 = math.sqrt(2.0) / 2.0
        aj_r = RotationMatrix(
            1.0, 0.0, 0.0,
            0.0, s2, s2,
            0.0, -s2, s2,
        )
        bh_r = bh.RotationMatrix(
            1.0, 0.0, 0.0,
            0.0, s2, s2,
            0.0, -s2, s2,
        )

        aj_q = aj_r.to_quaternion()
        bh_q = bh_r.to_quaternion()

        np.testing.assert_allclose(
            np.array(aj_q.to_vector()), _brahe_q_to_array(bh_q), atol=ATOL
        )

    @pytest.mark.parametrize("order", list(EulerAngleOrder))
    def test_rotation_matrix_to_euler_angle(self, order):
        bh_order = _ORDER_MAP[order]

        # Build a non-trivial rotation matrix: Rx(30) * Ry(45) * Rx(60)
        aj_r = (
            RotationMatrix.rotation_x(30.0, use_degrees=True)
            * RotationMatrix.rotation_y(45.0, use_degrees=True)
            * RotationMatrix.rotation_x(60.0, use_degrees=True)
        )

        bh_rx30 = bh.RotationMatrix.Rx(30.0, bh.AngleFormat.DEGREES)
        bh_ry45 = bh.RotationMatrix.Ry(45.0, bh.AngleFormat.DEGREES)
        bh_rx60 = bh.RotationMatrix.Rx(60.0, bh.AngleFormat.DEGREES)
        bh_r = bh_rx30 * bh_ry45 * bh_rx60

        aj_e = aj_r.to_euler_angle(order)
        bh_e = bh_r.to_euler_angle(bh_order)

        np.testing.assert_allclose(float(aj_e.phi), float(bh_e.phi), atol=ATOL)
        np.testing.assert_allclose(float(aj_e.theta), float(bh_e.theta), atol=ATOL)
        np.testing.assert_allclose(float(aj_e.psi), float(bh_e.psi), atol=ATOL)
