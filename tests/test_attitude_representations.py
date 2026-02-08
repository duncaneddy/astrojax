"""Unit tests for attitude representation classes.

Mirrors the Rust test suite from refs/brahe_rust/tests/attitude/.
Uses float64 for precision parity with the Rust reference.
"""

import math

import jax
import jax.numpy as jnp
import pytest

# Enable float64 before importing astrojax classes
jax.config.update("jax_enable_x64", True)

from astrojax.config import set_dtype  # noqa: E402

set_dtype(jnp.float64)

from astrojax import (  # noqa: E402
    EulerAngle,
    EulerAngleOrder,
    EulerAxis,
    Quaternion,
    RotationMatrix,
)

DEG2RAD = math.pi / 180.0
SQRT2_2 = math.sqrt(2.0) / 2.0
PI = math.pi
ATOL = 1e-12


@pytest.fixture(autouse=True)
def _ensure_float64():
    """Ensure float64 is active for precision parity with Rust reference."""
    set_dtype(jnp.float64)
    yield
    set_dtype(jnp.float64)


# ===========================================================================
# EulerAngleOrder
# ===========================================================================


class TestEulerAngleOrder:
    def test_all_orders_exist(self):
        orders = list(EulerAngleOrder)
        assert len(orders) == 12

    def test_values_contiguous(self):
        for i, order in enumerate(EulerAngleOrder):
            assert order.value == i

    def test_names(self):
        expected = [
            "XYX",
            "XYZ",
            "XZX",
            "XZY",
            "YXY",
            "YXZ",
            "YZX",
            "YZY",
            "ZXY",
            "ZXZ",
            "ZYX",
            "ZYZ",
        ]
        for name in expected:
            assert hasattr(EulerAngleOrder, name)


# ===========================================================================
# Quaternion
# ===========================================================================


class TestQuaternion:
    def test_new_identity(self):
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        assert float(q.w) == pytest.approx(1.0, abs=ATOL)
        assert float(q.x) == pytest.approx(0.0, abs=ATOL)
        assert float(q.y) == pytest.approx(0.0, abs=ATOL)
        assert float(q.z) == pytest.approx(0.0, abs=ATOL)

    def test_new_normalizes(self):
        q = Quaternion(1.0, 1.0, 1.0, 1.0)
        assert float(q.w) == pytest.approx(0.5, abs=ATOL)
        assert float(q.x) == pytest.approx(0.5, abs=ATOL)
        assert float(q.y) == pytest.approx(0.5, abs=ATOL)
        assert float(q.z) == pytest.approx(0.5, abs=ATOL)

    def test_from_vector_scalar_first(self):
        v = jnp.array([1.0, 0.0, 0.0, 0.0])
        q = Quaternion.from_vector(v, scalar_first=True)
        assert float(q.w) == pytest.approx(1.0, abs=ATOL)

    def test_from_vector_scalar_last(self):
        v = jnp.array([0.0, 0.0, 0.0, 1.0])
        q = Quaternion.from_vector(v, scalar_first=False)
        assert float(q.w) == pytest.approx(1.0, abs=ATOL)

    def test_to_vector_scalar_first(self):
        q = Quaternion(1.0, 1.0, 1.0, 1.0)
        v = q.to_vector(scalar_first=True)
        assert float(v[0]) == pytest.approx(0.5, abs=ATOL)

    def test_to_vector_scalar_last(self):
        q = Quaternion(1.0, 1.0, 1.0, 1.0)
        v = q.to_vector(scalar_first=False)
        assert float(v[3]) == pytest.approx(0.5, abs=ATOL)

    def test_norm(self):
        q = Quaternion(1.0, 1.0, 1.0, 1.0)
        assert float(q.norm()) == pytest.approx(1.0, abs=ATOL)

    def test_conjugate(self):
        q = Quaternion(1.0, 1.0, 1.0, 1.0)
        qc = q.conjugate()
        v = qc.to_vector()
        assert float(v[0]) == pytest.approx(0.5, abs=ATOL)
        assert float(v[1]) == pytest.approx(-0.5, abs=ATOL)
        assert float(v[2]) == pytest.approx(-0.5, abs=ATOL)
        assert float(v[3]) == pytest.approx(-0.5, abs=ATOL)

    def test_inverse(self):
        q = Quaternion(1.0, 1.0, 1.0, 1.0)
        qi = q.inverse()
        product = q * qi
        identity = Quaternion(1.0, 0.0, 0.0, 0.0)
        assert product == identity

    def test_inverse_arbitrary(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        qi = q.inverse()
        product = q * qi
        identity = Quaternion(1.0, 0.0, 0.0, 0.0)
        assert product == identity

    def test_add(self):
        q1 = Quaternion(0.5, 1.0, 0.0, 0.5)
        q2 = Quaternion(0.5, 0.0, 1.0, 0.5)
        q = q1 + q2
        assert q == Quaternion(0.5, 0.5, 0.5, 0.5)

    def test_sub(self):
        q1 = Quaternion(0.5, 0.5, 0.0, 0.0)
        q2 = Quaternion(-0.5, 0.0, 0.0, -0.5)
        q = q1 - q2
        expected = Quaternion(1.0, 0.5, 0.0, 0.5)
        assert float(q[0]) == pytest.approx(float(expected[0]), abs=ATOL)
        assert float(q[1]) == pytest.approx(float(expected[1]), abs=ATOL)
        assert float(q[2]) == pytest.approx(float(expected[2]), abs=ATOL)
        assert float(q[3]) == pytest.approx(float(expected[3]), abs=ATOL)

    def test_mul_hamilton(self):
        q1 = Quaternion(1.0, 1.0, 0.0, 0.0)
        q2 = Quaternion(1.0, 0.0, 1.0, 0.0)
        q = q1 * q2
        assert q == Quaternion(1.0, 1.0, 1.0, 1.0)

    def test_neg(self):
        q = Quaternion(1.0, 1.0, 1.0, 1.0)
        qn = -q
        assert float(qn[0]) == pytest.approx(0.5, abs=ATOL)
        assert float(qn[1]) == pytest.approx(-0.5, abs=ATOL)
        assert float(qn[2]) == pytest.approx(-0.5, abs=ATOL)
        assert float(qn[3]) == pytest.approx(-0.5, abs=ATOL)

    def test_getitem(self):
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        assert float(q[0]) == pytest.approx(1.0, abs=ATOL)

    def test_eq(self):
        q1 = Quaternion(1.0, 0.0, 0.0, 0.0)
        q2 = Quaternion(1.0, 0.0, 0.0, 0.0)
        assert q1 == q2

    def test_ne(self):
        q1 = Quaternion(1.0, 0.0, 0.0, 0.0)
        q2 = Quaternion(0.0, 1.0, 0.0, 0.0)
        assert q1 != q2

    def test_slerp(self):
        q1 = EulerAngle(EulerAngleOrder.XYZ, 0.0, 0.0, 0.0, use_degrees=True).to_quaternion()
        q2 = EulerAngle(EulerAngleOrder.XYZ, 180.0, 0.0, 0.0, use_degrees=True).to_quaternion()
        q = q1.slerp(q2, 0.5)
        expected = EulerAngle(EulerAngleOrder.XYZ, 90.0, 0.0, 0.0, use_degrees=True).to_quaternion()
        assert q == expected


# ===========================================================================
# Quaternion conversion round-trips
# ===========================================================================


class TestQuaternionConversions:
    def test_from_euler_axis_identity(self):
        ea = EulerAxis(jnp.array([1.0, 0.0, 0.0]), 0.0)
        q = Quaternion.from_euler_axis(ea)
        assert q == Quaternion(1.0, 0.0, 0.0, 0.0)

    def test_from_euler_axis_90deg(self):
        ea = EulerAxis(jnp.array([1.0, 0.0, 0.0]), 90.0, use_degrees=True)
        q = Quaternion.from_euler_axis(ea)
        assert q == Quaternion(0.5, 0.5, 0.0, 0.0)

    def test_from_euler_angle_xyz(self):
        e = EulerAngle(EulerAngleOrder.XYZ, 90.0, 0.0, 0.0, use_degrees=True)
        q = Quaternion.from_euler_angle(e)
        expected = Quaternion(SQRT2_2, SQRT2_2, 0.0, 0.0)
        assert q == expected

    def test_from_rotation_matrix(self):
        r = RotationMatrix(
            1.0,
            0.0,
            0.0,
            0.0,
            SQRT2_2,
            -SQRT2_2,
            0.0,
            SQRT2_2,
            SQRT2_2,
        )
        q = Quaternion.from_rotation_matrix(r)
        expected = Quaternion(0.9238795325112867, -0.3826834323650898, 0.0, 0.0)
        assert q == expected

    def test_to_euler_axis_identity(self):
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        ea = q.to_euler_axis()
        assert ea == EulerAxis(jnp.array([1.0, 0.0, 0.0]), 0.0)

    def test_to_euler_axis_roundtrip(self):
        q = Quaternion(0.675, 0.42, 0.5, 0.71)
        ea = q.to_euler_axis()
        q2 = Quaternion.from_euler_axis(ea)
        assert q == q2

    def test_to_rotation_matrix_identity(self):
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        r = q.to_rotation_matrix()
        expected = RotationMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        assert r == expected

    def test_to_rotation_matrix_roundtrip(self):
        q = Quaternion(0.675, 0.42, 0.5, 0.71)
        r = q.to_rotation_matrix()
        q2 = Quaternion.from_rotation_matrix(r)
        assert q == q2

    @pytest.mark.parametrize("order", list(EulerAngleOrder))
    def test_to_euler_angle_roundtrip(self, order):
        q = Quaternion(0.675, 0.42, 0.5, 0.71)
        e = q.to_euler_angle(order)
        q2 = Quaternion.from_euler_angle(e)
        assert q == q2

    def test_from_euler_angle_xyz_values(self):
        e = EulerAngle(EulerAngleOrder.XYZ, 30.0, 45.0, 60.0, use_degrees=True)
        q = e.to_quaternion()
        assert float(q[0]) == pytest.approx(0.8223631719059994, abs=ATOL)
        assert float(q[1]) == pytest.approx(0.022260026714733844, abs=1e-10)
        assert float(q[2]) == pytest.approx(0.43967973954090955, abs=ATOL)
        assert float(q[3]) == pytest.approx(0.3604234056503559, abs=ATOL)


# ===========================================================================
# RotationMatrix
# ===========================================================================


class TestRotationMatrix:
    def test_new_valid(self):
        r = RotationMatrix(
            1.0,
            0.0,
            0.0,
            0.0,
            SQRT2_2,
            SQRT2_2,
            0.0,
            -SQRT2_2,
            SQRT2_2,
        )
        assert r.r11 == pytest.approx(1.0, abs=ATOL)

    def test_new_invalid_raises(self):
        with pytest.raises(ValueError):
            RotationMatrix(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0)

    def test_from_matrix(self):
        m = jnp.eye(3)
        r = RotationMatrix.from_matrix(m)
        assert r == RotationMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    def test_to_matrix(self):
        r = RotationMatrix(
            1.0,
            0.0,
            0.0,
            0.0,
            SQRT2_2,
            SQRT2_2,
            0.0,
            -SQRT2_2,
            SQRT2_2,
        )
        m = r.to_matrix()
        assert m.shape == (3, 3)
        assert float(m[0, 0]) == pytest.approx(1.0, abs=ATOL)

    def test_rotation_x(self):
        r = RotationMatrix.rotation_x(45.0, use_degrees=True)
        expected = RotationMatrix(
            1.0,
            0.0,
            0.0,
            0.0,
            SQRT2_2,
            SQRT2_2,
            0.0,
            -SQRT2_2,
            SQRT2_2,
        )
        assert r == expected

    def test_rotation_y(self):
        r = RotationMatrix.rotation_y(45.0, use_degrees=True)
        expected = RotationMatrix(
            SQRT2_2,
            0.0,
            -SQRT2_2,
            0.0,
            1.0,
            0.0,
            SQRT2_2,
            0.0,
            SQRT2_2,
        )
        assert r == expected

    def test_rotation_z(self):
        r = RotationMatrix.rotation_z(45.0, use_degrees=True)
        expected = RotationMatrix(
            SQRT2_2,
            SQRT2_2,
            0.0,
            -SQRT2_2,
            SQRT2_2,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        assert r == expected

    def test_mul_rotation_matrices(self):
        r1 = RotationMatrix.rotation_x(30.0, use_degrees=True)
        r2 = RotationMatrix.rotation_y(45.0, use_degrees=True)
        r3 = r1 * r2
        assert isinstance(r3, RotationMatrix)

    def test_mul_vector(self):
        r = RotationMatrix.rotation_z(90.0, use_degrees=True)
        v = jnp.array([1.0, 0.0, 0.0])
        result = r * v
        assert float(result[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(result[1]) == pytest.approx(-1.0, abs=1e-6)

    def test_getitem(self):
        r = RotationMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        assert float(r[0, 0]) == pytest.approx(1.0, abs=ATOL)

    def test_eq(self):
        r1 = RotationMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        r2 = RotationMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        assert r1 == r2


class TestRotationMatrixConversions:
    def test_from_quaternion_identity(self):
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        r = RotationMatrix.from_quaternion(q)
        expected = RotationMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        assert r == expected

    def test_from_euler_axis_rx(self):
        ea = EulerAxis(jnp.array([1.0, 0.0, 0.0]), 45.0, use_degrees=True)
        r = RotationMatrix.from_euler_axis(ea)
        expected = RotationMatrix(
            1.0,
            0.0,
            0.0,
            0.0,
            SQRT2_2,
            SQRT2_2,
            0.0,
            -SQRT2_2,
            SQRT2_2,
        )
        assert r == expected

    def test_from_euler_axis_ry(self):
        ea = EulerAxis(jnp.array([0.0, 1.0, 0.0]), 45.0, use_degrees=True)
        r = RotationMatrix.from_euler_axis(ea)
        expected = RotationMatrix(
            SQRT2_2,
            0.0,
            -SQRT2_2,
            0.0,
            1.0,
            0.0,
            SQRT2_2,
            0.0,
            SQRT2_2,
        )
        assert r == expected

    def test_from_euler_axis_rz(self):
        ea = EulerAxis(jnp.array([0.0, 0.0, 1.0]), 45.0, use_degrees=True)
        r = RotationMatrix.from_euler_axis(ea)
        expected = RotationMatrix(
            SQRT2_2,
            SQRT2_2,
            0.0,
            -SQRT2_2,
            SQRT2_2,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        assert r == expected

    def test_from_euler_angle_xyz(self):
        e = EulerAngle(EulerAngleOrder.XYZ, 45.0, 0.0, 0.0, use_degrees=True)
        r = RotationMatrix.from_euler_angle(e)
        expected = RotationMatrix(
            1.0,
            0.0,
            0.0,
            0.0,
            SQRT2_2,
            SQRT2_2,
            0.0,
            -SQRT2_2,
            SQRT2_2,
        )
        assert r == expected

    def test_to_quaternion(self):
        r = RotationMatrix(
            1.0,
            0.0,
            0.0,
            0.0,
            SQRT2_2,
            SQRT2_2,
            0.0,
            -SQRT2_2,
            SQRT2_2,
        )
        q = r.to_quaternion()
        expected = Quaternion(0.9238795325112867, 0.3826834323650898, 0.0, 0.0)
        assert q == expected

    def test_to_euler_axis_rx(self):
        r = RotationMatrix(
            1.0,
            0.0,
            0.0,
            0.0,
            SQRT2_2,
            SQRT2_2,
            0.0,
            -SQRT2_2,
            SQRT2_2,
        )
        ea = r.to_euler_axis()
        expected = EulerAxis(jnp.array([1.0, 0.0, 0.0]), 45.0, use_degrees=True)
        assert ea == expected

    def test_to_euler_axis_ry(self):
        r = RotationMatrix(
            SQRT2_2,
            0.0,
            -SQRT2_2,
            0.0,
            1.0,
            0.0,
            SQRT2_2,
            0.0,
            SQRT2_2,
        )
        ea = r.to_euler_axis()
        expected = EulerAxis(jnp.array([0.0, 1.0, 0.0]), 45.0, use_degrees=True)
        assert ea == expected

    def test_to_euler_axis_rz(self):
        r = RotationMatrix(
            SQRT2_2,
            SQRT2_2,
            0.0,
            -SQRT2_2,
            SQRT2_2,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        ea = r.to_euler_axis()
        expected = EulerAxis(jnp.array([0.0, 0.0, 1.0]), 45.0, use_degrees=True)
        assert ea == expected

    @pytest.mark.parametrize("order", list(EulerAngleOrder))
    def test_to_euler_angle_circular(self, order):
        r = (
            RotationMatrix.rotation_x(30.0, use_degrees=True)
            * RotationMatrix.rotation_y(45.0, use_degrees=True)
            * RotationMatrix.rotation_x(60.0, use_degrees=True)
        )
        e = r.to_euler_angle(order)
        r2 = e.to_rotation_matrix()
        assert r == r2

    @pytest.mark.parametrize("order", list(EulerAngleOrder))
    def test_from_euler_angle_all_orders(self, order):
        e = EulerAngle(order, 45.0, 30.0, 60.0, use_degrees=True)
        r = RotationMatrix.from_euler_angle(e)
        e2 = r.to_euler_angle(order)
        assert e == e2


# ===========================================================================
# EulerAngle
# ===========================================================================


class TestEulerAngle:
    def test_new_degrees(self):
        e = EulerAngle(EulerAngleOrder.XYZ, 30.0, 45.0, 60.0, use_degrees=True)
        assert float(e.phi) == pytest.approx(30.0 * DEG2RAD, abs=ATOL)
        assert float(e.theta) == pytest.approx(45.0 * DEG2RAD, abs=ATOL)
        assert float(e.psi) == pytest.approx(60.0 * DEG2RAD, abs=ATOL)
        assert e.order == EulerAngleOrder.XYZ

    def test_new_radians(self):
        e = EulerAngle(EulerAngleOrder.XYZ, PI / 6, PI / 4, PI / 3)
        assert float(e.phi) == pytest.approx(PI / 6, abs=ATOL)
        assert float(e.theta) == pytest.approx(PI / 4, abs=ATOL)
        assert float(e.psi) == pytest.approx(PI / 3, abs=ATOL)

    def test_degree_radian_equivalence(self):
        e1 = EulerAngle(EulerAngleOrder.XYZ, 30.0, 45.0, 60.0, use_degrees=True)
        e2 = EulerAngle(EulerAngleOrder.XYZ, PI / 6, PI / 4, PI / 3)
        assert e1 == e2

    def test_all_orders(self):
        for order in EulerAngleOrder:
            e = EulerAngle(order, 30.0, 45.0, 60.0, use_degrees=True)
            assert e.order == order

    def test_from_vector(self):
        v = jnp.array([30.0, 45.0, 60.0])
        e = EulerAngle.from_vector(v, EulerAngleOrder.XYZ, use_degrees=True)
        assert float(e.phi) == pytest.approx(30.0 * DEG2RAD, abs=ATOL)

    def test_from_quaternion(self):
        q = Quaternion(SQRT2_2, 0.0, 0.0, SQRT2_2)
        e = EulerAngle.from_quaternion(q, EulerAngleOrder.XYZ)
        assert float(e.phi) == pytest.approx(0.0, abs=ATOL)
        assert float(e.theta) == pytest.approx(0.0, abs=ATOL)
        assert float(e.psi) == pytest.approx(PI / 2, abs=ATOL)

    def test_from_euler_axis_x(self):
        ea = EulerAxis(jnp.array([1.0, 0.0, 0.0]), 45.0, use_degrees=True)
        e = EulerAngle.from_euler_axis(ea, EulerAngleOrder.XYZ)
        assert float(e.phi) == pytest.approx(45.0 * DEG2RAD, abs=ATOL)
        assert float(e.theta) == pytest.approx(0.0, abs=ATOL)
        assert float(e.psi) == pytest.approx(0.0, abs=ATOL)

    def test_from_euler_axis_y(self):
        ea = EulerAxis(jnp.array([0.0, 1.0, 0.0]), 45.0, use_degrees=True)
        e = EulerAngle.from_euler_axis(ea, EulerAngleOrder.XYZ)
        assert float(e.phi) == pytest.approx(0.0, abs=ATOL)
        assert float(e.theta) == pytest.approx(45.0 * DEG2RAD, abs=ATOL)
        assert float(e.psi) == pytest.approx(0.0, abs=ATOL)

    def test_from_euler_axis_z(self):
        ea = EulerAxis(jnp.array([0.0, 0.0, 1.0]), 45.0, use_degrees=True)
        e = EulerAngle.from_euler_axis(ea, EulerAngleOrder.XYZ)
        assert float(e.phi) == pytest.approx(0.0, abs=ATOL)
        assert float(e.theta) == pytest.approx(0.0, abs=ATOL)
        assert float(e.psi) == pytest.approx(45.0 * DEG2RAD, abs=ATOL)

    def test_from_euler_angle_order_change(self):
        e1 = EulerAngle(EulerAngleOrder.XYZ, 30.0, 45.0, 60.0, use_degrees=True)
        e2 = EulerAngle.from_euler_angle(e1, EulerAngleOrder.ZYX)
        assert e2.order == EulerAngleOrder.ZYX

    def test_from_rotation_matrix(self):
        r = RotationMatrix(
            1.0,
            0.0,
            0.0,
            0.0,
            SQRT2_2,
            SQRT2_2,
            0.0,
            -SQRT2_2,
            SQRT2_2,
        )
        e = EulerAngle.from_rotation_matrix(r, EulerAngleOrder.XYZ)
        assert float(e.phi) == pytest.approx(PI / 4, abs=ATOL)
        assert float(e.theta) == pytest.approx(0.0, abs=ATOL)
        assert float(e.psi) == pytest.approx(0.0, abs=ATOL)

    def test_to_quaternion_identity(self):
        e = EulerAngle(EulerAngleOrder.XYZ, 0.0, 0.0, 0.0, use_degrees=True)
        q = e.to_quaternion()
        assert float(q[0]) == pytest.approx(1.0, abs=ATOL)
        assert float(q[1]) == pytest.approx(0.0, abs=ATOL)
        assert float(q[2]) == pytest.approx(0.0, abs=ATOL)
        assert float(q[3]) == pytest.approx(0.0, abs=ATOL)

    def test_to_euler_axis_x(self):
        e = EulerAngle(EulerAngleOrder.XYZ, 45.0, 0.0, 0.0, use_degrees=True)
        ea = e.to_euler_axis()
        assert float(ea.axis[0]) == pytest.approx(1.0, abs=ATOL)
        assert float(ea.axis[1]) == pytest.approx(0.0, abs=ATOL)
        assert float(ea.axis[2]) == pytest.approx(0.0, abs=ATOL)
        assert float(ea.angle) == pytest.approx(PI / 4, abs=ATOL)

    def test_to_euler_axis_y(self):
        e = EulerAngle(EulerAngleOrder.XYZ, 0.0, 45.0, 0.0, use_degrees=True)
        ea = e.to_euler_axis()
        assert float(ea.axis[0]) == pytest.approx(0.0, abs=ATOL)
        assert float(ea.axis[1]) == pytest.approx(1.0, abs=ATOL)
        assert float(ea.axis[2]) == pytest.approx(0.0, abs=ATOL)
        assert float(ea.angle) == pytest.approx(PI / 4, abs=ATOL)

    def test_to_euler_axis_z(self):
        e = EulerAngle(EulerAngleOrder.XYZ, 0.0, 0.0, 45.0, use_degrees=True)
        ea = e.to_euler_axis()
        assert float(ea.axis[0]) == pytest.approx(0.0, abs=ATOL)
        assert float(ea.axis[1]) == pytest.approx(0.0, abs=ATOL)
        assert float(ea.axis[2]) == pytest.approx(1.0, abs=ATOL)
        assert float(ea.angle) == pytest.approx(PI / 4, abs=ATOL)

    def test_to_euler_angle_order_change(self):
        e = EulerAngle(EulerAngleOrder.XYZ, 30.0, 45.0, 60.0, use_degrees=True)
        e2 = e.to_euler_angle(EulerAngleOrder.ZYX)
        assert e2.order == EulerAngleOrder.ZYX

    def test_to_rotation_matrix_rx(self):
        e = EulerAngle(EulerAngleOrder.XYZ, 45.0, 0.0, 0.0, use_degrees=True)
        r = e.to_rotation_matrix()
        assert float(r[0, 0]) == pytest.approx(1.0, abs=ATOL)
        assert float(r[1, 1]) == pytest.approx(SQRT2_2, abs=ATOL)
        assert float(r[1, 2]) == pytest.approx(SQRT2_2, abs=ATOL)
        assert float(r[2, 1]) == pytest.approx(-SQRT2_2, abs=ATOL)
        assert float(r[2, 2]) == pytest.approx(SQRT2_2, abs=ATOL)

    def test_to_rotation_matrix_ry(self):
        e = EulerAngle(EulerAngleOrder.XYZ, 0.0, 45.0, 0.0, use_degrees=True)
        r = e.to_rotation_matrix()
        assert float(r[0, 0]) == pytest.approx(SQRT2_2, abs=ATOL)
        assert float(r[0, 2]) == pytest.approx(-SQRT2_2, abs=ATOL)
        assert float(r[1, 1]) == pytest.approx(1.0, abs=ATOL)
        assert float(r[2, 0]) == pytest.approx(SQRT2_2, abs=ATOL)
        assert float(r[2, 2]) == pytest.approx(SQRT2_2, abs=ATOL)

    def test_to_rotation_matrix_rz(self):
        e = EulerAngle(EulerAngleOrder.XYZ, 0.0, 0.0, 45.0, use_degrees=True)
        r = e.to_rotation_matrix()
        assert float(r[0, 0]) == pytest.approx(SQRT2_2, abs=ATOL)
        assert float(r[0, 1]) == pytest.approx(SQRT2_2, abs=ATOL)
        assert float(r[1, 0]) == pytest.approx(-SQRT2_2, abs=ATOL)
        assert float(r[1, 1]) == pytest.approx(SQRT2_2, abs=ATOL)
        assert float(r[2, 2]) == pytest.approx(1.0, abs=ATOL)

    @pytest.mark.parametrize("order", list(EulerAngleOrder))
    def test_euler_angle_circular_roundtrip(self, order):
        e = EulerAngle(order, 30.0, 45.0, 60.0, use_degrees=True)
        e2 = e.to_euler_angle(order)
        assert e == e2

    def test_eq(self):
        e1 = EulerAngle(EulerAngleOrder.XYZ, 30.0, 45.0, 60.0, use_degrees=True)
        e2 = EulerAngle(EulerAngleOrder.XYZ, 30.0, 45.0, 60.0, use_degrees=True)
        assert e1 == e2

    def test_ne_different_order(self):
        e1 = EulerAngle(EulerAngleOrder.XYZ, 30.0, 45.0, 60.0, use_degrees=True)
        e2 = EulerAngle(EulerAngleOrder.ZYX, 30.0, 45.0, 60.0, use_degrees=True)
        assert e1 != e2


# ===========================================================================
# EulerAxis
# ===========================================================================


class TestEulerAxis:
    def test_new_degrees(self):
        ea = EulerAxis(jnp.array([1.0, 1.0, 1.0]), 45.0, use_degrees=True)
        assert float(ea.angle) == pytest.approx(45.0 * DEG2RAD, abs=ATOL)

    def test_new_radians(self):
        ea = EulerAxis(jnp.array([1.0, 0.0, 0.0]), PI / 4)
        assert float(ea.angle) == pytest.approx(PI / 4, abs=ATOL)

    def test_from_values(self):
        ea = EulerAxis.from_values(1.0, 1.0, 1.0, 45.0, use_degrees=True)
        assert float(ea.axis[0]) == pytest.approx(1.0, abs=ATOL)
        assert float(ea.angle) == pytest.approx(45.0 * DEG2RAD, abs=ATOL)

    def test_from_vector_vector_first(self):
        v = jnp.array([1.0, 1.0, 1.0, 45.0])
        ea = EulerAxis.from_vector(v, use_degrees=True, vector_first=True)
        assert float(ea.axis[0]) == pytest.approx(1.0, abs=ATOL)
        assert float(ea.angle) == pytest.approx(45.0 * DEG2RAD, abs=ATOL)

    def test_from_vector_angle_first(self):
        v = jnp.array([45.0, 1.0, 1.0, 1.0])
        ea = EulerAxis.from_vector(v, use_degrees=True, vector_first=False)
        assert float(ea.axis[0]) == pytest.approx(1.0, abs=ATOL)
        assert float(ea.angle) == pytest.approx(45.0 * DEG2RAD, abs=ATOL)

    def test_to_vector_vector_first(self):
        ea = EulerAxis.from_values(1.0, 1.0, 1.0, 45.0, use_degrees=True)
        v = ea.to_vector(use_degrees=True, vector_first=True)
        assert float(v[0]) == pytest.approx(1.0, abs=ATOL)
        assert float(v[3]) == pytest.approx(45.0, abs=ATOL)

    def test_to_vector_angle_first(self):
        ea = EulerAxis.from_values(1.0, 1.0, 1.0, 45.0, use_degrees=True)
        v = ea.to_vector(use_degrees=True, vector_first=False)
        assert float(v[0]) == pytest.approx(45.0, abs=ATOL)
        assert float(v[1]) == pytest.approx(1.0, abs=ATOL)

    def test_from_quaternion(self):
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        ea = EulerAxis.from_quaternion(q)
        assert float(ea.axis[0]) == pytest.approx(1.0, abs=ATOL)
        assert float(ea.angle) == pytest.approx(0.0, abs=ATOL)

    def test_from_euler_axis(self):
        ea = EulerAxis.from_values(1.0, 1.0, 1.0, 45.0, use_degrees=True)
        ea2 = EulerAxis.from_euler_axis(ea)
        assert ea == ea2

    def test_from_euler_angle_x(self):
        e = EulerAngle(EulerAngleOrder.XYZ, 45.0, 0.0, 0.0, use_degrees=True)
        ea = EulerAxis.from_euler_angle(e)
        assert float(ea.axis[0]) == pytest.approx(1.0, abs=ATOL)
        assert float(ea.angle) == pytest.approx(PI / 4, abs=ATOL)

    def test_from_euler_angle_y(self):
        e = EulerAngle(EulerAngleOrder.XYZ, 0.0, 45.0, 0.0, use_degrees=True)
        ea = EulerAxis.from_euler_angle(e)
        assert float(ea.axis[1]) == pytest.approx(1.0, abs=ATOL)
        assert float(ea.angle) == pytest.approx(PI / 4, abs=ATOL)

    def test_from_euler_angle_z(self):
        e = EulerAngle(EulerAngleOrder.XYZ, 0.0, 0.0, 45.0, use_degrees=True)
        ea = EulerAxis.from_euler_angle(e)
        assert float(ea.axis[2]) == pytest.approx(1.0, abs=ATOL)
        assert float(ea.angle) == pytest.approx(PI / 4, abs=ATOL)

    def test_from_rotation_matrix_rx(self):
        r = RotationMatrix(
            1.0,
            0.0,
            0.0,
            0.0,
            SQRT2_2,
            SQRT2_2,
            0.0,
            -SQRT2_2,
            SQRT2_2,
        )
        ea = EulerAxis.from_rotation_matrix(r)
        assert float(ea.axis[0]) == pytest.approx(1.0, abs=ATOL)
        assert float(ea.angle) == pytest.approx(PI / 4, abs=ATOL)

    def test_from_rotation_matrix_ry(self):
        r = RotationMatrix(
            SQRT2_2,
            0.0,
            -SQRT2_2,
            0.0,
            1.0,
            0.0,
            SQRT2_2,
            0.0,
            SQRT2_2,
        )
        ea = EulerAxis.from_rotation_matrix(r)
        assert float(ea.axis[1]) == pytest.approx(1.0, abs=ATOL)
        assert float(ea.angle) == pytest.approx(PI / 4, abs=ATOL)

    def test_from_rotation_matrix_rz(self):
        r = RotationMatrix(
            SQRT2_2,
            SQRT2_2,
            0.0,
            -SQRT2_2,
            SQRT2_2,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        ea = EulerAxis.from_rotation_matrix(r)
        assert float(ea.axis[2]) == pytest.approx(1.0, abs=ATOL)
        assert float(ea.angle) == pytest.approx(PI / 4, abs=ATOL)

    def test_to_quaternion_identity(self):
        ea = EulerAxis.from_values(1.0, 0.0, 0.0, 0.0)
        q = ea.to_quaternion()
        assert q == Quaternion(1.0, 0.0, 0.0, 0.0)

    def test_to_euler_axis_copy(self):
        ea = EulerAxis.from_values(1.0, 1.0, 1.0, 45.0, use_degrees=True)
        ea2 = ea.to_euler_axis()
        assert ea == ea2

    def test_to_euler_angle_rx(self):
        ea = EulerAxis.from_values(1.0, 0.0, 0.0, PI / 4)
        e = ea.to_euler_angle(EulerAngleOrder.XYZ)
        assert float(e.phi) == pytest.approx(PI / 4, abs=ATOL)
        assert float(e.theta) == pytest.approx(0.0, abs=ATOL)
        assert float(e.psi) == pytest.approx(0.0, abs=ATOL)

    def test_to_rotation_matrix_rx(self):
        ea = EulerAxis.from_values(1.0, 0.0, 0.0, PI / 4)
        r = ea.to_rotation_matrix()
        assert float(r[0, 0]) == pytest.approx(1.0, abs=ATOL)
        assert float(r[1, 1]) == pytest.approx(SQRT2_2, abs=ATOL)
        assert float(r[1, 2]) == pytest.approx(SQRT2_2, abs=ATOL)
        assert float(r[2, 1]) == pytest.approx(-SQRT2_2, abs=ATOL)
        assert float(r[2, 2]) == pytest.approx(SQRT2_2, abs=ATOL)

    def test_eq(self):
        ea1 = EulerAxis.from_values(1.0, 0.0, 0.0, PI / 4)
        ea2 = EulerAxis.from_values(1.0, 0.0, 0.0, PI / 4)
        assert ea1 == ea2

    def test_ne(self):
        ea1 = EulerAxis.from_values(1.0, 0.0, 0.0, PI / 4)
        ea2 = EulerAxis.from_values(0.0, 1.0, 0.0, PI / 4)
        assert ea1 != ea2

    def test_getitem(self):
        ea = EulerAxis.from_values(1.0, 2.0, 3.0, PI / 4)
        assert float(ea[0]) == pytest.approx(1.0, abs=ATOL)
        assert float(ea[1]) == pytest.approx(2.0, abs=ATOL)
        assert float(ea[2]) == pytest.approx(3.0, abs=ATOL)


# ===========================================================================
# JIT compatibility
# ===========================================================================


class TestJIT:
    def test_quaternion_to_rotation_matrix_jit(self):
        @jax.jit
        def convert(q):
            return q.to_rotation_matrix()

        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        r = convert(q)
        assert isinstance(r, RotationMatrix)
        expected = RotationMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        assert r == expected

    def test_rotation_matrix_to_quaternion_jit(self):
        @jax.jit
        def convert(r):
            return r.to_quaternion()

        r = RotationMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        q = convert(r)
        assert isinstance(q, Quaternion)
        assert q == Quaternion(1.0, 0.0, 0.0, 0.0)

    def test_euler_axis_to_quaternion_jit(self):
        @jax.jit
        def convert(ea):
            return ea.to_quaternion()

        ea = EulerAxis(jnp.array([1.0, 0.0, 0.0]), PI / 4)
        q = convert(ea)
        assert isinstance(q, Quaternion)

    def test_euler_angle_to_quaternion_jit(self):
        @jax.jit
        def convert(e):
            return e.to_quaternion()

        e = EulerAngle(EulerAngleOrder.XYZ, 0.5, 0.3, 0.1)
        q = convert(e)
        assert isinstance(q, Quaternion)

    def test_full_roundtrip_jit(self):
        @jax.jit
        def roundtrip(q):
            r = q.to_rotation_matrix()
            return r.to_quaternion()

        q = Quaternion(0.675, 0.42, 0.5, 0.71)
        q2 = roundtrip(q)
        assert q == q2

    def test_hamilton_product_jit(self):
        @jax.jit
        def product(q1, q2):
            return q1 * q2

        q1 = Quaternion(1.0, 1.0, 0.0, 0.0)
        q2 = Quaternion(1.0, 0.0, 1.0, 0.0)
        q = product(q1, q2)
        assert q == Quaternion(1.0, 1.0, 1.0, 1.0)

    def test_slerp_jit(self):
        @jax.jit
        def interp(q1, q2):
            return q1.slerp(q2, 0.5)

        q1 = Quaternion(1.0, 0.0, 0.0, 0.0)
        q2 = Quaternion(0.0, 1.0, 0.0, 0.0)
        q = interp(q1, q2)
        assert isinstance(q, Quaternion)


# ===========================================================================
# Pytree
# ===========================================================================


class TestPytree:
    def test_quaternion_pytree(self):
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        leaves, treedef = jax.tree_util.tree_flatten(q)
        assert len(leaves) == 1
        q2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert q == q2

    def test_rotation_matrix_pytree(self):
        r = RotationMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        leaves, treedef = jax.tree_util.tree_flatten(r)
        assert len(leaves) == 1
        r2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert r == r2

    def test_euler_axis_pytree(self):
        ea = EulerAxis(jnp.array([1.0, 0.0, 0.0]), PI / 4)
        leaves, treedef = jax.tree_util.tree_flatten(ea)
        assert len(leaves) == 2
        ea2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert ea == ea2

    def test_euler_angle_pytree(self):
        e = EulerAngle(EulerAngleOrder.XYZ, 0.5, 0.3, 0.1)
        leaves, treedef = jax.tree_util.tree_flatten(e)
        assert len(leaves) == 3
        e2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert e == e2
        assert e2.order == EulerAngleOrder.XYZ
