"""Pure conversion functions between attitude representations.

All functions operate on raw JAX arrays (no class instances) to avoid
circular imports between class modules.  The classes in ``quaternion.py``,
``rotation_matrix.py``, ``euler_angle.py``, and ``euler_axis.py`` call
these kernels and wrap the results.

Convention:
    Quaternion layout is scalar-first: ``[w, x, y, z]`` (shape ``(4,)``).
    Rotation matrix layout is row-major: shape ``(3, 3)``.
    Euler axis is ``(axis(3,), angle_scalar)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Quaternion <-> Rotation Matrix
# ---------------------------------------------------------------------------

def quaternion_to_rotation_matrix(q: jax.Array) -> jax.Array:
    """Convert a unit quaternion to a 3x3 rotation matrix.

    Uses the bilinear product form (Diebel eq. 125).

    Args:
        q (jax.Array): Quaternion array of shape ``(4,)`` in scalar-first order ``[w, x, y, z]``.

    Returns:
        jnp.ndarray: Rotation matrix of shape ``(3, 3)``.
    """
    qs, q1, q2, q3 = q[0], q[1], q[2], q[3]

    return jnp.array([
        [qs*qs + q1*q1 - q2*q2 - q3*q3,  2.0*q1*q2 + 2.0*qs*q3,          2.0*q1*q3 - 2.0*qs*q2],
        [2.0*q1*q2 - 2.0*qs*q3,           qs*qs - q1*q1 + q2*q2 - q3*q3,  2.0*q2*q3 + 2.0*qs*q1],
        [2.0*q1*q3 + 2.0*qs*q2,           2.0*q2*q3 - 2.0*qs*q1,          qs*qs - q1*q1 - q2*q2 + q3*q3],
    ])


def rotation_matrix_to_quaternion(R: jax.Array) -> jax.Array:
    """Convert a 3x3 rotation matrix to a unit quaternion.

    Uses Shepperd's method with ``jax.lax.switch`` on ``argmax`` for
    numerical stability and JIT compatibility.

    Args:
        R (jax.Array): Rotation matrix of shape ``(3, 3)``.

    Returns:
        jnp.ndarray: Quaternion array of shape ``(4,)`` in scalar-first order ``[w, x, y, z]``.
    """
    # Diebel eqs. 131-134: the four candidate traces
    qvec = jnp.array([
        1.0 + R[0, 0] + R[1, 1] + R[2, 2],
        1.0 + R[0, 0] - R[1, 1] - R[2, 2],
        1.0 - R[0, 0] + R[1, 1] - R[2, 2],
        1.0 - R[0, 0] - R[1, 1] + R[2, 2],
    ])

    ind_max = jnp.argmax(qvec)
    q_max = qvec[ind_max]

    # Four branch functions, one per argmax case
    def _case0(_):
        sq = jnp.sqrt(q_max)
        return 0.5 * jnp.array([
            sq,
            (R[1, 2] - R[2, 1]) / sq,
            (R[2, 0] - R[0, 2]) / sq,
            (R[0, 1] - R[1, 0]) / sq,
        ])

    def _case1(_):
        sq = jnp.sqrt(q_max)
        return 0.5 * jnp.array([
            (R[1, 2] - R[2, 1]) / sq,
            sq,
            (R[0, 1] + R[1, 0]) / sq,
            (R[2, 0] + R[0, 2]) / sq,
        ])

    def _case2(_):
        sq = jnp.sqrt(q_max)
        return 0.5 * jnp.array([
            (R[2, 0] - R[0, 2]) / sq,
            (R[0, 1] + R[1, 0]) / sq,
            sq,
            (R[1, 2] + R[2, 1]) / sq,
        ])

    def _case3(_):
        sq = jnp.sqrt(q_max)
        return 0.5 * jnp.array([
            (R[0, 1] - R[1, 0]) / sq,
            (R[2, 0] + R[0, 2]) / sq,
            (R[1, 2] + R[2, 1]) / sq,
            sq,
        ])

    return jax.lax.switch(ind_max, [_case0, _case1, _case2, _case3], None)


# ---------------------------------------------------------------------------
# Euler Axis <-> Quaternion
# ---------------------------------------------------------------------------

def euler_axis_to_quaternion(axis: jax.Array, angle: jax.Array) -> jax.Array:
    """Convert an Euler axis-angle representation to a quaternion.

    The axis is used as-is with half-angle trig, then the result is
    normalized.  This matches the brahe Rust convention where
    ``Quaternion::new`` normalizes after construction.

    Args:
        axis (jax.Array): Rotation axis vector of shape ``(3,)``.
        angle (jax.Array): Rotation angle in radians (scalar).

    Returns:
        jnp.ndarray: Unit quaternion of shape ``(4,)`` in scalar-first order.
    """
    half = angle / 2.0
    s = jnp.sin(half)
    q = jnp.array([jnp.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])
    return q / jnp.linalg.norm(q)


def quaternion_to_euler_axis(q: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Convert a unit quaternion to Euler axis-angle representation.

    When the rotation angle is zero the axis is undefined; the default
    axis ``[1, 0, 0]`` is returned.

    Args:
        q (jax.Array): Quaternion of shape ``(4,)`` in scalar-first order.

    Returns:
        tuple: ``(axis, angle)`` where ``axis`` has shape ``(3,)`` and
        ``angle`` is a scalar in radians.
    """
    angle = 2.0 * jnp.arccos(jnp.clip(q[0], -1.0, 1.0))

    # When angle ~ 0 the vector part is ~ 0 and axis is undefined
    v = jnp.array([q[1], q[2], q[3]])
    v_norm = jnp.linalg.norm(v)
    default_axis = jnp.array([1.0, 0.0, 0.0])

    axis = jnp.where(v_norm > 1e-15, v / v_norm, default_axis)

    return axis, angle


# ---------------------------------------------------------------------------
# Euler Angle -> Quaternion (12 branches)
# ---------------------------------------------------------------------------

def _ea_to_q_xyx(cp: jax.Array, ct: jax.Array, cs: jax.Array, sp: jax.Array, st: jax.Array, ss: jax.Array) -> jax.Array:
    return jnp.array([
        cp*ct*cs - sp*ct*ss,
        cp*ct*ss + ct*cs*sp,
        cp*cs*st + sp*st*ss,
        cp*st*ss - sp*cs*st,
    ])

def _ea_to_q_xyz(cp: jax.Array, ct: jax.Array, cs: jax.Array, sp: jax.Array, st: jax.Array, ss: jax.Array) -> jax.Array:
    return jnp.array([
        cp*ct*cs + sp*st*ss,
        -cp*st*ss + ct*cs*sp,
        cp*cs*st + sp*ct*ss,
        cp*ct*ss - sp*cs*st,
    ])

def _ea_to_q_xzx(cp: jax.Array, ct: jax.Array, cs: jax.Array, sp: jax.Array, st: jax.Array, ss: jax.Array) -> jax.Array:
    return jnp.array([
        cp*ct*cs - sp*ct*ss,
        cp*ct*ss + ct*cs*sp,
        -cp*st*ss + sp*cs*st,
        cp*cs*st + sp*st*ss,
    ])

def _ea_to_q_xzy(cp: jax.Array, ct: jax.Array, cs: jax.Array, sp: jax.Array, st: jax.Array, ss: jax.Array) -> jax.Array:
    return jnp.array([
        cp*ct*cs - sp*st*ss,
        cp*st*ss + ct*cs*sp,
        cp*ct*ss + sp*cs*st,
        cp*cs*st - sp*ct*ss,
    ])

def _ea_to_q_yxy(cp: jax.Array, ct: jax.Array, cs: jax.Array, sp: jax.Array, st: jax.Array, ss: jax.Array) -> jax.Array:
    return jnp.array([
        cp*ct*cs - sp*ct*ss,
        cp*cs*st + sp*st*ss,
        cp*ct*ss + ct*cs*sp,
        -cp*st*ss + sp*cs*st,
    ])

def _ea_to_q_yxz(cp: jax.Array, ct: jax.Array, cs: jax.Array, sp: jax.Array, st: jax.Array, ss: jax.Array) -> jax.Array:
    return jnp.array([
        cp*ct*cs - sp*st*ss,
        cp*cs*st - sp*ct*ss,
        cp*st*ss + ct*cs*sp,
        cp*ct*ss + sp*cs*st,
    ])

def _ea_to_q_yzx(cp: jax.Array, ct: jax.Array, cs: jax.Array, sp: jax.Array, st: jax.Array, ss: jax.Array) -> jax.Array:
    return jnp.array([
        cp*ct*cs + sp*st*ss,
        cp*ct*ss - sp*cs*st,
        -cp*st*ss + ct*cs*sp,
        cp*cs*st + sp*ct*ss,
    ])

def _ea_to_q_yzy(cp: jax.Array, ct: jax.Array, cs: jax.Array, sp: jax.Array, st: jax.Array, ss: jax.Array) -> jax.Array:
    return jnp.array([
        cp*ct*cs - sp*ct*ss,
        cp*st*ss - sp*cs*st,
        cp*ct*ss + ct*cs*sp,
        cp*cs*st + sp*st*ss,
    ])

def _ea_to_q_zxy(cp: jax.Array, ct: jax.Array, cs: jax.Array, sp: jax.Array, st: jax.Array, ss: jax.Array) -> jax.Array:
    return jnp.array([
        cp*ct*cs + sp*st*ss,
        cp*cs*st + sp*ct*ss,
        cp*ct*ss - sp*cs*st,
        -cp*st*ss + ct*cs*sp,
    ])

def _ea_to_q_zxz(cp: jax.Array, ct: jax.Array, cs: jax.Array, sp: jax.Array, st: jax.Array, ss: jax.Array) -> jax.Array:
    return jnp.array([
        cp*ct*cs - sp*ct*ss,
        cp*cs*st + sp*st*ss,
        cp*st*ss - sp*cs*st,
        cp*ct*ss + ct*cs*sp,
    ])

def _ea_to_q_zyx(cp: jax.Array, ct: jax.Array, cs: jax.Array, sp: jax.Array, st: jax.Array, ss: jax.Array) -> jax.Array:
    return jnp.array([
        cp*ct*cs - sp*st*ss,
        cp*ct*ss + sp*cs*st,
        cp*cs*st - sp*ct*ss,
        cp*st*ss + ct*cs*sp,
    ])

def _ea_to_q_zyz(cp: jax.Array, ct: jax.Array, cs: jax.Array, sp: jax.Array, st: jax.Array, ss: jax.Array) -> jax.Array:
    return jnp.array([
        cp*ct*cs - sp*ct*ss,
        -cp*st*ss + sp*cs*st,
        cp*cs*st + sp*st*ss,
        cp*ct*ss + ct*cs*sp,
    ])

_EA_TO_Q_BRANCHES = [
    _ea_to_q_xyx,   # 0
    _ea_to_q_xyz,   # 1
    _ea_to_q_xzx,   # 2
    _ea_to_q_xzy,   # 3
    _ea_to_q_yxy,   # 4
    _ea_to_q_yxz,   # 5
    _ea_to_q_yzx,   # 6
    _ea_to_q_yzy,   # 7
    _ea_to_q_zxy,   # 8
    _ea_to_q_zxz,   # 9
    _ea_to_q_zyx,   # 10
    _ea_to_q_zyz,   # 11
]


def euler_angle_to_quaternion(order_idx: jax.Array, phi: jax.Array, theta: jax.Array, psi: jax.Array) -> jax.Array:
    """Convert Euler angles to a quaternion via ``jax.lax.switch``.

    Args:
        order_idx (jax.Array): Integer index 0--11 matching ``EulerAngleOrder``.
        phi (jax.Array): First rotation angle in radians.
        theta (jax.Array): Second rotation angle in radians.
        psi (jax.Array): Third rotation angle in radians.

    Returns:
        jnp.ndarray: Quaternion of shape ``(4,)`` in scalar-first order.
    """
    cp = jnp.cos(phi / 2.0)
    ct = jnp.cos(theta / 2.0)
    cs = jnp.cos(psi / 2.0)
    sp = jnp.sin(phi / 2.0)
    st = jnp.sin(theta / 2.0)
    ss = jnp.sin(psi / 2.0)

    trig = (cp, ct, cs, sp, st, ss)

    branches = [lambda t, f=f: f(*t) for f in _EA_TO_Q_BRANCHES]
    q = jax.lax.switch(order_idx, branches, trig)

    # Normalize
    return q / jnp.linalg.norm(q)


# ---------------------------------------------------------------------------
# Rotation Matrix -> Euler Angle (12 branches)
# ---------------------------------------------------------------------------

def _rm_to_ea_xyx(R: jax.Array) -> jax.Array:
    return jnp.array([jnp.arctan2(R[1, 0], R[2, 0]),  jnp.arccos(R[0, 0]),   jnp.arctan2(R[0, 1], -R[0, 2])])

def _rm_to_ea_xyz(R: jax.Array) -> jax.Array:
    return jnp.array([jnp.arctan2(R[1, 2], R[2, 2]),  -jnp.arcsin(R[0, 2]),  jnp.arctan2(R[0, 1], R[0, 0])])

def _rm_to_ea_xzx(R: jax.Array) -> jax.Array:
    return jnp.array([jnp.arctan2(R[2, 0], -R[1, 0]), jnp.arccos(R[0, 0]),   jnp.arctan2(R[0, 2], R[0, 1])])

def _rm_to_ea_xzy(R: jax.Array) -> jax.Array:
    return jnp.array([jnp.arctan2(-R[2, 1], R[1, 1]), jnp.arcsin(R[0, 1]),   jnp.arctan2(-R[0, 2], R[0, 0])])

def _rm_to_ea_yxy(R: jax.Array) -> jax.Array:
    return jnp.array([jnp.arctan2(R[0, 1], -R[2, 1]), jnp.arccos(R[1, 1]),   jnp.arctan2(R[1, 0], R[1, 2])])

def _rm_to_ea_yxz(R: jax.Array) -> jax.Array:
    return jnp.array([jnp.arctan2(-R[0, 2], R[2, 2]), jnp.arcsin(R[1, 2]),   jnp.arctan2(-R[1, 0], R[1, 1])])

def _rm_to_ea_yzx(R: jax.Array) -> jax.Array:
    return jnp.array([jnp.arctan2(R[2, 0], R[0, 0]),  -jnp.arcsin(R[1, 0]),  jnp.arctan2(R[1, 2], R[1, 1])])

def _rm_to_ea_yzy(R: jax.Array) -> jax.Array:
    return jnp.array([jnp.arctan2(R[2, 1], R[0, 1]),  jnp.arccos(R[1, 1]),   jnp.arctan2(R[1, 2], -R[1, 0])])

def _rm_to_ea_zxy(R: jax.Array) -> jax.Array:
    return jnp.array([jnp.arctan2(R[0, 1], R[1, 1]),  -jnp.arcsin(R[2, 1]),  jnp.arctan2(R[2, 0], R[2, 2])])

def _rm_to_ea_zxz(R: jax.Array) -> jax.Array:
    return jnp.array([jnp.arctan2(R[0, 2], R[1, 2]),  jnp.arccos(R[2, 2]),   jnp.arctan2(R[2, 0], -R[2, 1])])

def _rm_to_ea_zyx(R: jax.Array) -> jax.Array:
    return jnp.array([jnp.arctan2(-R[1, 0], R[0, 0]), jnp.arcsin(R[2, 0]),   jnp.arctan2(-R[2, 1], R[2, 2])])

def _rm_to_ea_zyz(R: jax.Array) -> jax.Array:
    return jnp.array([jnp.arctan2(R[1, 2], -R[0, 2]), jnp.arccos(R[2, 2]),   jnp.arctan2(R[2, 1], R[2, 0])])

_RM_TO_EA_BRANCHES = [
    _rm_to_ea_xyx,   # 0
    _rm_to_ea_xyz,   # 1
    _rm_to_ea_xzx,   # 2
    _rm_to_ea_xzy,   # 3
    _rm_to_ea_yxy,   # 4
    _rm_to_ea_yxz,   # 5
    _rm_to_ea_yzx,   # 6
    _rm_to_ea_yzy,   # 7
    _rm_to_ea_zxy,   # 8
    _rm_to_ea_zxz,   # 9
    _rm_to_ea_zyx,   # 10
    _rm_to_ea_zyz,   # 11
]


def rotation_matrix_to_euler_angle(order_idx: jax.Array, R: jax.Array) -> jax.Array:
    """Extract Euler angles from a rotation matrix.

    Args:
        order_idx (jax.Array): Integer index 0--11 matching ``EulerAngleOrder``.
        R (jax.Array): Rotation matrix of shape ``(3, 3)``.

    Returns:
        jnp.ndarray: Array ``[phi, theta, psi]`` in radians.
    """
    branches = [lambda r, f=f: f(r) for f in _RM_TO_EA_BRANCHES]
    return jax.lax.switch(order_idx, branches, R)


# ---------------------------------------------------------------------------
# Quaternion multiplication and SLERP
# ---------------------------------------------------------------------------

def quaternion_multiply(q1: jax.Array, q2: jax.Array) -> jax.Array:
    """Hamilton product of two quaternions.

    Args:
        q1 (jax.Array): First quaternion of shape ``(4,)`` in scalar-first order.
        q2 (jax.Array): Second quaternion of shape ``(4,)`` in scalar-first order.

    Returns:
        jnp.ndarray: Product quaternion of shape ``(4,)``.
    """
    s1, v1 = q1[0], q1[1:]
    s2, v2 = q2[0], q2[1:]

    s = s1 * s2 - jnp.dot(v1, v2)
    v = s1 * v2 + s2 * v1 + jnp.cross(v1, v2)

    result = jnp.concatenate([jnp.array([s]), v])
    return result / jnp.linalg.norm(result)


def quaternion_slerp(q1: jax.Array, q2: jax.Array, t: float | jax.Array) -> jax.Array:
    """Spherical linear interpolation between two quaternions.

    Falls back to linear interpolation when the quaternions are nearly
    parallel (dot product > 0.9995).

    Args:
        q1 (jax.Array): Start quaternion of shape ``(4,)``.
        q2 (jax.Array): End quaternion of shape ``(4,)``.
        t (float | jax.Array): Interpolation parameter in ``[0, 1]``.

    Returns:
        jnp.ndarray: Interpolated quaternion of shape ``(4,)``.
    """
    dot = jnp.dot(q1, q2)

    # Flip sign if needed for shortest path
    q2_adj = jnp.where(dot < 0.0, -q2, q2)
    dot = jnp.abs(dot)

    def _linear_interp(_):
        qt = q1 + (q2_adj - q1) * t
        return qt / jnp.linalg.norm(qt)

    def _slerp_interp(_):
        theta_0 = jnp.arccos(jnp.clip(dot, -1.0, 1.0))
        theta = theta_0 * t
        s0 = jnp.cos(theta) - dot * jnp.sin(theta) / jnp.sin(theta_0)
        s1 = jnp.sin(theta) / jnp.sin(theta_0)
        qt = q1 * s0 + q2_adj * s1
        return qt / jnp.linalg.norm(qt)

    return jax.lax.cond(dot > 0.9995, _linear_interp, _slerp_interp, None)
