# Conversions

The `astrojax.attitude_representations.conversions` module contains
pure conversion functions that operate on raw JAX arrays.  These
kernels avoid circular imports between the class modules and serve
as the computational backbone for the `to_*` / `from_*` methods on
the representation classes.

## Module Design

The conversion functions accept and return plain `jax.Array` values
rather than class instances.  This separation keeps the conversion
math independent of the class hierarchy and allows direct use in
JIT-compiled pipelines without class overhead.

**Convention:**

- Quaternion layout: scalar-first `[w, x, y, z]`, shape `(4,)`
- Rotation matrix layout: row-major, shape `(3, 3)`
- Euler axis: `(axis(3,), angle_scalar)`

## Quaternion ↔ Rotation Matrix

### `quaternion_to_rotation_matrix`

Converts a unit quaternion to a 3x3 rotation matrix using the
bilinear product form:

```python
import jax.numpy as jnp
from astrojax.attitude_representations.conversions import quaternion_to_rotation_matrix

q = jnp.array([1.0, 0.0, 0.0, 0.0])  # identity
R = quaternion_to_rotation_matrix(q)    # shape (3, 3)
```

### `rotation_matrix_to_quaternion`

Converts a 3x3 rotation matrix to a unit quaternion using
Shepperd's method.  Uses `jax.lax.switch` on `argmax` for
numerical stability and JIT compatibility:

```python
from astrojax.attitude_representations.conversions import rotation_matrix_to_quaternion

R = jnp.eye(3)
q = rotation_matrix_to_quaternion(R)  # shape (4,)
```

!!! info "Shepperd's method"
    Shepperd's method computes four candidate trace values and
    selects the branch with the largest trace to avoid numerical
    instability from small denominators.  The `jax.lax.switch`
    dispatch makes this fully JIT-compatible.

## Euler Axis ↔ Quaternion

### `euler_axis_to_quaternion`

Converts an axis-angle pair to a quaternion using half-angle
trigonometry:

```python
from astrojax.attitude_representations.conversions import euler_axis_to_quaternion

axis = jnp.array([0.0, 0.0, 1.0])
angle = jnp.float32(0.7854)  # ~45 degrees
q = euler_axis_to_quaternion(axis, angle)
```

### `quaternion_to_euler_axis`

Extracts the axis and angle from a quaternion.  Returns a default
axis of `[1, 0, 0]` when the rotation angle is zero:

```python
from astrojax.attitude_representations.conversions import quaternion_to_euler_axis

q = jnp.array([1.0, 0.0, 0.0, 0.0])
axis, angle = quaternion_to_euler_axis(q)
```

## Euler Angle → Quaternion

### `euler_angle_to_quaternion`

Converts Euler angles to a quaternion.  Dispatches to one of 12
branch functions (one per `EulerAngleOrder`) using
`jax.lax.switch`:

```python
from astrojax.attitude_representations.conversions import euler_angle_to_quaternion

order_idx = jnp.int32(1)  # XYZ
phi = jnp.float32(0.5236)
theta = jnp.float32(0.7854)
psi = jnp.float32(1.0472)
q = euler_angle_to_quaternion(order_idx, phi, theta, psi)
```

The 12 internal branch functions correspond to:

| Index | Order | Index | Order |
|-------|-------|-------|-------|
| 0 | XYX | 6 | YZX |
| 1 | XYZ | 7 | YZY |
| 2 | XZX | 8 | ZXY |
| 3 | XZY | 9 | ZXZ |
| 4 | YXY | 10 | ZYX |
| 5 | YXZ | 11 | ZYZ |

## Rotation Matrix → Euler Angle

### `rotation_matrix_to_euler_angle`

Extracts Euler angles from a rotation matrix.  Dispatches to one of
12 branch functions matching the order index:

```python
from astrojax.attitude_representations.conversions import rotation_matrix_to_euler_angle

R = jnp.eye(3)
order_idx = jnp.int32(10)  # ZYX
angles = rotation_matrix_to_euler_angle(order_idx, R)  # [phi, theta, psi]
```

## Quaternion Operations

### `quaternion_multiply`

Hamilton product of two quaternions.  The result is normalized:

```python
from astrojax.attitude_representations.conversions import quaternion_multiply

q1 = jnp.array([1.0, 0.0, 0.0, 0.0])
q2 = jnp.array([0.5, 0.5, 0.5, 0.5])
q3 = quaternion_multiply(q1, q2)
```

### `quaternion_slerp`

Spherical linear interpolation between two quaternions.  Falls back
to linear interpolation when the quaternions are nearly parallel
(dot product > 0.9995):

```python
from astrojax.attitude_representations.conversions import quaternion_slerp

q_mid = quaternion_slerp(q1, q2, 0.5)  # halfway interpolation
```

The SLERP implementation:

1. Computes the dot product between the two quaternions
2. Flips the sign of `q2` if needed for shortest-path interpolation
3. Uses `jax.lax.cond` to branch between linear and spherical interpolation
