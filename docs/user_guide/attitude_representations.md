# Attitude Representations

The `astrojax.attitude_representations` module provides four
interconvertible 3D rotation representations. All types use JAX
primitives and are compatible with `jax.jit`, `jax.vmap`, and
`jax.grad`.

## Representation Types

| Type | Internal Storage | Degrees of Freedom |
|------|------------------|--------------------|
| `Quaternion` | `[w, x, y, z]` shape `(4,)` | 4 (unit constraint) |
| `RotationMatrix` | shape `(3, 3)` | 9 (SO(3) constraint) |
| `EulerAngle` | `(order, phi, theta, psi)` | 3 angles + order |
| `EulerAxis` | `(axis(3,), angle)` | 3 axis + 1 angle |

## Quaternion

A unit quaternion in scalar-first convention. The constructor
normalizes automatically:

```python
from astrojax import Quaternion

q = Quaternion(1.0, 0.0, 0.0, 0.0)   # identity rotation
q2 = Quaternion(1.0, 1.0, 1.0, 1.0)  # auto-normalized to unit length
```

### Factory methods

```python
import jax.numpy as jnp

# From a 4-element vector (scalar-first by default)
q = Quaternion.from_vector(jnp.array([1.0, 0.0, 0.0, 0.0]))

# Scalar-last convention
q = Quaternion.from_vector(jnp.array([0.0, 0.0, 0.0, 1.0]), scalar_first=False)
```

### Operations

```python
q1 = Quaternion(1.0, 0.0, 0.0, 0.0)
q2 = Quaternion(0.5, 0.5, 0.5, 0.5)

# Hamilton product
q3 = q1 * q2

# Conjugate and inverse
qc = q2.conjugate()
qi = q2.inverse()

# Spherical linear interpolation
q_mid = q1.slerp(q2, 0.5)
```

## Rotation Matrix

A 3×3 direction cosine matrix (DCM) in SO(3). The constructor
validates that the matrix is orthogonal with determinant +1:

```python
import math
from astrojax import RotationMatrix

s2 = math.sqrt(2.0) / 2.0
r = RotationMatrix(
    1.0, 0.0, 0.0,
    0.0, s2,  s2,
    0.0, -s2, s2,
)
```

### Axis rotations

The class provides factory methods that delegate to the existing
`Rx`, `Ry`, `Rz` functions:

```python
rx = RotationMatrix.rotation_x(45.0, use_degrees=True)
ry = RotationMatrix.rotation_y(30.0, use_degrees=True)
rz = RotationMatrix.rotation_z(60.0, use_degrees=True)
```

### Composition

Matrix multiplication composes rotations or rotates vectors:

```python
import jax.numpy as jnp

# Compose two rotations
r_combined = rx * ry

# Rotate a vector
v = jnp.array([1.0, 0.0, 0.0])
v_rotated = r_combined * v
```

## Euler Angles

Three successive rotations about specified axes. The 12 standard
rotation sequences are defined by `EulerAngleOrder`:

```python
from astrojax import EulerAngle, EulerAngleOrder

# XYZ Tait-Bryan angles (roll-pitch-yaw)
e = EulerAngle(EulerAngleOrder.XYZ, 30.0, 45.0, 60.0, use_degrees=True)

# Access angles (always in radians internally)
print(e.phi, e.theta, e.psi)
```

### Available orders

| Symmetric | Tait-Bryan |
|-----------|------------|
| XYX, XZX | XYZ, XZY |
| YXY, YZY | YXZ, YZX |
| ZXZ, ZYZ | ZXY, ZYX |

## Euler Axis (Axis-Angle)

A rotation defined by a unit axis and an angle:

```python
import jax.numpy as jnp
from astrojax import EulerAxis

# 45-degree rotation about z-axis
ea = EulerAxis(jnp.array([0.0, 0.0, 1.0]), 45.0, use_degrees=True)

# From individual components
ea = EulerAxis.from_values(0.0, 0.0, 1.0, 45.0, use_degrees=True)
```

## Conversions

All four types provide `to_*` instance methods and `from_*` class
methods for full interconvertibility. Most conversions route through
the quaternion representation internally:

```python
from astrojax import Quaternion, RotationMatrix, EulerAngle, EulerAngleOrder, EulerAxis

q = Quaternion(0.675, 0.42, 0.5, 0.71)

# Quaternion -> RotationMatrix
r = q.to_rotation_matrix()

# RotationMatrix -> EulerAngle (requires order)
e = r.to_euler_angle(EulerAngleOrder.ZYX)

# EulerAngle -> EulerAxis
ea = e.to_euler_axis()

# EulerAxis -> Quaternion (round-trip)
q_back = ea.to_quaternion()
```

The `from_*` factory methods provide the same conversions in the
opposite direction:

```python
r = RotationMatrix.from_quaternion(q)
e = EulerAngle.from_rotation_matrix(r, EulerAngleOrder.XYZ)
ea = EulerAxis.from_euler_angle(e)
q2 = Quaternion.from_euler_axis(ea)
```

## JAX Compatibility

All four types are registered as JAX pytrees and work with `jax.jit`:

```python
import jax

@jax.jit
def rotate_and_convert(q):
    r = q.to_rotation_matrix()
    return r.to_euler_angle(EulerAngleOrder.XYZ)

q = Quaternion(0.675, 0.42, 0.5, 0.71)
e = rotate_and_convert(q)
```

The pytree structure for each type:

| Type | Leaves | Auxiliary |
|------|--------|----------|
| `Quaternion` | `(data,)` | `None` |
| `RotationMatrix` | `(data,)` | `None` |
| `EulerAngle` | `(phi, theta, psi)` | `order` |
| `EulerAxis` | `(axis, angle)` | `None` |
