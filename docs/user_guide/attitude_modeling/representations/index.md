# Attitude Representations

The `astrojax.attitude_representations` module provides four
interconvertible 3D rotation representations, elementary rotation
functions, and pure conversion kernels. All types use JAX
primitives and are compatible with `jax.jit`, `jax.vmap`, and
`jax.grad`.

## Representation Types

| Type | Internal Storage | Degrees of Freedom | Page |
|------|------------------|--------------------|------|
| [`Quaternion`](quaternion.md) | `[w, x, y, z]` shape `(4,)` | 4 (unit constraint) | [Quaternion](quaternion.md) |
| [`RotationMatrix`](rotation_matrix.md) | shape `(3, 3)` | 9 (SO(3) constraint) | [Rotation Matrix](rotation_matrix.md) |
| [`EulerAngle`](euler_angles.md) | `(order, phi, theta, psi)` | 3 angles + order | [Euler Angles](euler_angles.md) |
| [`EulerAxis`](euler_axis.md) | `(axis(3,), angle)` | 3 axis + 1 angle | [Euler Axis](euler_axis.md) |

## Additional Components

| Component | Description | Page |
|-----------|-------------|------|
| Elementary Rotations | `Rx`, `Ry`, `Rz` axis-aligned rotation matrix constructors | [Elementary Rotations](elementary_rotations.md) |
| Conversions | Pure conversion kernels operating on raw JAX arrays | [Conversions](conversions.md) |

## Conversion Graph

All four types provide `to_*` instance methods and `from_*` class
methods for full interconvertibility. Most conversions route through
the quaternion representation internally:

```python
from astrojax import Quaternion, RotationMatrix, EulerAngle, EulerAngleOrder, EulerAxis

q = Quaternion(0.675, 0.42, 0.5, 0.71)

# Quaternion -> RotationMatrix -> EulerAngle -> EulerAxis -> Quaternion
r = q.to_rotation_matrix()
e = r.to_euler_angle(EulerAngleOrder.ZYX)
ea = e.to_euler_axis()
q_back = ea.to_quaternion()
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
