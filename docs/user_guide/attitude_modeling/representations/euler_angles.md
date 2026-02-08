# Euler Angles

Three successive rotations about specified axes.  The `EulerAngle`
class stores three angles `(phi, theta, psi)` and a rotation
`order` specifying which axes are used and in what sequence.

## Rotation Orders

The 12 standard rotation sequences are defined by the
`EulerAngleOrder` enum.  Values are contiguous integers 0--11,
suitable for use as branch indices in `jax.lax.switch` for
JIT-compatible dispatch.

### Symmetric Sequences

In symmetric (proper Euler) sequences, the first and third axes
are the same:

| Order | Axes | Index |
|-------|------|-------|
| `XYX` | X-Y-X | 0 |
| `XZX` | X-Z-X | 2 |
| `YXY` | Y-X-Y | 4 |
| `YZY` | Y-Z-Y | 7 |
| `ZXZ` | Z-X-Z | 9 |
| `ZYZ` | Z-Y-Z | 11 |

### Tait-Bryan Sequences

In Tait-Bryan sequences, all three axes are distinct:

| Order | Axes | Index | Also known as |
|-------|------|-------|---------------|
| `XYZ` | X-Y-Z | 1 | Roll-Pitch-Yaw |
| `XZY` | X-Z-Y | 3 | |
| `YXZ` | Y-X-Z | 5 | |
| `YZX` | Y-Z-X | 6 | |
| `ZXY` | Z-X-Y | 8 | |
| `ZYX` | Z-Y-X | 10 | Yaw-Pitch-Roll |

## Construction

```python
from astrojax import EulerAngle, EulerAngleOrder

# XYZ Tait-Bryan angles (roll-pitch-yaw)
e = EulerAngle(EulerAngleOrder.XYZ, 30.0, 45.0, 60.0, use_degrees=True)

# Angles in radians (default)
import math
e = EulerAngle(EulerAngleOrder.ZYX, math.pi/6, math.pi/4, math.pi/3)
```

### From a Vector

```python
import jax.numpy as jnp

vec = jnp.array([0.5236, 0.7854, 1.0472])  # radians
e = EulerAngle.from_vector(vec, EulerAngleOrder.XYZ)
```

## Properties

Angles are stored in radians internally:

```python
e = EulerAngle(EulerAngleOrder.XYZ, 30.0, 45.0, 60.0, use_degrees=True)
print(e.order)  # EulerAngleOrder.XYZ
print(e.phi)    # ~0.5236 rad
print(e.theta)  # ~0.7854 rad
print(e.psi)    # ~1.0472 rad
```

## Order Conversion

Convert between different rotation orders by going through the
quaternion representation:

```python
e_xyz = EulerAngle(EulerAngleOrder.XYZ, 30.0, 45.0, 60.0, use_degrees=True)
e_zyx = e_xyz.to_euler_angle(EulerAngleOrder.ZYX)
```

## Conversions

```python
from astrojax import Quaternion, RotationMatrix, EulerAxis

e = EulerAngle(EulerAngleOrder.XYZ, 30.0, 45.0, 60.0, use_degrees=True)

# To other representations
q = e.to_quaternion()
r = e.to_rotation_matrix()
ea = e.to_euler_axis()

# From other representations
e2 = EulerAngle.from_quaternion(q, EulerAngleOrder.XYZ)
e3 = EulerAngle.from_rotation_matrix(r, EulerAngleOrder.XYZ)
e4 = EulerAngle.from_euler_axis(ea, EulerAngleOrder.XYZ)
```

!!! warning "Gimbal lock"
    Euler angles suffer from gimbal lock when the second rotation
    angle `theta` approaches $\pm\pi/2$ (Tait-Bryan) or $0$ / $\pi$
    (symmetric).  Near these singularities, `phi` and `psi` become
    ambiguous.  Consider using quaternions for singularity-free
    representation.
