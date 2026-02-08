# Euler Axis (Axis-Angle)

A rotation defined by a unit axis vector and a rotation angle.  The
`EulerAxis` class stores a shape `(3,)` axis and a scalar angle,
both in the configured float dtype.  Angles are always stored in
radians internally.

## Construction

### From Axis Array and Angle

```python
import jax.numpy as jnp
from astrojax import EulerAxis

# 45-degree rotation about z-axis
ea = EulerAxis(jnp.array([0.0, 0.0, 1.0]), 45.0, use_degrees=True)
```

### From Individual Components

```python
ea = EulerAxis.from_values(0.0, 0.0, 1.0, 45.0, use_degrees=True)
```

### From a 4-Element Vector

```python
import jax.numpy as jnp

# Default: vector-first [x, y, z, angle]
ea = EulerAxis.from_vector(jnp.array([0.0, 0.0, 1.0, 0.7854]))

# Angle-first [angle, x, y, z]
ea = EulerAxis.from_vector(jnp.array([0.7854, 0.0, 0.0, 1.0]), vector_first=False)
```

## Properties

```python
ea = EulerAxis.from_values(0.0, 0.0, 1.0, 45.0, use_degrees=True)
print(ea.axis)   # [0.0, 0.0, 1.0]
print(ea.angle)  # ~0.7854 radians
```

## Serialization

Export to a 4-element vector:

```python
v = ea.to_vector()                           # [x, y, z, angle_rad]
v = ea.to_vector(use_degrees=True)           # [x, y, z, angle_deg]
v = ea.to_vector(vector_first=False)         # [angle_rad, x, y, z]
```

## Conversions

```python
from astrojax import Quaternion, RotationMatrix, EulerAngle, EulerAngleOrder

ea = EulerAxis.from_values(0.0, 0.0, 1.0, 45.0, use_degrees=True)

# To other representations
q = ea.to_quaternion()
r = ea.to_rotation_matrix()
e = ea.to_euler_angle(EulerAngleOrder.XYZ)

# From other representations
ea2 = EulerAxis.from_quaternion(q)
ea3 = EulerAxis.from_rotation_matrix(r)
ea4 = EulerAxis.from_euler_angle(e)
```

!!! note "Zero rotation"
    When the rotation angle is zero, the axis is undefined.  The
    conversion from quaternion returns a default axis of `[1, 0, 0]`
    in this case.
