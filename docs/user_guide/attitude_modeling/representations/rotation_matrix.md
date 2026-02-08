# Rotation Matrix

A 3x3 direction cosine matrix (DCM) in SO(3).  The `RotationMatrix`
class wraps a `(3, 3)` JAX array and validates that the matrix is
orthogonal with determinant +1 on construction.

## Construction

### From Nine Elements

Pass all nine matrix elements in row-major order:

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

### From a 3x3 Array

```python
import jax.numpy as jnp

data = jnp.eye(3)
r = RotationMatrix.from_matrix(data)

# Skip SO(3) validation when the matrix is known to be valid
r = RotationMatrix.from_matrix(data, validate=False)
```

### Axis Rotation Factory Methods

The class provides factory methods that delegate to the
[elementary rotation functions](elementary_rotations.md):

```python
rx = RotationMatrix.rotation_x(45.0, use_degrees=True)
ry = RotationMatrix.rotation_y(30.0, use_degrees=True)
rz = RotationMatrix.rotation_z(60.0, use_degrees=True)
```

## Properties

All nine matrix elements are accessible as read-only properties
`r11` through `r33`:

```python
r = RotationMatrix.rotation_x(45.0, use_degrees=True)
print(r.r11, r.r12, r.r13)  # first row
```

The underlying array can be retrieved with `to_matrix()`:

```python
data = r.to_matrix()  # shape (3, 3) JAX array
```

## Composition

Matrix multiplication composes rotations or rotates vectors:

```python
import jax.numpy as jnp

rx = RotationMatrix.rotation_x(45.0, use_degrees=True)
ry = RotationMatrix.rotation_y(30.0, use_degrees=True)

# Compose two rotations
r_combined = rx * ry

# Rotate a vector
v = jnp.array([1.0, 0.0, 0.0])
v_rotated = r_combined * v
```

## SO(3) Validation

The constructor and `from_matrix(validate=True)` check that:

- $R^T R \approx I$ (orthogonality)
- $\det(R) \approx +1$ (proper rotation, not a reflection)

If validation fails, a `ValueError` is raised.  Use
`_from_internal()` or `from_matrix(validate=False)` when the matrix
is already known to be valid (e.g., from conversion outputs or
pytree unflatten).

## Conversions

```python
from astrojax import Quaternion, EulerAngle, EulerAngleOrder, EulerAxis

r = RotationMatrix.rotation_z(45.0, use_degrees=True)

# To other representations
q = r.to_quaternion()
e = r.to_euler_angle(EulerAngleOrder.ZYX)
ea = r.to_euler_axis()

# From other representations
r2 = RotationMatrix.from_quaternion(q)
r3 = RotationMatrix.from_euler_angle(e)
r4 = RotationMatrix.from_euler_axis(ea)
```
