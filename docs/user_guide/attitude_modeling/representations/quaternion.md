# Quaternion

A unit quaternion in scalar-first convention `[w, x, y, z]`.  The
quaternion serves as the **central hub** for attitude conversions --
most cross-type conversions route through `Quaternion` internally.

## Construction

The constructor normalizes automatically:

```python
from astrojax import Quaternion

q = Quaternion(1.0, 0.0, 0.0, 0.0)   # identity rotation
q2 = Quaternion(1.0, 1.0, 1.0, 1.0)  # auto-normalized to unit length
```

### Factory Methods

```python
import jax.numpy as jnp

# From a 4-element vector (scalar-first by default)
q = Quaternion.from_vector(jnp.array([1.0, 0.0, 0.0, 0.0]))

# Scalar-last convention
q = Quaternion.from_vector(jnp.array([0.0, 0.0, 0.0, 1.0]), scalar_first=False)
```

## Properties

The four quaternion components are accessible as read-only
properties:

```python
q = Quaternion(0.5, 0.5, 0.5, 0.5)
print(q.w, q.x, q.y, q.z)
```

## Operations

### Hamilton Product

Quaternion multiplication implements the Hamilton product, composing
two rotations:

```python
q1 = Quaternion(1.0, 0.0, 0.0, 0.0)
q2 = Quaternion(0.5, 0.5, 0.5, 0.5)

q3 = q1 * q2  # Hamilton product
```

### Conjugate and Inverse

For a unit quaternion, the conjugate equals the inverse.  For
non-unit quaternions, `inverse()` divides by the squared norm:

```python
qc = q2.conjugate()  # [w, -x, -y, -z]
qi = q2.inverse()     # conjugate / norm^2
```

### Spherical Linear Interpolation (SLERP)

SLERP provides constant-speed interpolation along the shortest
great-circle arc on the unit sphere.  Falls back to linear
interpolation when quaternions are nearly parallel:

```python
q_mid = q1.slerp(q2, 0.5)  # halfway between q1 and q2
```

### Arithmetic

Addition, subtraction, and negation are also supported:

```python
q_sum = q1 + q2   # component-wise add, then renormalize
q_diff = q1 - q2  # component-wise subtract, then renormalize
q_neg = -q1       # negate vector part
```

## Conversions

```python
from astrojax import RotationMatrix, EulerAngle, EulerAngleOrder, EulerAxis

q = Quaternion(0.675, 0.42, 0.5, 0.71)

# To other representations
r = q.to_rotation_matrix()
e = q.to_euler_angle(EulerAngleOrder.ZYX)
ea = q.to_euler_axis()

# From other representations
q2 = Quaternion.from_rotation_matrix(r)
q3 = Quaternion.from_euler_angle(e)
q4 = Quaternion.from_euler_axis(ea)
```
