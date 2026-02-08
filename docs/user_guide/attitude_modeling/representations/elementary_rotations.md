# Elementary Rotations

The `rotation_matrices` module provides three functions -- `Rx`,
`Ry`, `Rz` -- that construct 3x3 rotation matrices for rotations
about the principal axes.  These are the building blocks for
composed rotations and are used internally by
`RotationMatrix.rotation_x()`, `rotation_y()`, and `rotation_z()`.

## Functions

### Rx -- Rotation About the X-Axis

$$
R_x(\theta) = \begin{bmatrix}
1 & 0 & 0 \\
0 & \cos\theta & \sin\theta \\
0 & -\sin\theta & \cos\theta
\end{bmatrix}
$$

```python
from astrojax import Rx

# Angle in radians (default)
r = Rx(0.7854)

# Angle in degrees
r = Rx(45.0, use_degrees=True)
```

### Ry -- Rotation About the Y-Axis

$$
R_y(\theta) = \begin{bmatrix}
\cos\theta & 0 & -\sin\theta \\
0 & 1 & 0 \\
\sin\theta & 0 & \cos\theta
\end{bmatrix}
$$

```python
from astrojax import Ry

r = Ry(30.0, use_degrees=True)
```

### Rz -- Rotation About the Z-Axis

$$
R_z(\theta) = \begin{bmatrix}
\cos\theta & \sin\theta & 0 \\
-\sin\theta & \cos\theta & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

```python
from astrojax import Rz

r = Rz(60.0, use_degrees=True)
```

## Composing Rotations

Elementary rotations can be composed by matrix multiplication to
build arbitrary rotation sequences:

```python
import jax.numpy as jnp
from astrojax import Rx, Ry, Rz

# ZYX Euler rotation sequence
R = Rz(60.0, use_degrees=True) @ Ry(30.0, use_degrees=True) @ Rx(45.0, use_degrees=True)

# Rotate a vector
v = jnp.array([1.0, 0.0, 0.0])
v_rotated = R @ v
```

!!! note "Return type"
    `Rx`, `Ry`, and `Rz` return raw `jnp.ndarray` arrays (shape
    `(3, 3)`), not `RotationMatrix` instances.  Use
    `RotationMatrix.rotation_x()` etc. if you need the class wrapper.

## Convention

The rotation direction follows the right-hand rule: a positive
angle produces a counter-clockwise rotation when viewed from the
positive end of the axis looking toward the origin.
