# Polyhedral Gravity

## Overview

The polyhedral gravity model computes the full gravitational field
(potential, acceleration, and gravity gradient tensor) of a homogeneous,
irregularly-shaped body represented as a triangulated surface mesh.
This is the standard approach for modelling asteroid gravity using shape
models from databases like DAMIT.

The implementation follows the Tsoulis (2012) line-integral method,
evaluated entirely in JAX for JIT compilation and automatic
differentiation.

## Basic Usage

The low-level function `polyhedral_gravity` operates in the body-fixed
frame and returns the full gravity field:

```python
import jax.numpy as jnp
from astrojax.orbit_dynamics import polyhedral_gravity

# Define a unit cube (8 vertices, 12 triangular faces)
vertices = jnp.array([
    [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
    [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
], dtype=jnp.float64)

faces = jnp.array([
    [1,3,2],[0,3,1],[0,1,5],[0,5,4],[0,7,3],[0,4,7],
    [1,2,6],[1,6,5],[2,3,6],[3,7,6],[4,5,6],[4,6,7],
])

# Evaluate at a point outside the body
r = jnp.array([5.0, 5.0, 5.0])
potential, acceleration, tensor = polyhedral_gravity(r, vertices, faces, density=1.0)
```

The returned quantities are:

| Output | Shape | Units | Description |
|--------|-------|-------|-------------|
| `potential` | scalar | m²/s² | Gravitational potential |
| `acceleration` | `(3,)` | m/s² | Attractive acceleration (toward body) |
| `tensor` | `(6,)` | 1/s² | Gradient tensor `[Vxx, Vyy, Vzz, Vxy, Vxz, Vyz]` |

## Inertial-Frame Acceleration

For orbit propagation, use `accel_polyhedral_gravity` which handles the
coordinate transformation between inertial and body-fixed frames:

```python
from astrojax.orbit_dynamics import accel_polyhedral_gravity

r_point = jnp.array([5.0, 5.0, 5.0])   # spacecraft position (inertial)
r_body = jnp.zeros(3)                    # body centre (inertial)
R = jnp.eye(3)                           # body-to-inertial rotation

a = accel_polyhedral_gravity(r_point, r_body, R, vertices, faces, density=1.0)
```

The function:

1. Transforms the computation point to the body frame: `r_bf = R.T @ (r_point - r_body)`
2. Evaluates `polyhedral_gravity` in the body frame
3. Rotates the acceleration back: `a_inertial = R @ a_body`

## Integration with DAMIT

For asteroid applications, combine with the DAMIT dataset functions:

```python
from astrojax.datasets import (
    damit_spin_to_rotation,
    scale_shape_vertices,
)
from astrojax.orbit_dynamics import accel_polyhedral_gravity

# Load shape model vertices and faces from DAMIT
# vertices, faces = ...  (from DAMIT provider)

# Scale vertices to physical size (e.g. 500 m max extent)
scaled_verts = scale_shape_vertices(vertices, 500.0)

# Compute body-to-ecliptic rotation at a given epoch
spin_params = jnp.array([30.0, 60.0, 5.0, 2460000.5, 0.0, 0.0])
R_body_to_ecl = damit_spin_to_rotation(spin_params, target_jd=2460001.5)

# Evaluate gravity at a spacecraft position
r_sc = jnp.array([1000.0, 0.0, 0.0])   # ecliptic frame
r_ast = jnp.zeros(3)                     # asteroid at origin
a = accel_polyhedral_gravity(r_sc, r_ast, R_body_to_ecl, scaled_verts, faces, density=2000.0)
```

## Precision

The default `float32` precision is sufficient for many applications, but
for high-accuracy reference comparisons use `float64`:

```python
from astrojax.config import set_dtype
import jax.numpy as jnp

set_dtype(jnp.float64)  # call before any JIT compilation
```

## JIT and vmap

Both functions are fully JIT-compatible.  Use `jax.vmap` to evaluate
gravity at multiple computation points in parallel:

```python
import jax

# Batch evaluation over multiple points
points = jnp.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])

@jax.jit
@jax.vmap
def batch_gravity(r):
    return polyhedral_gravity(r, vertices, faces, density=1.0)

pots, accels, tensors = batch_gravity(points)
```

The gradient of the potential with respect to position is available via
`jax.grad`, providing an independent computation of the acceleration:

```python
def potential_only(r):
    pot, _, _ = polyhedral_gravity(r, vertices, faces, density=1.0)
    return pot

grad_V = jax.grad(potential_only)(r)  # should equal -acceleration
```
