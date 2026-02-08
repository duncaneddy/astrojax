# Attitude Modeling

The attitude modeling subsystem in astrojax provides tools for
representing, converting, and propagating spacecraft orientations.
All implementations use JAX primitives and are compatible with
`jax.jit`, `jax.vmap`, and `jax.grad`.

## Subsystems

| Subsystem | Description |
|-----------|-------------|
| [Attitude Representations](representations/index.md) | Four interconvertible 3D rotation types (quaternion, rotation matrix, Euler angles, axis-angle) plus elementary rotation functions and conversion kernels |
| [Attitude Dynamics](dynamics.md) | Rigid-body attitude propagation including quaternion kinematics, Euler's equation, and gravity gradient torque |

## Architecture

Attitude representations are centered around the **Quaternion** as
the conversion hub.  Most cross-type conversions route through the
quaternion internally, matching the design of the Rust brahe
library.  All four representation types are registered as JAX
pytrees, enabling seamless use with JAX transformations.

The attitude dynamics module composes quaternion kinematics with
torque models (gravity gradient, etc.) into integrator-compatible
closures for propagation via `jax.lax.scan`.

```
Quaternion  <──>  RotationMatrix
    ^                   ^
    |                   |
    v                   v
EulerAxis   <──>  EulerAngle (12 orders)
```
