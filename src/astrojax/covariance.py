"""Covariance propagation via variational equations.

Provides a generic mechanism to augment *any* ODE dynamics function with
State Transition Matrix (STM) propagation. The Jacobian ``A = ∂f/∂x`` is
computed automatically via ``jax.jacfwd``, so no hand-derived partial
derivatives are needed.

The augmented state vector is ``[x(n), vec(Φ)(n²)]`` and can be integrated
with any existing integrator (RK4, RKF45, DP54, etc.). This approach works
with orbit dynamics, attitude dynamics, relative motion, or any other
differentiable ODE system.

Functions:
    create_variational_dynamics: Augment a dynamics function with STM propagation.
    augmented_initial_state: Build the initial augmented state vector.
    extract_state_and_stm: Split an augmented state into (x, Φ).
    propagate_covariance: Map a covariance through the STM.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype


def create_variational_dynamics(
    dynamics: Callable[[ArrayLike, ArrayLike], Array],
    n: int,
) -> Callable[[ArrayLike, ArrayLike], Array]:
    """Augment a dynamics function with STM propagation.

    Returns a new dynamics function whose state is the original state
    concatenated with the vectorised STM. The Jacobian ``A = ∂f/∂x`` is
    computed inside the closure via ``jax.jacfwd``, so the user never
    needs to derive partials by hand.

    Args:
        dynamics: Original ODE right-hand side ``f(t, x) -> dx/dt``.
            Must be differentiable by JAX.
        n: Dimension of the original state vector ``x``.

    Returns:
        Augmented dynamics function ``f_aug(t, aug_state) -> d(aug_state)/dt``
        where ``aug_state`` has length ``n + n²``.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.covariance import (
            create_variational_dynamics,
            augmented_initial_state,
            extract_state_and_stm,
        )
        from astrojax.integrators import rk4_step

        def harmonic(t, x):
            return jnp.array([x[1], -x[0]])

        aug_dynamics = create_variational_dynamics(harmonic, n=2)
        aug_x0 = augmented_initial_state(jnp.array([1.0, 0.0]), n=2)
        result = rk4_step(aug_dynamics, 0.0, aug_x0, 0.01)
        x, Phi = extract_state_and_stm(result.state, n=2)
        ```
    """

    def aug_dynamics(t: ArrayLike, aug_state: ArrayLike) -> Array:
        dtype = get_dtype()
        aug_state = jnp.asarray(aug_state, dtype=dtype)

        # Split augmented state
        x = aug_state[:n]
        Phi_flat = aug_state[n:]
        Phi = Phi_flat.reshape(n, n)

        # Original derivative
        dx = dynamics(t, x)

        # Jacobian A = ∂f/∂x evaluated at current (t, x)
        A = jax.jacfwd(lambda x_: dynamics(t, x_))(x)

        # STM derivative: dΦ/dt = A @ Φ
        dPhi = A @ Phi
        dPhi_flat = dPhi.reshape(n * n)

        return jnp.concatenate([dx, dPhi_flat])

    return aug_dynamics


def augmented_initial_state(x0: ArrayLike, n: int) -> Array:
    """Build the initial augmented state vector.

    Concatenates the initial state ``x0`` with the vectorised identity
    matrix ``vec(I_n)``, since the STM at ``t=t₀`` is the identity.

    Args:
        x0: Initial state vector of shape ``(n,)``.
        n: Dimension of the state vector.

    Returns:
        Augmented state of shape ``(n + n²,)``.
    """
    dtype = get_dtype()
    x0 = jnp.asarray(x0, dtype=dtype)
    Phi0_flat = jnp.eye(n, dtype=dtype).reshape(n * n)
    return jnp.concatenate([x0, Phi0_flat])


def extract_state_and_stm(aug_state: ArrayLike, n: int) -> tuple[Array, Array]:
    """Split an augmented state into the state vector and STM.

    Args:
        aug_state: Augmented state of shape ``(n + n²,)``.
        n: Dimension of the original state vector.

    Returns:
        Tuple ``(x, Phi)`` where ``x`` has shape ``(n,)`` and ``Phi``
        has shape ``(n, n)``.
    """
    dtype = get_dtype()
    aug_state = jnp.asarray(aug_state, dtype=dtype)
    x = aug_state[:n]
    Phi = aug_state[n:].reshape(n, n)
    return x, Phi


def propagate_covariance(
    Phi: ArrayLike,
    P0: ArrayLike,
    Q: ArrayLike | None = None,
) -> Array:
    """Propagate a covariance matrix through the STM.

    Computes ``P = Φ P₀ Φᵀ + Q`` where ``Q`` defaults to zero.

    Args:
        Phi: State transition matrix of shape ``(n, n)``.
        P0: Initial covariance matrix of shape ``(n, n)``.
        Q: Process noise covariance of shape ``(n, n)``. Defaults to zero.

    Returns:
        Propagated covariance matrix of shape ``(n, n)``.
    """
    dtype = get_dtype()
    Phi = jnp.asarray(Phi, dtype=dtype)
    P0 = jnp.asarray(P0, dtype=dtype)
    P = Phi @ P0 @ Phi.T
    if Q is not None:
        P = P + jnp.asarray(Q, dtype=dtype)
    return P
