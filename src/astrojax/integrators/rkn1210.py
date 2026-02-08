"""Runge-Kutta-Nyström 12(10) adaptive integrator (RKN1210).

Implements the Dormand-El-Mikkawy-Prince embedded Runge-Kutta-Nyström method
with a 12th-order solution for propagation and a 10th-order solution for error
estimation. The method uses 17 stages per step.

RKN methods are specialized for **second-order ODEs** of the form ``y'' = f(t, y)``,
making them highly efficient for orbital mechanics and other problems with this
structure. The state is split into position and velocity halves, and only the
acceleration (second derivative) is computed at each stage. This gives better
accuracy per function evaluation than standard RK methods for second-order systems.

Despite this internal specialization, the **public API matches the other
integrators exactly**: ``dynamics(t, state) -> state_dot``. The function
internally extracts only the acceleration half, so users can swap
``dp54_step`` for ``rkn1210_step`` with zero changes to their dynamics function.

Coefficients from: Dormand, El-Mikkawy, & Prince (1987), "High-Order Embedded
Runge-Kutta-Nyström Formulae". Based on implementation by Rody Oldenhuis
(FEX-RKN1210), used under BSD 2-Clause License.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype
from astrojax.integrators._adaptive import compute_error_norm, compute_next_step_size
from astrojax.integrators._types import AdaptiveConfig, StepResult

# ──────────────────────────────────────────────
# Butcher tableau coefficients (Python tuples, cast at call time)
# ──────────────────────────────────────────────

# Nodes (c) — 17 stages
_C = (
    0.0,
    2.0e-2,
    4.0e-2,
    1.0e-1,
    1.333333333333333333333e-1,
    1.6e-1,
    5.0e-2,
    2.0e-1,
    2.5e-1,
    3.333333333333333333333e-1,
    5.0e-1,
    5.555555555555555555556e-1,
    7.5e-1,
    8.571428571428571428571e-1,
    9.452162222720143401300e-1,
    1.0,
    1.0,
)

# Coupling coefficients (lower-triangular rows)
# _A[i] has i entries for stages j = 0..i-1
_A = (
    (),  # stage 0: no dependencies
    # Row 2
    (2.0e-4,),
    # Row 3
    (2.666666666666666666667e-4, 5.333333333333333333333e-4),
    # Row 4
    (2.916666666666666666667e-3, -4.166666666666666666667e-3, 6.25e-3),
    # Row 5
    (1.646090534979423868313e-3, 0.0, 5.486968449931412894376e-3,
     1.755829903978052126200e-3),
    # Row 6
    (1.9456e-3, 0.0, 7.151746031746031746032e-3,
     2.912711111111111111111e-3, 7.899428571428571428571e-4),
    # Row 7
    (5.6640625e-4, 0.0, 8.809730489417989417989e-4,
     -4.369212962962962962963e-4, 3.390066964285714285714e-4,
     -9.946469907407407407407e-5),
    # Row 8
    (3.083333333333333333333e-3, 0.0, 0.0,
     1.777777777777777777778e-3, 2.7e-3,
     1.578282828282828282828e-3, 1.086060606060606060606e-2),
    # Row 9
    (3.651839374801129713751e-3, 0.0, 3.965171714072343066176e-3,
     3.197258262930628223501e-3, 8.221467306855435369687e-3,
     -1.313092695957237983620e-3, 9.771586968064867815626e-3,
     3.755769069232833794879e-3),
    # Row 10
    (3.707241068718500810196e-3, 0.0, 5.082045854555285980761e-3,
     1.174708002175412044736e-3, -2.114762991512699149962e-2,
     6.010463698107880812226e-2, 2.010573476850618818467e-2,
     -2.835075012293358084304e-2, 1.487956891858193275559e-2),
    # Row 11
    (3.512537656073344153113e-2, 0.0, -8.615749195138479103406e-3,
     -5.791448051007916521676e-3, 1.945554823782615842394e0,
     -3.435123867456513596368e0, -1.093070110747522175839e-1,
     2.349638311899516639432e0, -7.560094086870229780272e-1,
     1.095289722215692642465e-1),
    # Row 12
    (2.052779253748249665097e-2, 0.0, -7.286446764480179917782e-3,
     -2.115355607961840240693e-3, 9.275807968723522242568e-1,
     -1.652282484425736679073e0, -2.107956300568656981919e-2,
     1.206536432620787154477e0, -4.137144770010661413247e-1,
     9.079873982809653759568e-2, 5.355552600533985049169e-3),
    # Row 13
    (-1.432407887554551504589e-1, 0.0, 1.252870377309181727785e-2,
     6.826019163969827128681e-3, -4.799555395574387265502e0,
     5.698625043951941433792e0, 7.553430369523645222494e-1,
     -1.275548785828108371754e-1, -1.960592605111738432891e0,
     9.185609056635262409762e-1, -2.388008550528443105348e-1,
     1.591108135723421551387e-1),
    # Row 14
    (8.045019205520489486972e-1, 0.0, -1.665852706701124517785e-2,
     -2.141583404262973481173e-2, 1.682723592896246587020e1,
     -1.117283535717609792679e1, -3.377159297226323741489e0,
     -1.524332665536084564618e1, 1.717983573821541656202e1,
     -5.437719239823994645354e0, 1.387867161836465575513e0,
     -5.925827732652811653477e-1, 2.960387317129735279616e-2),
    # Row 15
    (-9.132967666973580820963e-1, 0.0, 2.411272575780517839245e-3,
     1.765812269386174198207e-2, -1.485164977972038382461e1,
     2.158970867004575600308e0, 3.997915583117879901153e0,
     2.843415180023223189845e1, -2.525936435494159843788e1,
     7.733878542362237365534e0, -1.891302894847867461038e0,
     1.001484507022471780367e0, 4.641199599109051905105e-3,
     1.121875502214895703398e-2),
    # Row 16
    (-2.751962972055939382061e-1, 0.0, 3.661188877915492013423e-2,
     9.789519688231562624651e-3, -1.229306234588621030421e1,
     1.420722645393790269429e1, 1.586647690678953683225e0,
     2.457773532759594543903e0, -8.935193694403271905523e0,
     4.373672731613406948393e0, -1.834718176544949163043e0,
     1.159208528906149120781e0, -1.729025316538392215180e-2,
     1.932597790446076667276e-2, 5.204442937554993111849e-3),
    # Row 17
    (1.307639184740405758800e0, 0.0, 1.736410918974584186709e-2,
     -1.854445645426579502436e-2, 1.481152203286772689685e1,
     9.383176308482470907879e0, -5.228426199944542254147e0,
     -4.895128052584765080401e1, 3.829709603433792256258e1,
     -1.058738133697597970916e1, 2.433230437622627635851e0,
     -1.045340604257544428487e0, 7.177320950867259451982e-2,
     2.162210970808278269055e-3, 7.009595759602514236993e-3,
     0.0),
)

# High-order position weights (12th order)
_B_POS_HIGH = (
    1.212786851718541497689e-2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    8.629746251568874443638e-2,
    2.525469581187147194323e-1,
    -1.974186799326823033583e-1,
    2.031869190789725908093e-1,
    -2.077580807771491661219e-2,
    1.096780487450201362501e-1,
    3.806513252646650573449e-2,
    1.163406880432422964409e-2,
    4.658029704024878686936e-3,
    0.0,
    0.0,
)

# High-order velocity weights (12th order)
_B_VEL_HIGH = (
    1.212786851718541497689e-2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    9.083943422704078361724e-2,
    3.156836976483933992904e-1,
    -2.632249065769097378111e-1,
    3.047803786184588862139e-1,
    -4.155161615542983322439e-2,
    2.467756096762953065628e-1,
    1.522605301058660229380e-1,
    8.143848163026960750865e-2,
    8.502571193890811280080e-2,
    -9.155189630077962873141e-3,
    2.5e-2,
)

# Low-order position weights (10th order)
_B_POS_LOW = (
    1.700870190700699175275e-2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    7.225933593083140694886e-2,
    3.720261773267530453882e-1,
    -4.018211450093035214393e-1,
    3.354550683013516666966e-1,
    -1.313065010753318084303e-1,
    1.894319066160486527227e-1,
    2.684080204002904790537e-2,
    1.630566560591792389352e-2,
    3.799988356696594561666e-3,
    0.0,
    0.0,
)

# Low-order velocity weights (10th order)
_B_VEL_LOW = (
    1.700870190700699175275e-2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    7.606245887455937573564e-2,
    4.650327216584413067353e-1,
    -5.357615266790713619191e-1,
    5.031826024520275000449e-1,
    -2.626130021506636168606e-1,
    4.262217898861094686260e-1,
    1.073632081601161916215e-1,
    1.141396592414254672546e-1,
    6.936338665004867700906e-2,
    2.0e-2,
    0.0,
)


def rkn1210_step(
    dynamics: Callable[[ArrayLike, ArrayLike], Array],
    t: ArrayLike,
    state: ArrayLike,
    dt: ArrayLike,
    config: Optional[AdaptiveConfig] = None,
    control: Optional[Callable[[ArrayLike, ArrayLike], Array]] = None,
) -> StepResult:
    """Perform a single adaptive RKN1210 integration step.

    Advances the state from time ``t`` by up to ``dt`` using the
    Runge-Kutta-Nyström 12(10) method with adaptive step-size control. This
    integrator exploits the second-order structure of ``y'' = f(t, y)``
    problems internally, while accepting the standard ``dynamics(t, state)``
    interface.

    The state vector must have an **even** number of elements, split as
    ``[position, velocity]``. The dynamics function must return
    ``[velocity, acceleration]``. Only the acceleration half is used
    internally for stage computation.

    Compatible with ``jax.jit`` and ``jax.vmap``. Not compatible with
    reverse-mode ``jax.grad`` due to the internal ``lax.while_loop``.

    Args:
        dynamics: ODE right-hand side function ``f(t, x) -> dx/dt``.
            Must return ``[velocity, acceleration]`` for a state vector
            ``[position, velocity]``.
        t: Current time.
        state: Current state vector. Must be even-length with the first
            half representing positions and the second half velocities.
        dt: Requested timestep. May be negative for backward integration.
            The actual timestep used may be smaller if the adaptive
            controller rejects the initial attempt.
        config: Adaptive step-size configuration. Uses default
            :class:`AdaptiveConfig` if ``None``.
        control: Optional additive control function ``u(t, x) -> force``.
            When provided, the effective derivative is
            ``f(t, x) + u(t, x)``.

    Returns:
        StepResult: Named tuple with fields:
            - ``state``: State at ``t + dt_used``.
            - ``dt_used``: Actual timestep taken (<= ``|dt|``).
            - ``error_estimate``: Normalized error of the accepted step.
            - ``dt_next``: Suggested timestep for the next step.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.integrators import rkn1210_step
        def harmonic(t, x):
            return jnp.array([x[1], -x[0]])
        result = rkn1210_step(harmonic, 0.0, jnp.array([1.0, 0.0]), 0.1)
        result.state  # ~[cos(0.1), -sin(0.1)]
        ```
    """
    if config is None:
        config = AdaptiveConfig()

    dtype = get_dtype()
    t = jnp.asarray(t, dtype=dtype)
    state = jnp.asarray(state, dtype=dtype)
    dt = jnp.asarray(dt, dtype=dtype)

    half_dim = state.shape[0] // 2

    def f(ti, xi):
        dx = dynamics(ti, xi)
        if control is not None:
            dx = dx + control(ti, xi)
        return dx

    def _attempt_step(h):
        """Compute one RKN1210 trial step with step size h."""
        pos = state[:half_dim]
        vel = state[half_dim:]
        h2 = h * h

        # Stage accelerations stored as a list (unrolled by XLA)
        k_list = []

        for i in range(17):
            # Compute position perturbation: h² * Σ(a[i,j] * k[j])
            pos_pert = jnp.zeros(half_dim, dtype=dtype)
            a_row = _A[i]
            for j in range(len(a_row)):
                pos_pert = pos_pert + a_row[j] * k_list[j]

            # Stage position: pos + c[i]*h*vel + h²*pos_pert
            stage_pos = pos + _C[i] * h * vel + h2 * pos_pert

            # Reconstruct full state for dynamics evaluation
            stage_state = jnp.concatenate([stage_pos, vel])

            # Evaluate dynamics and extract acceleration
            state_dot = f(t + _C[i] * h, stage_state)
            accel = state_dot[half_dim:]
            k_list.append(accel)

        # Assemble solutions using the four weight vectors
        pos_update_high = jnp.zeros(half_dim, dtype=dtype)
        vel_update_high = jnp.zeros(half_dim, dtype=dtype)
        pos_update_low = jnp.zeros(half_dim, dtype=dtype)
        vel_update_low = jnp.zeros(half_dim, dtype=dtype)

        for i in range(17):
            pos_update_high = pos_update_high + _B_POS_HIGH[i] * k_list[i]
            vel_update_high = vel_update_high + _B_VEL_HIGH[i] * k_list[i]
            pos_update_low = pos_update_low + _B_POS_LOW[i] * k_list[i]
            vel_update_low = vel_update_low + _B_VEL_LOW[i] * k_list[i]

        pos_high = pos + h * vel + h2 * pos_update_high
        vel_high = vel + h * vel_update_high
        pos_low = pos + h * vel + h2 * pos_update_low
        vel_low = vel + h * vel_update_low

        state_high = jnp.concatenate([pos_high, vel_high])
        state_low = jnp.concatenate([pos_low, vel_low])

        error_vec = state_high - state_low
        error = compute_error_norm(
            error_vec, state_high, state, config.abs_tol, config.rel_tol
        )
        return state_high, error

    # Adaptive step-rejection loop via lax.while_loop.
    # Carry: (h, attempts, accepted, state_out, error_out)
    def cond_fn(carry):
        _h, attempts, accepted, _state_out, _error_out = carry
        return (~accepted) & (attempts < config.max_step_attempts)

    def body_fn(carry):
        h, attempts, _accepted, _state_out, _error_out = carry
        state_new, error = _attempt_step(h)

        at_min_step = jnp.abs(h) <= config.min_step
        step_accepted = (error <= 1.0) | at_min_step

        # If rejected, shrink step size for next attempt (10th-order error)
        h_reduced = compute_next_step_size(
            error, h, 9.0, config.safety_factor,
            config.min_scale_factor, config.max_scale_factor,
            config.min_step, config.max_step,
        )
        h_next = jnp.where(step_accepted, h, h_reduced)

        return (h_next, attempts + 1, step_accepted, state_new, error)

    init_carry = (
        dt,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
        state,
        jnp.asarray(jnp.inf, dtype=dtype),
    )

    h_final, _attempts, _accepted, state_out, error_out = jax.lax.while_loop(
        cond_fn, body_fn, init_carry
    )

    # Compute suggested next step size from accepted error (12th-order solution)
    dt_next = compute_next_step_size(
        error_out, h_final, 11.0, config.safety_factor,
        config.min_scale_factor, config.max_scale_factor,
        config.min_step, config.max_step,
    )

    return StepResult(
        state=state_out,
        dt_used=h_final,
        error_estimate=error_out,
        dt_next=dt_next,
    )
