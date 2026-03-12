# %% [markdown]
# Propagate orbit covariance using variational equations.
#
# Demonstrates covariance propagation for a 500 km circular LEO orbit using
# two-body dynamics. The STM is integrated alongside the state via augmented
# variational equations, then used to map the initial covariance forward.
#
# Produces two plots:
#   1. Position 1-sigma uncertainties (R, T, N) over one orbit
#   2. Velocity 1-sigma uncertainties (R, T, N) over one orbit
#
# No external data is needed beyond zero EOP (two-body only).
#
# Usage:
#     uv run examples/propagate_covariance.py

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from astrojax import set_dtype
from astrojax.config import get_dtype
from astrojax.constants import GM_EARTH, R_EARTH
from astrojax.covariance import (
    augmented_initial_state,
    create_variational_dynamics,
    extract_state_and_stm,
    propagate_covariance,
)
from astrojax.eop import zero_eop
from astrojax.epoch import Epoch
from astrojax.integrators import rk4_step
from astrojax.orbit_dynamics.factory import create_orbit_dynamics
from astrojax.orbits import orbital_period
from astrojax.relative_motion import rotation_eci_to_rtn

# %% ── Setup ─────────────────────────────────────────────────────────────────

set_dtype(jnp.float64)

# 500 km circular LEO
altitude = 500e3  # m
a = R_EARTH + altitude
v_circular = jnp.sqrt(GM_EARTH / a)

r0 = jnp.array([a, 0.0, 0.0])
v0 = jnp.array([0.0, v_circular, 0.0])
x0 = jnp.concatenate([r0, v0])

# Initial covariance: 100 m position, 0.1 m/s velocity (1-sigma)
P0 = jnp.diag(jnp.array([100.0**2, 100.0**2, 100.0**2, 0.1**2, 0.1**2, 0.1**2]))

# Integration parameters
T_orbit = orbital_period(a)
dt = 10.0  # seconds
n_steps = int(float(T_orbit) / float(dt))

print(f"Orbit: {altitude / 1e3:.0f} km circular LEO")
print(f"Period: {float(T_orbit):.1f} s ({float(T_orbit) / 60:.1f} min)")
print(f"Integration: dt={dt:.0f} s, {n_steps} steps")

# %% ── Build dynamics ────────────────────────────────────────────────────────

epoch_0 = Epoch(2026, 3, 11, 0, 0, 0.0)
eop = zero_eop()
dynamics = create_orbit_dynamics(eop, epoch_0)

# %% ── Augment with variational equations ────────────────────────────────────

n = 6  # state dimension
aug_dynamics = create_variational_dynamics(dynamics, n)
aug_x0 = augmented_initial_state(x0, n)

# %% ── Propagate one orbit (recording trajectory) ───────────────────────────

dtype = get_dtype()


def scan_step(carry, _):
    t, aug_state = carry
    result = rk4_step(aug_dynamics, t, aug_state, dt)
    t_next = t + jnp.asarray(dt, dtype=dtype)
    return (t_next, result.state), result.state


print("\nCompiling and propagating (first call triggers XLA)...")
init_carry = (jnp.asarray(0.0, dtype=dtype), aug_x0)
(t_final, aug_final), aug_history = jax.lax.scan(scan_step, init_carry, None, length=n_steps)

# aug_history has shape (n_steps, n + n*n)
# Prepend initial state for a complete trajectory
aug_history = jnp.concatenate([aug_x0[None, :], aug_history], axis=0)
times = np.arange(n_steps + 1) * float(dt)

x_final, Phi = extract_state_and_stm(aug_final, n)

# %% ── Compute RTN covariance at each timestep ───────────────────────────────


def covariance_rtn_at_step(aug_state):
    """Compute RTN 1-sigma from augmented state."""
    x_i, Phi_i = extract_state_and_stm(aug_state, n)
    P_eci = propagate_covariance(Phi_i, P0)

    # Build 6x6 rotation: T = [[R, 0], [0, R]]
    R_eci2rtn = rotation_eci_to_rtn(x_i)
    T = jnp.zeros((6, 6), dtype=dtype)
    T = T.at[:3, :3].set(R_eci2rtn)
    T = T.at[3:, 3:].set(R_eci2rtn)

    P_rtn = T @ P_eci @ T.T
    return jnp.sqrt(jnp.diag(P_rtn))


sigma_rtn_history = jax.vmap(covariance_rtn_at_step)(aug_history)
# Shape: (n_steps+1, 6) — columns are [sigma_R, sigma_T, sigma_N, sigma_vR, sigma_vT, sigma_vN]

# Convert to numpy for plotting
sigma_rtn = np.asarray(sigma_rtn_history)
t_minutes = times / 60.0

# %% ── Print final results ───────────────────────────────────────────────────

P_final = propagate_covariance(Phi, P0)
sigma_0 = jnp.sqrt(jnp.diag(P0))
sigma_f = jnp.sqrt(jnp.diag(P_final))

print(f"\nPropagated {float(t_final):.1f} s ({float(t_final) / 60:.1f} min)")

r_final_mag = jnp.linalg.norm(x_final[:3])
alt_final = (r_final_mag - R_EARTH) / 1e3
print(f"Final altitude: {float(alt_final):.1f} km")

print("\n1-sigma uncertainties (ECI):")
print(f"  {'Component':<12} {'Initial':>12} {'Final':>12} {'Growth':>8}")
print(f"  {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 8}")
labels = ["x (m)", "y (m)", "z (m)", "vx (m/s)", "vy (m/s)", "vz (m/s)"]
for i, label in enumerate(labels):
    growth = float(sigma_f[i]) / float(sigma_0[i])
    print(f"  {label:<12} {float(sigma_0[i]):>12.4f} {float(sigma_f[i]):>12.4f} {growth:>7.1f}x")

print("\n1-sigma uncertainties (RTN, final):")
print(f"  {'Component':<12} {'Final':>12}")
print(f"  {'─' * 12} {'─' * 12}")
rtn_labels = ["R (m)", "T (m)", "N (m)", "vR (m/s)", "vT (m/s)", "vN (m/s)"]
for i, label in enumerate(rtn_labels):
    print(f"  {label:<12} {sigma_rtn[-1, i]:>12.4f}")

det_Phi = jnp.linalg.det(Phi)
print(f"\nSTM determinant: {float(det_Phi):.6f} (should be ≈ 1.0 for Hamiltonian flow)")

# %% ── Plot ──────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Position uncertainties
ax1.plot(t_minutes, sigma_rtn[:, 0], label="Radial (R)", color="tab:red")
ax1.plot(t_minutes, sigma_rtn[:, 1], label="Along-track (T)", color="tab:blue")
ax1.plot(t_minutes, sigma_rtn[:, 2], label="Cross-track (N)", color="tab:green")
ax1.set_ylabel("1-sigma position (m)")
ax1.set_title("Covariance Growth in RTN Frame — 500 km LEO (Two-Body)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Velocity uncertainties
ax2.plot(t_minutes, sigma_rtn[:, 3], label="Radial (R)", color="tab:red")
ax2.plot(t_minutes, sigma_rtn[:, 4], label="Along-track (T)", color="tab:blue")
ax2.plot(t_minutes, sigma_rtn[:, 5], label="Cross-track (N)", color="tab:green")
ax2.set_ylabel("1-sigma velocity (m/s)")
ax2.set_xlabel("Time (minutes)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("covariance_rtn.png", dpi=150)
print("\nSaved plot to covariance_rtn.png")
plt.show()
