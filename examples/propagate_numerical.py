# /// script
# requires-python = ">=3.11"
# dependencies = ["typer>=0.9.0", "astrojax[cuda13]"]
#
# [tool.uv.sources]
# astrojax = { path = ".." }
# ///
"""Propagate satellites using numerical integration with multi-device parallelization.

Downloads GP elements from Celestrak or Space-Track.org, generates initial
ECI states via SGP4, constructs a configurable force model (spherical
harmonics gravity, atmospheric drag, solar radiation pressure, third-body
perturbations), and numerically propagates orbits across all available JAX
devices (GPUs or CPUs) using lax.scan + vmap + pmap.

Requires astrojax to be installed (``uv pip install -e .`` from the repo root).

Usage:
    uv run examples/propagate_numerical.py [OPTIONS]

Examples:
    # Quick smoke test (few satellites, short duration, large timestep)
    uv run examples/propagate_numerical.py --limit 4 --timestep 600 --duration 0.01

    # Small batch with full force model
    uv run examples/propagate_numerical.py --limit 32 --duration 0.1

    # Full run: 100 satellites for 1 day at 60s timestep
    uv run examples/propagate_numerical.py --limit 100 --duration 1.0

    # Two-body + SH gravity only (no drag, SRP, or third-body)
    uv run examples/propagate_numerical.py --limit 32 --no-drag --no-srp \\
        --no-third-body-sun --no-third-body-moon

    # NRLMSISE-00 drag model (slower XLA compilation, ~6 min first call)
    uv run examples/propagate_numerical.py --limit 32 --drag-model nrlmsise00

    # Lower gravity field resolution
    uv run examples/propagate_numerical.py --limit 32 --gravity-degree 8 --gravity-order 8
"""

import enum
import os
import sys
import time
from datetime import UTC, datetime
from typing import Annotated

import jax
import jax.numpy as jnp
import typer

from astrojax import set_dtype
from astrojax.constants import R_EARTH
from astrojax.eop import load_cached_eop
from astrojax.epoch import Epoch
from astrojax.integrators import rk4_step
from astrojax.orbit_dynamics.config import ForceModelConfig
from astrojax.orbit_dynamics.factory import create_orbit_dynamics
from astrojax.orbit_dynamics.gravity import GravityModel
from astrojax.sgp4 import (
    WGS72,
    gp_record_to_array,
    sgp4_init_jax,
    sgp4_propagate_unified,
)
from astrojax.space_weather import load_cached_sw

# ── JAX setup ────────────────────────────────────────────────────────────────

try:
    _GPUS = jax.devices("gpu")
except RuntimeError:
    _GPUS = []

set_dtype(jnp.float64)  # Must be before any JIT compilation

# ── JIT-compiled SGP4 init (reused from propagate.py) ────────────────────────

_init_batch = jax.jit(
    jax.vmap(sgp4_init_jax, in_axes=(0, None, None)),
    static_argnames=("gravity", "opsmode"),
)


class Source(enum.StrEnum):
    """Ephemeris data source."""

    celestrak = "celestrak"
    spacetrack = "spacetrack"


class DragModel(enum.StrEnum):
    """Atmospheric density model."""

    harris_priester = "harris_priester"
    nrlmsise00 = "nrlmsise00"


def main(
    source: Annotated[Source, typer.Option(help="Ephemeris data source")] = Source.celestrak,
    group: Annotated[
        str, typer.Option(help="Celestrak GP group (ignored for spacetrack)")
    ] = "active",
    identity: Annotated[
        str | None,
        typer.Option(help="Space-Track.org login email (or set SPACETRACK_USER env var)"),
    ] = None,
    password: Annotated[
        str | None,
        typer.Option(help="Space-Track.org password (or set SPACETRACK_PASS env var)"),
    ] = None,
    limit: Annotated[int, typer.Option(help="Max number of spacecraft to propagate")] = 100,
    timestep: Annotated[float, typer.Option(help="Integration timestep in seconds")] = 60.0,
    duration: Annotated[float, typer.Option(help="Propagation duration in days")] = 1.0,
    batch_size: Annotated[int, typer.Option(help="Spacecraft per device per batch")] = 32,
    gravity_degree: Annotated[int, typer.Option(help="Spherical harmonic degree")] = 20,
    gravity_order: Annotated[int, typer.Option(help="Spherical harmonic order")] = 20,
    drag: Annotated[bool, typer.Option(help="Enable atmospheric drag")] = True,
    drag_model: Annotated[
        DragModel, typer.Option(help="Atmospheric density model")
    ] = DragModel.harris_priester,
    srp: Annotated[bool, typer.Option(help="Enable solar radiation pressure")] = True,
    third_body_sun: Annotated[bool, typer.Option(help="Enable Sun perturbation")] = True,
    third_body_moon: Annotated[bool, typer.Option(help="Enable Moon perturbation")] = True,
) -> None:
    """Propagate satellites with numerical integration on multiple devices."""
    devices = jax.devices()
    n_devices = len(devices)
    print(f"JAX devices: {n_devices} x {devices[0].platform.upper()}")
    print(f"  Devices: {[str(d) for d in devices]}")

    # ── Stage 1: Download GP Records ─────────────────────────────────────
    if source == Source.celestrak:
        print(f"\n── Stage 1: Downloading GP records from Celestrak (group={group}) ──")
        t0 = time.perf_counter()
        from astrojax.celestrak import CelestrakClient

        records = CelestrakClient().get_gp(group=group)
        print(f"  Downloaded {len(records)} GP records in {time.perf_counter() - t0:.1f}s")
    else:
        resolved_identity = identity or os.environ.get("SPACETRACK_USER")
        resolved_password = password or os.environ.get("SPACETRACK_PASS")
        if not resolved_identity or not resolved_password:
            print(
                "ERROR: Space-Track credentials required. Provide --identity and --password "
                "or set SPACETRACK_USER and SPACETRACK_PASS environment variables."
            )
            sys.exit(1)

        print("\n── Stage 1: Downloading GP records from Space-Track ──")
        t0 = time.perf_counter()
        from astrojax.spacetrack import (
            OutputFormat,
            RequestClass,
            SpaceTrackClient,
            SpaceTrackQuery,
        )

        client = SpaceTrackClient(resolved_identity, resolved_password)
        query = SpaceTrackQuery(RequestClass.GP).format(OutputFormat.JSON)
        records = client.query_gp(query)
        print(f"  Downloaded {len(records)} GP records in {time.perf_counter() - t0:.1f}s")

    if not records:
        print("ERROR: No GP records returned. Exiting.")
        sys.exit(1)

    # Apply limit
    records = records[:limit]
    print(f"  Using {len(records)} satellites (limit={limit})")

    # ── Stage 2: Generate initial ECI states via SGP4 ────────────────────
    print("\n── Stage 2: Generating initial ECI states via SGP4 ──")
    t0 = time.perf_counter()

    # Convert GP records to element arrays
    elements_list: list[jax.Array] = []
    valid_records: list = []
    n_failures = 0
    for record in records:
        try:
            elements_list.append(gp_record_to_array(record))
            valid_records.append(record)
        except Exception:
            n_failures += 1

    if not elements_list:
        print("ERROR: No satellites converted successfully. Exiting.")
        sys.exit(1)

    n_sats = len(elements_list)
    elements_all = jnp.stack(elements_list)  # (n_sats, 11)
    print(f"  Converted {n_sats} GP records to element arrays")
    if n_failures > 0:
        print(f"  Failed: {n_failures}")

    # Initialize SGP4 parameters
    params_all = _init_batch(elements_all, WGS72, "i")
    params_all.block_until_ready()
    print(f"  Initialized SGP4 parameters for {n_sats} satellites")

    # Use current time as reference epoch
    now = datetime.now(UTC)
    epoch_0 = Epoch(now.year, now.month, now.day, now.hour, now.minute, now.second)
    print(f"  Reference epoch: {epoch_0}")

    # Propagate each satellite from its TLE epoch to epoch_0 via SGP4
    # SGP4 tsince is in minutes from TLE epoch
    # TLE epoch JD = jdsatepoch + jdsatepochF (first two elements)
    tle_epoch_jd = elements_all[:, 0] + elements_all[:, 1]  # (n_sats,)
    epoch_0_jd = epoch_0.jd()

    # Time from TLE epoch to reference epoch in minutes
    tsince_minutes = (epoch_0_jd - tle_epoch_jd) * 24.0 * 60.0  # (n_sats,)

    # Propagate all satellites to epoch_0 via SGP4
    # vmap over satellites, each with its own tsince
    sgp4_propagate_batch = jax.jit(jax.vmap(sgp4_propagate_unified, in_axes=(0, 0)))
    r_teme, v_teme = sgp4_propagate_batch(params_all, tsince_minutes)
    r_teme.block_until_ready()

    # SGP4 outputs are in km and km/s (TEME frame ≈ ECI for our purposes)
    # Convert to m and m/s for the dynamics factory
    r_eci = r_teme * 1e3  # (n_sats, 3) [m]
    v_eci = v_teme * 1e3  # (n_sats, 3) [m/s]

    # Assemble initial state vectors: [x, y, z, vx, vy, vz]
    x0_all = jnp.concatenate([r_eci, v_eci], axis=1)  # (n_sats, 6)

    # Filter out any satellites with NaN states (SGP4 failures)
    valid_mask = ~jnp.any(jnp.isnan(x0_all), axis=1)
    n_valid = int(jnp.sum(valid_mask))
    if n_valid < n_sats:
        print(f"  Warning: {n_sats - n_valid} satellites had SGP4 propagation failures, removed")
        x0_all = x0_all[valid_mask]
        n_sats = n_valid

    if n_sats == 0:
        print("ERROR: No valid initial states. Exiting.")
        sys.exit(1)

    # Print altitude statistics
    r_mag = jnp.linalg.norm(x0_all[:, :3], axis=1)
    alt_km = (r_mag - R_EARTH) / 1e3
    print(
        f"  Initial altitudes: min={float(jnp.min(alt_km)):.1f} km, "
        f"max={float(jnp.max(alt_km)):.1f} km, "
        f"mean={float(jnp.mean(alt_km)):.1f} km"
    )
    print(f"  Generated {n_sats} initial ECI states in {time.perf_counter() - t0:.1f}s")

    # ── Stage 3: Load EOP and Space Weather Data ─────────────────────────
    print("\n── Stage 3: Loading EOP and auxiliary data ──")
    t0 = time.perf_counter()

    eop = load_cached_eop()
    print("  Loaded EOP data")

    space_weather = None
    if drag and drag_model == DragModel.nrlmsise00:
        space_weather = load_cached_sw()
        print("  Loaded space weather data (NRLMSISE-00)")

    gravity_model = GravityModel.from_type("JGM3")
    print(f"  Loaded gravity model: JGM3 (degree={gravity_degree}, order={gravity_order})")
    print(f"  Data loading took {time.perf_counter() - t0:.1f}s")

    # ── Stage 4: Build Force Model Configuration ─────────────────────────
    print("\n── Stage 4: Building force model ──")

    config = ForceModelConfig(
        gravity_type="spherical_harmonics",
        gravity_model=gravity_model,
        gravity_degree=gravity_degree,
        gravity_order=gravity_order,
        drag=drag,
        density_model=drag_model.value,
        srp=srp,
        third_body_sun=third_body_sun,
        third_body_moon=third_body_moon,
    )

    dynamics = create_orbit_dynamics(eop, epoch_0, config, space_weather)

    print(f"  Gravity: spherical harmonics {gravity_degree}x{gravity_order}")
    print(f"  Drag: {'enabled (' + drag_model.value + ')' if drag else 'disabled'}")
    print(f"  SRP: {'enabled' if srp else 'disabled'}")
    print(f"  Third-body Sun: {'enabled' if third_body_sun else 'disabled'}")
    print(f"  Third-body Moon: {'enabled' if third_body_moon else 'disabled'}")

    # ── Stage 5: Multi-device Numerical Propagation ──────────────────────
    print("\n── Stage 5: Multi-device numerical propagation ──")

    dt = timestep
    duration_seconds = duration * 86400.0
    n_steps = int(duration_seconds / dt)

    print(f"  Duration: {duration} days = {duration_seconds:.0f} s")
    print(f"  Timestep: {dt} s")
    print(f"  Total timesteps: {n_steps}")
    print(f"  Satellites: {n_sats}")

    # Memory estimate: only final states stored (not full trajectory)
    output_bytes = n_sats * 6 * 8
    output_mb = output_bytes / (1024**2)
    print(f"  Output size (final states): {output_mb:.2f} MB")

    # Define single-spacecraft propagation using lax.scan.
    # n_steps is captured in the closure so it is concrete at trace time.
    # Only the final state is returned (not the full trajectory) to keep
    # memory usage low and speed up XLA compilation.
    def propagate_one(x0: jax.Array, dt_val: jax.Array) -> jax.Array:
        """Numerically propagate one spacecraft using RK4 + lax.scan.

        Args:
            x0: Initial state [x, y, z, vx, vy, vz] in ECI [m, m/s].
            dt_val: Integration timestep [s].

        Returns:
            Final state vector of shape (6,).
        """

        def scan_step(carry, _):
            t, state = carry
            result = rk4_step(dynamics, t, state, dt_val)
            return (t + dt_val, result.state), None

        init_carry = (jnp.float64(0.0), x0)
        (_, x_final), _ = jax.lax.scan(scan_step, init_carry, None, length=n_steps)
        return x_final  # (6,)

    # vmap over spacecraft: each gets a different x0
    propagate_batch = jax.vmap(propagate_one, in_axes=(0, None))

    if _GPUS:
        propagate_parallel = jax.pmap(propagate_batch, in_axes=(0, None))
    else:
        propagate_parallel = jax.jit(propagate_batch)

    total_per_pass = batch_size * n_devices if _GPUS else batch_size

    # Pad satellite count to nearest multiple of total_per_pass.
    # Use a valid satellite state for padding to avoid singularities
    # (zero states cause division-by-zero in gravity/drag).
    n_padded = ((n_sats + total_per_pass - 1) // total_per_pass) * total_per_pass
    if n_padded > n_sats:
        pad_state = x0_all[0:1]  # replicate first valid satellite
        padding = jnp.tile(pad_state, (n_padded - n_sats, 1))
        x0_padded = jnp.concatenate([x0_all, padding], axis=0)
    else:
        x0_padded = x0_all

    n_sat_batches = n_padded // total_per_pass
    dt_jax = jnp.float64(dt)

    print(f"\n  Propagating {n_sats} satellites ({n_padded} padded) in {n_sat_batches} batch(es)")
    print(f"  Batch size per device: {batch_size}")
    if _GPUS:
        print(f"  Devices: {n_devices} GPUs")
    else:
        print(f"  Devices: {n_devices} CPU(s)")

    t_prop_start = time.perf_counter()

    final_states = []
    for sat_batch_idx in range(n_sat_batches):
        sat_start = sat_batch_idx * total_per_pass
        sat_end = sat_start + total_per_pass
        batch_x0 = x0_padded[sat_start:sat_end]

        # Reshape for pmap: (n_devices, batch_size, 6)
        if _GPUS:
            batch_x0 = batch_x0.reshape(n_devices, batch_size, 6)

        if sat_batch_idx == 0:
            print("  Compiling dynamics (first call triggers XLA compilation)...", flush=True)

        # Propagate: returns final states (batch_size, 6) or (n_devices, batch_size, 6)
        batch_final = propagate_parallel(batch_x0, dt_jax)

        # Block until computation completes for accurate timing
        batch_final.block_until_ready()

        if sat_batch_idx == 0:
            t_compiled = time.perf_counter() - t_prop_start
            print(f"  Compilation + first batch took {t_compiled:.1f}s")

        # Flatten pmap device dimension if needed
        if _GPUS:
            batch_final = batch_final.reshape(-1, 6)

        final_states.append(batch_final)

        print(
            f"\r  Batch {sat_batch_idx + 1}/{n_sat_batches} complete",
            end="",
            flush=True,
        )

    print()

    elapsed = time.perf_counter() - t_prop_start
    total_propagations = n_sats * n_steps

    # Combine final states and trim padding
    all_final = jnp.concatenate(final_states, axis=0)[:n_sats]

    print(
        f"  Propagated {n_sats} satellites x {n_steps} timesteps "
        f"= {total_propagations:,} RK4 steps in {elapsed:.1f}s"
    )
    if elapsed > 0:
        print(f"  Throughput: {total_propagations / elapsed:,.0f} RK4 steps/s")

    # ── Stage 6: Report Results ──────────────────────────────────────────
    print("\n── Stage 6: Results ──")

    r_final = jnp.linalg.norm(all_final[:, :3], axis=1)
    alt_final_km = (r_final - R_EARTH) / 1e3

    print(
        f"  Final altitudes: min={float(jnp.min(alt_final_km)):.1f} km, "
        f"max={float(jnp.max(alt_final_km)):.1f} km, "
        f"mean={float(jnp.mean(alt_final_km)):.1f} km"
    )

    # Check for decayed satellites (altitude < 0)
    n_decayed = int(jnp.sum(alt_final_km < 0))
    if n_decayed > 0:
        print(f"  Warning: {n_decayed} satellite(s) decayed (altitude < 0 km)")

    v_final = jnp.linalg.norm(all_final[:, 3:6], axis=1) / 1e3  # km/s
    print(
        f"  Final velocities: min={float(jnp.min(v_final)):.3f} km/s, "
        f"max={float(jnp.max(v_final)):.3f} km/s"
    )

    print("\nDone.")


if __name__ == "__main__":
    typer.run(main)
