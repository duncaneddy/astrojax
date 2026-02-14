# /// script
# requires-python = ">=3.11"
# dependencies = ["typer>=0.9.0", "astrojax[cuda13]"]
#
# [tool.uv.sources]
# astrojax = { path = ".." }
# ///
"""Propagate satellites using SGP4 with multi-device parallelization.

Downloads GP elements from Celestrak or Space-Track.org, initializes SGP4
parameters using JIT-compiled vmap'd initialization, and propagates orbits
across all available JAX devices (GPUs or CPUs) using vmap + pmap.

Requires astrojax to be installed (``uv pip install -e .`` from the repo root).

Usage:
    uv run examples/propagate.py [OPTIONS]

Examples:
    # Celestrak: quick smoke test
    uv run examples/propagate.py --timestep 3600 --duration 0.01 --batch-size 4

    # Celestrak: full 7-day propagation of active satellites at 60s timestep
    uv run examples/propagate.py --timestep 60 --duration 7.0

    # Celestrak: propagate a specific group
    uv run examples/propagate.py --group stations --duration 1.0

    # Space-Track: full catalog propagation
    SPACETRACK_USER=user@email.com SPACETRACK_PASS=pass \\
        uv run examples/propagate.py --source spacetrack --timestep 60 --duration 7.0
"""

import enum
import os
import sys
import time
from typing import Annotated

import jax
import jax.numpy as jnp
import typer

from astrojax import set_dtype
from astrojax.sgp4 import (
    WGS72,
    gp_record_to_array,
    sgp4_init_jax,
    sgp4_propagate_unified,
)

# ── JAX setup ────────────────────────────────────────────────────────────────

try:
    _GPUS = jax.devices("gpu")
except RuntimeError:
    _GPUS = []

set_dtype(jnp.float64)  # Must be before any JIT compilation

# ── JIT-compiled init and propagation ────────────────────────────────────────

_init_batch = jax.jit(
    jax.vmap(sgp4_init_jax, in_axes=(0, None, None)),
    static_argnames=("gravity", "opsmode"),
)


class Source(enum.StrEnum):
    """Ephemeris data source."""

    celestrak = "celestrak"
    spacetrack = "spacetrack"


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
    timestep: Annotated[float, typer.Option(help="Propagation timestep in seconds")] = 5.0,
    duration: Annotated[float, typer.Option(help="Propagation duration in days")] = 7.0,
    batch_size: Annotated[int, typer.Option(help="Satellites per device per batch")] = 512,
    time_chunk: Annotated[int, typer.Option(help="Timesteps per propagation chunk")] = 10080,
    init_batch_size: Annotated[int, typer.Option(help="Satellites per init batch (vmap)")] = 4096,
) -> None:
    """Propagate satellites with SGP4 on multiple devices."""
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

    # ── Stage 2: Convert GP records to element arrays ─────────────────────
    print("\n── Stage 2: Converting GP records to element arrays ──")
    t0 = time.perf_counter()

    elements_list: list[jax.Array] = []
    n_failures = 0
    for record in records:
        try:
            elements_list.append(gp_record_to_array(record))
        except Exception:
            n_failures += 1

    n_sats = len(elements_list)
    elements_all = jnp.stack(elements_list)  # (n_sats, 11)
    print(f"  Converted {n_sats} records in {time.perf_counter() - t0:.1f}s")
    if n_failures > 0:
        print(f"  Failed: {n_failures}")

    if n_sats == 0:
        print("ERROR: No satellites converted successfully. Exiting.")
        sys.exit(1)

    # ── Stage 3: JIT-compiled batch initialization ────────────────────────
    print("\n── Stage 3: Batch-initializing SGP4 parameters (JIT + vmap) ──")
    t0 = time.perf_counter()

    params_batches: list[jax.Array] = []
    for batch_start in range(0, n_sats, init_batch_size):
        batch_end = min(batch_start + init_batch_size, n_sats)
        batch_elements = elements_all[batch_start:batch_end]
        batch_params = _init_batch(batch_elements, WGS72, "i")
        batch_params.block_until_ready()
        params_batches.append(batch_params)
        print(f"\r  Initialized {batch_end}/{n_sats} satellites", end="", flush=True)
    print()

    params_all = jnp.concatenate(params_batches, axis=0)  # (n_sats, 97)

    from astrojax.sgp4._propagation import _IDX

    is_deep = params_all[:, _IDX["method"]] > 0.5
    n_deep_space = int(jnp.sum(is_deep))
    n_near_earth = n_sats - n_deep_space

    elapsed_init = time.perf_counter() - t0
    print(
        f"  Initialized: {n_sats} satellites ({n_near_earth} near-earth, {n_deep_space} deep-space)"
    )
    print(f"  Batch init took {elapsed_init:.1f}s")

    # ── Stage 4: Multi-device SGP4 propagation ───────────────────────────
    print("\n── Stage 4: Multi-device SGP4 propagation ──")

    duration_minutes = duration * 24.0 * 60.0
    timestep_minutes = timestep / 60.0
    tsince = jnp.arange(0.0, duration_minutes, timestep_minutes)
    n_steps = tsince.shape[0]

    print(f"  Duration: {duration} days = {duration_minutes:.0f} minutes")
    print(f"  Timestep: {timestep}s = {timestep_minutes:.4f} min")
    print(f"  Total timesteps: {n_steps}")

    # Memory estimate: positions + velocities in float64
    output_bytes = n_sats * n_steps * 6 * 8  # 6 floats (pos+vel) * 8 bytes (float64)
    output_gb = output_bytes / (1024**3)
    print(f"  Estimated output size: {output_gb:.2f} GB")
    if output_gb > 100:
        print(
            "  WARNING: Output exceeds 100 GB! "
            "Consider reducing --duration or increasing --timestep."
        )

    # Build parallelized propagation: vmap(time) -> vmap(sats) -> pmap(devices)
    propagate_over_time = jax.vmap(sgp4_propagate_unified, in_axes=(None, 0))
    propagate_batch = jax.vmap(propagate_over_time, in_axes=(0, None))

    if _GPUS:
        propagate_parallel = jax.pmap(propagate_batch, in_axes=(0, None))
    else:
        propagate_parallel = jax.jit(propagate_batch)

    total_per_pass = batch_size * n_devices if _GPUS else batch_size

    # Pad satellite count to nearest multiple of total_per_pass
    n_padded = ((n_sats + total_per_pass - 1) // total_per_pass) * total_per_pass
    if n_padded > n_sats:
        padding = jnp.zeros((n_padded - n_sats, params_all.shape[1]))
        params_padded = jnp.concatenate([params_all, padding], axis=0)
    else:
        params_padded = params_all

    n_time_chunks = (n_steps + time_chunk - 1) // time_chunk
    n_sat_batches = n_padded // total_per_pass

    t_prop_start = time.perf_counter()

    for sat_batch_idx in range(n_sat_batches):
        sat_start = sat_batch_idx * total_per_pass
        sat_end = sat_start + total_per_pass
        batch_params = params_padded[sat_start:sat_end]

        # Reshape for pmap: (n_devices, batch_size, n_params) or (batch_size, n_params)
        if _GPUS:
            batch_params = batch_params.reshape(n_devices, batch_size, -1)

        for time_chunk_idx in range(n_time_chunks):
            t_start = time_chunk_idx * time_chunk
            t_end = min(t_start + time_chunk, n_steps)
            tsince_chunk = tsince[t_start:t_end]

            # Propagate
            r, v = propagate_parallel(batch_params, tsince_chunk)

            # Block until computation completes for accurate timing
            r.block_until_ready()
            v.block_until_ready()

            print(
                f"\r  Batch {sat_batch_idx + 1}/{n_sat_batches}, "
                f"time chunk {time_chunk_idx + 1}/{n_time_chunks}",
                end="",
                flush=True,
            )

    print()

    elapsed = time.perf_counter() - t_prop_start
    total_propagations = n_sats * n_steps
    print(
        f"  Propagated {n_sats} satellites x {n_steps} timesteps = {total_propagations:,} evaluations in {elapsed:.1f}s"
    )
    if elapsed > 0:
        print(f"  Throughput: {total_propagations / elapsed:,.0f} propagations/s")

    print("\nDone.")


if __name__ == "__main__":
    typer.run(main)
