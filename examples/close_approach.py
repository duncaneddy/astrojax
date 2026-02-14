# /// script
# requires-python = ">=3.11"
# dependencies = ["typer>=0.9.0", "astrojax[cuda13]"]
#
# [tool.uv.sources]
# astrojax = { path = ".." }
# ///
"""Detect and refine close approaches between satellites using SGP4.

Downloads GP elements from Celestrak or Space-Track.org, propagates all objects
using JIT-compiled vmap'd SGP4, screens for pairwise close approaches,
de-duplicates encounters, and refines times of closest approach (TCA) using
Newton-Raphson with JAX autodiff.

Requires astrojax to be installed (``uv pip install -e .`` from the repo root).

Usage:
    uv run examples/close_approach.py [OPTIONS]

Examples:
    # Celestrak: quick smoke test with space stations
    uv run examples/close_approach.py --group stations --duration 0.1

    # Celestrak: full 1-day screening of stations
    uv run examples/close_approach.py --group stations --duration 1.0

    # Celestrak: larger catalog with object limit
    uv run examples/close_approach.py --group active --limit 200 --duration 0.5

    # Space-Track: close approach screening
    SPACETRACK_USER=user@email.com SPACETRACK_PASS=pass \\
        uv run examples/close_approach.py --source spacetrack --limit 50 --duration 0.1
"""

import csv
import enum
import os
import sys
import time
from collections import defaultdict
from typing import Annotated

import jax
import jax.numpy as jnp
import numpy as np
import typer

from astrojax import set_dtype
from astrojax.coordinates.keplerian import state_eci_to_koe
from astrojax.orbits.keplerian import perigee_velocity
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
    ] = "stations",
    identity: Annotated[
        str | None,
        typer.Option(help="Space-Track.org login email (or set SPACETRACK_USER env var)"),
    ] = None,
    password: Annotated[
        str | None,
        typer.Option(help="Space-Track.org password (or set SPACETRACK_PASS env var)"),
    ] = None,
    threshold: Annotated[
        float, typer.Option(help="Close approach distance threshold in meters")
    ] = 10000.0,
    duration: Annotated[float, typer.Option(help="Screening duration in days")] = 1.0,
    limit: Annotated[int | None, typer.Option(help="Process only the first N objects")] = None,
    pair_chunk: Annotated[
        int, typer.Option(help="Satellite block size for chunked pairwise distance computation")
    ] = 1000,
    time_chunk: Annotated[int, typer.Option(help="Timesteps per propagation chunk")] = 10080,
    batch_size: Annotated[int, typer.Option(help="Satellites per device per batch")] = 512,
    init_batch_size: Annotated[int, typer.Option(help="Satellites per init batch (vmap)")] = 4096,
    max_refine_iter: Annotated[
        int, typer.Option(help="Max Newton-Raphson iterations for TCA refinement")
    ] = 10,
) -> None:
    """Detect close approaches between satellites using SGP4 + JAX."""
    devices = jax.devices()
    n_devices = len(devices)
    print(f"JAX devices: {n_devices} x {devices[0].platform.upper()}")
    print(f"  Devices: {[str(d) for d in devices]}")
    print(f"  Source: {source.value}")
    print(f"  Threshold: {threshold:.0f} m")
    print(f"  Duration: {duration} days")

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

    if limit is not None:
        records = records[:limit]
        print(f"  Limited to first {limit} objects")

    n_total = len(records)
    if n_total > 5000:
        print(
            f"  WARNING: {n_total} objects will require O(N^2) = O({n_total**2:,}) pairwise comparisons."
        )

    # Store names and NORAD IDs for output labeling
    names: list[str] = []
    norad_ids: list[int] = []
    for rec in records:
        names.append(rec.object_name or "UNKNOWN")
        norad_ids.append(rec.norad_cat_id or 0)

    # ── Stage 2: Convert to Element Arrays + Batch SGP4 Init ─────────────
    print("\n── Stage 2: Converting GP records to element arrays ──")
    t0 = time.perf_counter()

    elements_list: list[jax.Array] = []
    valid_indices: list[int] = []
    n_failures = 0
    for idx, record in enumerate(records):
        try:
            elements_list.append(gp_record_to_array(record))
            valid_indices.append(idx)
        except Exception:
            n_failures += 1

    n_sats = len(elements_list)
    elements_all = jnp.stack(elements_list)  # (n_sats, 11)
    print(f"  Converted {n_sats} records in {time.perf_counter() - t0:.1f}s")
    if n_failures > 0:
        print(f"  Failed: {n_failures}")

    # Filter names/norad_ids to match valid elements
    names = [names[i] for i in valid_indices]
    norad_ids = [norad_ids[i] for i in valid_indices]

    if n_sats < 2:
        print("ERROR: Need at least 2 satellites for close approach detection. Exiting.")
        sys.exit(1)

    print("\n── Stage 2b: Batch-initializing SGP4 parameters (JIT + vmap) ──")
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

    print(
        f"  Initialized: {n_sats} satellites ({n_near_earth} near-earth, {n_deep_space} deep-space)"
    )
    print(f"  Batch init took {time.perf_counter() - t0:.1f}s")

    # ── Stage 3: Compute Adaptive Timestep from Perigee Velocities ───────
    print("\n── Stage 3: Computing adaptive timestep from perigee velocities ──")
    t0 = time.perf_counter()

    # Propagate all objects to t=0
    propagate_t0 = jax.jit(jax.vmap(sgp4_propagate_unified, in_axes=(0, None)))
    r_km, v_kms = propagate_t0(params_all, jnp.float64(0.0))
    r_km.block_until_ready()

    # Convert to SI units (SGP4 outputs km, km/s)
    states = jnp.concatenate([r_km * 1000.0, v_kms * 1000.0], axis=-1)  # (N, 6)

    # Compute orbital elements via vmap'd state_eci_to_koe
    oe = jax.vmap(state_eci_to_koe)(states)  # (N, 6): [a, e, i, RAAN, omega, M]

    # Compute perigee velocities
    v_p = perigee_velocity(oe[:, 0], oe[:, 1])  # (N,) in m/s
    max_v_p = float(jnp.max(jnp.abs(v_p)))

    # Set timestep: threshold / (2 * max_v_p)
    threshold_m = threshold
    dt_seconds = threshold_m / (2.0 * max_v_p)
    dt_minutes = dt_seconds / 60.0

    print(f"  Max perigee velocity: {max_v_p:.1f} m/s ({max_v_p / 1000.0:.2f} km/s)")
    print(f"  Adaptive timestep: {dt_seconds:.2f} s ({dt_minutes:.4f} min)")
    print(f"  Computed in {time.perf_counter() - t0:.1f}s")

    # ── Stage 4: Propagate All Objects ───────────────────────────────────
    print("\n── Stage 4: Propagating all objects ──")

    duration_minutes = duration * 24.0 * 60.0
    tsince = jnp.arange(0.0, duration_minutes, dt_minutes)
    n_steps = int(tsince.shape[0])

    print(f"  Duration: {duration} days = {duration_minutes:.0f} minutes")
    print(f"  Timestep: {dt_seconds:.2f}s = {dt_minutes:.4f} min")
    print(f"  Total timesteps: {n_steps}")

    # Build propagation functions
    propagate_over_time = jax.vmap(sgp4_propagate_unified, in_axes=(None, 0))
    propagate_batch = jax.jit(jax.vmap(propagate_over_time, in_axes=(0, None)))

    # Propagate in time chunks, screening each chunk for close approaches
    n_time_chunks = (n_steps + time_chunk - 1) // time_chunk
    threshold_km = threshold_m / 1000.0

    events: list[tuple[int, int, int, float]] = []  # (idx_i, idx_j, global_t_idx, dist_km)
    t_prop_start = time.perf_counter()

    for tc_idx in range(n_time_chunks):
        t_start = tc_idx * time_chunk
        t_end = min(t_start + time_chunk, n_steps)
        tsince_chunk = tsince[t_start:t_end]
        chunk_size = int(tsince_chunk.shape[0])

        # Propagate ALL satellites for this time chunk
        # Process in satellite batches to manage memory
        all_r_chunks: list[np.ndarray] = []

        for sat_start in range(0, n_sats, batch_size):
            sat_end = min(sat_start + batch_size, n_sats)
            batch_params = params_all[sat_start:sat_end]
            r_chunk, _ = propagate_batch(batch_params, tsince_chunk)
            r_chunk.block_until_ready()
            all_r_chunks.append(np.asarray(r_chunk))  # (batch, T_chunk, 3) in km

        # Concatenate all satellite positions: (N, T_chunk, 3) in km
        pos_all = np.concatenate(all_r_chunks, axis=0)  # (N, T_chunk, 3)

        # ── Stage 5: Detect Close Approaches (Coarse Screen) ────────────
        for t_local in range(chunk_size):
            pos = pos_all[:, t_local, :]  # (N, 3) in km
            global_t_idx = t_start + t_local

            # Chunked pairwise distance computation
            for i_start in range(0, n_sats, pair_chunk):
                i_end = min(i_start + pair_chunk, n_sats)
                for j_start in range(i_start, n_sats, pair_chunk):
                    j_end = min(j_start + pair_chunk, n_sats)
                    pos_i = pos[i_start:i_end]  # (chunk_i, 3)
                    pos_j = pos[j_start:j_end]  # (chunk_j, 3)
                    diff = pos_i[:, None, :] - pos_j[None, :, :]  # (chunk_i, chunk_j, 3)
                    dist = np.linalg.norm(diff, axis=-1)  # (chunk_i, chunk_j)

                    if i_start == j_start:
                        # Same block: upper triangle only (avoid self-pairs and duplicates)
                        ii, jj = np.where(np.triu(dist < threshold_km, k=1))
                    else:
                        ii, jj = np.where(dist < threshold_km)

                    for i_local, j_local in zip(ii, jj, strict=True):
                        events.append(
                            (
                                i_start + int(i_local),
                                j_start + int(j_local),
                                global_t_idx,
                                float(dist[i_local, j_local]),
                            )
                        )

        print(
            f"\r  Time chunk {tc_idx + 1}/{n_time_chunks}, raw detections so far: {len(events)}",
            end="",
            flush=True,
        )

    print()
    elapsed_prop = time.perf_counter() - t_prop_start
    print(f"  Propagation + screening took {elapsed_prop:.1f}s")
    print(f"  Raw detections: {len(events)}")

    if not events:
        print("\nNo close approaches detected within the threshold.")
        print("Done.")
        return

    # ── Stage 6: De-duplicate Encounters ─────────────────────────────────
    print("\n── Stage 6: De-duplicating encounters ──")

    pair_events: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
    for idx_i, idx_j, t_idx, dist_km in events:
        pair_events[(idx_i, idx_j)].append((t_idx, dist_km))

    encounters: list[tuple[int, int, int]] = []  # (idx_i, idx_j, t_guess_idx)
    for (idx_i, idx_j), detections in pair_events.items():
        detections.sort(key=lambda x: x[0])

        # Split into clusters: gap > 2 timesteps means new encounter
        clusters: list[list[tuple[int, float]]] = []
        current_cluster = [detections[0]]
        for det in detections[1:]:
            if det[0] - current_cluster[-1][0] <= 2:
                current_cluster.append(det)
            else:
                clusters.append(current_cluster)
                current_cluster = [det]
        clusters.append(current_cluster)

        for cluster in clusters:
            best = min(cluster, key=lambda x: x[1])
            encounters.append((idx_i, idx_j, best[0]))

    print(f"  Unique encounters: {len(encounters)} (from {len(events)} raw detections)")

    # ── Stage 7: Newton-Raphson TCA Refinement ───────────────────────────
    print("\n── Stage 7: Refining TCA with Newton-Raphson (JAX autodiff) ──")
    t0 = time.perf_counter()

    def range_rate_dot_range(
        params_i: jax.Array, params_j: jax.Array, t_min: jax.Array
    ) -> jax.Array:
        """Dot product of relative position and relative velocity.

        Equals zero at the time of closest approach (local extremum of distance).
        """
        r_i, v_i = sgp4_propagate_unified(params_i, t_min)
        r_j, v_j = sgp4_propagate_unified(params_j, t_min)
        dr = r_i - r_j
        dv = v_i - v_j
        return jnp.dot(dr, dv)

    d_range_rate = jax.grad(range_rate_dot_range, argnums=2)

    def refine_tca(
        params_i: jax.Array,
        params_j: jax.Array,
        t_guess_min: jax.Array,
        max_iter: int = 10,
    ) -> jax.Array:
        """Refine TCA using Newton-Raphson with JAX autodiff."""

        def step(_: int, t: jax.Array) -> jax.Array:
            f = range_rate_dot_range(params_i, params_j, t)
            df = d_range_rate(params_i, params_j, t)
            dt = jnp.where(jnp.abs(df) > 1e-30, f / df, 0.0)
            return t - dt

        return jax.lax.fori_loop(0, max_iter, step, t_guess_min)

    refine_tca_jit = jax.jit(refine_tca, static_argnames=("max_iter",))

    def compute_distance_km(
        params_i: jax.Array, params_j: jax.Array, t_min: jax.Array
    ) -> jax.Array:
        """Compute distance between two objects at time t_min."""
        r_i, _ = sgp4_propagate_unified(params_i, t_min)
        r_j, _ = sgp4_propagate_unified(params_j, t_min)
        return jnp.linalg.norm(r_i - r_j)

    compute_distance_jit = jax.jit(compute_distance_km)

    results: list[tuple[str, str, int, int, float, float]] = []
    for enc_idx, (idx_i, idx_j, t_guess_idx) in enumerate(encounters):
        t_guess_min = float(tsince[t_guess_idx])
        t_refined = float(
            refine_tca_jit(
                params_all[idx_i],
                params_all[idx_j],
                jnp.float64(t_guess_min),
                max_iter=max_refine_iter,
            )
        )

        min_dist_km = float(
            compute_distance_jit(
                params_all[idx_i],
                params_all[idx_j],
                jnp.float64(t_refined),
            )
        )
        min_dist_m = min_dist_km * 1000.0

        results.append(
            (
                names[idx_i],
                names[idx_j],
                norad_ids[idx_i],
                norad_ids[idx_j],
                t_refined,
                min_dist_m,
            )
        )

        if (enc_idx + 1) % 10 == 0 or enc_idx + 1 == len(encounters):
            print(f"\r  Refined {enc_idx + 1}/{len(encounters)} encounters", end="", flush=True)

    print()
    print(f"  TCA refinement took {time.perf_counter() - t0:.1f}s")

    # ── Output Results ───────────────────────────────────────────────────
    results.sort(key=lambda x: x[5])

    print("\n── Close Approach Results ──")
    print(f"Found {len(encounters)} unique encounters (from {len(events)} raw detections)\n")

    header = f"  {'Object 1':<25} {'Object 2':<25} {'TCA (min)':>12} {'Min Dist (m)':>14}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for name1, name2, _, _, tca, dist_m in results:
        print(f"  {name1:<25} {name2:<25} {tca:>12.2f} {dist_m:>14.1f}")

    # Write CSV
    csv_path = "close_approaches.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "object_1_name",
                "object_1_norad",
                "object_2_name",
                "object_2_norad",
                "tca_minutes",
                "min_distance_m",
            ]
        )
        for name1, name2, norad1, norad2, tca, dist_m in results:
            writer.writerow([name1, norad1, name2, norad2, f"{tca:.6f}", f"{dist_m:.3f}"])

    print(f"\nResults written to {csv_path}")
    print("Done.")


if __name__ == "__main__":
    typer.run(main)
