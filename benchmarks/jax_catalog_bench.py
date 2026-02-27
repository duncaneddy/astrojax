#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["astrojax[cuda12]>=0.5.2", "typer>=0.9"]
# # For CUDA 13, use: "astrojax[cuda13]>=0.5.2"
# ///

"""Benchmark: propagate the entire active Celestrak catalog to a single future time.

This measures the "catalog sweep" pattern -- batch-initializing and propagating
~10,000 satellites via jit+vmap on a single device (or pmap+vmap across
multiple GPUs). Durations tested are 1, 3, 5, and 7 days.

Usage (CPU only)::

    JAX_PLATFORMS=cpu uv run benchmarks/jax_catalog_bench.py

Usage (single GPU, auto-detected)::

    uv run benchmarks/jax_catalog_bench.py

Usage (specific GPU via CUDA_VISIBLE_DEVICES)::

    CUDA_VISIBLE_DEVICES=0 uv run benchmarks/jax_catalog_bench.py

Usage (multi-GPU pmap, 4 GPUs)::

    uv run benchmarks/jax_catalog_bench.py --num-gpus 4

Usage (float32 only, 20 iterations)::

    uv run benchmarks/jax_catalog_bench.py --dtype float32 --iterations 20

Options::

    --platform     cpu|gpu   Force platform (default: auto-detect).
    --num-gpus     INT       Number of GPUs for pmap. 0 = all available,
                             1 = single-device jit+vmap (default: 1).
    --dtype        TEXT      Precision: float32, float64, or both (default: both).
    --iterations   INT       Timing iterations per scenario (default: 10).
"""

import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import typer

from astrojax import set_dtype
from astrojax.celestrak import CelestrakClient
from astrojax.sgp4 import gp_record_to_array, sgp4_init_jax, sgp4_propagate_unified
from astrojax.sgp4._constants import WGS72

# Duration scenarios: (label, tsince in minutes)
DURATIONS = [
    ("1 day", 1 * 24 * 60.0),
    ("3 days", 3 * 24 * 60.0),
    ("5 days", 5 * 24 * 60.0),
    ("7 days", 7 * 24 * 60.0),
]

app = typer.Typer(add_completion=False)


def _download_catalog() -> tuple[jnp.ndarray, int]:
    """Download the active catalog and convert to a stacked element array.

    Returns:
        Tuple of (element_array of shape (N, 11), num_failed).
    """
    print("Downloading active satellite catalog from Celestrak...")
    client = CelestrakClient()
    records = client.get_gp(group="active")
    print(f"  Retrieved {len(records)} GP records")

    arrays = []
    failed = 0
    for record in records:
        try:
            arrays.append(gp_record_to_array(record))
        except Exception:
            failed += 1

    if not arrays:
        print("ERROR: No satellites parsed successfully. Exiting.")
        sys.exit(1)

    return jnp.stack(arrays), failed


def _detect_device() -> str:
    """Detect the best available device and return its description."""
    try:
        gpus = jax.devices("gpu")
        if gpus:
            return gpus[0].device_kind
    except RuntimeError:
        pass
    return "CPU"


def _get_gpus() -> list:
    """Return available GPU devices, or empty list if none."""
    try:
        return jax.devices("gpu")
    except RuntimeError:
        return []


def _run_single_device(
    elements_batch: jnp.ndarray,
    num_failed: int,
    dtype: type,
    iterations: int,
    device: object | None = None,
) -> None:
    """Run single-device jit+vmap benchmark.

    Args:
        elements_batch: Stacked element arrays of shape (N, 11).
        num_failed: Number of records that failed to parse.
        dtype: JAX dtype (jnp.float64 or jnp.float32).
        iterations: Number of timing iterations per scenario.
        device: JAX device to place computation on, or None for default.
    """
    dtype_name = "float64" if dtype == jnp.float64 else "float32"
    set_dtype(dtype)

    n_sats = elements_batch.shape[0]
    device_name = device.device_kind if device is not None else _detect_device()

    print(f"\nJAX Catalog SGP4 Benchmark ({dtype_name})")
    print("=" * 50)
    print(f"Device: {device_name}")
    print(f"JAX version: {jax.__version__}")
    print(f"Precision: {dtype_name}")
    print(f"Catalog: {n_sats} satellites ({num_failed} failed to parse)")

    elements_batch_typed = elements_batch.astype(dtype)

    # Bind static args before vmap (gravity/opsmode are not JAX types)
    init_fn = partial(sgp4_init_jax, gravity=WGS72, opsmode="i")
    if device is not None:
        batch_init = jax.jit(jax.vmap(init_fn), device=device)
        batch_propagate = jax.jit(
            jax.vmap(sgp4_propagate_unified, in_axes=(0, None)),
            device=device,
        )
    else:
        batch_init = jax.jit(jax.vmap(init_fn))
        batch_propagate = jax.jit(
            jax.vmap(sgp4_propagate_unified, in_axes=(0, None)),
        )

    # --- Initialization ---
    print("\nInitializing all satellites...")
    params_batch = batch_init(elements_batch_typed)
    params_batch.block_until_ready()
    print(f"  Initialized {params_batch.shape[0]} satellites (params shape: {params_batch.shape})")

    # --- Warmup ---
    print("Warming up (compiling propagation)...")
    for _label, tsince_min in DURATIONS:
        tsince = dtype(tsince_min)
        r, v = batch_propagate(params_batch, tsince)
        r.block_until_ready()
    print("  Warmup complete")

    # --- Timed benchmark ---
    print("\n--- Batch Propagation (jit+vmap) ---")

    total_props_per_sec = 0.0
    for label, tsince_min in DURATIONS:
        tsince = dtype(tsince_min)

        total_ns = 0
        for _ in range(iterations):
            start = time.perf_counter_ns()
            r, v = batch_propagate(params_batch, tsince)
            r.block_until_ready()
            total_ns += time.perf_counter_ns() - start

        avg_ms = total_ns / iterations / 1_000_000
        props_per_sec = n_sats / (avg_ms / 1000)
        total_props_per_sec += props_per_sec
        print(f"{label:<30} {avg_ms:>10.3f} ms  ({props_per_sec:>12.2f} prop/s)")

    avg_throughput = total_props_per_sec / len(DURATIONS)
    print(f"{'Average':<30} {avg_throughput:>23.2f} prop/s")


def _run_multi_gpu(
    elements_batch: jnp.ndarray,
    num_failed: int,
    dtype: type,
    iterations: int,
    num_gpus: int,
) -> None:
    """Run multi-GPU pmap+vmap benchmark.

    Args:
        elements_batch: Stacked element arrays of shape (N, 11).
        num_failed: Number of records that failed to parse.
        dtype: JAX dtype (jnp.float64 or jnp.float32).
        iterations: Number of timing iterations per scenario.
        num_gpus: Number of GPUs to use.
    """
    dtype_name = "float64" if dtype == jnp.float64 else "float32"
    set_dtype(dtype)

    n_sats = elements_batch.shape[0]

    print(f"\nJAX Catalog SGP4 Benchmark ({dtype_name}, {num_gpus} GPUs)")
    print("=" * 50)
    gpus = _get_gpus()
    for i in range(num_gpus):
        print(f"  [{i}] {gpus[i].device_kind}")
    print(f"JAX version: {jax.__version__}")
    print(f"Precision: {dtype_name}")
    print(f"Catalog: {n_sats} satellites ({num_failed} failed to parse)")

    elements_batch_typed = elements_batch.astype(dtype)

    # --- Batch init on first GPU, then replicate params ---
    init_fn = partial(sgp4_init_jax, gravity=WGS72, opsmode="i")
    batch_init = jax.jit(jax.vmap(init_fn), device=gpus[0])

    print("\nInitializing all satellites...")
    params_batch = batch_init(elements_batch_typed)
    params_batch.block_until_ready()
    print(f"  Initialized {params_batch.shape[0]} satellites (params shape: {params_batch.shape})")

    # Pad satellite count to evenly divide across GPUs
    padded_n = ((n_sats + num_gpus - 1) // num_gpus) * num_gpus
    if padded_n > n_sats:
        pad_rows = padded_n - n_sats
        padding = jnp.zeros((pad_rows, params_batch.shape[1]), dtype=params_batch.dtype)
        params_padded = jnp.concatenate([params_batch, padding], axis=0)
    else:
        params_padded = params_batch
    # Reshape to (num_gpus, sats_per_gpu, params) and place each shard on its GPU
    params_reshaped = params_padded.reshape(num_gpus, -1, params_padded.shape[1])
    params_sharded = jax.device_put_sharded(
        [params_reshaped[i] for i in range(num_gpus)],
        gpus[:num_gpus],
    )

    pmap_propagate = jax.pmap(
        jax.vmap(sgp4_propagate_unified, in_axes=(0, None)), in_axes=(0, None)
    )

    # --- Warmup ---
    print("Warming up (compiling pmap propagation)...")
    for _label, tsince_min in DURATIONS:
        tsince = dtype(tsince_min)
        r, v = pmap_propagate(params_sharded, tsince)
        r.block_until_ready()
    print("  Warmup complete")

    # --- Timed benchmark ---
    print(f"\n--- Multi-GPU Propagation (pmap+vmap, {num_gpus} GPUs) ---")

    total_props_per_sec = 0.0
    for label, tsince_min in DURATIONS:
        tsince = dtype(tsince_min)

        total_ns = 0
        for _ in range(iterations):
            start = time.perf_counter_ns()
            r, v = pmap_propagate(params_sharded, tsince)
            r.block_until_ready()
            total_ns += time.perf_counter_ns() - start

        avg_ms = total_ns / iterations / 1_000_000
        # Use original (unpadded) count for throughput
        props_per_sec = n_sats / (avg_ms / 1000)
        total_props_per_sec += props_per_sec
        print(f"{label:<30} {avg_ms:>10.3f} ms  ({props_per_sec:>12.2f} prop/s)")

    avg_throughput = total_props_per_sec / len(DURATIONS)
    print(f"{'Average':<30} {avg_throughput:>23.2f} prop/s")


@app.command()
def main(
    platform: str | None = typer.Option(
        None,
        help="Force platform: 'cpu' or 'gpu'. Default: auto-detect.",
    ),
    num_gpus: int = typer.Option(
        1,
        help="Number of GPUs. 0 = all available, 1 = single-device jit+vmap, >1 = pmap+vmap.",
    ),
    dtype: str = typer.Option(
        "both",
        help="Precision: 'float32', 'float64', or 'both'.",
    ),
    iterations: int = typer.Option(
        10,
        help="Number of timing iterations per scenario.",
    ),
) -> None:
    """Propagate the entire active Celestrak catalog and measure throughput."""
    # --- Platform / device resolution ---
    gpus = _get_gpus()
    force_cpu = False

    if platform is not None:
        platform = platform.lower()
        if platform not in ("cpu", "gpu"):
            print(f"ERROR: Unknown platform '{platform}'. Use 'cpu' or 'gpu'.")
            sys.exit(1)
        if platform == "cpu":
            force_cpu = True
        elif platform == "gpu" and not gpus:
            print("ERROR: --platform gpu requested but no GPUs available.")
            sys.exit(1)

    # --- Resolve GPU count ---
    use_multi_gpu = False

    if force_cpu:
        num_gpus = 0
    elif num_gpus == 0:
        num_gpus = len(gpus) if gpus else 0
    if num_gpus > 1:
        if len(gpus) < num_gpus:
            print(f"WARNING: Requested {num_gpus} GPUs but only {len(gpus)} available.")
            num_gpus = len(gpus)
        if num_gpus > 1:
            use_multi_gpu = True

    # --- Resolve dtypes ---
    dtype_lower = dtype.lower()
    if dtype_lower == "both":
        dtypes = [jnp.float64, jnp.float32]
    elif dtype_lower == "float64":
        dtypes = [jnp.float64]
    elif dtype_lower == "float32":
        dtypes = [jnp.float32]
    else:
        print(f"ERROR: Unknown dtype '{dtype}'. Use 'float32', 'float64', or 'both'.")
        sys.exit(1)

    # --- Download catalog once ---
    elements_batch, num_failed = _download_catalog()

    # --- Resolve target device ---
    if force_cpu:
        device = jax.devices("cpu")[0]
    elif gpus:
        device = gpus[0]
    else:
        device = None

    # --- Run benchmarks ---
    for dt in dtypes:
        if use_multi_gpu:
            _run_multi_gpu(elements_batch, num_failed, dt, iterations, num_gpus)
        else:
            _run_single_device(elements_batch, num_failed, dt, iterations, device)

    print()


if __name__ == "__main__":
    app()
