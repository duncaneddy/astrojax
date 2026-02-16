# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "typer>=0.9.0",
#     "astrojax[extras,cuda13]",
#     "pygfx>=0.7",
#     "pylinalg",
#     "rendercanvas",
#     "imageio",
#     "moviepy>=1.0",
# ]
#
# [tool.uv.sources]
# astrojax = { path = ".."}
# ///
"""Simulate spacecraft dynamics near a randomly-generated asteroid.

Loads real asteroid datasets (MPC orbits, DAMIT shapes, ASTMass masses),
samples from them to create a synthetic asteroid, constructs a
JIT-compiled dynamics model with polyhedral gravity, third-body
perturbations, and solar radiation pressure, propagates a trajectory,
and renders a 3D fly-around video with a starfield background.

Requires astrojax to be installed (``uv pip install -e '.[extras]'``
from the repo root), plus the visualization dependencies listed in the
script header.

Usage:
    uv run examples/asteroid_dynamics.py [OPTIONS]

Examples:
    # Quick smoke test (short propagation, few video frames)
    uv run examples/asteroid_dynamics.py --duration 0.1 --n-frames 36

    # Full run with default settings
    uv run examples/asteroid_dynamics.py

    # Custom asteroid orbit radius and output directory
    uv run examples/asteroid_dynamics.py --orbit-radius-factor 5.0 \\
        --output-dir output/my_asteroid
"""

import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import typer

from astrojax import set_dtype
from astrojax.attitude_representations import Rx
from astrojax.constants import (
    AU,
    GM_EARTH,
    GM_JUPITER,
    GM_MARS,
    GM_MERCURY,
    GM_NEPTUNE,
    GM_SATURN,
    GM_SUN,
    GM_URANUS,
    GM_VENUS,
    OBLIQUITY_J2000,
    P_SUN,
)
from astrojax.datasets import (
    asteroid_state_ecliptic,
    damit_spin_to_rotation,
    export_shape_glb,
    export_shape_stl,
    get_damit_shape,
    get_damit_spin,
    load_asteroid_masses,
    load_damit_models,
    load_mpc_asteroids,
    scale_shape_vertices,
    shape_to_trimesh,
)
from astrojax.epoch import Epoch
from astrojax.integrators import rk4_step
from astrojax.orbit_dynamics import (
    EMB_ID,
    JUPITER_ID,
    MARS_ID,
    MERCURY_ID,
    NEPTUNE_ID,
    SATURN_ID,
    URANUS_ID,
    VENUS_ID,
    SpacecraftParams,
    accel_point_mass,
    accel_polyhedral_gravity,
    accel_srp,
    planet_position_jpl_approx,
)
from astrojax.time import jd_to_caldate

# ── JAX setup ────────────────────────────────────────────────────────────────

set_dtype(jnp.float64)

# ── Constants ─────────────────────────────────────────────────────────────────

_G = 6.67430e-11  # Gravitational constant [m^3 kg^-1 s^-2]
_OBLIQUITY_DEG = OBLIQUITY_J2000 / 3600.0  # Mean obliquity [deg]

# Ecliptic-to-equatorial rotation matrix
_R_ECL_TO_EQ = Rx(-_OBLIQUITY_DEG, use_degrees=True)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _epoch_from_jd(jd_value: float) -> Epoch:
    """Create an Epoch from a Julian Date value.

    Args:
        jd_value: Julian Date (TT).

    Returns:
        Epoch instance.
    """
    y, mo, d, h, mi, s = jd_to_caldate(jd_value)
    return Epoch(int(y), int(mo), int(d), int(h), int(mi), float(s))


def _ecliptic_to_equatorial_state(state_ecl: jax.Array) -> jax.Array:
    """Rotate a 6-element state from ecliptic to equatorial (EME2000).

    Args:
        state_ecl: ``[x, y, z, vx, vy, vz]`` in heliocentric ecliptic J2000.

    Returns:
        ``[x, y, z, vx, vy, vz]`` in heliocentric equatorial EME2000.
    """
    r_eq = _R_ECL_TO_EQ @ state_ecl[:3]
    v_eq = _R_ECL_TO_EQ @ state_ecl[3:6]
    return jnp.concatenate([r_eq, v_eq])


# ── Stage 4: Dynamics factory ─────────────────────────────────────────────────


def create_asteroid_dynamics(
    epoch_0: Epoch,
    r_asteroid_heliocentric: jax.Array,
    vertices: jax.Array,
    faces: jax.Array,
    density: float,
    spin_params: jax.Array,
    spacecraft: SpacecraftParams,
) -> Callable:
    """Build a JIT-friendly dynamics closure for spacecraft near an asteroid.

    The returned function computes the time derivative of a 6-element
    state vector ``[x, y, z, vx, vy, vz]`` in an asteroid-centered
    equatorial inertial frame.

    Forces included:
        1. Polyhedral gravity (primary, via DAMIT shape + ASTMass density)
        2. Third-body gravity (Sun + 8 planets via JPL approx ephemerides)
        3. Solar radiation pressure

    The asteroid's heliocentric position is treated as constant over the
    propagation window (valid for near-asteroid operations).

    Args:
        epoch_0: Reference epoch (seconds-since-epoch origin).
        r_asteroid_heliocentric: Heliocentric equatorial position of the
            asteroid at epoch_0 [m], shape ``(3,)``.
        vertices: Scaled DAMIT shape vertices in body frame [m], ``(N, 3)``.
        faces: Face indices (0-indexed), ``(M, 3)`` int.
        density: Bulk density [kg/m^3].
        spin_params: DAMIT spin parameters ``[lambda_deg, beta_deg,
            period_hours, jd0, phi0_deg, yorp]``.
        spacecraft: Physical properties of the spacecraft.

    Returns:
        Callable ``dynamics(t, state) -> derivative`` compatible with
        astrojax integrators.
    """
    # Capture static configuration
    _r_ast = r_asteroid_heliocentric
    _verts = vertices
    _faces = faces
    _rho = density
    _spin = spin_params
    _mass = spacecraft.mass
    _cr = spacecraft.cr
    _srp_area = spacecraft.srp_area

    # Sun position relative to asteroid (constant since asteroid pos is fixed)
    _r_sun_ast = -_r_ast

    # JD at epoch_0 for DAMIT rotation computation
    _jd0 = epoch_0.jd()

    def dynamics(t: jax.typing.ArrayLike, state: jax.typing.ArrayLike) -> jax.Array:
        """Compute state derivative for spacecraft near asteroid.

        Args:
            t: Seconds since epoch_0.
            state: ``[x, y, z, vx, vy, vz]`` asteroid-centered equatorial [m, m/s].

        Returns:
            ``[vx, vy, vz, ax, ay, az]`` [m/s, m/s^2].
        """
        r_sc = state[:3]
        v_sc = state[3:6]

        # Current epoch for planetary ephemerides
        epc = epoch_0 + t
        target_jd = _jd0 + t / 86400.0

        # ── 1. Polyhedral gravity (primary) ──
        # Body-to-ecliptic rotation from DAMIT spin model
        R_body_to_ecl = damit_spin_to_rotation(_spin, target_jd)
        # Body-to-equatorial rotation
        R_body_to_eq = _R_ECL_TO_EQ @ R_body_to_ecl
        # Asteroid body center is at origin in asteroid-centered frame
        r_body_center = jnp.zeros(3)
        a = accel_polyhedral_gravity(r_sc, r_body_center, R_body_to_eq, _verts, _faces, _rho)

        # ── 2. Third-body gravity: Sun ──
        a = a + accel_point_mass(r_sc, _r_sun_ast, GM_SUN)

        # ── 3. Third-body gravity: planets ──
        # Each planet's heliocentric position is recomputed at the current epoch
        a = a + accel_point_mass(
            r_sc, planet_position_jpl_approx(MERCURY_ID, epc) - _r_ast, GM_MERCURY
        )
        a = a + accel_point_mass(r_sc, planet_position_jpl_approx(VENUS_ID, epc) - _r_ast, GM_VENUS)
        a = a + accel_point_mass(r_sc, planet_position_jpl_approx(EMB_ID, epc) - _r_ast, GM_EARTH)
        a = a + accel_point_mass(r_sc, planet_position_jpl_approx(MARS_ID, epc) - _r_ast, GM_MARS)
        a = a + accel_point_mass(
            r_sc, planet_position_jpl_approx(JUPITER_ID, epc) - _r_ast, GM_JUPITER
        )
        a = a + accel_point_mass(
            r_sc, planet_position_jpl_approx(SATURN_ID, epc) - _r_ast, GM_SATURN
        )
        a = a + accel_point_mass(
            r_sc, planet_position_jpl_approx(URANUS_ID, epc) - _r_ast, GM_URANUS
        )
        a = a + accel_point_mass(
            r_sc, planet_position_jpl_approx(NEPTUNE_ID, epc) - _r_ast, GM_NEPTUNE
        )

        # ── 4. Solar radiation pressure ──
        a = a + accel_srp(r_sc, _r_sun_ast, _mass, _cr, _srp_area, P_SUN)

        return jnp.concatenate([v_sc, a])

    return dynamics


# ── Stage 5: Visualization helpers ────────────────────────────────────────────


def render_flyaround_video(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: Path,
    starmap_dir: Path,
    *,
    width: int = 512,
    height: int = 512,
    n_frames: int = 360,
    fps: int = 30,
    orbit_distance_factor: float = 2.0,
) -> Path:
    """Render a fly-around video of the asteroid mesh with a starfield.

    Uses pygfx for offscreen rendering and moviepy for video assembly.

    Args:
        vertices: Mesh vertices ``(N, 3)`` in metres.
        faces: Face indices ``(M, 3)``.
        output_path: Destination path for the MP4 file.
        starmap_dir: Directory containing NASA starmap cubemap face PNGs.
        width: Render width in pixels.
        height: Render height in pixels.
        n_frames: Number of frames for the full 360-degree orbit.
        fps: Video frames per second.
        orbit_distance_factor: Camera orbit distance as multiple of
            the mesh's maximum extent.

    Returns:
        Resolved path to the written MP4 file.
    """
    import imageio.v3 as iio
    import pygfx as gfx
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

    # On headless servers the default Vulkan GPU adapter often produces blank
    # frames because VK_EXT_physical_device_drm is missing.  Fall back to the
    # llvmpipe software rasteriser which works reliably without a display.
    if not os.environ.get("DISPLAY") and "WGPU_ADAPTER_NAME" not in os.environ:
        os.environ["WGPU_ADAPTER_NAME"] = "llvmpipe"

    # ── Build scene ──
    scene = gfx.Scene()

    # Asteroid mesh
    positions = vertices.astype(np.float32)
    indices = faces.astype(np.uint32).reshape(-1, 3)
    geometry = gfx.Geometry(positions=positions, indices=indices)
    material = gfx.MeshStandardMaterial(color=(0.55, 0.52, 0.50), roughness=0.9, metalness=0.1)
    mesh = gfx.Mesh(geometry, material)
    mesh.receive_shadow = True
    mesh.cast_shadow = True
    scene.add(mesh)

    # Sunlight
    sun_light = gfx.DirectionalLight(intensity=2.0)
    sun_light.local.position = (1.0, 0.5, 0.5)
    sun_light.cast_shadow = True
    scene.add(sun_light)

    # Ambient fill
    scene.add(gfx.AmbientLight(intensity=0.15))

    # Starfield background (NASA cubemap)
    if starmap_dir.exists():
        face_names = ["px", "nx", "py", "ny", "pz", "nz"]
        face_images = []
        for name in face_names:
            img_path = starmap_dir / f"{name}.png"
            if img_path.exists():
                face_images.append(iio.imread(img_path))
        if len(face_images) == 6:
            cubemap_arr = np.stack(face_images, axis=0)
            cubemap_tex = gfx.Texture(
                cubemap_arr, dim=2, size=(face_images[0].shape[1], face_images[0].shape[0], 6)
            )
            bg = gfx.Background(None, gfx.BackgroundSkyboxMaterial(map=cubemap_tex))
            scene.add(bg)

    # ── Camera orbit parameters ──
    max_extent = float(np.max(np.linalg.norm(positions, axis=1)))
    cam_dist = orbit_distance_factor * max_extent

    # ── Offscreen canvas and renderer ──
    try:
        from rendercanvas.offscreen import OffscreenRenderCanvas as _OffscreenCanvas
    except Exception:
        from wgpu.gui.offscreen import WgpuCanvas as _OffscreenCanvas  # type: ignore[assignment]
    canvas = _OffscreenCanvas(size=(width, height))
    renderer = gfx.renderers.WgpuRenderer(canvas)

    # ── Render each frame ──
    frames_rgb = []
    for i in range(n_frames):
        angle = 2.0 * np.pi * i / n_frames
        cam_x = cam_dist * np.cos(angle)
        cam_y = cam_dist * 0.3  # slight elevation
        cam_z = cam_dist * np.sin(angle)

        camera = gfx.PerspectiveCamera(fov=60, aspect=width / height)
        camera.local.position = (cam_x, cam_y, cam_z)
        camera.look_at((0, 0, 0))

        renderer.render(scene, camera)
        data = renderer.target.draw()

        # Convert to RGB numpy array
        frame = np.asarray(data)
        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        frames_rgb.append(frame)

    # ── Assemble video ──
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip = ImageSequenceClip([f for f in frames_rgb], fps=fps)
    clip.write_videofile(str(output_path), logger=None)

    return output_path.resolve()


# ── Main ──────────────────────────────────────────────────────────────────────


def main(
    seed: Annotated[int, typer.Option(help="Random seed for reproducibility")] = 42,
    train_ratio: Annotated[float, typer.Option(help="Train/test split ratio")] = 0.8,
    timestep: Annotated[float, typer.Option(help="Integration timestep in seconds")] = 10.0,
    duration: Annotated[float, typer.Option(help="Propagation duration in orbital periods")] = 1.0,
    orbit_radius_factor: Annotated[
        float, typer.Option(help="Spacecraft orbit radius as multiple of asteroid radius")
    ] = 3.0,
    output_dir: Annotated[str, typer.Option(help="Output directory")] = "output/asteroid_dynamics",
    image_width: Annotated[int, typer.Option(help="Render width")] = 512,
    image_height: Annotated[int, typer.Option(help="Render height")] = 512,
    video_fps: Annotated[int, typer.Option(help="Video frames per second")] = 30,
    n_frames: Annotated[int, typer.Option(help="Number of frames for fly-around video")] = 360,
) -> None:
    """Simulate spacecraft dynamics near a randomly-generated asteroid."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print(f"JAX devices: {jax.devices()}")
    print(f"Output directory: {output_path.resolve()}")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 1: Load Datasets
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Stage 1: Loading datasets ──")
    t0 = time.perf_counter()

    mpc_df = load_mpc_asteroids()
    print(f"  MPC asteroids: {len(mpc_df)} rows, columns: {mpc_df.columns}")

    astmass_df = load_asteroid_masses()
    print(f"  ASTMass records: {len(astmass_df)} rows, columns: {astmass_df.columns}")

    damit_models_df = load_damit_models()
    print(f"  DAMIT models: {len(damit_models_df)} rows, columns: {damit_models_df.columns}")

    print(f"  Datasets loaded in {time.perf_counter() - t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 2: Train/Test Split
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Stage 2: Train/test split ──")

    def _train_test_split(
        df: pl.DataFrame, fraction: float, seed: int
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Split a DataFrame into train/test by row index to avoid null-join issues."""
        idx_col = "__row_idx__"
        indexed = df.with_row_index(idx_col)
        train = indexed.sample(fraction=fraction, seed=seed)
        train_indices = train.select(idx_col)
        test = indexed.join(train_indices, on=idx_col, how="anti").drop(idx_col)
        train = train.drop(idx_col)
        return train, test

    mpc_train, mpc_test = _train_test_split(mpc_df, train_ratio, seed)
    astmass_train, astmass_test = _train_test_split(astmass_df, train_ratio, seed)
    damit_train, damit_test = _train_test_split(damit_models_df, train_ratio, seed)

    print(f"  MPC:     train={len(mpc_train)}, test={len(mpc_test)}")
    print(f"  ASTMass: train={len(astmass_train)}, test={len(astmass_test)}")
    print(f"  DAMIT:   train={len(damit_train)}, test={len(damit_test)}")

    # Save split indices
    split_info = {
        "seed": seed,
        "train_ratio": train_ratio,
        "mpc_train_count": len(mpc_train),
        "mpc_test_count": len(mpc_test),
        "astmass_train_count": len(astmass_train),
        "astmass_test_count": len(astmass_test),
        "damit_train_count": len(damit_train),
        "damit_test_count": len(damit_test),
    }
    split_path = output_path / "split_indices.json"
    split_path.write_text(json.dumps(split_info, indent=2))
    print(f"  Saved split info to {split_path}")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 3: Random Asteroid Generation
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Stage 3: Random asteroid generation ──")

    # ── (a) Sample MPC orbit ──
    print("\n  (a) Sampling MPC orbit...")
    mpc_row_idx = rng.integers(0, len(mpc_train))
    mpc_row = mpc_train.row(int(mpc_row_idx), named=True)

    ast_name = mpc_row.get("name", "Unknown")
    ast_number = mpc_row.get("number", "?")
    epoch_jd = mpc_row["epoch_jd"]
    a_au = mpc_row["a"]
    e_val = mpc_row["e"]
    i_deg = mpc_row["i"]
    node_deg = mpc_row["node"]
    peri_deg = mpc_row["peri"]
    M_deg = mpc_row["M"]

    print(f"    Asteroid: {ast_name} (#{ast_number})")
    print(f"    Epoch JD: {epoch_jd}")
    print(f"    a={a_au:.4f} AU, e={e_val:.4f}, i={i_deg:.2f} deg")
    print(f"    node={node_deg:.2f} deg, peri={peri_deg:.2f} deg, M={M_deg:.2f} deg")

    # Compute heliocentric ecliptic state at epoch
    oe = jnp.array([a_au, e_val, i_deg, node_deg, peri_deg, M_deg])
    state_ecl = asteroid_state_ecliptic(epoch_jd, oe, epoch_jd)  # state at own epoch
    state_eq = _ecliptic_to_equatorial_state(state_ecl)
    r_ast_helio_eq = state_eq[:3]

    r_ast_mag_au = float(jnp.linalg.norm(r_ast_helio_eq)) / AU
    print(f"    Heliocentric distance: {r_ast_mag_au:.4f} AU")

    # Reference epoch for the simulation
    epoch_0 = _epoch_from_jd(epoch_jd)
    print(f"    Reference epoch: {epoch_0}")

    # ── (b) Sample DAMIT shape ──
    print("\n  (b) Sampling DAMIT shape...")
    damit_row_idx = rng.integers(0, len(damit_train))
    damit_row = damit_train.row(int(damit_row_idx), named=True)
    model_id = damit_row["id"]
    asteroid_id = damit_row["asteroid_id"]

    print(f"    DAMIT model ID: {model_id}, asteroid_id: {asteroid_id}")

    vertices, faces = get_damit_shape(model_id=model_id)
    print(f"    Shape: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # Get spin parameters (values may be str, int, float, or None from CSV)
    spin_dict = get_damit_spin(damit_models_df, asteroid_id=asteroid_id)
    spin_lambda = float(spin_dict.get("lambda") or 0.0)
    spin_beta = float(spin_dict.get("beta") or 0.0)
    spin_period = float(spin_dict.get("period") or 6.0)
    spin_jd0 = float(spin_dict.get("jd0") or epoch_jd)
    spin_phi0 = float(spin_dict.get("phi0") or 0.0)
    spin_yorp = float(spin_dict.get("yorp") or 0.0)

    spin_params = jnp.array([spin_lambda, spin_beta, spin_period, spin_jd0, spin_phi0, spin_yorp])
    print(
        f"    Spin: lambda={spin_lambda:.1f} deg, beta={spin_beta:.1f} deg, P={spin_period:.3f} h"
    )
    print(f"    Spin: jd0={spin_jd0:.1f}, phi0={spin_phi0:.1f} deg, YORP={spin_yorp}")

    # ── (c) Sample ASTMass physical properties ──
    print("\n  (c) Sampling ASTMass physical properties...")
    # Filter for rows with valid density and axis_a
    astmass_valid = astmass_train.filter(
        pl.col("bulk_density").is_not_null() & pl.col("axis_a").is_not_null()
    )
    if len(astmass_valid) == 0:
        print("    WARNING: No valid ASTMass rows with density+axis. Using defaults.")
        axis_a_km = 5.0
        bulk_density = 2500.0
    else:
        am_row_idx = rng.integers(0, len(astmass_valid))
        am_row = astmass_valid.row(int(am_row_idx), named=True)
        axis_a_km = am_row["axis_a"]
        bulk_density = am_row["bulk_density"]
        print(
            f"    Asteroid: {am_row.get('ast_name', 'Unknown')} (#{am_row.get('ast_number', '?')})"
        )

    axis_a_m = axis_a_km * 1000.0  # km -> m
    print(f"    Radius (axis_a): {axis_a_km:.2f} km = {axis_a_m:.0f} m")
    print(f"    Bulk density: {bulk_density:.0f} kg/m^3")

    # ── (d) Scale DAMIT shape ──
    print("\n  (d) Scaling DAMIT shape...")
    scaled_vertices = scale_shape_vertices(vertices, axis_a_m)
    scale_factor = axis_a_m / float(jnp.max(jnp.linalg.norm(vertices, axis=1)))
    print(f"    Scale factor: {scale_factor:.4f}")
    print(
        f"    Max vertex distance after scaling: {float(jnp.max(jnp.linalg.norm(scaled_vertices, axis=1))):.1f} m"
    )

    # ══════════════════════════════════════════════════════════════════════
    # Stage 4: Set Up JIT-Compiled Dynamics
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Stage 4: Building dynamics model ──")

    spacecraft = SpacecraftParams(mass=100.0, srp_area=1.0, cr=1.3)

    # Estimate asteroid GM from shape volume and density
    mesh = shape_to_trimesh(scaled_vertices, faces)
    volume_m3 = float(abs(mesh.volume))
    asteroid_gm = _G * bulk_density * volume_m3
    print(f"    Shape volume: {volume_m3:.3e} m^3")
    print(f"    Asteroid GM: {asteroid_gm:.6e} m^3/s^2")

    # Initial spacecraft state: circular orbit at orbit_radius_factor * axis_a
    orbit_radius = orbit_radius_factor * axis_a_m
    v_circular = math.sqrt(asteroid_gm / orbit_radius)
    orbital_period = 2.0 * math.pi * orbit_radius / v_circular

    x0 = jnp.array([orbit_radius, 0.0, 0.0, 0.0, v_circular, 0.0])

    print(f"    Orbit radius: {orbit_radius:.0f} m ({orbit_radius_factor:.1f} x axis_a)")
    print(f"    Circular velocity: {v_circular:.6f} m/s")
    print(f"    Orbital period: {orbital_period:.1f} s ({orbital_period / 3600.0:.2f} h)")

    # Build dynamics closure
    dynamics = create_asteroid_dynamics(
        epoch_0=epoch_0,
        r_asteroid_heliocentric=r_ast_helio_eq,
        vertices=scaled_vertices,
        faces=faces,
        density=bulk_density,
        spin_params=spin_params,
        spacecraft=spacecraft,
    )

    # Propagation parameters
    duration_seconds = duration * orbital_period
    dt = timestep
    n_steps = int(duration_seconds / dt)

    print("\n  Propagation:")
    print(f"    Duration: {duration:.2f} periods = {duration_seconds:.0f} s")
    print(f"    Timestep: {dt:.1f} s")
    print(f"    Total steps: {n_steps}")

    # JIT-compiled propagation via lax.scan (stores full trajectory)
    def propagate(x0_in: jax.Array, dt_val: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Propagate spacecraft and return final state + trajectory.

        Args:
            x0_in: Initial state ``[x, y, z, vx, vy, vz]`` [m, m/s].
            dt_val: Integration timestep [s].

        Returns:
            Tuple of (final_state, trajectory positions ``(n_steps, 3)``).
        """

        def scan_step(carry, _unused):
            t, state = carry
            result = rk4_step(dynamics, t, state, dt_val)
            return (t + dt_val, result.state), result.state[:3]

        init_carry = (jnp.float64(0.0), x0_in)
        (_, x_final), positions = jax.lax.scan(scan_step, init_carry, None, length=n_steps)
        return x_final, positions

    print("\n  Compiling dynamics (first call triggers XLA compilation)...")
    t_compile_start = time.perf_counter()
    dt_jax = jnp.float64(dt)
    x_final, trajectory = jax.jit(propagate)(x0, dt_jax)
    x_final.block_until_ready()
    t_compile = time.perf_counter() - t_compile_start
    print(f"  Compilation + propagation took {t_compile:.1f}s")

    # Report trajectory statistics
    r_traj = jnp.linalg.norm(trajectory, axis=1)
    print("\n  Trajectory statistics:")
    print(f"    Min distance: {float(jnp.min(r_traj)):.1f} m")
    print(f"    Max distance: {float(jnp.max(r_traj)):.1f} m")
    print(f"    Mean distance: {float(jnp.mean(r_traj)):.1f} m")
    print(f"    Initial state: {np.array(x0)}")
    print(f"    Final state:   {np.array(x_final)}")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 5: Visualization
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Stage 5: Visualization ──")

    # ── (a) Export mesh files ──
    print("\n  (a) Exporting mesh files...")
    glb_path = export_shape_glb(scaled_vertices, faces, output_path / "asteroid.glb")
    print(f"    GLB: {glb_path}")

    stl_path = export_shape_stl(scaled_vertices, faces, output_path / "asteroid.stl")
    print(f"    STL: {stl_path}")

    # ── (b) Render fly-around video ──
    print("\n  (b) Rendering fly-around video...")
    starmap_dir = (
        Path(__file__).resolve().parent.parent
        / "refs"
        / "seamstress"
        / "src"
        / "seamstress"
        / "graphics"
        / "resources"
        / "nasa_starmap_2020"
    )

    video_path = output_path / "asteroid_flyaround.mp4"
    try:
        t_render_start = time.perf_counter()
        video_result = render_flyaround_video(
            vertices=np.asarray(scaled_vertices),
            faces=np.asarray(faces),
            output_path=video_path,
            starmap_dir=starmap_dir,
            width=image_width,
            height=image_height,
            n_frames=n_frames,
            fps=video_fps,
        )
        t_render = time.perf_counter() - t_render_start
        print(f"    Video: {video_result} ({t_render:.1f}s)")
    except Exception as exc:
        print(f"    WARNING: Video rendering failed: {exc}")
        print("    (This may happen in headless environments without GPU/display support)")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 6: Summary
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Stage 6: Summary ──")

    print("\n  Output files:")
    for p in sorted(output_path.iterdir()):
        size_kb = p.stat().st_size / 1024
        print(f"    {p.name}: {size_kb:.1f} KB")

    print("\n  Asteroid parameters:")
    print(f"    MPC orbit: {ast_name} (#{ast_number})")
    print(f"      a={a_au:.4f} AU, e={e_val:.4f}, i={i_deg:.2f} deg")
    print(f"    DAMIT shape: model {model_id} ({vertices.shape[0]} verts, {faces.shape[0]} faces)")
    print(f"      lambda={spin_lambda:.1f} deg, beta={spin_beta:.1f} deg, P={spin_period:.3f} h")
    print(f"    Physical: radius={axis_a_km:.2f} km, density={bulk_density:.0f} kg/m^3")
    print(f"      GM={asteroid_gm:.6e} m^3/s^2, volume={volume_m3:.3e} m^3")

    print("\n  Propagation:")
    print(f"    Duration: {duration:.2f} orbital periods ({duration_seconds:.0f} s)")
    print(f"    Timestep: {dt:.1f} s, steps: {n_steps}")
    print(f"    Initial pos: [{float(x0[0]):.1f}, {float(x0[1]):.1f}, {float(x0[2]):.1f}] m")
    print(
        f"    Final pos:   [{float(x_final[0]):.1f}, {float(x_final[1]):.1f}, {float(x_final[2]):.1f}] m"
    )
    r_final = float(jnp.linalg.norm(x_final[:3]))
    print(f"    Final distance: {r_final:.1f} m (initial: {orbit_radius:.1f} m)")

    print("\nDone.")


if __name__ == "__main__":
    typer.run(main)
