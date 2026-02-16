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
"""Generate random asteroid GLB files from DAMIT shapes scaled by ASTMass radii.

Randomly samples asteroid shapes from the DAMIT dataset and physical
radii (``axis_a``) from the ASTMass dataset, then scales each shape
and exports it as a ``.glb`` file with accompanying metadata JSON.

Optionally renders a fly-around ``.mp4`` video for each asteroid and
applies a texture image (e.g. ``.tif``) to both the GLB and the video.

Requires astrojax to be installed (``uv pip install -e '.[extras]'``
from the repo root).

Usage:
    uv run examples/generate_asteroid_glbs.py [OPTIONS]

Examples:
    # Generate 3 asteroids with a fixed seed
    uv run examples/generate_asteroid_glbs.py --count 3 --seed 42

    # Generate 10 asteroids (default) with random sampling
    uv run examples/generate_asteroid_glbs.py

    # Custom output directory
    uv run examples/generate_asteroid_glbs.py --output-dir output/my_asteroids

    # With fly-around video
    uv run examples/generate_asteroid_glbs.py --count 1 --seed 42 --video --n-frames 36

    # With texture and video
    uv run examples/generate_asteroid_glbs.py --count 1 --seed 42 --texture bennu_texture.tiff --video
"""

import json
import os
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer

from astrojax.datasets import (
    compute_spherical_uvs,
    export_shape_glb,
    export_shape_glb_textured,
    get_damit_shape,
    load_asteroid_masses,
    load_damit_models,
    scale_shape_vertices,
)

_STARMAP_DIR = (
    Path(__file__).resolve().parent.parent
    / "refs"
    / "seamstress"
    / "src"
    / "seamstress"
    / "graphics"
    / "resources"
    / "nasa_starmap_2020"
)


def render_flyaround_video(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: Path,
    *,
    texture_image: "np.ndarray | None" = None,
    starmap_dir: Path = _STARMAP_DIR,
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
        texture_image: Optional texture as an RGBA/RGB numpy array
            (H, W, 3 or 4).  When provided, spherical UV projection is
            used to map the texture onto the mesh.
        starmap_dir: Directory containing NASA starmap cubemap face PNGs
            (``px.png``, ``nx.png``, …).  When available, a skybox
            background is rendered behind the asteroid.
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

    # On headless servers fall back to the llvmpipe software rasteriser.
    if not os.environ.get("DISPLAY") and "WGPU_ADAPTER_NAME" not in os.environ:
        os.environ["WGPU_ADAPTER_NAME"] = "llvmpipe"

    # ── Build scene ──
    scene = gfx.Scene()

    # Asteroid mesh
    positions = vertices.astype(np.float32)
    indices = faces.astype(np.uint32).reshape(-1, 3)

    if texture_image is not None:
        uvs = compute_spherical_uvs(vertices)
        geometry = gfx.Geometry(positions=positions, indices=indices, texcoords=uvs)
        tex = gfx.Texture(texture_image, dim=2)
        material = gfx.MeshStandardMaterial(map=tex, roughness=0.9, metalness=0.1)
    else:
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
                cubemap_arr,
                dim=2,
                size=(face_images[0].shape[1], face_images[0].shape[0], 6),
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


def main(
    count: Annotated[
        int, typer.Option("--count", "-n", help="Number of asteroids to generate")
    ] = 10,
    seed: Annotated[int | None, typer.Option(help="Random seed (omit for random each run)")] = None,
    output_dir: Annotated[str, typer.Option(help="Output directory")] = "output/asteroid_glbs",
    video: Annotated[
        bool, typer.Option("--video/--no-video", help="Render fly-around video for each asteroid")
    ] = False,
    texture: Annotated[
        str | None, typer.Option(help="Path to a texture image file (e.g. .tif, .jpg, .png)")
    ] = None,
    n_frames: Annotated[int, typer.Option(help="Number of frames for fly-around video")] = 360,
    video_fps: Annotated[int, typer.Option(help="Video frames per second")] = 30,
    image_width: Annotated[int, typer.Option(help="Render width in pixels")] = 512,
    image_height: Annotated[int, typer.Option(help="Render height in pixels")] = 512,
) -> None:
    """Generate random asteroid GLB files from DAMIT shapes and ASTMass radii."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print(f"Output directory: {output_path.resolve()}")
    print(f"Count: {count}, Seed: {seed}")

    # ── Load texture (once, if provided) ──────────────────────────────────
    texture_pil = None
    texture_array = None
    if texture is not None:
        from PIL import Image

        texture_path = Path(texture)
        if not texture_path.exists():
            print(f"ERROR: Texture file not found: {texture_path}")
            raise typer.Exit(code=1)
        texture_pil = Image.open(texture_path).convert("RGBA")
        texture_array = np.asarray(texture_pil)
        print(f"Texture: {texture_path} ({texture_pil.size[0]}x{texture_pil.size[1]})")

    # ── Load datasets ─────────────────────────────────────────────────────
    print("\n── Loading datasets ──")

    damit_models_df = load_damit_models()
    print(f"  DAMIT models: {len(damit_models_df)} rows")

    astmass_df = load_asteroid_masses()
    print(f"  ASTMass records: {len(astmass_df)} rows")

    # Filter ASTMass to rows with valid, positive axis_a
    astmass_valid = astmass_df.filter(pl.col("axis_a").is_not_null() & (pl.col("axis_a") > 0))
    print(f"  ASTMass with valid axis_a: {len(astmass_valid)} rows")

    if len(astmass_valid) == 0:
        print("ERROR: No ASTMass rows with valid axis_a found.")
        raise typer.Exit(code=1)

    if len(damit_models_df) == 0:
        print("ERROR: No DAMIT models found.")
        raise typer.Exit(code=1)

    # ── Pre-sample all choices ────────────────────────────────────────────
    print(f"\n── Sampling {count} asteroid configurations ──")

    samples = []
    for i in range(count):
        damit_idx = int(rng.integers(0, len(damit_models_df)))
        damit_row = damit_models_df.row(damit_idx, named=True)
        model_id = damit_row["id"]

        am_idx = int(rng.integers(0, len(astmass_valid)))
        am_row = astmass_valid.row(am_idx, named=True)
        axis_a_km = am_row["axis_a"]
        axis_a_m = axis_a_km * 1000.0
        ast_number = am_row.get("ast_number", None)
        ast_name = am_row.get("ast_name", None)

        samples.append(
            {
                "index": i,
                "model_id": model_id,
                "axis_a_km": axis_a_km,
                "axis_a_m": axis_a_m,
                "ast_number": ast_number,
                "ast_name": ast_name,
            }
        )

    unique_model_ids = list({s["model_id"] for s in samples})
    print(f"  {count} samples use {len(unique_model_ids)} unique DAMIT shapes")

    # ── Load shapes and generate GLB files ────────────────────────────────
    print(f"\n── Generating {count} asteroid GLB files ──")

    # Cache shapes to avoid redundant reads for the same model_id
    shape_cache: dict[int, tuple] = {}

    for s in samples:
        i = s["index"]
        model_id = s["model_id"]
        axis_a_km = s["axis_a_km"]
        axis_a_m = s["axis_a_m"]

        if model_id not in shape_cache:
            shape_cache[model_id] = get_damit_shape(model_id=model_id)

        vertices, faces = shape_cache[model_id]
        scaled_vertices = scale_shape_vertices(vertices, axis_a_m)

        # Export GLB (textured or plain)
        glb_filename = f"asteroid_{i:04d}.glb"
        if texture_pil is not None:
            export_shape_glb_textured(
                scaled_vertices, faces, output_path / glb_filename, texture_pil
            )
        else:
            export_shape_glb(scaled_vertices, faces, output_path / glb_filename)

        # Save metadata JSON
        metadata = {
            "damit_model_id": int(model_id),
            "axis_a_km": float(axis_a_km),
            "axis_a_m": float(axis_a_m),
            "ast_number": int(s["ast_number"]) if s["ast_number"] is not None else None,
            "ast_name": str(s["ast_name"]) if s["ast_name"] is not None else None,
        }
        json_filename = f"asteroid_{i:04d}.json"
        json_path = output_path / json_filename
        json_path.write_text(json.dumps(metadata, indent=2))

        print(
            f"  [{i + 1}/{count}] model_id={model_id}, axis_a={axis_a_km:.2f} km -> {glb_filename}"
        )

        # Render fly-around video (if requested)
        if video:
            video_filename = f"asteroid_{i:04d}.mp4"
            video_path = output_path / video_filename
            try:
                render_flyaround_video(
                    vertices=np.asarray(scaled_vertices),
                    faces=np.asarray(faces),
                    output_path=video_path,
                    texture_image=texture_array,
                    width=image_width,
                    height=image_height,
                    n_frames=n_frames,
                    fps=video_fps,
                )
                print(f"         -> {video_filename}")
            except Exception as exc:
                print(f"         WARNING: Video rendering failed: {exc}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n── Summary ──")
    print(f"  Generated {count} asteroid GLB files in {output_path.resolve()}")
    print("\n  Output files:")
    for p in sorted(output_path.iterdir()):
        size_kb = p.stat().st_size / 1024
        print(f"    {p.name}: {size_kb:.1f} KB")

    print("\nDone.")


if __name__ == "__main__":
    typer.run(main)
