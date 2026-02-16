"""Mesh export utilities for DAMIT asteroid shape models.

Provides functions to convert DAMIT shape data (vertices and facets)
into standard 3D mesh formats (GLB, STL) via the optional ``trimesh``
library.  Functions raise :class:`ImportError` with install instructions
if ``trimesh`` is not available.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jax.typing import ArrayLike

if TYPE_CHECKING:
    import PIL.Image

try:
    import trimesh

    _HAS_TRIMESH = True
except ImportError:
    _HAS_TRIMESH = False

_TRIMESH_INSTALL_MSG = (
    "trimesh is required for mesh export. Install it with: pip install 'astrojax[extras]'"
)


def _require_trimesh() -> None:
    """Raise ImportError if trimesh is not available.

    Raises:
        ImportError: If ``trimesh`` is not installed.
    """
    if not _HAS_TRIMESH:
        raise ImportError(_TRIMESH_INSTALL_MSG)


def shape_to_trimesh(
    vertices: ArrayLike,
    facets: ArrayLike,
) -> trimesh.Trimesh:
    """Convert DAMIT shape arrays to a trimesh Trimesh object.

    Args:
        vertices: ``(N, 3)`` array of vertex coordinates.
        facets: ``(M, 3)`` array of 0-indexed face vertex indices.

    Returns:
        A :class:`trimesh.Trimesh` instance.

    Raises:
        ImportError: If ``trimesh`` is not installed.

    Examples:
        ```python
        from astrojax.datasets import get_damit_shape, shape_to_trimesh
        verts, faces = get_damit_shape(model_id=42)
        mesh = shape_to_trimesh(verts, faces)
        print(mesh.is_watertight)
        ```
    """
    _require_trimesh()
    import numpy as np

    return trimesh.Trimesh(
        vertices=np.asarray(vertices),
        faces=np.asarray(facets),
    )


def export_shape_glb(
    vertices: ArrayLike,
    facets: ArrayLike,
    filepath: str | Path,
) -> Path:
    """Export DAMIT shape data to a GLB (binary glTF) file.

    Args:
        vertices: ``(N, 3)`` array of vertex coordinates.
        facets: ``(M, 3)`` array of 0-indexed face vertex indices.
        filepath: Destination path for the GLB file.

    Returns:
        Resolved :class:`~pathlib.Path` to the written file.

    Raises:
        ImportError: If ``trimesh`` is not installed.

    Examples:
        ```python
        from astrojax.datasets import get_damit_shape, export_shape_glb
        verts, faces = get_damit_shape(model_id=42)
        export_shape_glb(verts, faces, "asteroid.glb")
        ```
    """
    _require_trimesh()
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    mesh = shape_to_trimesh(vertices, facets)
    scene = trimesh.Scene(mesh)
    scene.export(str(filepath), file_type="glb")

    return filepath.resolve()


def export_shape_stl(
    vertices: ArrayLike,
    facets: ArrayLike,
    filepath: str | Path,
) -> Path:
    """Export DAMIT shape data to an STL file.

    Args:
        vertices: ``(N, 3)`` array of vertex coordinates.
        facets: ``(M, 3)`` array of 0-indexed face vertex indices.
        filepath: Destination path for the STL file.

    Returns:
        Resolved :class:`~pathlib.Path` to the written file.

    Raises:
        ImportError: If ``trimesh`` is not installed.

    Examples:
        ```python
        from astrojax.datasets import get_damit_shape, export_shape_stl
        verts, faces = get_damit_shape(model_id=42)
        export_shape_stl(verts, faces, "asteroid.stl")
        ```
    """
    _require_trimesh()
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    mesh = shape_to_trimesh(vertices, facets)
    mesh.export(str(filepath), file_type="stl")

    return filepath.resolve()


def compute_spherical_uvs(vertices: ArrayLike) -> np.ndarray:
    """Compute spherical-projection UV coordinates for mesh vertices.

    Projects each vertex onto a unit sphere centred at the mesh centroid
    and maps the resulting spherical coordinates to the ``[0, 1]`` UV range.

    Args:
        vertices: ``(N, 3)`` array of vertex coordinates.

    Returns:
        ``(N, 2)`` float32 array of UV coordinates with ``u`` in ``[0, 1]``
        (azimuth) and ``v`` in ``[0, 1]`` (elevation).

    Examples:
        ```python
        from astrojax.datasets import get_damit_shape, compute_spherical_uvs
        verts, faces = get_damit_shape(model_id=42)
        uvs = compute_spherical_uvs(verts)
        print(uvs.shape)  # (N, 2)
        ```
    """
    verts = np.asarray(vertices, dtype=np.float64)
    centroid = verts.mean(axis=0)
    dirs = verts - centroid

    # Spherical coordinates
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.clip(r, 1e-12, None)  # avoid division by zero

    theta = np.arctan2(y, x)  # [-pi, pi]
    phi = np.arccos(np.clip(z / r, -1.0, 1.0))  # [0, pi]

    u = (theta + np.pi) / (2.0 * np.pi)  # [0, 1]
    v = phi / np.pi  # [0, 1]

    return np.stack([u, v], axis=1).astype(np.float32)


def export_shape_glb_textured(
    vertices: ArrayLike,
    facets: ArrayLike,
    filepath: str | Path,
    texture_image: PIL.Image.Image,
) -> Path:
    """Export DAMIT shape data to a textured GLB (binary glTF) file.

    Applies a texture to the mesh using spherical UV projection.  The
    texture is embedded as a PBR base-color map in the exported GLB.

    Args:
        vertices: ``(N, 3)`` array of vertex coordinates.
        facets: ``(M, 3)`` array of 0-indexed face vertex indices.
        filepath: Destination path for the GLB file.
        texture_image: PIL Image to use as the base-color texture.

    Returns:
        Resolved :class:`~pathlib.Path` to the written file.

    Raises:
        ImportError: If ``trimesh`` is not installed.

    Examples:
        ```python
        from PIL import Image
        from astrojax.datasets import get_damit_shape, export_shape_glb_textured
        verts, faces = get_damit_shape(model_id=42)
        tex = Image.open("texture.tif")
        export_shape_glb_textured(verts, faces, "asteroid.glb", tex)
        ```
    """
    _require_trimesh()
    from trimesh.visual import TextureVisuals

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    uvs = compute_spherical_uvs(vertices)
    mesh = shape_to_trimesh(vertices, facets)
    mesh.visual = TextureVisuals(uv=uvs, image=texture_image)

    scene = trimesh.Scene(mesh)
    scene.export(str(filepath), file_type="glb")

    return filepath.resolve()
