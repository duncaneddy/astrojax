"""Mesh export utilities for DAMIT asteroid shape models.

Provides functions to convert DAMIT shape data (vertices and facets)
into standard 3D mesh formats (GLB, STL) via the optional ``trimesh``
library.  Functions raise :class:`ImportError` with install instructions
if ``trimesh`` is not available.
"""

from __future__ import annotations

from pathlib import Path

from jax.typing import ArrayLike

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
