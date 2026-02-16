"""Tests for the polyhedral gravity model."""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import polyhedral_gravity as pg  # ty: ignore[unresolved-import]
import pytest

from astrojax.config import set_dtype
from astrojax.orbit_dynamics.polyhedral_gravity import (
    accel_polyhedral_gravity,
    polyhedral_gravity,
)

# Enable float64 for high-precision reference comparisons
set_dtype(jnp.float64)

# ---------------------------------------------------------------------------
# Shared test data: unit cube centred at origin (2x2x2)
# ---------------------------------------------------------------------------

CUBE_VERTICES = jnp.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=jnp.float64,
)

CUBE_FACES = jnp.array(
    [
        [1, 3, 2],
        [0, 3, 1],
        [0, 1, 5],
        [0, 5, 4],
        [0, 7, 3],
        [0, 4, 7],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 6],
        [3, 7, 6],
        [4, 5, 6],
        [4, 6, 7],
    ],
    dtype=jnp.int32,
)


def _generate_analytic_reference(
    density: float, n_points: int = 25
) -> list[tuple[jnp.ndarray, float, jnp.ndarray]]:
    """Generate reference gravity solutions using the polyhedral-gravity package.

    Creates a grid of test points from -5 to 5 in 0.5 steps (matching the
    original reference file layout) and evaluates the polyhedral gravity model
    to produce reference potential and acceleration values.

    Args:
        density: Mass density of the polyhedron.
        n_points: Number of grid points to generate (default 25, tests use 20).

    Returns:
        List of (point, potential, acceleration) tuples.
    """

    # Build the same grid as the original reference files: -5 to 5 in 0.5 steps
    axis = [v / 2.0 for v in range(-10, 11)]  # -5.0, -4.5, ..., 5.0
    grid = list(itertools.product(axis, repeat=3))

    # Take only the needed number of points
    grid = grid[:n_points]

    vertices_list = CUBE_VERTICES.tolist()
    faces_list = CUBE_FACES.tolist()
    poly = pg.Polyhedron(
        polyhedral_source=(vertices_list, faces_list),
        density=density,
        integrity_check=pg.PolyhedronIntegrity.DISABLE,
    )

    results = pg.evaluate(poly, grid)

    data: list[tuple[jnp.ndarray, float, jnp.ndarray]] = []
    for (x, y, z), (pot, accel, _tensor) in zip(grid, results, strict=True):
        data.append(
            (
                jnp.array([x, y, z]),
                float(pot),
                jnp.array(accel),
            )
        )
    return data


# ---------------------------------------------------------------------------
# TestPolyhedralGravityBasic
# ---------------------------------------------------------------------------


class TestPolyhedralGravityBasic:
    """Basic output shape, direction, and edge-case tests."""

    def test_output_shapes(self) -> None:
        """Potential is scalar, acceleration is (3,), tensor is (6,)."""
        r = jnp.array([5.0, 5.0, 5.0])
        pot, accel, tensor = polyhedral_gravity(r, CUBE_VERTICES, CUBE_FACES, 1.0)
        assert pot.shape == ()
        assert accel.shape == (3,)
        assert tensor.shape == (6,)

    def test_acceleration_direction(self) -> None:
        """Acceleration points toward the body (negative radial)."""
        r = jnp.array([5.0, 0.0, 0.0])
        _, accel, _ = polyhedral_gravity(r, CUBE_VERTICES, CUBE_FACES, 1.0)
        # Acceleration should point in -x direction
        assert float(accel[0]) < 0.0

    def test_symmetry_x(self) -> None:
        """Acceleration is symmetric for +x and -x on-axis points."""
        r_pos = jnp.array([5.0, 0.0, 0.0])
        r_neg = jnp.array([-5.0, 0.0, 0.0])
        _, a_pos, _ = polyhedral_gravity(r_pos, CUBE_VERTICES, CUBE_FACES, 1.0)
        _, a_neg, _ = polyhedral_gravity(r_neg, CUBE_VERTICES, CUBE_FACES, 1.0)
        # Magnitudes should match
        npt.assert_allclose(np.abs(np.array(a_pos)), np.abs(np.array(a_neg)), atol=1e-15)

    def test_symmetry_corner(self) -> None:
        """Symmetric corner points should have equal acceleration magnitudes."""
        r1 = jnp.array([5.0, 5.0, 5.0])
        r2 = jnp.array([-5.0, -5.0, -5.0])
        _, a1, _ = polyhedral_gravity(r1, CUBE_VERTICES, CUBE_FACES, 1.0)
        _, a2, _ = polyhedral_gravity(r2, CUBE_VERTICES, CUBE_FACES, 1.0)
        npt.assert_allclose(float(jnp.linalg.norm(a1)), float(jnp.linalg.norm(a2)), atol=1e-15)

    def test_zero_density(self) -> None:
        """Zero density should produce zero potential and acceleration."""
        r = jnp.array([5.0, 5.0, 5.0])
        pot, accel, tensor = polyhedral_gravity(r, CUBE_VERTICES, CUBE_FACES, 0.0)
        assert float(pot) == 0.0
        npt.assert_allclose(np.array(accel), np.zeros(3), atol=1e-30)
        npt.assert_allclose(np.array(tensor), np.zeros(6), atol=1e-30)

    def test_potential_positive(self) -> None:
        """Gravitational potential should be positive for an external point."""
        r = jnp.array([5.0, 5.0, 5.0])
        pot, _, _ = polyhedral_gravity(r, CUBE_VERTICES, CUBE_FACES, 1.0)
        assert float(pot) > 0.0

    def test_density_scaling(self) -> None:
        """Doubling density should double potential and acceleration."""
        r = jnp.array([5.0, 5.0, 5.0])
        pot1, a1, t1 = polyhedral_gravity(r, CUBE_VERTICES, CUBE_FACES, 1.0)
        pot2, a2, t2 = polyhedral_gravity(r, CUBE_VERTICES, CUBE_FACES, 2.0)
        npt.assert_allclose(float(pot2), 2.0 * float(pot1), rtol=1e-12)
        npt.assert_allclose(np.array(a2), 2.0 * np.array(a1), rtol=1e-12)
        npt.assert_allclose(np.array(t2), 2.0 * np.array(t1), rtol=1e-12)


# ---------------------------------------------------------------------------
# TestPolyhedralGravityJIT
# ---------------------------------------------------------------------------


class TestPolyhedralGravityJIT:
    """JIT compilation and vmap compatibility tests."""

    def test_jit_parity(self) -> None:
        """JIT-compiled result matches eager execution."""
        r = jnp.array([5.0, 5.0, 5.0])
        pot_eager, a_eager, t_eager = polyhedral_gravity(r, CUBE_VERTICES, CUBE_FACES, 1.0)

        jit_fn = jax.jit(polyhedral_gravity, static_argnums=())
        pot_jit, a_jit, t_jit = jit_fn(r, CUBE_VERTICES, CUBE_FACES, 1.0)

        npt.assert_allclose(float(pot_jit), float(pot_eager), atol=1e-12)
        npt.assert_allclose(np.array(a_jit), np.array(a_eager), atol=1e-12)
        npt.assert_allclose(np.array(t_jit), np.array(t_eager), atol=1e-12)

    def test_vmap_over_points(self) -> None:
        """vmap over multiple computation points."""
        points = jnp.array(
            [
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 5.0],
            ]
        )

        def _eval_single(r: jnp.ndarray) -> tuple:
            return polyhedral_gravity(r, CUBE_VERTICES, CUBE_FACES, 1.0)

        pots, accels, tensors = jax.vmap(_eval_single)(points)
        assert pots.shape == (3,)
        assert accels.shape == (3, 3)
        assert tensors.shape == (3, 6)

        # On-axis symmetry: all three should have same potential
        npt.assert_allclose(float(pots[0]), float(pots[1]), rtol=1e-10)
        npt.assert_allclose(float(pots[0]), float(pots[2]), rtol=1e-10)


# ---------------------------------------------------------------------------
# TestPolyhedralGravityGrad
# ---------------------------------------------------------------------------


class TestPolyhedralGravityGrad:
    """Automatic differentiation tests."""

    def test_grad_potential_matches_acceleration(self) -> None:
        """grad(potential) should equal acceleration.

        The Tsoulis convention defines a positive gravitational potential
        V = G*M/r whose gradient equals the attractive acceleration:
        grad(V) = a.
        """
        r = jnp.array([5.0, 3.0, 4.0])

        def _potential(r_pt: jnp.ndarray) -> jnp.ndarray:
            pot, _, _ = polyhedral_gravity(r_pt, CUBE_VERTICES, CUBE_FACES, 1.0)
            return pot

        grad_pot = jax.grad(_potential)(r)
        _, accel, _ = polyhedral_gravity(r, CUBE_VERTICES, CUBE_FACES, 1.0)

        # In the positive-potential convention: grad(V) = acceleration
        npt.assert_allclose(np.array(grad_pot), np.array(accel), rtol=1e-5, atol=1e-15)


# ---------------------------------------------------------------------------
# TestPolyhedralGravityAnalytical
# ---------------------------------------------------------------------------


class TestPolyhedralGravityAnalytical:
    """Tests against analytical solutions from reference data files."""

    @pytest.fixture(scope="class")
    def analytic_data_d1(self) -> list[tuple[jnp.ndarray, float, jnp.ndarray]]:
        """Generate density=1 analytical reference solutions."""
        return _generate_analytic_reference(1.0)

    def test_analytic_cube_density1_potential_sample(self, analytic_data_d1: list[tuple]) -> None:
        """Check potential against analytical solution at sampled points."""
        # Test first 20 points to keep test time reasonable
        for point, expected_pot, _ in analytic_data_d1[:20]:
            pot, _, _ = polyhedral_gravity(point, CUBE_VERTICES, CUBE_FACES, 1.0)
            npt.assert_allclose(
                float(pot),
                expected_pot,
                atol=1e-7,
                err_msg=f"Potential mismatch at {point}",
            )

    def test_analytic_cube_density1_acceleration_sample(
        self, analytic_data_d1: list[tuple]
    ) -> None:
        """Check acceleration against analytical solution at sampled points."""
        for point, _, expected_accel in analytic_data_d1[:20]:
            _, accel, _ = polyhedral_gravity(point, CUBE_VERTICES, CUBE_FACES, 1.0)
            npt.assert_allclose(
                np.array(accel),
                np.array(expected_accel),
                atol=1e-7,
                err_msg=f"Acceleration mismatch at {point}",
            )

    def test_analytic_cube_density1_corner(self, analytic_data_d1: list[tuple]) -> None:
        """Verify the corner point (-5, -5, -5) matches analytical data."""
        point, expected_pot, expected_accel = analytic_data_d1[0]
        pot, accel, _ = polyhedral_gravity(point, CUBE_VERTICES, CUBE_FACES, 1.0)
        npt.assert_allclose(float(pot), expected_pot, atol=1e-7)
        npt.assert_allclose(np.array(accel), np.array(expected_accel), atol=1e-7)


# ---------------------------------------------------------------------------
# TestPolyhedralGravityReference
# ---------------------------------------------------------------------------


class TestPolyhedralGravityReference:
    """Comparison tests against the polyhedral-gravity C++ package."""

    def _evaluate_reference(self, point: list, density: float) -> tuple:
        """Evaluate with reference package and return (potential, accel, tensor)."""
        vertices_list = CUBE_VERTICES.tolist()
        faces_list = CUBE_FACES.tolist()
        poly = pg.Polyhedron(
            polyhedral_source=(vertices_list, faces_list),
            density=density,
            integrity_check=pg.PolyhedronIntegrity.DISABLE,
        )
        result = pg.evaluate(poly, [point])
        pot, accel, tensor = result[0]
        return pot, accel, tensor

    def test_reference_density1(self) -> None:
        """Match reference package at density=1.0 for several points."""
        test_points = [
            [5.0, 5.0, 5.0],
            [-5.0, -5.0, -5.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
            [2.0, 3.0, 4.0],
        ]
        for pt in test_points:
            ref_pot, ref_accel, ref_tensor = self._evaluate_reference(pt, 1.0)
            r = jnp.array(pt)
            pot, accel, tensor = polyhedral_gravity(r, CUBE_VERTICES, CUBE_FACES, 1.0)
            npt.assert_allclose(
                float(pot),
                ref_pot,
                atol=1e-7,
                err_msg=f"Potential mismatch at {pt}",
            )
            npt.assert_allclose(
                np.array(accel),
                np.array(ref_accel),
                atol=1e-7,
                err_msg=f"Acceleration mismatch at {pt}",
            )
            npt.assert_allclose(
                np.array(tensor),
                np.array(ref_tensor),
                atol=1e-7,
                err_msg=f"Tensor mismatch at {pt}",
            )

    def test_reference_density42(self) -> None:
        """Match reference package at density=42.0."""
        test_points = [
            [5.0, 5.0, 5.0],
            [-3.0, 2.0, -4.0],
            [0.0, 0.0, 5.0],
        ]
        for pt in test_points:
            ref_pot, ref_accel, ref_tensor = self._evaluate_reference(pt, 42.0)
            r = jnp.array(pt)
            pot, accel, tensor = polyhedral_gravity(r, CUBE_VERTICES, CUBE_FACES, 42.0)
            npt.assert_allclose(
                float(pot),
                ref_pot,
                atol=1e-5,
                err_msg=f"Potential mismatch at {pt}",
            )
            npt.assert_allclose(
                np.array(accel),
                np.array(ref_accel),
                atol=1e-5,
                err_msg=f"Acceleration mismatch at {pt}",
            )
            npt.assert_allclose(
                np.array(tensor),
                np.array(ref_tensor),
                atol=1e-5,
                err_msg=f"Tensor mismatch at {pt}",
            )


# ---------------------------------------------------------------------------
# TestAccelPolyhedralGravity
# ---------------------------------------------------------------------------


class TestAccelPolyhedralGravity:
    """Tests for the inertial-frame wrapper function."""

    def test_identity_rotation(self) -> None:
        """With identity rotation, inertial == body frame."""
        r_point = jnp.array([5.0, 5.0, 5.0])
        r_body = jnp.zeros(3)
        R = jnp.eye(3)

        a_inertial = accel_polyhedral_gravity(r_point, r_body, R, CUBE_VERTICES, CUBE_FACES, 1.0)
        _, a_body, _ = polyhedral_gravity(r_point, CUBE_VERTICES, CUBE_FACES, 1.0)
        npt.assert_allclose(np.array(a_inertial), np.array(a_body), atol=1e-12)

    def test_offset_body(self) -> None:
        """Displaced body: acceleration should still point toward the body."""
        r_body = jnp.array([10.0, 0.0, 0.0])
        r_point = jnp.array([15.0, 0.0, 0.0])
        R = jnp.eye(3)

        a = accel_polyhedral_gravity(r_point, r_body, R, CUBE_VERTICES, CUBE_FACES, 1.0)
        # Should point in -x (toward body at x=10)
        assert float(a[0]) < 0.0
        npt.assert_allclose(float(a[1]), 0.0, atol=1e-12)
        npt.assert_allclose(float(a[2]), 0.0, atol=1e-12)

    def test_rotation_consistency(self) -> None:
        """Rotated body: magnitude should be independent of orientation."""
        r_point = jnp.array([5.0, 0.0, 0.0])
        r_body = jnp.zeros(3)

        # 90-degree rotation about z
        R_z90 = jnp.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        a_id = accel_polyhedral_gravity(r_point, r_body, jnp.eye(3), CUBE_VERTICES, CUBE_FACES, 1.0)
        a_rot = accel_polyhedral_gravity(r_point, r_body, R_z90, CUBE_VERTICES, CUBE_FACES, 1.0)

        # For a cube, 90-degree rotation about z is a symmetry, so the
        # acceleration magnitude should be the same
        npt.assert_allclose(
            float(jnp.linalg.norm(a_id)),
            float(jnp.linalg.norm(a_rot)),
            rtol=1e-10,
        )

    def test_jit_accel_polyhedral(self) -> None:
        """JIT compilation of the inertial-frame function."""
        r_point = jnp.array([5.0, 5.0, 5.0])
        r_body = jnp.zeros(3)
        R = jnp.eye(3)

        jit_fn = jax.jit(accel_polyhedral_gravity)
        a_jit = jit_fn(r_point, r_body, R, CUBE_VERTICES, CUBE_FACES, 1.0)
        a_eager = accel_polyhedral_gravity(r_point, r_body, R, CUBE_VERTICES, CUBE_FACES, 1.0)
        npt.assert_allclose(np.array(a_jit), np.array(a_eager), atol=1e-12)


# ---------------------------------------------------------------------------
# TestPolyhedralGravityDegenerateFaces
# ---------------------------------------------------------------------------


class TestPolyhedralGravityDegenerateFaces:
    """Tests that degenerate (padded) faces are handled gracefully.

    When batching polyhedra with different face counts (e.g., via ``jax.vmap``),
    vertex/face arrays must be padded to a uniform size.  Both NaN-padded and
    zero-padded vertices create degenerate triangles whose per-face contributions
    are NaN.  The ``nan_to_num`` guard in ``polyhedral_gravity`` should zero
    these out so the final result is unaffected.
    """

    # Reference result from the clean cube
    R_TEST = jnp.array([5.0, 3.0, 4.0])
    DENSITY = 1.0

    @pytest.fixture(scope="class")
    def clean_result(self) -> tuple:
        """Compute the reference result from the clean (unpadded) cube."""
        return polyhedral_gravity(self.R_TEST, CUBE_VERTICES, CUBE_FACES, self.DENSITY)

    # -- NaN-padded vertices --------------------------------------------------

    @staticmethod
    def _make_nan_padded() -> tuple[jnp.ndarray, jnp.ndarray]:
        """Append NaN-vertex rows and degenerate faces that reference them."""
        n_pad = 4
        nan_verts = jnp.full((n_pad, 3), jnp.nan, dtype=jnp.float64)
        padded_verts = jnp.concatenate([CUBE_VERTICES, nan_verts], axis=0)

        # Degenerate faces pointing to the NaN vertices
        base = CUBE_VERTICES.shape[0]
        pad_faces = jnp.array(
            [[base, base + 1, base + 2], [base + 1, base + 2, base + 3]],
            dtype=jnp.int32,
        )
        padded_faces = jnp.concatenate([CUBE_FACES, pad_faces], axis=0)
        return padded_verts, padded_faces

    def test_nan_padded_faces_ignored(self, clean_result: tuple) -> None:
        """NaN-padded faces produce the same result as the clean cube."""
        padded_verts, padded_faces = self._make_nan_padded()
        pot, accel, tensor = polyhedral_gravity(
            self.R_TEST, padded_verts, padded_faces, self.DENSITY
        )
        clean_pot, clean_accel, clean_tensor = clean_result

        assert jnp.isfinite(pot), "Potential must be finite"
        assert jnp.all(jnp.isfinite(accel)), "Acceleration must be finite"
        assert jnp.all(jnp.isfinite(tensor)), "Tensor must be finite"

        npt.assert_allclose(float(pot), float(clean_pot), atol=1e-12)
        npt.assert_allclose(np.array(accel), np.array(clean_accel), atol=1e-12)
        npt.assert_allclose(np.array(tensor), np.array(clean_tensor), atol=1e-12)

    # -- Zero-padded vertices -------------------------------------------------

    @staticmethod
    def _make_zero_padded() -> tuple[jnp.ndarray, jnp.ndarray]:
        """Append zero-vertex rows and degenerate faces (all same index)."""
        n_pad = 4
        zero_verts = jnp.zeros((n_pad, 3), dtype=jnp.float64)
        padded_verts = jnp.concatenate([CUBE_VERTICES, zero_verts], axis=0)

        # Degenerate faces: all three indices point to the same zero vertex
        base = CUBE_VERTICES.shape[0]
        pad_faces = jnp.array(
            [[base, base, base], [base + 1, base + 1, base + 1]],
            dtype=jnp.int32,
        )
        padded_faces = jnp.concatenate([CUBE_FACES, pad_faces], axis=0)
        return padded_verts, padded_faces

    def test_zero_padded_faces_ignored(self, clean_result: tuple) -> None:
        """Zero-padded faces produce the same result as the clean cube."""
        padded_verts, padded_faces = self._make_zero_padded()
        pot, accel, tensor = polyhedral_gravity(
            self.R_TEST, padded_verts, padded_faces, self.DENSITY
        )
        clean_pot, clean_accel, clean_tensor = clean_result

        assert jnp.isfinite(pot), "Potential must be finite"
        assert jnp.all(jnp.isfinite(accel)), "Acceleration must be finite"
        assert jnp.all(jnp.isfinite(tensor)), "Tensor must be finite"

        npt.assert_allclose(float(pot), float(clean_pot), atol=1e-12)
        npt.assert_allclose(np.array(accel), np.array(clean_accel), atol=1e-12)
        npt.assert_allclose(np.array(tensor), np.array(clean_tensor), atol=1e-12)

    # -- Gradient through padded faces ----------------------------------------

    def test_padded_grad_clean(self) -> None:
        """grad(potential) with NaN-padded faces produces finite values."""
        padded_verts, padded_faces = self._make_nan_padded()

        def _potential(r_pt: jnp.ndarray) -> jnp.ndarray:
            pot, _, _ = polyhedral_gravity(r_pt, padded_verts, padded_faces, self.DENSITY)
            return pot

        grad_pot = jax.grad(_potential)(self.R_TEST)
        assert jnp.all(jnp.isfinite(grad_pot)), f"Gradient must be finite, got {grad_pot}"

    # -- JIT with padded faces ------------------------------------------------

    def test_padded_jit(self) -> None:
        """JIT compilation works with NaN-padded faces."""
        padded_verts, padded_faces = self._make_nan_padded()

        jit_fn = jax.jit(polyhedral_gravity)
        pot, accel, tensor = jit_fn(self.R_TEST, padded_verts, padded_faces, self.DENSITY)

        assert jnp.isfinite(pot), "JIT potential must be finite"
        assert jnp.all(jnp.isfinite(accel)), "JIT acceleration must be finite"
        assert jnp.all(jnp.isfinite(tensor)), "JIT tensor must be finite"
