"""Polyhedral gravity model using the Tsoulis (2012) line-integral method.

Computes gravitational potential, acceleration, and gravity gradient tensor
for irregularly-shaped bodies represented as triangulated surface meshes.
This is the standard approach for modelling the gravitational field of
asteroids and small bodies using shape models such as those from the DAMIT
database.

The algorithm follows the formulation in:

    D. Tsoulis, "Analytical computation of the full gravity tensor of a
    homogeneous arbitrarily shaped polyhedral source using line integrals,"
    *Geophysics*, vol. 77, no. 2, pp. F1-F11, 2012.

with singularity-handling refinements from:

    D. Tsoulis and K. Petrovic, "On the singularities of the gravity field
    of a homogeneous polyhedral body," *Geophysics*, vol. 66, no. 2,
    pp. 535-539, 2001.

All functions use JAX primitives and are compatible with ``jax.jit``,
``jax.vmap``, and ``jax.grad``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype

# Gravitational constant [m^3 kg^-1 s^-2]
_G = 6.67430e-11


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------


def _safe_norm(x: Array) -> Array:
    """Euclidean norm with safe gradient at the origin.

    ``jnp.linalg.norm`` has an undefined gradient at ``x = 0`` because
    ``d||x||/dx = x / ||x||``.  Adding a tiny epsilon inside the sqrt
    makes the gradient well-defined everywhere.

    Args:
        x: Input vector, shape ``(N,)``.

    Returns:
        Scalar norm.
    """
    return jnp.sqrt(jnp.dot(x, x) + 1e-300)


def _sgn(x: Array) -> Array:
    """Signum with epsilon deadband.

    Returns +1, -1, or 0 when ``|x|`` is below a dtype-appropriate
    epsilon threshold.

    Args:
        x: Scalar or array.

    Returns:
        Element-wise sign with deadband.
    """
    eps = jnp.finfo(x.dtype).eps * 100.0
    return jnp.where(jnp.abs(x) < eps, 0.0, jnp.sign(x))


def _build_segment_vectors(v0: Array, v1: Array, v2: Array) -> Array:
    """Edge vectors G_pq for a triangular face.

    Args:
        v0: First vertex, shape ``(3,)``.
        v1: Second vertex, shape ``(3,)``.
        v2: Third vertex, shape ``(3,)``.

    Returns:
        Segment vectors, shape ``(3, 3)`` where row *q* is G_pq.
    """
    return jnp.stack([v1 - v0, v2 - v1, v0 - v2])


def _build_plane_unit_normal(g0: Array, g1: Array) -> Array:
    """Face unit normal via cross product of two edge vectors.

    Args:
        g0: First segment vector G_p0, shape ``(3,)``.
        g1: Second segment vector G_p1, shape ``(3,)``.

    Returns:
        Unit normal N_p, shape ``(3,)``.
    """
    n = jnp.cross(g0, g1)
    return n / _safe_norm(n)


def _build_segment_unit_normals(seg_vecs: Array, plane_normal: Array) -> Array:
    """Segment unit normals n_pq = cross(G_pq, N_p) / |cross(G_pq, N_p)|.

    Args:
        seg_vecs: Segment vectors, shape ``(3, 3)``.
        plane_normal: Plane unit normal N_p, shape ``(3,)``.

    Returns:
        Segment unit normals, shape ``(3, 3)``.
    """

    def _one_segment(g: Array) -> Array:
        c = jnp.cross(g, plane_normal)
        return c / _safe_norm(c)

    return jax.vmap(_one_segment)(seg_vecs)


def _compute_hessian_plane(v0: Array, v1: Array, v2: Array) -> Array:
    """Hessian normal form [A, B, C, D] of the plane through three vertices.

    Args:
        v0: First vertex, shape ``(3,)``.
        v1: Second vertex, shape ``(3,)``.
        v2: Third vertex, shape ``(3,)``.

    Returns:
        Array of shape ``(4,)`` with ``[A, B, C, D]``.
    """
    cross = jnp.cross(v0 - v1, v0 - v2)
    d = jnp.dot(-v0, cross)
    return jnp.append(cross, d)


def _distance_origin_to_plane(hessian: Array) -> Array:
    """Perpendicular distance h_p from origin to the plane.

    Args:
        hessian: Hessian plane ``[A, B, C, D]``, shape ``(4,)``.

    Returns:
        Scalar distance h_p (always non-negative).
    """
    abc = hessian[:3]
    d = hessian[3]
    return jnp.abs(d / _safe_norm(abc))


def _project_origin_onto_plane(
    plane_normal: Array,
    h_p: Array,
    hessian: Array,
) -> Array:
    """Orthogonal projection P' of the origin onto the face plane.

    Applies sign corrections based on the Hessian plane intersections
    with the coordinate axes, following the reference implementation.

    Args:
        plane_normal: Plane unit normal N_p, shape ``(3,)``.
        h_p: Perpendicular distance from origin to plane.
        hessian: Hessian plane ``[A, B, C, D]``, shape ``(4,)``.

    Returns:
        Projection point P', shape ``(3,)``.
    """
    proj = plane_normal * h_p

    abc = hessian[:3]
    d = hessian[3]

    # Axis intersections: d/A, d/B, d/C (avoid division by zero)
    safe_abc = jnp.where(abc == 0.0, 1.0, abc)
    intersections = jnp.where(abc == 0.0, 0.0, d / safe_abc)

    # Sign correction logic from reference:
    # If intersection < 0 (meaning -intersection >= 0): coordinate is |proj|
    # Else if plane_normal > 0: coordinate is -proj
    # Else: coordinate is proj (unchanged)
    corrected = jnp.where(
        intersections < 0.0,
        jnp.abs(proj),
        jnp.where(plane_normal > 0.0, -proj, proj),
    )
    return corrected


def _compute_segment_normal_orientations(
    face_verts: Array,
    proj_point: Array,
    seg_unit_normals: Array,
) -> Array:
    """Segment normal orientations sigma_pq.

    Args:
        face_verts: Triangle vertices, shape ``(3, 3)``.
        proj_point: Projection point P', shape ``(3,)``.
        seg_unit_normals: Segment unit normals, shape ``(3, 3)``.

    Returns:
        Orientations sigma_pq, shape ``(3,)``.
    """

    def _one(n_pq: Array, vertex: Array) -> Array:
        return _sgn(jnp.dot(n_pq, proj_point - vertex)) * -1.0

    return jax.vmap(_one)(seg_unit_normals, face_verts)


def _project_onto_segment(v1: Array, v2: Array, p_prime: Array) -> Array:
    """Project P' orthogonally onto a line segment to get P''.

    Uses Cramer's rule on a 3x3 linear system, following the reference
    implementation.

    Args:
        v1: First segment endpoint, shape ``(3,)``.
        v2: Second segment endpoint, shape ``(3,)``.
        p_prime: Projection point P' on the plane, shape ``(3,)``.

    Returns:
        Projection point P'' on the segment, shape ``(3,)``.
    """
    row1 = v2 - v1
    row2 = jnp.cross(v1 - p_prime, row1)
    row3 = jnp.cross(row2, row1)
    d = jnp.array(
        [
            jnp.dot(row1, p_prime),
            jnp.dot(row2, p_prime),
            jnp.dot(row3, v1),
        ]
    )

    # Row matrix A (Cramer's rule solves A * P'' = d)
    A = jnp.stack([row1, row2, row3])
    det_A = jnp.linalg.det(A)

    # Cramer's rule: replace column i of A with d
    def _replace_col(col_idx: int) -> Array:
        m = A.at[:, col_idx].set(d)
        return jnp.linalg.det(m)

    x = _replace_col(0) / det_A
    y = _replace_col(1) / det_A
    z = _replace_col(2) / det_A
    return jnp.array([x, y, z])


def _project_onto_segments(
    proj_point: Array,
    sigma_pq: Array,
    face_verts: Array,
) -> Array:
    """Project P' onto each of the 3 segments to get P''.

    When sigma_pq == 0, P' is already on the segment, so P'' = P'.

    Args:
        proj_point: Projection point P', shape ``(3,)``.
        sigma_pq: Segment normal orientations, shape ``(3,)``.
        face_verts: Triangle vertices, shape ``(3, 3)``.

    Returns:
        Projection points P'' for each segment, shape ``(3, 3)``.
    """

    def _one(j: int) -> Array:
        v1 = face_verts[j]
        v2 = face_verts[(j + 1) % 3]
        p_double_prime = _project_onto_segment(v1, v2, proj_point)
        # If sigma_pq == 0, P' is on the segment already
        return jnp.where(sigma_pq[j] == 0.0, proj_point, p_double_prime)

    return jnp.stack([_one(0), _one(1), _one(2)])


def _compute_distances(
    seg_vecs: Array,
    p_double_primes: Array,
    face_verts: Array,
) -> Array:
    """Compute signed distances l1, l2, s1, s2 for each segment.

    Implements the 4-case sign convention from Tsoulis (2021).

    Args:
        seg_vecs: Segment vectors G_pq, shape ``(3, 3)``.
        p_double_primes: Projection points P'', shape ``(3, 3)``.
        face_verts: Triangle vertices, shape ``(3, 3)``.

    Returns:
        Distances array of shape ``(3, 4)`` with columns ``[l1, l2, s1, s2]``.
    """
    eps = jnp.finfo(face_verts.dtype).eps * 100.0

    def _one(j: int) -> Array:
        v1 = face_verts[j]
        v2 = face_verts[(j + 1) % 3]
        p_pp = p_double_primes[j]
        g = seg_vecs[j]

        # 3D distances from origin to vertices
        l1 = _safe_norm(v1)
        l2 = _safe_norm(v2)

        # 1D distances from P'' to vertices
        s1 = _safe_norm(p_pp - v1)
        s2 = _safe_norm(p_pp - v2)

        g_norm = _safe_norm(g)

        # Case 4: |s1 - l1| < eps AND |s2 - l2| < eps
        # P coincides with P' and P'' (on segment direction)
        is_case4 = (jnp.abs(s1 - l1) < eps) & (jnp.abs(s2 - l2) < eps)

        # Case 4 sub-cases
        # Case 4.2: s2 < s1 → all negative
        case4_2 = s2 < s1 - eps
        # Case 4.1: s2 ≈ s1 → s1 negative, s2 positive
        case4_1 = jnp.abs(s2 - s1) < eps

        s1_c4 = jnp.where(case4_2, -s1, jnp.where(case4_1, -s1, s1))
        s2_c4 = jnp.where(case4_2, -s2, s2)
        l1_c4 = jnp.where(case4_2, -l1, jnp.where(case4_1, -l1, l1))
        l2_c4 = jnp.where(case4_2, -l2, l2)

        # Cases 1-3 (not case 4)
        # Case 1: s1 < |G| AND s2 < |G| → P'' inside segment
        is_case1 = (s1 < g_norm) & (s2 < g_norm)
        # Case 2: s2 < s1 → P'' on right side
        is_case2 = s2 < s1

        # Case 1: s1 = -|s1|
        # Case 2: s1 = -|s1|, s2 = -|s2|
        # Case 3: all positive (default)
        s1_other = jnp.where(is_case1, -s1, jnp.where(is_case2, -s1, s1))
        s2_other = jnp.where(is_case1, s2, jnp.where(is_case2, -s2, s2))

        s1_final = jnp.where(is_case4, s1_c4, s1_other)
        s2_final = jnp.where(is_case4, s2_c4, s2_other)
        l1_final = jnp.where(is_case4, l1_c4, l1)
        l2_final = jnp.where(is_case4, l2_c4, l2)

        return jnp.array([l1_final, l2_final, s1_final, s2_final])

    return jnp.stack([_one(0), _one(1), _one(2)])


def _compute_norms_projection_to_vertices(
    proj_point: Array,
    face_verts: Array,
) -> Array:
    """Euclidean norms from P' to each vertex.

    Args:
        proj_point: Projection point P', shape ``(3,)``.
        face_verts: Triangle vertices, shape ``(3, 3)``.

    Returns:
        Norms, shape ``(3,)``.
    """
    return jnp.array(
        [
            _safe_norm(proj_point - face_verts[0]),
            _safe_norm(proj_point - face_verts[1]),
            _safe_norm(proj_point - face_verts[2]),
        ]
    )


def _compute_transcendental_expressions(
    distances: Array,
    h_p: Array,
    h_pq: Array,
    sigma_pq: Array,
    proj_vertex_norms: Array,
) -> Array:
    """Compute LN_pq and AN_pq for each segment.

    LN_pq = ln((s2 + l2) / (s1 + l1))
    AN_pq = atan2(h_p * s2, h_pq * l2) - atan2(h_p * s1, h_pq * l1)

    With singularity handling when P' is on a vertex or distances vanish.

    Args:
        distances: Shape ``(3, 4)`` with columns ``[l1, l2, s1, s2]``.
        h_p: Perpendicular distance from origin to plane.
        h_pq: Segment distances, shape ``(3,)``.
        sigma_pq: Segment normal orientations, shape ``(3,)``.
        proj_vertex_norms: Norms from P' to vertices, shape ``(3,)``.

    Returns:
        Shape ``(3, 2)`` with columns ``[LN_pq, AN_pq]``.
    """
    eps = jnp.finfo(distances.dtype).eps * 100.0

    def _one(j: int) -> Array:
        l1 = distances[j, 0]
        l2 = distances[j, 1]
        s1 = distances[j, 2]
        s2 = distances[j, 3]

        # Norms from P' to segment endpoints
        # r1Norm corresponds to vertex (j+1)%3, r2Norm to vertex j
        r1_norm = proj_vertex_norms[(j + 1) % 3]
        r2_norm = proj_vertex_norms[j]

        # LN_pq
        # Zero if: (sigma_pq==0 and either endpoint norm is ~0)
        #       or (|s1+s2| < eps and |l1+l2| < eps)
        ln_zero_cond = ((sigma_pq[j] == 0.0) & ((r1_norm < eps) | (r2_norm < eps))) | (
            (jnp.abs(s1 + s2) < eps) & (jnp.abs(l1 + l2) < eps)
        )
        # Safe computation: avoid log of negative or zero
        num = s2 + l2
        den = s1 + l1
        safe_ratio = jnp.where(jnp.abs(den) < eps, 1.0, num / den)
        safe_ratio = jnp.where(safe_ratio <= 0.0, 1.0, safe_ratio)
        ln_val = jnp.where(ln_zero_cond, 0.0, jnp.log(safe_ratio))

        # AN_pq
        # Zero if h_p < eps or h_pq[j] < eps
        an_zero_cond = (h_p < eps) | (h_pq[j] < eps)
        an_val_raw = jnp.arctan2(h_p * s2, h_pq[j] * l2) - jnp.arctan2(h_p * s1, h_pq[j] * l1)
        an_val = jnp.where(an_zero_cond, 0.0, an_val_raw)

        return jnp.array([ln_val, an_val])

    return jnp.stack([_one(0), _one(1), _one(2)])


def _compute_singularity_terms(
    seg_vecs: Array,
    sigma_pq: Array,
    proj_vertex_norms: Array,
    plane_normal: Array,
    h_p: Array,
    sigma_p: Array,
) -> tuple[Array, Array]:
    """Compute singularity correction terms sing_alpha and sing_beta.

    Four cases:
    1. All sigma_pq == 1 → P' inside the face → -2π·h_p, -2π·σ_p·N_p
    2. sigma_pq == 0 and P' on a segment (not vertex) → -π·h_p, -π·σ_p·N_p
    3. sigma_pq == 0 and P' at a vertex → -θ·h_p, -θ·σ_p·N_p
    4. Otherwise → 0, [0,0,0]

    Args:
        seg_vecs: Segment vectors, shape ``(3, 3)``.
        sigma_pq: Segment normal orientations, shape ``(3,)``.
        proj_vertex_norms: Norms from P' to vertices, shape ``(3,)``.
        plane_normal: Plane unit normal, shape ``(3,)``.
        h_p: Perpendicular distance to plane.
        sigma_p: Plane normal orientation.

    Returns:
        Tuple of (sing_alpha scalar, sing_beta shape ``(3,)``).
    """
    eps = jnp.finfo(seg_vecs.dtype).eps * 100.0
    pi = jnp.array(jnp.pi, dtype=seg_vecs.dtype)
    two_pi = 2.0 * pi

    # Case 1: All sigma_pq == 1 → P' inside the face
    all_inside = (sigma_pq[0] == 1.0) & (sigma_pq[1] == 1.0) & (sigma_pq[2] == 1.0)

    # Case 2: Any sigma_pq == 0 AND P' on segment (not at vertex)
    # For segment j: sigma_pq[j]==0 and both endpoint norms < |G_j|
    g_norms = jnp.array(
        [
            _safe_norm(seg_vecs[0]),
            _safe_norm(seg_vecs[1]),
            _safe_norm(seg_vecs[2]),
        ]
    )

    on_seg_0 = (
        (jnp.abs(sigma_pq[0]) < eps)
        & (proj_vertex_norms[1] < g_norms[0])
        & (proj_vertex_norms[0] < g_norms[0])
    )
    on_seg_1 = (
        (jnp.abs(sigma_pq[1]) < eps)
        & (proj_vertex_norms[2] < g_norms[1])
        & (proj_vertex_norms[1] < g_norms[1])
    )
    on_seg_2 = (
        (jnp.abs(sigma_pq[2]) < eps)
        & (proj_vertex_norms[0] < g_norms[2])
        & (proj_vertex_norms[2] < g_norms[2])
    )
    on_segment = on_seg_0 | on_seg_1 | on_seg_2

    # Case 3: Any sigma_pq == 0 AND P' at a vertex (endpoint norm ≈ 0)
    at_vtx_0 = (jnp.abs(sigma_pq[0]) < eps) & (
        (proj_vertex_norms[1] < eps) | (proj_vertex_norms[0] < eps)
    )
    at_vtx_1 = (jnp.abs(sigma_pq[1]) < eps) & (
        (proj_vertex_norms[2] < eps) | (proj_vertex_norms[1] < eps)
    )
    at_vtx_2 = (jnp.abs(sigma_pq[2]) < eps) & (
        (proj_vertex_norms[0] < eps) | (proj_vertex_norms[2] < eps)
    )
    at_vertex = at_vtx_0 | at_vtx_1 | at_vtx_2

    # Compute theta for vertex case
    # Need to identify which segment j and which endpoint (r1 or r2)
    # r1Norm = proj_vertex_norms[(j+1)%3], r2Norm = proj_vertex_norms[j]
    # If r1Norm == 0: g1 = seg_vecs[j], g2 = seg_vecs[(j+1)%3]
    # If r2Norm == 0: g1 = seg_vecs[(j-1+3)%3], g2 = seg_vecs[j]

    def _theta_for_seg(j: int) -> Array:
        r1_norm = proj_vertex_norms[(j + 1) % 3]
        r1_zero = r1_norm < eps

        g1 = jnp.where(r1_zero, seg_vecs[j], seg_vecs[(j - 1 + 3) % 3])
        g2 = jnp.where(r1_zero, seg_vecs[(j + 1) % 3], seg_vecs[j])

        g1_norm = _safe_norm(g1)
        g2_norm = _safe_norm(g2)
        gdot = jnp.dot(-g1, g2)
        cos_theta = gdot / (g1_norm * g2_norm)
        cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
        return jnp.where(jnp.abs(gdot) < eps, pi / 2.0, jnp.arccos(cos_theta))

    theta_0 = _theta_for_seg(0)
    theta_1 = _theta_for_seg(1)
    theta_2 = _theta_for_seg(2)

    # Pick the theta from the first matching segment
    theta = jnp.where(at_vtx_0, theta_0, jnp.where(at_vtx_1, theta_1, theta_2))

    # Combine cases with priority: case1 > case2 > case3 > case4
    angle = jnp.where(
        all_inside,
        two_pi,
        jnp.where(on_segment, pi, jnp.where(at_vertex, theta, 0.0)),
    )

    sing_alpha = -angle * h_p
    sing_beta = plane_normal * (-angle * sigma_p)

    return sing_alpha, sing_beta


def _evaluate_face(
    face_verts: Array,
    seg_vecs: Array,
    plane_normal: Array,
    seg_unit_normals: Array,
) -> tuple[Array, Array, Array]:
    """Evaluate the gravitational contribution of a single face.

    The computation point is assumed to have been relocated to the origin.

    Args:
        face_verts: Triangle vertices (already shifted), shape ``(3, 3)``.
        seg_vecs: Segment vectors G_pq, shape ``(3, 3)``.
        plane_normal: Plane unit normal N_p, shape ``(3,)``.
        seg_unit_normals: Segment unit normals n_pq, shape ``(3, 3)``.

    Returns:
        Tuple of:
        - potential_contrib: scalar
        - accel_contrib: shape ``(3,)``
        - tensor_contrib: shape ``(6,)`` as ``[Vxx, Vyy, Vzz, Vxy, Vxz, Vyz]``
    """
    # 1-04: Plane normal orientation sigma_p
    sigma_p = _sgn(jnp.dot(plane_normal, face_verts[0]))

    # 1-05: Hessian normal plane
    hessian = _compute_hessian_plane(face_verts[0], face_verts[1], face_verts[2])

    # 1-06: Distance h_p
    h_p = _distance_origin_to_plane(hessian)

    # 1-07: Projection P'
    proj_point = _project_origin_onto_plane(plane_normal, h_p, hessian)

    # 1-08: Segment normal orientations sigma_pq
    sigma_pq = _compute_segment_normal_orientations(face_verts, proj_point, seg_unit_normals)

    # 1-09: Projections P'' onto segments
    p_double_primes = _project_onto_segments(proj_point, sigma_pq, face_verts)

    # 1-10: Segment distances h_pq
    h_pq = jnp.array(
        [
            _safe_norm(p_double_primes[0] - proj_point),
            _safe_norm(p_double_primes[1] - proj_point),
            _safe_norm(p_double_primes[2] - proj_point),
        ]
    )

    # 1-11: Distances l1, l2, s1, s2
    distances = _compute_distances(seg_vecs, p_double_primes, face_verts)

    # 1-12: Norms from P' to vertices
    proj_vertex_norms = _compute_norms_projection_to_vertices(proj_point, face_verts)

    # 1-13: Transcendental expressions LN_pq and AN_pq
    trans = _compute_transcendental_expressions(distances, h_p, h_pq, sigma_pq, proj_vertex_norms)
    ln_pq = trans[:, 0]
    an_pq = trans[:, 1]

    # 1-14: Singularity terms
    sing_alpha, sing_beta = _compute_singularity_terms(
        seg_vecs, sigma_pq, proj_vertex_norms, plane_normal, h_p, sigma_p
    )

    # Step 2: Sum1 for potential/acceleration = Σ sigma_pq * h_pq * LN_pq
    sum1_pot_accel = jnp.sum(sigma_pq * h_pq * ln_pq)

    # Step 3: Sum1 for tensor = Σ n_pq * LN_pq
    sum1_tensor = jnp.sum(seg_unit_normals * ln_pq[:, None], axis=0)

    # Step 4: Sum2 = Σ sigma_pq * AN_pq
    sum2 = jnp.sum(sigma_pq * an_pq)

    # Step 5: Combined sum for potential and acceleration
    plane_sum = sum1_pot_accel + h_p * sum2 + sing_alpha

    # Potential contribution: sigma_p * h_p * plane_sum (eq. 11)
    potential_contrib = sigma_p * h_p * plane_sum

    # Acceleration contribution: N_p * plane_sum (eq. 12)
    accel_contrib = plane_normal * plane_sum

    # Step 6: Tensor (eq. 13)
    sub_sum = sum1_tensor + plane_normal * (sigma_p * sum2) + sing_beta
    # Diagonal: N_p * sub_sum (element-wise)
    diag = plane_normal * sub_sum  # [Vxx, Vyy, Vzz]
    # Off-diagonal: reordering
    off_diag = jnp.array(
        [
            plane_normal[0] * sub_sum[1],  # Vxy
            plane_normal[0] * sub_sum[2],  # Vxz
            plane_normal[1] * sub_sum[2],  # Vyz
        ]
    )
    tensor_contrib = jnp.concatenate([diag, off_diag])

    return potential_contrib, accel_contrib, tensor_contrib


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def polyhedral_gravity(
    r_body_frame: ArrayLike,
    vertices: ArrayLike,
    faces: ArrayLike,
    density: ArrayLike,
) -> tuple[Array, Array, Array]:
    """Gravitational potential, acceleration, and tensor from a polyhedral body.

    Evaluates the full gravity field of a homogeneous polyhedron at a
    single computation point using the Tsoulis (2012) line-integral
    method.  All quantities are computed in the body-fixed frame.

    Args:
        r_body_frame: Computation point in body frame [m], shape ``(3,)``.
        vertices: Mesh vertex coordinates [m], shape ``(N, 3)``.
        faces: Triangle face indices (0-indexed), shape ``(M, 3)`` of int.
        density: Bulk density of the body [kg/m^3], scalar.

    Returns:
        Tuple of:

        - **potential** — Gravitational potential [m^2/s^2], scalar.
        - **acceleration** — Gravitational acceleration [m/s^2],
            shape ``(3,)``.  The vector points *toward* the body
            (attractive).
        - **tensor** — Gravity gradient tensor [1/s^2],
            shape ``(6,)`` as ``[Vxx, Vyy, Vzz, Vxy, Vxz, Vyz]``.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.orbit_dynamics import polyhedral_gravity

        # Unit cube: 8 vertices, 12 triangular faces
        vertices = jnp.array([
            [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
            [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
        ], dtype=jnp.float64)
        faces = jnp.array([
            [1,3,2],[0,3,1],[0,1,5],[0,5,4],[0,7,3],[0,4,7],
            [1,2,6],[1,6,5],[2,3,6],[3,7,6],[4,5,6],[4,6,7],
        ])

        r = jnp.array([5.0, 5.0, 5.0])
        potential, accel, tensor = polyhedral_gravity(r, vertices, faces, 1.0)
        ```
    """
    _float = get_dtype()
    r = jnp.asarray(r_body_frame, dtype=_float)
    verts = jnp.asarray(vertices, dtype=_float)
    face_idx = jnp.asarray(faces, dtype=jnp.int32)
    rho = jnp.asarray(density, dtype=_float)

    # Relocate: shift all vertices so computation point is at origin
    shifted_verts = verts - r[None, :]

    # Gather face vertices: (M, 3, 3)
    face_v = shifted_verts[face_idx]

    # Detect degenerate faces with NaN vertices (e.g., from NaN-padded arrays
    # used when batching polyhedra with different face counts via jax.vmap).
    face_valid = jnp.all(jnp.isfinite(face_v), axis=(-2, -1))  # (M,)

    # Substitute a valid dummy triangle for NaN faces so the forward pass
    # stays NaN-free. This ensures clean gradient propagation: jnp.where
    # with lax.select blocks gradient flow through the unselected branch,
    # unlike nan_to_num which suffers from 0 * NaN = NaN in the backward pass.
    _dummy_tri = jnp.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=_float,
    )
    face_v = jnp.where(face_valid[:, None, None], face_v, _dummy_tri[None, :, :])

    # Pre-compute per-face invariants
    def _prepare_face(fv: Array) -> tuple[Array, Array, Array, Array]:
        sv = _build_segment_vectors(fv[0], fv[1], fv[2])
        pn = _build_plane_unit_normal(sv[0], sv[1])
        sn = _build_segment_unit_normals(sv, pn)
        return fv, sv, pn, sn

    all_fv, all_sv, all_pn, all_sn = jax.vmap(_prepare_face)(face_v)

    # Evaluate all faces
    all_pot, all_accel, all_tensor = jax.vmap(_evaluate_face)(all_fv, all_sv, all_pn, all_sn)

    # Zero out contributions from NaN-padded degenerate faces.
    all_pot = jnp.where(face_valid, all_pot, 0.0)
    all_accel = jnp.where(face_valid[:, None], all_accel, 0.0)
    all_tensor = jnp.where(face_valid[:, None], all_tensor, 0.0)

    # Safety net: zero out any remaining NaN/Inf from other degenerate faces
    # (e.g., zero-padded vertices creating zero-area triangles).
    all_pot = jnp.nan_to_num(all_pot, nan=0.0, posinf=0.0, neginf=0.0)
    all_accel = jnp.nan_to_num(all_accel, nan=0.0, posinf=0.0, neginf=0.0)
    all_tensor = jnp.nan_to_num(all_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    # Reduce: sum over faces
    total_pot = jnp.sum(all_pot)
    total_accel = jnp.sum(all_accel, axis=0)
    total_tensor = jnp.sum(all_tensor, axis=0)

    # Apply prefix: G * density
    prefix = jnp.asarray(_G, dtype=_float) * rho

    potential = (total_pot * prefix) / 2.0
    acceleration = total_accel * (-1.0 * prefix)
    tensor = total_tensor * prefix

    return potential, acceleration, tensor


def accel_polyhedral_gravity(
    r_point: ArrayLike,
    r_body: ArrayLike,
    R_body_to_inertial: ArrayLike,
    vertices: ArrayLike,
    faces: ArrayLike,
    density: ArrayLike,
) -> Array:
    """Gravitational acceleration from a polyhedral body in an inertial frame.

    Transforms the computation point into the body-fixed frame, evaluates
    the polyhedral gravity acceleration there, and rotates the result back
    to the inertial frame.

    This function is designed to integrate with
    :func:`~astrojax.datasets.damit_spin_to_rotation` and
    :func:`~astrojax.datasets.scale_shape_vertices`.

    Args:
        r_point: Position of the computation point in the inertial
            frame [m], shape ``(3,)``.
        r_body: Position of the body centre in the inertial
            frame [m], shape ``(3,)``.
        R_body_to_inertial: Rotation matrix from body-fixed to inertial
            frame, shape ``(3, 3)``.
        vertices: Mesh vertex coordinates in the body frame [m],
            shape ``(N, 3)``.
        faces: Triangle face indices (0-indexed), shape ``(M, 3)`` of int.
        density: Bulk density [kg/m^3], scalar.

    Returns:
        Gravitational acceleration in the inertial frame [m/s^2],
        shape ``(3,)``.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.orbit_dynamics import accel_polyhedral_gravity

        # Body at origin, identity rotation (body = inertial)
        vertices = jnp.array([
            [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
            [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
        ], dtype=jnp.float64)
        faces = jnp.array([
            [1,3,2],[0,3,1],[0,1,5],[0,5,4],[0,7,3],[0,4,7],
            [1,2,6],[1,6,5],[2,3,6],[3,7,6],[4,5,6],[4,6,7],
        ])

        r_point = jnp.array([5.0, 5.0, 5.0])
        r_body = jnp.zeros(3)
        R = jnp.eye(3)
        a = accel_polyhedral_gravity(r_point, r_body, R, vertices, faces, 1.0)
        ```
    """
    _float = get_dtype()
    r_p = jnp.asarray(r_point, dtype=_float)
    r_b = jnp.asarray(r_body, dtype=_float)
    R = jnp.asarray(R_body_to_inertial, dtype=_float)

    # Transform computation point to body frame
    r_bf = R.T @ (r_p - r_b)

    # Evaluate in body frame
    _, accel_bf, _ = polyhedral_gravity(r_bf, vertices, faces, density)

    # Rotate back to inertial
    return R @ accel_bf
