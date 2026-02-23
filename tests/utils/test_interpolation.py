"""Tests for the Hermite cubic interpolation utilities."""

import jax
import jax.numpy as jnp

from astrojax.utils._interpolation import hermite_interp, make_hermite_position_fn


class TestHermiteInterp:
    def test_exact_at_endpoints(self):
        """Hermite interpolation returns p0 at t0 and p1 at t1."""
        p0 = jnp.array([1.0, 2.0, 3.0])
        p1 = jnp.array([4.0, 5.0, 6.0])
        v0 = jnp.array([0.5, 0.5, 0.5])
        v1 = jnp.array([0.3, 0.3, 0.3])

        result_at_t0 = hermite_interp(0.0, 0.0, 1.0, p0, p1, v0, v1)
        result_at_t1 = hermite_interp(1.0, 0.0, 1.0, p0, p1, v0, v1)

        assert jnp.allclose(result_at_t0, p0, atol=1e-6)
        assert jnp.allclose(result_at_t1, p1, atol=1e-6)

    def test_midpoint_accuracy(self):
        """Hermite exactly reproduces a cubic polynomial.

        For f(t) = t^3, f'(t) = 3t^2, Hermite should be exact since it's a
        cubic interpolant matching a cubic function.
        """
        # f(t) = t^3 on [1, 3]
        t0, t1 = 1.0, 3.0
        p0 = jnp.array([t0**3])
        p1 = jnp.array([t1**3])
        v0 = jnp.array([3 * t0**2])
        v1 = jnp.array([3 * t1**2])

        t_mid = 2.0
        result = hermite_interp(t_mid, t0, t1, p0, p1, v0, v1)
        expected = jnp.array([t_mid**3])

        assert jnp.allclose(result, expected, atol=1e-5)

    def test_vs_linear(self):
        """Hermite should be more accurate than linear for a curved trajectory.

        For f(t) = sin(t), Hermite uses derivative information and should
        beat simple linear interpolation.
        """
        t0, t1 = 0.0, 1.0
        p0 = jnp.array([jnp.sin(t0)])
        p1 = jnp.array([jnp.sin(t1)])
        v0 = jnp.array([jnp.cos(t0)])
        v1 = jnp.array([jnp.cos(t1)])

        t_query = 0.5
        exact = jnp.sin(t_query)

        hermite_result = hermite_interp(t_query, t0, t1, p0, p1, v0, v1)
        linear_result = 0.5 * p0 + 0.5 * p1

        hermite_err = jnp.abs(hermite_result[0] - exact)
        linear_err = jnp.abs(linear_result[0] - exact)

        assert hermite_err < linear_err


class TestMakeHermitePositionFn:
    def test_circular_orbit(self):
        """Interpolate a circular orbit; position error should be tiny."""
        n = 100
        period = 5400.0
        r = 7000e3
        times = jnp.linspace(0.0, period, n)

        omega = 2.0 * jnp.pi / period
        positions = jnp.stack(
            [r * jnp.cos(omega * times), r * jnp.sin(omega * times), jnp.zeros(n)],
            axis=-1,
        )
        velocities = jnp.stack(
            [
                -r * omega * jnp.sin(omega * times),
                r * omega * jnp.cos(omega * times),
                jnp.zeros(n),
            ],
            axis=-1,
        )

        pos_fn = make_hermite_position_fn(times, positions, velocities)

        # Query at midpoints between samples
        dt = period / n
        t_query = times[50] + dt * 0.37  # arbitrary offset within bracket
        result = pos_fn(t_query)
        expected = jnp.array([r * jnp.cos(omega * t_query), r * jnp.sin(omega * t_query), 0.0])

        # With 100 points over one orbit, Hermite error should be very small
        err = jnp.linalg.norm(result - expected)
        assert err < 1.0  # sub-metre for ~54s intervals on a 7000km orbit

    def test_jit_compatible(self):
        """make_hermite_position_fn result works under jax.jit."""
        times = jnp.array([0.0, 1.0, 2.0])
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        velocities = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        pos_fn = make_hermite_position_fn(times, positions, velocities)
        jitted = jax.jit(pos_fn)

        result_eager = pos_fn(0.5)
        result_jit = jitted(0.5)

        assert jnp.allclose(result_eager, result_jit, atol=1e-6)

    def test_vmap_compatible(self):
        """make_hermite_position_fn result works with jax.vmap."""
        times = jnp.array([0.0, 1.0, 2.0])
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        velocities = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        pos_fn = make_hermite_position_fn(times, positions, velocities)
        query_times = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])
        results = jax.vmap(pos_fn)(query_times)

        assert results.shape == (5, 3)
        # At t=0, should be [0,0,0]; at t=1 should be [1,0,0]
        assert jnp.allclose(results[0], positions[0], atol=1e-6)
        assert jnp.allclose(results[2], positions[1], atol=1e-6)

    def test_endpoint_values(self):
        """Returns exact positions at sample times."""
        times = jnp.array([0.0, 10.0, 20.0])
        positions = jnp.array([[100.0, 200.0, 300.0], [400.0, 500.0, 600.0], [700.0, 800.0, 900.0]])
        velocities = jnp.array([[30.0, 30.0, 30.0], [30.0, 30.0, 30.0], [30.0, 30.0, 30.0]])

        pos_fn = make_hermite_position_fn(times, positions, velocities)

        for i in range(len(times)):
            result = pos_fn(times[i])
            assert jnp.allclose(result, positions[i], atol=1e-5)
