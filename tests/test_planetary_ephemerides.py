"""Tests for JPL approximate planetary ephemerides.

Validates heliocentric position output, JIT compatibility, vmap support,
and differentiability for all eight major planets.
"""

import jax
import jax.numpy as jnp

from astrojax import Epoch
from astrojax.constants import AU
from astrojax.orbit_dynamics import (
    EMB_ID,
    JUPITER_ID,
    MARS_ID,
    MERCURY_ID,
    NEPTUNE_ID,
    SATURN_ID,
    URANUS_ID,
    VENUS_ID,
    emb_position_jpl_approx,
    jupiter_position_jpl_approx,
    mars_position_jpl_approx,
    mercury_position_jpl_approx,
    neptune_position_jpl_approx,
    planet_position_jpl_approx,
    saturn_position_jpl_approx,
    uranus_position_jpl_approx,
    venus_position_jpl_approx,
)

# ---------------------------------------------------------------------------
# Orbital distance ranges (heliocentric, in AU) — generous bounds covering
# perihelion to aphelion with margin for the approximate model.
# ---------------------------------------------------------------------------
_DISTANCE_RANGES_AU = {
    "mercury": (0.30, 0.48),
    "venus": (0.71, 0.73),
    "emb": (0.98, 1.02),
    "mars": (1.36, 1.67),
    "jupiter": (4.9, 5.5),
    "saturn": (9.0, 10.1),
    "uranus": (18.3, 20.1),
    "neptune": (29.8, 30.4),
}

_PLANET_FUNCS = {
    "mercury": mercury_position_jpl_approx,
    "venus": venus_position_jpl_approx,
    "emb": emb_position_jpl_approx,
    "mars": mars_position_jpl_approx,
    "jupiter": jupiter_position_jpl_approx,
    "saturn": saturn_position_jpl_approx,
    "uranus": uranus_position_jpl_approx,
    "neptune": neptune_position_jpl_approx,
}

_PLANET_IDS = {
    "mercury": MERCURY_ID,
    "venus": VENUS_ID,
    "emb": EMB_ID,
    "mars": MARS_ID,
    "jupiter": JUPITER_ID,
    "saturn": SATURN_ID,
    "uranus": URANUS_ID,
    "neptune": NEPTUNE_ID,
}


def _epc_2024():
    return Epoch(2024, 6, 15)


# ===========================================================================
# Per-planet tests
# ===========================================================================

class TestMercury:
    def test_mercury_distance_magnitude(self):
        r = mercury_position_jpl_approx(_epc_2024())
        dist_au = float(jnp.linalg.norm(r)) / AU
        lo, hi = _DISTANCE_RANGES_AU["mercury"]
        assert lo < dist_au < hi

    def test_mercury_shape(self):
        r = mercury_position_jpl_approx(_epc_2024())
        assert r.shape == (3,)

    def test_mercury_jit_compatible(self):
        epc = _epc_2024()
        r_eager = mercury_position_jpl_approx(epc)
        r_jit = jax.jit(mercury_position_jpl_approx)(epc)
        assert jnp.allclose(r_eager, r_jit, atol=1.0)

    def test_mercury_different_epochs(self):
        r1 = mercury_position_jpl_approx(Epoch(2024, 1, 1))
        r2 = mercury_position_jpl_approx(Epoch(2024, 7, 1))
        assert not jnp.allclose(r1, r2, atol=1e6)


class TestVenus:
    def test_venus_distance_magnitude(self):
        r = venus_position_jpl_approx(_epc_2024())
        dist_au = float(jnp.linalg.norm(r)) / AU
        lo, hi = _DISTANCE_RANGES_AU["venus"]
        assert lo < dist_au < hi

    def test_venus_shape(self):
        r = venus_position_jpl_approx(_epc_2024())
        assert r.shape == (3,)

    def test_venus_jit_compatible(self):
        epc = _epc_2024()
        r_eager = venus_position_jpl_approx(epc)
        r_jit = jax.jit(venus_position_jpl_approx)(epc)
        assert jnp.allclose(r_eager, r_jit, rtol=2e-3)

    def test_venus_different_epochs(self):
        r1 = venus_position_jpl_approx(Epoch(2024, 1, 1))
        r2 = venus_position_jpl_approx(Epoch(2024, 7, 1))
        assert not jnp.allclose(r1, r2, atol=1e6)


class TestEMB:
    def test_emb_distance_magnitude(self):
        r = emb_position_jpl_approx(_epc_2024())
        dist_au = float(jnp.linalg.norm(r)) / AU
        lo, hi = _DISTANCE_RANGES_AU["emb"]
        assert lo < dist_au < hi

    def test_emb_shape(self):
        r = emb_position_jpl_approx(_epc_2024())
        assert r.shape == (3,)

    def test_emb_jit_compatible(self):
        epc = _epc_2024()
        r_eager = emb_position_jpl_approx(epc)
        r_jit = jax.jit(emb_position_jpl_approx)(epc)
        assert jnp.allclose(r_eager, r_jit, atol=1.0)

    def test_emb_different_epochs(self):
        r1 = emb_position_jpl_approx(Epoch(2024, 1, 1))
        r2 = emb_position_jpl_approx(Epoch(2024, 7, 1))
        assert not jnp.allclose(r1, r2, atol=1e6)


class TestMars:
    def test_mars_distance_magnitude(self):
        r = mars_position_jpl_approx(_epc_2024())
        dist_au = float(jnp.linalg.norm(r)) / AU
        lo, hi = _DISTANCE_RANGES_AU["mars"]
        assert lo < dist_au < hi

    def test_mars_shape(self):
        r = mars_position_jpl_approx(_epc_2024())
        assert r.shape == (3,)

    def test_mars_jit_compatible(self):
        epc = _epc_2024()
        r_eager = mars_position_jpl_approx(epc)
        r_jit = jax.jit(mars_position_jpl_approx)(epc)
        assert jnp.allclose(r_eager, r_jit, rtol=2e-3)

    def test_mars_different_epochs(self):
        r1 = mars_position_jpl_approx(Epoch(2024, 1, 1))
        r2 = mars_position_jpl_approx(Epoch(2024, 7, 1))
        assert not jnp.allclose(r1, r2, atol=1e6)


class TestJupiter:
    def test_jupiter_distance_magnitude(self):
        r = jupiter_position_jpl_approx(_epc_2024())
        dist_au = float(jnp.linalg.norm(r)) / AU
        lo, hi = _DISTANCE_RANGES_AU["jupiter"]
        assert lo < dist_au < hi

    def test_jupiter_shape(self):
        r = jupiter_position_jpl_approx(_epc_2024())
        assert r.shape == (3,)

    def test_jupiter_jit_compatible(self):
        epc = _epc_2024()
        r_eager = jupiter_position_jpl_approx(epc)
        r_jit = jax.jit(jupiter_position_jpl_approx)(epc)
        assert jnp.allclose(r_eager, r_jit, atol=1.0)

    def test_jupiter_different_epochs(self):
        r1 = jupiter_position_jpl_approx(Epoch(2020, 1, 1))
        r2 = jupiter_position_jpl_approx(Epoch(2024, 7, 1))
        assert not jnp.allclose(r1, r2, atol=1e6)


class TestSaturn:
    def test_saturn_distance_magnitude(self):
        r = saturn_position_jpl_approx(_epc_2024())
        dist_au = float(jnp.linalg.norm(r)) / AU
        lo, hi = _DISTANCE_RANGES_AU["saturn"]
        assert lo < dist_au < hi

    def test_saturn_shape(self):
        r = saturn_position_jpl_approx(_epc_2024())
        assert r.shape == (3,)

    def test_saturn_jit_compatible(self):
        epc = _epc_2024()
        r_eager = saturn_position_jpl_approx(epc)
        r_jit = jax.jit(saturn_position_jpl_approx)(epc)
        assert jnp.allclose(r_eager, r_jit, atol=1.0)

    def test_saturn_different_epochs(self):
        r1 = saturn_position_jpl_approx(Epoch(2020, 1, 1))
        r2 = saturn_position_jpl_approx(Epoch(2024, 7, 1))
        assert not jnp.allclose(r1, r2, atol=1e6)


class TestUranus:
    def test_uranus_distance_magnitude(self):
        r = uranus_position_jpl_approx(_epc_2024())
        dist_au = float(jnp.linalg.norm(r)) / AU
        lo, hi = _DISTANCE_RANGES_AU["uranus"]
        assert lo < dist_au < hi

    def test_uranus_shape(self):
        r = uranus_position_jpl_approx(_epc_2024())
        assert r.shape == (3,)

    def test_uranus_jit_compatible(self):
        epc = _epc_2024()
        r_eager = uranus_position_jpl_approx(epc)
        r_jit = jax.jit(uranus_position_jpl_approx)(epc)
        assert jnp.allclose(r_eager, r_jit, atol=1.0)

    def test_uranus_different_epochs(self):
        r1 = uranus_position_jpl_approx(Epoch(2020, 1, 1))
        r2 = uranus_position_jpl_approx(Epoch(2024, 7, 1))
        assert not jnp.allclose(r1, r2, atol=1e6)


class TestNeptune:
    def test_neptune_distance_magnitude(self):
        r = neptune_position_jpl_approx(_epc_2024())
        dist_au = float(jnp.linalg.norm(r)) / AU
        lo, hi = _DISTANCE_RANGES_AU["neptune"]
        assert lo < dist_au < hi

    def test_neptune_shape(self):
        r = neptune_position_jpl_approx(_epc_2024())
        assert r.shape == (3,)

    def test_neptune_jit_compatible(self):
        epc = _epc_2024()
        r_eager = neptune_position_jpl_approx(epc)
        r_jit = jax.jit(neptune_position_jpl_approx)(epc)
        assert jnp.allclose(r_eager, r_jit, rtol=2e-3)

    def test_neptune_different_epochs(self):
        r1 = neptune_position_jpl_approx(Epoch(2020, 1, 1))
        r2 = neptune_position_jpl_approx(Epoch(2024, 7, 1))
        assert not jnp.allclose(r1, r2, atol=1e6)


# ===========================================================================
# Algorithm tests
# ===========================================================================

class TestAlgorithm:
    def test_planet_ordering_by_distance(self):
        """Mean heliocentric distances should follow Mercury < Venus < ... < Neptune."""
        epc = _epc_2024()
        distances = []
        for name in ["mercury", "venus", "emb", "mars", "jupiter", "saturn", "uranus", "neptune"]:
            r = _PLANET_FUNCS[name](epc)
            distances.append(float(jnp.linalg.norm(r)))
        for i in range(len(distances) - 1):
            assert distances[i] < distances[i + 1]

    def test_general_dispatcher_matches_individual(self):
        """planet_position_jpl_approx(ID, epc) must match per-planet functions."""
        epc = _epc_2024()
        for name, planet_id in _PLANET_IDS.items():
            r_individual = _PLANET_FUNCS[name](epc)
            r_general = planet_position_jpl_approx(planet_id, epc)
            assert jnp.allclose(r_individual, r_general, atol=1.0), (
                f"Mismatch for {name}: individual vs general dispatcher"
            )

    def test_emb_approximately_1au(self):
        """Earth-Moon Barycenter should be approximately 1 AU from the Sun."""
        epc = _epc_2024()
        r = emb_position_jpl_approx(epc)
        dist_au = float(jnp.linalg.norm(r)) / AU
        assert 0.98 < dist_au < 1.02

    def test_epoch_at_j2000(self):
        """Sanity check at J2000.0 — positions should be finite and have correct magnitude."""
        epc = Epoch(2000, 1, 1, 12, 0, 0.0)
        for name in _PLANET_FUNCS:
            r = _PLANET_FUNCS[name](epc)
            assert jnp.all(jnp.isfinite(r)), f"{name} has non-finite values at J2000"
            dist_au = float(jnp.linalg.norm(r)) / AU
            lo, hi = _DISTANCE_RANGES_AU[name]
            assert lo < dist_au < hi, f"{name} distance {dist_au:.3f} AU out of range at J2000"


# ===========================================================================
# JAX integration tests
# ===========================================================================

class TestJAXIntegration:
    def test_vmap_over_epochs(self):
        """vmap should work over multiple epochs for each planet."""
        epochs = [Epoch(2024, m, 1) for m in range(1, 7)]

        # Stack Epoch pytree leaves manually for vmap
        batched_epc = Epoch._from_internal(
            jnp.stack([e._jd for e in epochs]),
            jnp.stack([e._seconds for e in epochs]),
            jnp.stack([e._kahan_c for e in epochs]),
        )

        results = jax.vmap(mars_position_jpl_approx)(batched_epc)
        assert results.shape == (6, 3)
        assert jnp.all(jnp.isfinite(results))

    def test_grad_through_ephemeris(self):
        """Ephemeris should be differentiable w.r.t. epoch seconds offset."""
        epc = _epc_2024()

        def loss_fn(seconds_offset):
            shifted = Epoch._from_internal(
                epc._jd, epc._seconds + seconds_offset, epc._kahan_c
            )
            r = mars_position_jpl_approx(shifted)
            return jnp.sum(r**2)

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(0.0)
        assert jnp.isfinite(g)
        assert float(g) != 0.0
