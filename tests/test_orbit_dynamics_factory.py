"""Tests for the orbit dynamics factory.

Tests cover:
- Configuration dataclasses and validation
- Preset factory methods
- Dynamics function output shape and correctness
- Physical sanity checks (drag opposes velocity, SRP direction, etc.)
- End-to-end integration with RK4, DP54
- JIT compatibility
- Brahe comparison (propagation)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from astrojax import Epoch
from astrojax.config import get_dtype, set_dtype
from astrojax.constants import GM_EARTH, R_EARTH
from astrojax.integrators import dp54_step, rk4_step, AdaptiveConfig
from astrojax.orbit_dynamics import (
    GravityModel,
    accel_gravity,
)
from astrojax.orbit_dynamics.config import ForceModelConfig, SpacecraftParams
from astrojax.orbit_dynamics.factory import create_orbit_dynamics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _leo_state(alt_km: float = 500.0) -> jnp.ndarray:
    """Circular equatorial LEO state at given altitude."""
    _float = get_dtype()
    sma = _float(R_EARTH + alt_km * 1e3)
    v_circ = jnp.sqrt(_float(GM_EARTH) / sma)
    return jnp.array([sma, 0.0, 0.0, 0.0, float(v_circ), 0.0], dtype=_float)


def _geo_state() -> jnp.ndarray:
    """Circular equatorial GEO state."""
    _float = get_dtype()
    sma = _float(42164e3)
    v_circ = jnp.sqrt(_float(GM_EARTH) / sma)
    return jnp.array([sma, 0.0, 0.0, 0.0, float(v_circ), 0.0], dtype=_float)


def _epoch() -> Epoch:
    """Reference epoch: 2024-06-15 12:00:00."""
    return Epoch(2024, 6, 15, 12, 0, 0)


# ===========================================================================
# Configuration dataclass tests
# ===========================================================================


class TestSpacecraftParams:
    """Tests for SpacecraftParams."""

    def test_defaults(self):
        """Default SpacecraftParams has expected values."""
        sc = SpacecraftParams()
        assert sc.mass == 1000.0
        assert sc.drag_area == 10.0
        assert sc.srp_area == 10.0
        assert sc.cd == 2.2
        assert sc.cr == 1.3

    def test_custom(self):
        """SpacecraftParams accepts custom values."""
        sc = SpacecraftParams(mass=500.0, cd=2.5, cr=1.0)
        assert sc.mass == 500.0
        assert sc.cd == 2.5
        assert sc.cr == 1.0

    def test_frozen(self):
        """SpacecraftParams is immutable."""
        sc = SpacecraftParams()
        with pytest.raises(AttributeError):
            sc.mass = 42.0


class TestForceModelConfig:
    """Tests for ForceModelConfig."""

    def test_defaults(self):
        """Default config is point-mass gravity only."""
        cfg = ForceModelConfig()
        assert cfg.gravity_type == "point_mass"
        assert cfg.gravity_model is None
        assert cfg.drag is False
        assert cfg.srp is False
        assert cfg.third_body_sun is False
        assert cfg.third_body_moon is False

    def test_invalid_gravity_type(self):
        """Invalid gravity_type raises ValueError."""
        with pytest.raises(ValueError, match="gravity_type"):
            ForceModelConfig(gravity_type="j2_only")

    def test_invalid_eclipse_model(self):
        """Invalid eclipse_model raises ValueError."""
        with pytest.raises(ValueError, match="eclipse_model"):
            ForceModelConfig(eclipse_model="penumbra")

    def test_frozen(self):
        """ForceModelConfig is immutable."""
        cfg = ForceModelConfig()
        with pytest.raises(AttributeError):
            cfg.drag = True


class TestPresets:
    """Tests for ForceModelConfig preset factory methods."""

    def test_two_body(self):
        """two_body() preset has point-mass gravity only."""
        cfg = ForceModelConfig.two_body()
        assert cfg.gravity_type == "point_mass"
        assert cfg.drag is False
        assert cfg.srp is False

    def test_leo_default(self):
        """leo_default() includes full perturbation set."""
        cfg = ForceModelConfig.leo_default()
        assert cfg.gravity_type == "spherical_harmonics"
        assert cfg.gravity_model is not None
        assert cfg.gravity_degree == 20
        assert cfg.gravity_order == 20
        assert cfg.drag is True
        assert cfg.srp is True
        assert cfg.third_body_sun is True
        assert cfg.third_body_moon is True

    def test_leo_default_custom_model(self):
        """leo_default() accepts a custom gravity model."""
        model = GravityModel.from_type("JGM3")
        cfg = ForceModelConfig.leo_default(gravity_model=model)
        assert cfg.gravity_model is model

    def test_geo_default(self):
        """geo_default() has no drag but includes SRP and third-body."""
        cfg = ForceModelConfig.geo_default()
        assert cfg.gravity_type == "spherical_harmonics"
        assert cfg.gravity_degree == 8
        assert cfg.gravity_order == 8
        assert cfg.drag is False
        assert cfg.srp is True
        assert cfg.third_body_sun is True
        assert cfg.third_body_moon is True


# ===========================================================================
# Factory validation tests
# ===========================================================================


class TestCreateOrbitDynamicsValidation:
    """Tests for create_orbit_dynamics argument validation."""

    def test_default_config(self):
        """No config defaults to two-body."""
        dynamics = create_orbit_dynamics(_epoch())
        assert callable(dynamics)

    def test_spherical_harmonics_requires_model(self):
        """ValueError when spherical_harmonics has no gravity_model."""
        cfg = ForceModelConfig(gravity_type="spherical_harmonics")
        with pytest.raises(ValueError, match="gravity_model"):
            create_orbit_dynamics(_epoch(), cfg)


# ===========================================================================
# Two-body dynamics tests
# ===========================================================================


class TestTwoBodyDynamics:
    """Tests for point-mass gravity dynamics."""

    def test_derivative_shape(self):
        """Output is shape (6,)."""
        dynamics = create_orbit_dynamics(_epoch())
        x = _leo_state()
        dx = dynamics(0.0, x)
        assert dx.shape == (6,)

    def test_matches_manual(self):
        """Two-body acceleration matches manual -GM*r/|r|^3."""
        dynamics = create_orbit_dynamics(_epoch())
        x = _leo_state()
        dx = dynamics(0.0, x)

        r = x[:3]
        v = x[3:6]
        r_norm = jnp.linalg.norm(r)
        a_expected = -GM_EARTH * r / r_norm**3

        assert jnp.allclose(dx[:3], v, atol=1e-10)
        assert jnp.allclose(dx[3:6], a_expected, atol=1e-10)

    def test_matches_accel_gravity(self):
        """Factory two-body matches standalone accel_gravity."""
        dynamics = create_orbit_dynamics(_epoch())
        x = _leo_state()
        dx = dynamics(0.0, x)

        a_ref = accel_gravity(x[:3])
        assert jnp.allclose(dx[3:6], a_ref, atol=1e-10)

    def test_circular_orbit_period(self):
        """Propagate one orbit — position returns close to start."""
        _float = get_dtype()
        x0 = _leo_state()
        sma = float(jnp.linalg.norm(x0[:3]))
        T_period = 2.0 * jnp.pi * jnp.sqrt(_float(sma**3 / GM_EARTH))

        dynamics = create_orbit_dynamics(_epoch())

        # Use 1 s step for better accuracy at float32
        dt = 1.0
        n_steps = int(float(T_period) / dt)

        def scan_step(carry, _):
            t, state = carry
            result = rk4_step(dynamics, t, state, dt)
            return (t + dt, result.state), None

        (_, x_final), _ = jax.lax.scan(
            scan_step, (jnp.float32(0.0), x0), None, length=n_steps
        )

        # Position should return to near initial.
        # At float32 with dt=1s, RK4 truncation + float roundoff over ~5600
        # steps gives ~7 km error for a 500 km LEO orbit.
        pos_error = jnp.linalg.norm(x_final[:3] - x0[:3])
        assert float(pos_error) < 10000.0  # 10 km tolerance for float32


# ===========================================================================
# Spherical harmonics tests
# ===========================================================================


class TestSphericalHarmonicsDynamics:
    """Tests for spherical harmonic gravity dynamics."""

    def test_differs_from_point_mass(self):
        """J2 perturbation produces different acceleration than point-mass."""
        epc = _epoch()
        x = _leo_state()

        dyn_pm = create_orbit_dynamics(epc)
        dx_pm = dyn_pm(0.0, x)

        model = GravityModel.from_type("JGM3")
        cfg = ForceModelConfig(
            gravity_type="spherical_harmonics",
            gravity_model=model,
            gravity_degree=2,
            gravity_order=2,
        )
        dyn_sh = create_orbit_dynamics(epc, cfg)
        dx_sh = dyn_sh(0.0, x)

        # The J2 perturbation at LEO should be ~1e-3 to 1e-2 m/s^2
        diff = jnp.linalg.norm(dx_sh[3:6] - dx_pm[3:6])
        assert float(diff) > 1e-5  # Non-trivial difference

    def test_degree_0_matches_point_mass(self):
        """Degree-0 spherical harmonics should match point-mass."""
        epc = _epoch()
        x = _leo_state()

        dyn_pm = create_orbit_dynamics(epc)
        dx_pm = dyn_pm(0.0, x)

        model = GravityModel.from_type("JGM3")
        cfg = ForceModelConfig(
            gravity_type="spherical_harmonics",
            gravity_model=model,
            gravity_degree=0,
            gravity_order=0,
        )
        dyn_sh = create_orbit_dynamics(epc, cfg)
        dx_sh = dyn_sh(0.0, x)

        # Degree 0 = GM/r^2, should be very close to point-mass.
        # The SH path rotates to ECEF and back, introducing tiny cross-axis
        # components from the frame rotation at float32 precision.
        assert jnp.allclose(dx_sh[3:6], dx_pm[3:6], atol=1e-5)


# ===========================================================================
# Perturbation physics tests
# ===========================================================================


class TestDragPhysics:
    """Tests for atmospheric drag acceleration."""

    def test_drag_opposes_velocity(self):
        """Drag acceleration is anti-parallel to velocity."""
        epc = _epoch()
        x = _leo_state(alt_km=400.0)

        cfg_no_drag = ForceModelConfig()
        cfg_drag = ForceModelConfig(
            drag=True,
            spacecraft=SpacecraftParams(mass=100.0, drag_area=10.0, cd=2.2),
        )

        dx_no = create_orbit_dynamics(epc, cfg_no_drag)(0.0, x)
        dx_yes = create_orbit_dynamics(epc, cfg_drag)(0.0, x)

        # Drag contribution
        a_drag = dx_yes[3:6] - dx_no[3:6]
        v = x[3:6]

        # Drag should oppose velocity: dot(a_drag, v) < 0
        dot = float(jnp.dot(a_drag, v))
        assert dot < 0.0, f"Drag should oppose velocity, got dot={dot}"


class TestSRPPhysics:
    """Tests for solar radiation pressure acceleration."""

    def test_srp_direction(self):
        """SRP pushes object away from Sun."""
        epc = _epoch()
        x = _leo_state()

        cfg_srp = ForceModelConfig(
            srp=True,
            eclipse_model="none",  # no shadow
            spacecraft=SpacecraftParams(mass=100.0, srp_area=10.0, cr=1.3),
        )
        cfg_no = ForceModelConfig()

        dx_srp = create_orbit_dynamics(epc, cfg_srp)(0.0, x)
        dx_no = create_orbit_dynamics(epc, cfg_no)(0.0, x)

        a_srp = dx_srp[3:6] - dx_no[3:6]

        # SRP should push away from Sun
        from astrojax.orbit_dynamics import sun_position

        r_sun = sun_position(epc)
        d = x[:3] - r_sun  # object-to-sun direction (object minus sun is away from sun)
        # SRP force is in the direction from Sun to object (same as d)
        dot = float(jnp.dot(a_srp, d))
        assert dot > 0.0, f"SRP should push away from Sun, got dot={dot}"

    def test_eclipse_reduces_srp(self):
        """SRP is zero when spacecraft is in Earth's shadow."""
        epc = _epoch()

        # Place spacecraft behind Earth (shadow)
        from astrojax.orbit_dynamics import sun_position

        r_sun = sun_position(epc)
        sun_dir = r_sun / jnp.linalg.norm(r_sun)

        # Position behind Earth, opposite to Sun direction
        r_shadow = -(R_EARTH + 200e3) * sun_dir
        v = jnp.array([0.0, 7500.0, 0.0])
        x_shadow = jnp.concatenate([r_shadow, v])

        cfg_srp = ForceModelConfig(
            srp=True,
            eclipse_model="cylindrical",
            spacecraft=SpacecraftParams(mass=100.0, srp_area=10.0, cr=1.3),
        )
        cfg_no = ForceModelConfig()

        dx_srp = create_orbit_dynamics(epc, cfg_srp)(0.0, x_shadow)
        dx_no = create_orbit_dynamics(epc, cfg_no)(0.0, x_shadow)

        # In full shadow, SRP contribution should be zero
        a_srp = dx_srp[3:6] - dx_no[3:6]
        assert jnp.allclose(a_srp, 0.0, atol=1e-20)


class TestThirdBodyPhysics:
    """Tests for third-body gravitational perturbations."""

    def test_sun_perturbation_nonzero(self):
        """Sun third-body acceleration is nonzero at LEO."""
        epc = _epoch()
        x = _leo_state()

        cfg = ForceModelConfig(third_body_sun=True)
        cfg_no = ForceModelConfig()

        dx_sun = create_orbit_dynamics(epc, cfg)(0.0, x)
        dx_no = create_orbit_dynamics(epc, cfg_no)(0.0, x)

        a_sun = dx_sun[3:6] - dx_no[3:6]
        mag = float(jnp.linalg.norm(a_sun))
        # Sun perturbation at LEO is ~5e-7 m/s^2
        assert mag > 1e-8
        assert mag < 1e-4

    def test_moon_perturbation_nonzero(self):
        """Moon third-body acceleration is nonzero at LEO."""
        epc = _epoch()
        x = _leo_state()

        cfg = ForceModelConfig(third_body_moon=True)
        cfg_no = ForceModelConfig()

        dx_moon = create_orbit_dynamics(epc, cfg)(0.0, x)
        dx_no = create_orbit_dynamics(epc, cfg_no)(0.0, x)

        a_moon = dx_moon[3:6] - dx_no[3:6]
        mag = float(jnp.linalg.norm(a_moon))
        # Moon perturbation at LEO is ~1e-6 m/s^2
        assert mag > 1e-8
        assert mag < 1e-3


# ===========================================================================
# Full force model test
# ===========================================================================


class TestFullForceModel:
    """Tests with all perturbations enabled."""

    def test_full_force_model_propagation(self):
        """All forces on, propagate one orbit without NaN."""
        epc = _epoch()
        x0 = _leo_state(alt_km=400.0)
        cfg = ForceModelConfig.leo_default()
        dynamics = create_orbit_dynamics(epc, cfg)

        dt = 30.0
        n_steps = 200  # ~100 minutes

        def scan_step(carry, _):
            t, state = carry
            result = rk4_step(dynamics, t, state, dt)
            return (t + dt, result.state), result.state

        (_, x_final), traj = jax.lax.scan(
            scan_step, (jnp.float32(0.0), x0), None, length=n_steps
        )

        # No NaNs
        assert not jnp.any(jnp.isnan(x_final))
        # Altitude still reasonable (not crashed or escaped)
        r_final = float(jnp.linalg.norm(x_final[:3]))
        assert r_final > R_EARTH
        assert r_final < R_EARTH + 1000e3


# ===========================================================================
# JAX compatibility tests
# ===========================================================================


class TestJAXCompatibility:
    """Tests for JIT, vmap, and lax.scan compatibility."""

    def test_jit_compatible(self):
        """dynamics function is JIT-compilable."""
        dynamics = create_orbit_dynamics(_epoch())
        x = _leo_state()

        jit_dynamics = jax.jit(dynamics)
        dx = jit_dynamics(0.0, x)
        assert dx.shape == (6,)
        assert not jnp.any(jnp.isnan(dx))

    def test_jit_with_perturbations(self):
        """JIT works with spherical harmonics and perturbations."""
        cfg = ForceModelConfig.leo_default()
        dynamics = create_orbit_dynamics(_epoch(), cfg)
        x = _leo_state()

        jit_dynamics = jax.jit(dynamics)
        dx = jit_dynamics(0.0, x)
        assert dx.shape == (6,)
        assert not jnp.any(jnp.isnan(dx))


# ===========================================================================
# Integrator end-to-end tests
# ===========================================================================


class TestIntegratorEndToEnd:
    """Tests for dynamics function with actual integrators."""

    def test_rk4_step(self):
        """Single RK4 step produces valid result."""
        dynamics = create_orbit_dynamics(_epoch())
        x0 = _leo_state()
        result = rk4_step(dynamics, 0.0, x0, 60.0)

        assert result.state.shape == (6,)
        assert not jnp.any(jnp.isnan(result.state))
        # Position should have changed
        assert not jnp.allclose(result.state[:3], x0[:3])

    def test_dp54_step(self):
        """Single DP54 step produces valid result."""
        dynamics = create_orbit_dynamics(_epoch())
        x0 = _leo_state()
        config = AdaptiveConfig(abs_tol=1e-6, rel_tol=1e-3)
        result = dp54_step(dynamics, 0.0, x0, 60.0, config=config)

        assert result.state.shape == (6,)
        assert not jnp.any(jnp.isnan(result.state))

    def test_dynamics_with_control(self):
        """Thrust control via integrator control parameter."""
        dynamics = create_orbit_dynamics(_epoch())
        x0 = _leo_state()

        def thrust(t, x):
            """Constant prograde thrust."""
            v = x[3:6]
            v_hat = v / jnp.linalg.norm(v)
            return jnp.concatenate([jnp.zeros(3), 1e-3 * v_hat])

        result_no = rk4_step(dynamics, 0.0, x0, 60.0)
        result_thrust = rk4_step(dynamics, 0.0, x0, 60.0, control=thrust)

        # Thrust should increase velocity magnitude
        v_no = float(jnp.linalg.norm(result_no.state[3:6]))
        v_thrust = float(jnp.linalg.norm(result_thrust.state[3:6]))
        assert v_thrust > v_no


# ===========================================================================
# Brahe comparison tests
# ===========================================================================


class TestBraheComparison:
    """Comparison tests against brahe for propagation."""

    @pytest.fixture(autouse=True)
    def _use_float64(self):
        """Use float64 for brahe comparison tests."""
        set_dtype(jnp.float64)
        yield
        set_dtype(jnp.float32)

    def test_two_body_propagation_vs_brahe(self):
        """Two-body propagation should match brahe RK4."""
        try:
            import brahe as bh
        except ImportError:
            pytest.skip("brahe not installed")

        # Initial state: 500 km circular equatorial orbit
        sma = float(R_EARTH + 500e3)
        gm = float(GM_EARTH)
        v_circ = np.sqrt(gm / sma)

        x0_aj = jnp.array([sma, 0.0, 0.0, 0.0, v_circ, 0.0])
        x0_bh = np.array([sma, 0.0, 0.0, 0.0, v_circ, 0.0])

        # astrojax propagation
        epc = Epoch(2024, 6, 15, 12, 0, 0)
        dynamics = create_orbit_dynamics(epc)
        dt = 10.0

        state_aj = x0_aj
        for _ in range(10):
            result = rk4_step(dynamics, 0.0, state_aj, dt)
            state_aj = result.state

        # brahe propagation
        def bh_two_body(t, x):
            r = x[:3]
            v = x[3:]
            r_norm = np.linalg.norm(r)
            a = -gm * r / r_norm**3
            return np.concatenate([v, a])

        bh_rk4 = bh.RK4Integrator(6, bh_two_body)
        state_bh = x0_bh
        for _ in range(10):
            state_bh = bh_rk4.step(0.0, state_bh, dt)

        # Compare: RK4 is deterministic, so positions should match closely
        np.testing.assert_allclose(
            np.array(state_aj[:3]),
            state_bh[:3],
            rtol=1e-6,
            atol=1.0,
            err_msg="Two-body position mismatch vs brahe",
        )
        np.testing.assert_allclose(
            np.array(state_aj[3:6]),
            state_bh[3:6],
            rtol=1e-6,
            atol=1e-3,
            err_msg="Two-body velocity mismatch vs brahe",
        )
