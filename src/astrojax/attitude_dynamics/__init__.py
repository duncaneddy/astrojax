"""Rigid-body attitude dynamics for spacecraft attitude propagation.

Provides torque models and a configurable dynamics factory for
propagating the attitude state (quaternion + angular velocity)
of a rigid spacecraft:

- **Euler dynamics**: Quaternion kinematics and Euler's rotational equation
- **Gravity gradient**: Gravity gradient torque model
- **Factory**: Composable dynamics function compatible with all integrators
- **Utilities**: State normalization helpers
"""

from .config import AttitudeDynamicsConfig, SpacecraftInertia
from .euler_dynamics import euler_equation, quaternion_derivative
from .factory import create_attitude_dynamics
from .gravity_gradient import torque_gravity_gradient
from .utils import normalize_attitude_state

__all__ = [
    # Config
    "AttitudeDynamicsConfig",
    "SpacecraftInertia",
    # Euler dynamics
    "quaternion_derivative",
    "euler_equation",
    # Torque models
    "torque_gravity_gradient",
    # Factory
    "create_attitude_dynamics",
    # Utilities
    "normalize_attitude_state",
]
