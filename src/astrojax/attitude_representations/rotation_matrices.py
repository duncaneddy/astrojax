import jax.numpy as jnp

from astrojax.utils import to_radians


def Rx(angle:float, use_degrees:bool=False) -> jnp.ndarray:
    """Rotation matrix, for a rotation about the x-axis.

    Args:
        angle (float): Counter-clockwise angle of rotation as viewed 
            looking back along the postive direction of the rotation axis.
        use_degrees (bool): Handle input and output in degrees. Default: ``False``

    Returns:
        jnp.ndarray: Rotation matrix.

    References:

        1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and Applications*, 2012, p.27.
    """
    angle = to_radians(angle, use_degrees)

    c = jnp.cos(angle)
    s = jnp.sin(angle)

    return jnp.array([[1.0,  0.0,  0.0],
                      [0.0,   +c,   +s],
                      [0.0,   -s,   +c]])

def Ry(angle:float, use_degrees:bool=False) -> jnp.ndarray:
    """Rotation matrix, for a rotation about the y-axis.

    Args:
        angle (float): Counter-clockwise angle of rotation as viewed 
            looking back along the postive direction of the rotation axis.
        use_degrees (bool): Handle input and output in degrees. Default: ``False``

    Returns:
        jnp.ndarray: Rotation matrix.

    References:

        1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and Applications*, 2012, p.27.
    """
    angle = to_radians(angle, use_degrees)

    c = jnp.cos(angle)
    s = jnp.sin(angle)

    return jnp.array([[ +c,  0.0,   -s],
                      [0.0, +1.0,  0.0],
                      [ +s,  0.0,   +c]])

def Rz(angle:float, use_degrees:bool=False) -> jnp.ndarray:
    """Rotation matrix, for a rotation about the z-axis.

    Args:
        angle (float): Counter-clockwise angle of rotation as viewed 
            looking back along the postive direction of the rotation axis.
        use_degrees (bool): Handle input and output in degrees. Default: ``False``

    Returns:
        jnp.ndarray: Rotation matrix.

    References:

        1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and Applications*, 2012, p.27.
    """
    angle = to_radians(angle, use_degrees)

    c = jnp.cos(angle)
    s = jnp.sin(angle)

    return jnp.array([[ +c,   +s,  0.0],
                      [ -s,   +c,  0.0],
                      [0.0,  0.0,  1.0]])
