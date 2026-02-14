"""JPL approximate planetary ephemeris coefficients (Table 1, 1800-2050 AD).

Keplerian element coefficients for computing approximate heliocentric
positions of the eight major planets.  Each planet has six orbital
elements expressed as a linear function of time:

    element(T) = element_0 + element_dot * T

where *T* is Julian centuries from J2000 (TT).

The element order for each planet is:

    [a (AU), e, I (deg), L (deg), lon_peri (deg), lon_node (deg)]

References:
    E.M. Standish & J.G. Williams, "Keplerian Elements for
    Approximate Positions of the Major Planets",
    https://ssd.jpl.nasa.gov/planets/approx_pos.html
"""

import jax.numpy as jnp

# Planet ID constants
MERCURY_ID: int = 0
VENUS_ID: int = 1
EMB_ID: int = 2
MARS_ID: int = 3
JUPITER_ID: int = 4
SATURN_ID: int = 5
URANUS_ID: int = 6
NEPTUNE_ID: int = 7

# fmt: off
# Table 1: Keplerian elements and their rates for 1800-2050 AD.
# Shape: (8, 6, 2) — [planet, element, (value_at_epoch, rate_per_century)]
# Element order: [a, e, I, L, lon_peri, lon_node]
TABLE1_ELEMENTS = jnp.array([
    # Mercury
    [
        [0.38709927,    0.00000037],       # a (AU)
        [0.20563593,    0.00001906],       # e
        [7.00497902,   -0.00594749],       # I (deg)
        [252.25032350,  149472.67411175],  # L (deg)
        [77.45779628,   0.16047689],       # lon_peri (deg)
        [48.33076593,  -0.12534081],       # lon_node (deg)
    ],
    # Venus
    [
        [0.72333566,    0.00000390],
        [0.00677672,   -0.00004107],
        [3.39467605,   -0.00078890],
        [181.97909950,  58517.81538729],
        [131.60246718,  0.00268329],
        [76.67984255,  -0.27769418],
    ],
    # Earth-Moon Barycenter
    [
        [1.00000261,    0.00000562],
        [0.01671123,   -0.00004392],
        [-0.00001531,  -0.01294668],
        [100.46457166,  35999.37244981],
        [102.93768193,  0.32327364],
        [0.0,           0.0],
    ],
    # Mars
    [
        [1.52371034,    0.00001847],
        [0.09339410,    0.00007882],
        [1.84969142,   -0.00813131],
        [-4.55343205,   19140.30268499],
        [-23.94362959,  0.44441088],
        [49.55953891,  -0.29257343],
    ],
    # Jupiter
    [
        [5.20288700,   -0.00011607],
        [0.04838624,   -0.00013253],
        [1.30439695,   -0.00183714],
        [34.39644051,   3034.74612775],
        [14.72847983,   0.21252668],
        [100.47390909,  0.20469106],
    ],
    # Saturn
    [
        [9.53667594,   -0.00125060],
        [0.05386179,   -0.00050991],
        [2.48599187,    0.00193609],
        [49.95424423,   1222.49362201],
        [92.59887831,  -0.41897216],
        [113.66242448, -0.28867794],
    ],
    # Uranus
    [
        [19.18916464,  -0.00196176],
        [0.04725744,   -0.00004397],
        [0.77263783,   -0.00242939],
        [313.23810451,  428.48202785],
        [170.95427630,  0.40805281],
        [74.01692503,   0.04240589],
    ],
    # Neptune
    [
        [30.06992276,   0.00026291],
        [0.00859048,    0.00005105],
        [1.77004347,    0.00035372],
        [-55.12002969,  218.45945325],
        [44.96476227,  -0.32241464],
        [131.78422574, -0.00508664],
    ],
])
# fmt: on

# Obliquity of the ecliptic used by the JPL algorithm (degrees)
TABLE1_OBLIQUITY: float = 23.43928
