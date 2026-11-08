"""
Course definitions, independent of MuJoCo (used by scenario.py and by the
plotting code, which should not import MuJoCo).
"""

import numpy as np


def default_course():
    """
    (time, kind, parameter, label). Events are cumulative: the course gets
    harder over time.

       0- 4 s  flat hard ground, no payload   -- baseline behaviour
       4- 8 s  + off-centre payload            -- force AND moment disturbance
       8-12 s  + ground softens                -- contact-dependent disturbance
      12-16 s  + ground becomes slippery       -- friction cone violated
      16-20 s  + slope                         -- gravity rotates
    """
    return [
        (4.0,  "payload", 3.0,  "OFF-CENTRE PAYLOAD 3 kg"),
        (8.0,  "soft",    0.12, "ground SOFTENS"),
        (12.0, "mu",      0.25, "ground SLIPPERY"),
        (16.0, "slope",   6.0,  "SLOPE 6 deg"),
    ]


def random_payload_pos(rng, zmin=0.04, zmax=0.08):
    """
    Random payload position in the body frame, on top of the trunk and
    clearly off-centre (|x| >= 0.05) so that it creates a moment; a centred
    payload would excite only theta_z.
    """
    x = rng.uniform(0.05, 0.16) * rng.choice([-1.0, 1.0])
    y = rng.uniform(0.0, 0.05) * rng.choice([-1.0, 1.0])
    z = rng.uniform(zmin, zmax)
    return np.array([x, y, z])


def terrain_course():
    """
    Terrain course. Ground changes come with position, not time:

        x (m)        zone                  height
        -----------------------------------------------------
        0.0 - 1.5    flat start            0
        1.5 - 3.2    RAMP UP               0 -> 0.20 (steepest 10 deg)
        3.2 - 4.0    plateau               0.20
        4.0 - 5.7    ROUGH                 0.20 +- 2.3 cm, left/right differ
        5.7 - 7.4    RAMP DOWN             0.20 -> 0 (same angle as up)
        7.4 +        flat finish           0

    The only time-triggered event is the payload. At 0.3 m/s the robot is
    at ~1.5 m at t = 6 s, i.e. at the start of the ramp: payload and ramp
    arrive together.
    """
    return [
        (6.0, "payload", 3.0, "OFF-CENTRE PAYLOAD 3 kg"),
    ]


TERRAIN_ZONES = [
    (0.0, 1.5, "flat"),
    (1.5, 3.2, "RAMP UP"),
    (3.2, 4.0, "plateau"),
    (4.0, 5.7, "ROUGH"),
    (5.7, 7.4, "RAMP DOWN"),
    (7.4, 9.5, "flat finish"),
]
