"""Conductors used by the generated feeders.

Resistances and reactances are the phase values of the line codes of the Non-Synthetic European LV Test System
(aluminium; overhead bundles at 50 C, buried XLPE cables at 90 C). Ratings are rounded typical catalogue values and
are used only to reject implausible feeders, never by an identifier.
"""

#            ohm/km  ohm/km  A
CONDUCTORS = {
    "cable240": (0.160, 0.079, 355),
    "cable185": (0.210, 0.075, 305),
    "cable150": (0.264, 0.082, 270),
    "cable95":  (0.410, 0.083, 205),
    "cable25":  (1.230, 0.080, 100),
    "cable16":  (2.140, 0.090, 78),
    "bundle150": (0.231, 0.085, 285),
    "bundle95":  (0.359, 0.089, 215),
    "bundle50":  (0.718, 0.093, 140),
    "bundle25":  (1.340, 0.097, 90),
    "bundle16":  (2.140, 0.090, 70),
}


def impedance(conductor, length_m):
    """Phase resistance and reactance in ohm of a line."""
    r, x, _ = CONDUCTORS[conductor]
    return r*length_m/1000., x*length_m/1000.
