"""Built-in MATPOWER/PYPOWER-style IEEE 33-bus radial feeder data.

The branch impedances are stored in Ohm and loads in kW/kvar, following the
well-known Baran-Wu / MATPOWER ``case33bw`` data layout. The optional tie
lines are normally open and are not part of the closed radial tree.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RawCase33:
    """Container for the raw case33bw feeder."""

    buses: pd.DataFrame
    branches: pd.DataFrame
    root_bus: int
    base_kv: float
    base_mva: float


_BUS_LOADS: list[tuple[int, float, float]] = [
    (1, 0.0, 0.0),
    (2, 100.0, 60.0),
    (3, 90.0, 40.0),
    (4, 120.0, 80.0),
    (5, 60.0, 30.0),
    (6, 60.0, 20.0),
    (7, 200.0, 100.0),
    (8, 200.0, 100.0),
    (9, 60.0, 20.0),
    (10, 60.0, 20.0),
    (11, 45.0, 30.0),
    (12, 60.0, 35.0),
    (13, 60.0, 35.0),
    (14, 120.0, 80.0),
    (15, 60.0, 10.0),
    (16, 60.0, 20.0),
    (17, 60.0, 20.0),
    (18, 90.0, 40.0),
    (19, 90.0, 40.0),
    (20, 90.0, 40.0),
    (21, 90.0, 40.0),
    (22, 90.0, 40.0),
    (23, 90.0, 50.0),
    (24, 420.0, 200.0),
    (25, 420.0, 200.0),
    (26, 60.0, 25.0),
    (27, 60.0, 25.0),
    (28, 60.0, 20.0),
    (29, 120.0, 70.0),
    (30, 200.0, 600.0),
    (31, 150.0, 70.0),
    (32, 210.0, 100.0),
    (33, 60.0, 40.0),
]

_CLOSED_BRANCHES: list[tuple[int, int, float, float]] = [
    (1, 2, 0.0922, 0.0470),
    (2, 3, 0.4930, 0.2511),
    (3, 4, 0.3660, 0.1864),
    (4, 5, 0.3811, 0.1941),
    (5, 6, 0.8190, 0.7070),
    (6, 7, 0.1872, 0.6188),
    (7, 8, 0.7114, 0.2351),
    (8, 9, 1.0300, 0.7400),
    (9, 10, 1.0440, 0.7400),
    (10, 11, 0.1966, 0.0650),
    (11, 12, 0.3744, 0.1238),
    (12, 13, 1.4680, 1.1550),
    (13, 14, 0.5416, 0.7129),
    (14, 15, 0.5910, 0.5260),
    (15, 16, 0.7463, 0.5450),
    (16, 17, 1.2890, 1.7210),
    (17, 18, 0.7320, 0.5740),
    (2, 19, 0.1640, 0.1565),
    (19, 20, 1.5042, 1.3554),
    (20, 21, 0.4095, 0.4784),
    (21, 22, 0.7089, 0.9373),
    (3, 23, 0.4512, 0.3083),
    (23, 24, 0.8980, 0.7091),
    (24, 25, 0.8960, 0.7011),
    (6, 26, 0.2030, 0.1034),
    (26, 27, 0.2842, 0.1447),
    (27, 28, 1.0590, 0.9337),
    (28, 29, 0.8042, 0.7006),
    (29, 30, 0.5075, 0.2585),
    (30, 31, 0.9744, 0.9630),
    (31, 32, 0.3105, 0.3619),
    (32, 33, 0.3410, 0.5302),
]

_TIE_BRANCHES: list[tuple[int, int, float, float]] = [
    (8, 21, 2.0000, 2.0000),
    (9, 15, 2.0000, 2.0000),
    (12, 22, 2.0000, 2.0000),
    (18, 33, 0.5000, 0.5000),
    (25, 29, 0.5000, 0.5000),
]


def load_raw_case33bw(include_tie_lines: bool = False) -> RawCase33:
    """Load the built-in IEEE case33bw feeder.

    Args:
        include_tie_lines: Include the five normally open tie lines with
            ``status=False`` and ``is_tie=True``.

    Returns:
        RawCase33 with bus loads in kW/kvar and branch impedances in Ohm.
    """

    base_kv = 12.66
    base_mva = 10.0
    buses = pd.DataFrame(_BUS_LOADS, columns=["bus_id", "pd_kw", "qd_kvar"])
    buses["base_kv"] = base_kv

    rows = []
    for idx, (f_bus, t_bus, r_ohm, x_ohm) in enumerate(_CLOSED_BRANCHES, start=1):
        rows.append(
            {
                "branch_id": idx,
                "from_bus": f_bus,
                "to_bus": t_bus,
                "r_ohm": r_ohm,
                "x_ohm": x_ohm,
                "status": True,
                "is_tie": False,
            }
        )
    if include_tie_lines:
        start = len(rows) + 1
        for idx, (f_bus, t_bus, r_ohm, x_ohm) in enumerate(_TIE_BRANCHES, start=start):
            rows.append(
                {
                    "branch_id": idx,
                    "from_bus": f_bus,
                    "to_bus": t_bus,
                    "r_ohm": r_ohm,
                    "x_ohm": x_ohm,
                    "status": False,
                    "is_tie": True,
                }
            )
    branches = pd.DataFrame(rows)
    return RawCase33(buses=buses, branches=branches, root_bus=1, base_kv=base_kv, base_mva=base_mva)

