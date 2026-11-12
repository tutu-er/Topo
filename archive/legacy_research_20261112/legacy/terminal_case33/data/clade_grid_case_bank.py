"""Clade-grid terminal-only LV case bank for the aggregation phase diagram.

Each grid case is a hidden backbone chain rooted at bus 1 where every
attachment point feeds one service clade of ``clade_size`` observed terminals.
All hidden nodes keep degree >= 3 (``clade_size >= 2`` is enforced and the
chain tail hangs ``clade_size >= 2`` terminals), because degree-2 hidden
nodes are not identifiable from terminal-only measurements.

Impedances/lengths reuse the two-tier line library of
``paper_style_case_bank._make_case`` (overhead backbone vs. service drops).
Grid points are pre-registered into ``paper_style_case_bank.CASE_BUILDERS``
(``grid16_k2``/``grid16_k4``/``grid16_k8``, n=16 fixed), so
``experiments.common.simulate_case()`` picks them up by key unchanged.
"""

from __future__ import annotations

from terminal_case33.data.paper_style_case_bank import _make_case
from terminal_case33.models.network import TerminalizedNetwork

TERMINAL_ID_BASE = 101
BACKBONE_LENGTHS_M = (58.0, 74.0)
SERVICE_LENGTHS_M = (16.0, 27.0, 38.0, 52.0)


def make_clade_grid_case(
    n_terminals: int = 16,
    clade_size: int = 4,
    *,
    impedance_scale: float = 5.0,
    root_backbone_length_m: float = 70.0,
    case_name: str | None = None,
) -> TerminalizedNetwork:
    """Build a hidden backbone chain with uniform service clades.

    Layout: root bus 1 -> hidden chain 2-3-...-(m+1) with
    ``m = n_terminals // clade_size`` attachment points; each attachment point
    hangs ``clade_size`` observed terminals (ids ``101..100+n_terminals`` in
    chain order, so the chain-tail clade is the last ``clade_size`` ids).

    Degrees: interior attachment points have ``clade_size + 2`` (clade plus
    backbone predecessor/successor), the first has ``clade_size + 1`` plus the
    root edge, and the chain tail has ``clade_size + 1``; all >= 3 because
    ``clade_size >= 2``.
    """

    if clade_size < 2:
        raise ValueError("clade_size must be >= 2 so every hidden node has degree >= 3")
    if n_terminals <= 0 or n_terminals % clade_size != 0:
        raise ValueError("n_terminals must be a positive multiple of clade_size")
    n_attachment = n_terminals // clade_size

    hidden = list(range(2, 2 + n_attachment))
    edges: list[tuple[int, int, float, str]] = [(1, hidden[0], root_backbone_length_m, "backbone")]
    for idx in range(len(hidden) - 1):
        edges.append(
            (hidden[idx], hidden[idx + 1], BACKBONE_LENGTHS_M[idx % len(BACKBONE_LENGTHS_M)], "backbone")
        )

    terminals: list[tuple[int, float, float]] = []
    for clade_idx, attach in enumerate(hidden):
        for j in range(clade_size):
            bus = TERMINAL_ID_BASE + clade_idx * clade_size + j
            pd_kw = round(4.6 + 0.45 * ((5 * bus) % 11), 2)
            qd_kvar = round(0.34 * pd_kw + 0.15 * (bus % 3), 2)
            terminals.append((bus, pd_kw, qd_kvar))
            # Uneven service lengths (offset per clade) keep distances identifiable.
            length_m = SERVICE_LENGTHS_M[(j + clade_idx) % len(SERVICE_LENGTHS_M)]
            edges.append((attach, bus, length_m, "service"))

    net = _make_case(
        case_name=case_name or f"clade_grid_n{n_terminals}_k{clade_size}",
        hidden=hidden,
        terminals=terminals,
        edges=edges,
        impedance_scale=impedance_scale,
        note=(
            f"Clade-grid case: {n_attachment}-point hidden backbone chain, "
            f"one size-{clade_size} service clade per attachment point"
        ),
    )
    net.metadata.update(
        {
            "clade_grid": True,
            "clade_size": int(clade_size),
            "n_attachment_points": int(n_attachment),
            "terminal_ids": [bus for bus, _pd, _qd in terminals],
        }
    )
    return net


def build_grid16_k2_case() -> TerminalizedNetwork:
    """16 terminals as 8 backbone attachment points with size-2 clades.

    Uses a lower ``impedance_scale`` than the k4/k8 grids: the 8-segment
    backbone chain would otherwise accumulate enough series impedance to push
    the radial AC power flow into voltage collapse (non-convergence) under the
    standard load profiles, while k4/k8 converge at the default scale 5.0.
    """

    return make_clade_grid_case(16, 2, impedance_scale=2.0, case_name="grid16_k2")


def build_grid16_k4_case() -> TerminalizedNetwork:
    """16 terminals as 4 backbone attachment points with size-4 clades.

    Uses ``impedance_scale=3.5`` instead of the default 5.0: at 5.0 the
    4-segment backbone chain intermittently fails to converge in the radial
    AC power flow under the heavier load profiles (high_load, storm_front),
    while 3.5 converges for every profile/seed/T combination in the sweep
    matrix (worst Vmin about 0.67, comparable to the k2 grid at scale 2.0).
    """

    return make_clade_grid_case(16, 4, impedance_scale=3.5, case_name="grid16_k4")


def build_grid16_k8_case() -> TerminalizedNetwork:
    """16 terminals as 2 backbone attachment points with size-8 clades."""

    return make_clade_grid_case(16, 8, case_name="grid16_k8")


GRID_CASE_BUILDERS = {
    "grid16_k2": build_grid16_k2_case,
    "grid16_k4": build_grid16_k4_case,
    "grid16_k8": build_grid16_k8_case,
}
