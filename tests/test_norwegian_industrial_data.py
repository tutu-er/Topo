"""Integration checks for the optional Norwegian open dataset."""

from pathlib import Path

import networkx as nx
import pytest

from terminal_case33.data.norwegian_industrial import (
    canonicalize_norwegian_terminal_radial,
    load_norwegian_active_power,
    load_norwegian_industrial_radial,
)


DATA_DIR = Path("data_external/norwegian_industrial_zenodo_7123537")
pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "branch.csv").exists(),
    reason="optional Zenodo dataset is not downloaded",
)


def test_norwegian_radials_are_terminal_meter_trees() -> None:
    """Both published radials remain trees with metered loads at leaves."""

    expected = {1: (54, 40), 2: (17, 5)}
    for radial, (bus_count, terminal_count) in expected.items():
        net = load_norwegian_industrial_radial(DATA_DIR, radial=radial)
        assert len(net.buses) == bus_count
        assert len(net.load_buses()) == terminal_count
        assert nx.is_tree(net.to_networkx_graph())
        assert set(net.get_leaf_buses()) == set(net.load_buses())


def test_norwegian_active_power_alignment() -> None:
    """Hourly customer records align on a common timestamp intersection."""

    net = load_norwegian_industrial_radial(DATA_DIR, radial=1)
    power = load_norwegian_active_power(net, DATA_DIR, periods=48)
    assert power.shape == (48, 40)
    assert power.notna().all().all()
    intervals = power.index.to_series().diff().dropna().dt.total_seconds()
    assert (intervals == 3600.0).all()


def test_norwegian_radial2_canonical_form() -> None:
    """The benchmark form removes unidentifiable chains without losing meters."""

    raw = load_norwegian_industrial_radial(DATA_DIR, radial=2)
    canonical = canonicalize_norwegian_terminal_radial(raw)
    graph = canonical.to_networkx_graph()

    assert nx.is_tree(graph)
    assert set(canonical.load_buses()) == set(raw.load_buses())
    assert set(canonical.get_leaf_buses()) == set(raw.load_buses())
    assert len(canonical.buses) == 8
    assert len(canonical.hidden_buses()) == 2
    assert all(graph.degree(node) >= 3 for node in canonical.hidden_buses())
    assert all(
        abs(float(row.r_ohm)) + abs(float(row.x_ohm)) > 0.0
        for row in canonical.branches.itertuples()
    )
    power = load_norwegian_active_power(canonical, DATA_DIR, periods=24)
    assert list(power.columns) == canonical.load_buses()