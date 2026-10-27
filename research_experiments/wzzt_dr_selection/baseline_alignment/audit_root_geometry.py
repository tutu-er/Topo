"""Reproduce the declared paper15 exact-R/X geometry diagnostic; no optimizers.

The original three-condition receipt was produced by an inline Python stdin
command. This source was extracted afterward; it is not the source hash of
that original execution. See root_geometry_reproduction.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = Path(__file__).with_name("root_case_protocol.json")
SOURCE_FILES = (
    "rnj_wzzt_core/rnj_wzzt/data/paper_style_terminal_lv.py",
    "rnj_wzzt_core/rnj_wzzt/scenario/simulation.py",
    "rnj_wzzt_core/rnj_wzzt/scenario/settings.py",
    "rnj_wzzt_core/rnj_wzzt/models/lin_distflow.py",
    "rnj_wzzt_core/rnj_wzzt/graph/sensitivity_geometry.py",
    "rnj_wzzt_core/rnj_wzzt/graph/rooted_neighbor_joining.py",
    "rnj_wzzt_core/rnj_wzzt/graph/rooted_hierarchy.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialize(supports):
    return [sorted(s) for s in sorted(supports, key=lambda s: (len(s), sorted(s)))]


def compatible(a, b):
    return not (a & b) or a <= b or b <= a


def audit(protocol_path: Path, output_path: Path) -> None:
    # Refuse to rerun into an existing evidence file before numerical work.
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing evidence: {output_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol_id"] != "paper15_short_root_branches_v1":
        raise ValueError("This audit implements only the locked paper15 root protocol")
    sys.path.insert(0, str(REPO / "rnj_wzzt_core"))
    import numpy as np
    from rnj_wzzt.scenario.simulation import _prepare_case_context
    from rnj_wzzt.models.lin_distflow import build_reduced_sensitivity_matrices
    from rnj_wzzt.graph.sensitivity_geometry import sensitivity_geometry
    from rnj_wzzt.graph.rooted_neighbor_joining import rooted_neighbor_joining
    from rnj_wzzt.graph.rooted_hierarchy import rooted_clades

    root_truth = [frozenset(range(101, 106)), frozenset(range(106, 111)), frozenset(range(111, 116))]
    peripheral = [frozenset(range(103, 106)), frozenset(range(108, 111)), frozenset(range(113, 116))]
    truth = set(root_truth + peripheral)
    results = []
    for condition in protocol["conditions"]:
        context = _prepare_case_context("paper15", 96, scenario_suite="reference")
        net = context.net
        mask = net.branches["from_bus"].eq(1)
        net.branches.loc[mask, ["r_ohm", "x_ohm", "length_m"]] *= condition["root_branch_length_multiplier"]
        r, x = build_reduced_sensitivity_matrices(net, context.terminals, voltage_model="squared-voltage")
        geometry = sensitivity_geometry(r, x)
        threshold = .16 * float(np.median(geometry.root_depths))
        tree = rooted_neighbor_joining(geometry.shared_paths, geometry.root_depths,
                                     context.terminals, 1, group_tolerance=threshold)
        clades = set(rooted_clades(tree.edges, 1, context.terminals))
        position = {terminal: i for i, terminal in enumerate(context.terminals)}
        scores = [float(geometry.shared_paths[position[min(c)], position[max(c)]]) for c in root_truth]
        results.append({
            **condition,
            "default_group_tolerance": threshold,
            "root_shared_scores": scores,
            "score_to_tolerance": [value / threshold for value in scores],
            "root_clades": serialize(root_truth),
            "predicted_clades": serialize(clades),
            "missing_clades": serialize(truth - clades),
            "extra_clades": serialize(clades - truth),
            "root_matched": len(set(root_truth) & clades), "root_total": 3,
            "peripheral_matched": len(set(peripheral) & clades), "peripheral_total": 3,
            "all_missing_root_clades_compatible_with_all_retained_peripheral_clades": all(
                compatible(a, b) for a in set(root_truth) - clades for b in set(peripheral) & clades),
            "modified_physical_edges": net.branches.loc[
                mask, ["from_bus", "to_bus", "length_m", "r_ohm", "x_ohm"]].to_dict("records"),
        })
    record = {
        "protocol_id": protocol["protocol_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "oracle_exact_geometry_only_no_observations_no_optimization",
        "protocol_sha256": sha256(protocol_path),
        "source_hashes": {name: sha256(REPO / name) for name in SOURCE_FILES},
        "audit_source": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__)),
                         "origin": "extracted_after_original_inline_stdin_execution"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "condition_count": len(results), "successful_geometry_evaluations": len(results),
        "optimizer_calls": 0, "random_draws": 0, "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Saved {len(results)} exact-geometry conditions to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True,
                        help="New output file; existing results are never overwritten")
    args = parser.parse_args()
    audit(args.protocol, args.output)


if __name__ == "__main__":
    main()
