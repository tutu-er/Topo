"""Rank all one-line replacements at the retrospectively selected 10-child junction."""
# ruff: noqa: E402 -- set numeric library thread counts before importing their runtimes.
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
from time import perf_counter

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import gurobipy as gp

from ptid.cases import library
from ptid.cases.paths import OUTPUTS, STUDY
from ptid.method.model import Fitter, Model
from ptid.method.search import PRICING_TOL, node_miqp
from ptid.method.sets import canonical, children, unions
from ptid.runs.four_offline import digest
from ptid.runs.node_mechanism import Reference
from ptid.runs.root_ablation import physical_edges


CASE = "dev_huge_004_end"
RELEASE = tuple(range(4, 29))
PANEL = STUDY / "experiments/four_feeders/random_time_limits.json"


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_anchor_domain(outputs, panel, row, step):
    """Verify the saved domain of this particular anchored solver call."""
    archive = outputs / "topology_study_random_dev"
    old = json.loads((archive / "arms" / f"{CASE}__proposed.json").read_text(encoding="utf-8"))
    contract = dict(st=panel["new_settings"], inputs=row["observations"], original=old["run_contract"])
    folder = archive / "budget_30s" / json_hash(contract)
    for path in sorted(folder.glob("*.pkl")):
        saved = pickle.loads(path.read_bytes())
        if isinstance(saved, tuple) and len(saved) == 2 and isinstance(saved[1], dict):
            domain = saved[1].get("domain")
            if domain and json_hash(domain) == step["domain_sha256"]:
                if canonical(saved[0]) != canonical(step["tree"]):
                    continue
                return domain
    raise ValueError(f"Missing archived anchored domain in {folder}. Run random-budget first.")


def rank_replacements(model, tree, kids):
    """Use measurements and the contracted tree only. Report H (smaller is better)."""
    started = perf_counter()
    candidates = unions(kids)
    if len(kids) != 10 or len(set(candidates)) != 1012:
        raise ValueError("The archived junction no longer matches the 10-child experiment")
    values = {}
    for i in range(0, len(candidates), 100):
        block = candidates[i:i + 100]
        fitter = Fitter(model, [*tree, *block])
        for group in block:
            values[group] = fitter.crit([*tree, group]) * model.N / 2
    return values, perf_counter() - started


def run(outputs, seconds):
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    row = next(row for row in panel["rows"] if row["case"] == CASE)
    step = row["new"]["steps"][1]
    library.LIBRARY = outputs / "case_library_20261125"
    record = library.open_case(CASE)
    expected = next(item["sha256"] for item in library.manifest() if item["name"] == CASE)
    if expected != row["record_sha256"]:
        raise ValueError("The published case hash differs from the verified case manifest")
    observations = library.readings(record)
    data = {key: observations[key] for key in ("P", "Q", "Y", "V0sq")}
    if digest(data) != row["observations"]:
        raise ValueError("Observations differ from the published 30-second panel")
    settings = panel["new_settings"]
    model = Model(**data, lam=settings["lam"], p_s=settings["p_s"])
    domain = read_anchor_domain(outputs, panel, row, step)
    pool = set(canonical(domain["lines"]))
    anchors = {tuple(group[0]) for group in domain["require"] if len(group) == 1}
    if len(anchors) != len(domain["require"]) or RELEASE in anchors:
        raise ValueError("A required original grouping cannot be released by this exchange")
    original = canonical(step["tree"])
    if RELEASE not in original:
        raise ValueError("The selected release is missing from the archived tree")
    parent, = (group for group, kids in children(original, model.leaves).items() if RELEASE in kids)
    reduced = canonical(group for group in original if group != RELEASE)
    kids = children(reduced, model.leaves)[parent]
    values, enum_time = rank_replacements(model, reduced, kids)
    if RELEASE not in values:
        raise ValueError("Keeping the original tree is not an available choice")
    factor = model.N / 2
    tolerance = PRICING_TOL * factor
    ranked = sorted(values, key=lambda group: (values[group], group))
    optimum = values[ranked[0]]
    winners = [group for group in ranked if values[group] <= optimum + tolerance]
    began = perf_counter()
    line, objective, status, bound = node_miqp(model, reduced, kids, seconds,
                                              seed=settings["seed"], return_bound=True)
    solver_time = perf_counter() - began
    # Physical truth and the diagnostic target first enter after both searches.
    edges = physical_edges(record)
    original_final = set(canonical(row["new"]["tree"]))
    target, = (group for group, edge_list in edges.items()
               if any(edge["parent_bus"] == 8 and edge["child_bus"] == 11 for edge in edge_list))
    target_value = values[target]
    target_rank = 1 + sum(h < target_value - tolerance for h in values.values())
    target_ties = sum(abs(h - target_value) <= tolerance for h in values.values())
    if line is None or bound is None or status not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT):
        raise RuntimeError(f"Node MIQP returned status {status}, line {line}, bound {bound}")
    # Independent explicit voltage design with bounded SciPy least squares.
    ref = Reference(data, model)
    controls = dict(original=original, deletion=reduced, target=canonical([*reduced, target]),
                    best=canonical([*reduced, ranked[0]]), miqp=canonical([*reduced, line]))
    refits = {name: [v * factor for v in ref.score(tree)] for name, tree in controls.items()}
    check_groups = dict(original=RELEASE, target=target, best=ranked[0], miqp=line)
    if any(abs(refits[name][1] - values[group]) > tolerance for name, group in check_groups.items()):
        raise ValueError("Enumerated candidate disagrees with independent bounded refit")
    if abs(refits["original"][1] - step["H"]) > tolerance:
        raise ValueError("Original tree does not reproduce the archived objective")
    if bound * factor > optimum + tolerance or abs(objective * factor - values[line]) > tolerance:
        raise ValueError("Node MIQP bound or feasible objective contradicts enumeration")
    def mask(group):
        selected = set(group)
        return sum(1 << i for i, kid in enumerate(kids) if set(kid) <= selected)

    return dict(
        case=CASE, role="Retrospective fixed release selected using prior knowledge of the target; "
                        "the search compares every legal child union without target labels.",
        source=dict(panel="experiments/four_feeders/random_time_limits.json", stage=1,
                    data_sha256=row["record_sha256"], observation_sha256=row["observations"],
                    original_domain_sha256=step["domain_sha256"], environment=panel["environment"],
                    original_solver_settings=settings),
        pool_definition="Saved candidate family of the anchored Tree MIQP immediately preceding pricing.",
        original_tree=original, released=RELEASE, released_true=RELEASE in edges,
        anchors=canonical(anchors), parent=parent, children=kids, candidate_count=len(values),
        scoring="H = (N/2) times crit. Lower is better. Ties use PRICING_TOL in crit units.",
        tolerance_H=tolerance, current_H=refits["original"][1], deletion_H=refits["deletion"][1],
        target=dict(group=target, edges=edges[target], in_original_pool=target in pool,
                    in_published_final=target in original_final, H=target_value,
                    rank=target_rank, ties=target_ties),
        exhaustive=dict(seconds=enum_time, H=optimum, best=ranked[0], best_edges=edges.get(ranked[0], []),
                        best_true=ranked[0] in edges, in_original_pool=ranked[0] in pool,
                        in_published_final=ranked[0] in original_final,
                        tied_best_groups=winners, runner_up_H=values[ranked[1]], runner_up=ranked[1],
                        ranked_mask_H=[[mask(group), values[group]] for group in ranked]),
        miqp=dict(seconds=solver_time, limit_seconds=seconds, status=int(status), selected=line,
                  selected_true=line in edges, in_original_pool=line in pool, H=objective * factor,
                  lower_H=bound * factor, gap_to_exhaustive_best_H=values[line] - optimum,
                  matches_exhaustive_best=line in winners),
        independent_refit_H_intervals=refits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=OUTPUTS)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    output = args.out or args.outputs / "topology_study/line_exchange/fixed_node.json"
    result = run(args.outputs, args.seconds)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n",
                    encoding="utf-8")
    temp.replace(output)
    print(f"{result['candidate_count']} candidates, target rank {result['target']['rank']}, "
          f"MIQP matches best: {result['miqp']['matches_exhaustive_best']}. {output}")


if __name__ == "__main__":
    main()
