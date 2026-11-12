"""Read-only dictionary geometry study; not an AC theft-detection benchmark.

Assumes known fixed learned weights, common known Q/P=0.4, exact total
unmetered P, and the linear squared-voltage model. No LP or MILP is solved.
"""
from itertools import combinations
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

from theft_wzzt.theft.identified_tree import load_identified, to_theft_tree

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "theft_wzzt/outputs/theft/identified_tree.json",
    "theft_wzzt/outputs/theft_paper/identified_soumalas11.json",
    "theft_wzzt/outputs/theft_paper/identified_flynn16.json",
)


def analyze(source, kappa=.4):
    path = ROOT / source
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    identified = load_identified(path)
    tree, (r, x) = to_theft_tree(identified)
    z, a = tree.incidence()
    w = r + kappa * x
    dictionary = z.T @ (w[:, None] * a)
    # Power balance adds the final all-ones row. Row scaling is only for
    # numerical rank diagnostics; it is NOT calibrated noise whitening.
    augmented = np.vstack((dictionary, np.ones(len(tree.candidates))))
    normalized = augmented / np.linalg.norm(augmented, axis=1)[:, None]
    rank_checks = []
    for size in (2, 3, 4):
        minimum = float("inf")
        minimum_indices = None
        deficient = 0
        count = 0
        for indices in combinations(range(len(tree.candidates)), size):
            singular = np.linalg.svd(normalized[:, indices], compute_uv=False)
            count += 1
            deficient += int(singular[-1] < 1e-11)
            if singular[-1] < minimum:
                minimum = float(singular[-1])
                minimum_indices = indices
        rank_checks.append(dict(columns=size, checked=count,
                                numerically_rank_deficient=deficient,
                                minimum_singular_value=minimum,
                                worst_labels=[tree.candidates[j] for j in minimum_indices]))

    lookup = {h: j for j, h in enumerate(tree.candidates)}
    aliases = []
    for h in tree.candidates:
        if h in tree.observed:
            continue
        neighbors = [(v if u == h else u, e)
                     for e, (u, v) in enumerate(tree.edges) if u == h or v == h]
        # The root is not in the deployed candidate dictionary; do not add it.
        if len(neighbors) < 2 or any(v not in lookup for v, _ in neighbors):
            continue
        conductance = np.array([1 / w[e] for _, e in neighbors])
        fractions = conductance / conductance.sum()
        js = [lookup[v] for v, _ in neighbors]
        mixed = dictionary[:, js] @ fractions
        terminal_error = float(np.max(np.abs(dictionary[:, lookup[h]] - mixed)))
        total_error = abs(float(fractions.sum()) - 1.)
        # Hypothetical extra voltage measurement at h; this is a design
        # calculation, not a sensor installation or a new measured dataset.
        at_h = a[:, lookup[h]] @ (w[:, None] * a)
        added_sensor_gap = float(at_h[lookup[h]] - at_h[js] @ fractions)
        expected_gap = float(1 / conductance.sum())
        assert terminal_error < 1e-12 and total_error < 1e-12
        np.testing.assert_allclose(added_sensor_gap, expected_gap, rtol=1e-12, atol=1e-12)
        aliases.append(dict(
            one_source_label=h, multi_source_labels=[v for v, _ in neighbors],
            source_fractions=fractions.tolist(),
            for_total_10kw_allocations_kw=(10 * fractions).tolist(),
            max_terminal_response_error_per_pu_load=terminal_error,
            total_power_error_per_pu_load=total_error,
            proposed_extra_voltage_sensor=h,
            extra_sensor_response_separation_per_pu_load=added_sensor_gap,
        ))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    return dict(case=identified.case_key, source=source, source_sha256=before,
                measured_terminals=len(tree.observed), candidates=len(tree.candidates),
                augmented_dictionary_rank=int(np.linalg.matrix_rank(normalized)),
                row_scaling="unit Euclidean norm, not physical noise scaling",
                rank_tolerance=1e-11, subset_rank_checks=rank_checks,
                exact_positive_alias_examples=aliases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError("Output must be inside the Topo workspace")
    result = dict(date="2026-09-20", status="verified",
                  assumptions=dict(model="linear squared voltage", fixed_learned_rx=True,
                                   known_common_q_over_p=.4, exact_total_power=True),
                  limitations=["No noisy/AC recovery or calibration is evaluated.",
                               "Full column rank is a numerical check, not a symbolic proof for arbitrary networks.",
                               "Uncertain R/X, line losses, and power factors can create additional ambiguity.",
                               "A higher-cardinality mixture remains admissible unless a valid sparsity bound excludes it."],
                  cases=[analyze(source) for source in SOURCES])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"status": result["status"], "cases": [
        {"case": case["case"],
         "checked_subsets": sum(row["checked"] for row in case["subset_rank_checks"]),
         "rank_deficient": sum(row["numerically_rank_deficient"] for row in case["subset_rank_checks"]),
         "positive_aliases": len(case["exact_positive_alias_examples"]),
         "smallest_multi_source_count": min(len(row["multi_source_labels"]) for row in case["exact_positive_alias_examples"]),
         "max_alias_error": max(row["max_terminal_response_error_per_pu_load"] for row in case["exact_positive_alias_examples"])}
        for case in result["cases"]]}, ensure_ascii=False))
