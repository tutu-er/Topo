"""Audit selection of identified z atoms against existing path-based prediction.

Run with D:\\apps\\miniconda3\\envs\\Topo\\python.exe. No optimization,
simulation truth, or topology refitting is used. Historical files are read only.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

from theft_wzzt.theft.identified_tree import load_identified, to_theft_tree
from theft_wzzt.theft.theft_model import TheftData, predict

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "theft_wzzt/outputs/theft/identified_tree.json",
    "theft_wzzt/outputs/theft_paper/identified_soumalas11.json",
    "theft_wzzt/outputs/theft_paper/identified_flynn16.json",
)


def audit():
    rng = np.random.default_rng(20260920)
    cases = []
    for source in SOURCES:
        path = ROOT / source
        identified = load_identified(path)
        tree, weights = to_theft_tree(identified)
        z, c = tree.incidence()
        lookup = {identified.label_of(s): s for s in identified.supports}
        edge_supports = [lookup[child] for _, child in tree.edges]
        selected_supports = [lookup[h] for h in tree.candidates]
        a = np.array([[s <= parent for s in selected_supports]
                      for parent in edge_supports], dtype=int)
        assert np.array_equal(a, c), "Support inclusion differs from source paths"
        expected_z = np.array([[terminal in s for terminal in tree.observed]
                               for s in edge_supports], dtype=int)
        assert np.array_equal(z, expected_z), "An identified atom was changed"
        t, n = 4, len(tree.observed)
        p, q = rng.uniform(.001, .03, (2, t, n))
        amplitude = np.array([0., .003, .007, .011])
        extra_q = .4 * amplitude
        data = TheftData(p, q, np.zeros((t, n)), amplitude.copy(),
                         amplitude, extra_q, 1., 1.)
        maximum_error = 0.
        candidates = []
        # Also vary continuous weights: this identity does not require fixed R/X.
        for r, x in (weights, tuple(rng.uniform(.01, .2, (2, len(z))))):
            for j, h in enumerate(tree.candidates):
                selection = np.zeros((t, len(tree.candidates)), dtype=int)
                selection[amplitude > 0, j] = 1
                actual = predict(tree, data, r, x, selection)
                for k in range(t):
                    b = a @ selection[k]
                    extended_z = np.column_stack((z, b))
                    r_aug = extended_z.T @ (r[:, None] * extended_z)
                    x_aug = extended_z.T @ (x[:, None] * extended_z)
                    expected = (r_aug @ np.r_[p[k], amplitude[k]]
                                + x_aug @ np.r_[q[k], extra_q[k]])[:n]
                    maximum_error = max(maximum_error,
                                        float(np.max(np.abs(actual[k] - expected))))
                    np.testing.assert_allclose(actual[k], expected, rtol=1e-12, atol=1e-12)
        for j, h in enumerate(tree.candidates):
            support = selected_supports[j]
            candidates.append(dict(
                label=h, selected_support=sorted(support),
                identified_support_index=identified.supports.index(support),
                selected_z=[int(v in support) for v in tree.observed],
                activated_ancestor_labels=[tree.edges[e][1] for e in np.flatnonzero(a[:, j])],
            ))
        cases.append(dict(case=identified.case_key, source=source,
                          source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          candidate_count=len(candidates), atom_count=len(z),
                          inclusion_equals_path=True,
                          max_augmented_wzzt_prediction_error=maximum_error,
                          candidates=candidates))
    return dict(status="verified", date="2026-09-20", cases=cases,
                scope="Algebraic equivalence only; no new detection accuracy or calibration claim.",
                model="Select an existing z label; activate its ancestor chain; R/X may remain variable.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError("Output must stay inside the Topo workspace")
    result = audit()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"status": result["status"], "cases": [
        {k: c[k] for k in ("case", "candidate_count", "max_augmented_wzzt_prediction_error")}
        for c in result["cases"]]}, ensure_ascii=False))
