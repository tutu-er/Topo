"""Analyze saved pilot paths; truth is used only for explicit diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .run import ROOT, clade_score, from_observed, scenes, sha, write
from .model import canonical, frontiers, prediction_loss
from .matched import FiniteLP


def contract(point, calibration, factor):
    keep = [i for i, s in enumerate(point["supports"]) if len(s) == 1 or
            calibration["a"] * point["r"][i] + calibration["b"] * point["x"][i] + 1e-10 >=
            factor * max(calibration["median_root_depth"], 1e-12)]
    return dict(supports=[point["supports"][i] for i in keep],
                r=[point["r"][i] for i in keep], x=[point["x"][i] for i in keep])


def select_points(points, validation, calibration, *, fraction=.03, factor=.16, order="pre", refit_bank=None):
    contracted = [contract(p, calibration, factor) for p in points]
    if order == "post_refit":
        if refit_bank is None:
            raise ValueError("post_refit requires the training LP bank")
        contracted = [refit_bank.fit(p["supports"]) for p in contracted]
    scored = points if order == "pre" else contracted
    losses = [prediction_loss(validation, p["supports"], p["r"], p["x"]) for p in scored]
    eligible = [i for i, v in enumerate(losses) if v <= min(losses) * (1 + fraction) + 1e-12]
    index = min(eligible, key=lambda i: (len(scored[i]["supports"]), losses[i], i))
    return dict(point=contracted[index], index=index, factor=factor, fraction=fraction, order=order,
                ranked_validation=losses[index], validation=prediction_loss(validation,
                    contracted[index]["supports"], contracted[index]["r"], contracted[index]["x"]))


def diagnose(output, pilot):
    parent = ROOT / "outputs/wzzt_paper_20261029_full/jobs"
    previous = json.loads((pilot / "real.json").read_text(encoding="utf-8"))
    rows = []
    for row in previous:
        directory = parent / row["id"]
        payload = json.loads((directory / "payload.json").read_text(encoding="utf-8"))
        saved = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        cal = saved["calibration"]
        train, validation, test = (from_observed(payload[s], 48 if s == "train" else None)
                                   for s in ("train", "validation", "test"))
        clean = from_observed(payload["train_truth"], 48)
        clean_test = from_observed(payload["test_truth"])
        true = canonical(row["metrics"]["baseline__endpoint"]["truth"])
        baseline = row["baseline"]
        endpoint = row["rebuild"]["point"]
        n = train.n
        internal = lambda p: {tuple(s) for s, r, x in zip(p["supports"], p["r"], p["x"])
                              if 1 < len(s) < n and max(r, x) > 1e-9}
        b, e, truth = internal(baseline), internal(endpoint), set(true)
        counts = lambda f: dict(TP=len(f & truth), FP=len(f - truth), FN=len(truth - f))
        changes = []
        for s, r, x in zip(endpoint["supports"], endpoint["r"], endpoint["x"]):
            if tuple(s) in e - b:
                changes.append(dict(support=s, true=tuple(s) in truth, r=r, x=x,
                    depth_ratio=(cal["a"] * r + cal["b"] * x) / max(cal["median_root_depth"], 1e-12)))
        p, q = train.p[0], train.q[0]
        design = np.column_stack((p, q))
        scaled = (design - design.mean(axis=0)) / np.maximum(design.std(axis=0), 1e-12)
        correlations = [np.corrcoef(p[:, i], q[:, i])[0, 1] for i in range(n)]
        families = canonical([*((i,) for i in range(n)), tuple(range(n)), *true])
        bank = FiniteLP(train, canonical([*families, *baseline["supports"], *endpoint["supports"]]))
        oracle = bank.fit(families)
        # Fitting clean P/Q/Y on the true support domain separates the smooth
        # linear-model approximation error from errors in observed channels.
        clean_bank = FiniteLP(clean, families)
        clean_oracle = clean_bank.fit(families)
        metrics = {}
        for name, point in (("baseline", baseline), ("endpoint", endpoint), ("truth_family_fit", oracle)):
            metrics[name] = {split: prediction_loss(data, point["supports"], point["r"], point["x"])
                             for split, data in (("train", train), ("validation", validation), ("test", test),
                                                 ("clean_train", clean), ("clean_test", clean_test))}
            metrics[name].update(clade_score(point, n, true))
        sensitivity = []
        refit_bank = FiniteLP(train, canonical(s for p in row["rebuild"]["path"] for s in p["supports"]),
                              mandatory=[(i,) for i in range(n)])
        for order in ("pre", "post", "post_refit"):
            for fraction in (0., .01, .03):
                for factor in (.08, .16, .24):
                    decision = select_points(row["rebuild"]["path"], validation, cal,
                                             fraction=fraction, factor=factor, order=order, refit_bank=refit_bank)
                    decision.update(clade_score(decision["point"], n, true),
                                    test=prediction_loss(test, decision["point"]["supports"],
                                                        decision["point"]["r"], decision["point"]["x"]))
                    sensitivity.append(decision)
        all_frontiers = list(frontiers(baseline["supports"], n, max_blocks=5))
        candidate_universe = set()
        from .model import unions
        for v, blocks in all_frontiers:
            candidate_universe.update(s for _, s in unions(blocks))
        item = dict(case=row["case"], id=row["id"], before=counts(b), after=counts(e), added=changes,
                    removed=canonical(b - e), metrics=metrics, calibration=cal,
                    gate_mismatch=dict(previous_rebuild_relative=1e-8, previous_control_relative=.001),
                    design=dict(shape=list(design.shape), centered_scaled_rank=int(np.linalg.matrix_rank(scaled)),
                                centered_scaled_condition=float(np.linalg.cond(scaled)),
                                median_within_terminal_PQ_correlation=float(np.median(correlations))),
                    observed_minus_clean_target_MAE=float(abs(train.target[0] - clean.target[0]).mean()),
                    clean_truth_family_minimum_MAE=clean_oracle["objective"],
                    frontier_count=len(all_frontiers),
                    missing_true_supports_representable=canonical((truth - b) & candidate_universe),
                    sensitivity=sensitivity, refit_LP_diagnostics=refit_bank.solves, oracle_LP_diagnostics=bank.solves,
                    clean_oracle_LP_diagnostics=clean_bank.solves,
                    source_hashes={"pilot_real":sha(pilot/"real.json"), "parent_result":sha(directory/"result.json")},
                    interpretation="old four cases are diagnostic reuse; grid is not prospective validation")
        rows.append(item)
        write(output / "diagnosis.json", rows)
        print(row["case"], item["before"], item["after"], 'condition',item["design"], flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--pilot", type=Path, default=ROOT / "outputs/local_tree_rebuild_20261030_final")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    diagnose(args.output, args.pilot)
