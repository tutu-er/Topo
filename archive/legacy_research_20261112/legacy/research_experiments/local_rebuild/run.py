"""Small, frozen pilot: constructed mistakes and four existing AC data sets.

Run from the repository root:
  python -m research_experiments.local_rebuild.run
No old batch is resumed and no existing observations or results are changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rnj_wzzt_core"))
import numpy as np
import scipy

from rnj_wzzt.estimation import laminar_l1_milp as core
from .model import canonical, fit_fixed, prediction_loss, repair_control, scan, solve_rebuild


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return plain(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    Path(path).write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def scenes(p, q, y):
    return core._prepare_scenarios([dict(P_terminal=p, Q_terminal=q, drop_target=y)])


def clade_score(point, n, truth):
    predicted = {tuple(s) for s, r, x in zip(point["supports"], point["r"], point["x"])
                 if 1 < len(s) < n and max(r, x) > 1e-9}
    truth = {tuple(s) for s in truth if 1 < len(s) < n}
    denominator = len(predicted) + len(truth)
    return dict(f1=2 * len(predicted & truth) / denominator if denominator else 1.,
                exact=predicted == truth, predicted=canonical(predicted), truth=canonical(truth))


def synthetic(output):
    rows = []
    for n in (4, 5):
        base = [(i,) for i in range(n)] + [tuple(range(n))]
        wrong = [(0, 1), (2, 3)] + ([(0, 1, 2, 3)] if n == 5 else [])
        right = [(0, 2), (1, 3)] if n == 4 else [(0, 2), (1, 4), (0, 2, 3)]
        family, true = canonical([*base, *wrong]), canonical([*base, *right])
        for seed in range(3):
            for noise in (0., .01):
                ident = f"synthetic_n{n}_seed{seed}_noise{noise:g}"
                row = dict(id=ident, n=n, seed=seed, relative_y_noise=noise, status="failed")
                try:
                    rng = np.random.default_rng(20261030 + 100 * n + seed)
                    rr, xx = rng.uniform(.1, .5, (2, len(true)))
                    R, X = core.build_matrices_from_atoms(n, true, rr, xx)
                    inputs = []
                    for t in (20, 60):
                        p, q = rng.uniform(0., 1., (2, t, n))
                        clean = p @ R.T + q @ X.T
                        y = clean + noise * np.std(clean) * rng.normal(size=clean.shape)
                        inputs.append(scenes(p, q, y))
                    train, test = inputs
                    np.savez_compressed(output / (ident + ".npz"), P=train.p[0], Q=train.q[0],
                                        Y=train.target[0], testP=test.p[0], testQ=test.q[0], testY=test.target[0])
                    old = fit_fixed(train, family)
                    fitted = solve_rebuild(train, family, range(n), tuple((i,) for i in range(n)),
                                            seconds=5., cuts="subset")
                    point = fitted["point"]
                    # Ground truth is read for scoring only, after both fits.
                    row.update(status="completed" if point else "no_feasible_incumbent",
                               baseline=old, rebuild=fitted,
                               baseline_score=clade_score(old, n, true),
                               rebuild_score=clade_score(point, n, true) if point else None,
                               baseline_test_mae=prediction_loss(test, old["supports"], old["r"], old["x"]),
                               rebuild_test_mae=prediction_loss(test, point["supports"], point["r"], point["x"]) if point else None,
                               data_sha256=sha(output / (ident + ".npz")))
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
                write(output / "synthetic.json", rows)
                print(ident, row["status"], row.get("rebuild_score", {}), flush=True)
    return rows


def from_observed(path, n=None):
    from research_experiments.wzzt_paper.common import observed
    p, q, v, root = observed(path, n)
    return scenes(p.to_numpy(), q.to_numpy(), root.to_numpy()[:, None] ** 2 - v.to_numpy() ** 2)


def select_and_contract(points, validation, calibration):
    """Use the already frozen 3% validation / 0.16 depth contraction policy."""
    scores = [prediction_loss(validation, p["supports"], p["r"], p["x"]) for p in points]
    eligible = [i for i, score in enumerate(scores) if score <= min(scores) * 1.03 + 1e-12]
    index = min(eligible, key=lambda i: (len(points[i]["supports"]), scores[i], i))
    p = points[index]
    threshold = .16 * calibration["median_root_depth"]
    active = [i for i, s in enumerate(p["supports"]) if len(s) == 1 or
              calibration["a"] * p["r"][i] + calibration["b"] * p["x"][i] + 1e-10 >= threshold]
    return dict(supports=[p["supports"][i] for i in active], r=[p["r"][i] for i in active],
                x=[p["x"][i] for i in active], selected_index=index,
                pre_contraction_validation=scores[index], threshold=threshold)


def real_pilot(output, parent, budget):
    rows = []
    for case in ("paper15", "soumalas11", "flynn16", "pengwah18"):
        ident = f"main__{case}__b0__reference__v0005__N48"
        directory = parent / "jobs" / ident
        row = dict(id=ident, case=case, status="failed")
        try:
            payload = json.loads((directory / "payload.json").read_text(encoding="utf-8"))
            saved = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            train = from_observed(payload["train"], 48)
            initial = saved["models"]["ridge_union"]["supports"]
            baseline = fit_fixed(train, initial)
            print(case, "baseline", baseline["objective"], flush=True)
            # Matched extra wall budget, same start, objective, bounds and LP
            # backend. The earlier candidate pool is fixed before this pilot.
            control = repair_control(train, initial, saved["augmentation"]["pool"], seconds=budget)
            print(case, "repair", control["stop_reason"], control["seconds"], flush=True)
            rebuilt = scan(train, initial, seconds=budget, per_solve_seconds=3., max_solves=12)
            print(case, "rebuild", rebuilt["stop_reason"], rebuilt["seconds"], flush=True)
            # Only now open validation/test and physical truth. No feedback
            # from these evaluations changes search, budget, cuts or regions.
            validation, test = from_observed(payload["validation"]), from_observed(payload["test"])
            from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
            labels = saved["labels"]
            truth = physical_truth(Path(payload["truth_network"]), labels)["clades"]
            positions = {v: i for i, v in enumerate(labels)}
            true = canonical([tuple(positions[v] for v in c) for c in truth])
            methods = {"baseline": [baseline], "pool_repair": control["path"], "rebuild": rebuilt["path"]}
            metrics = {}
            for method, path in methods.items():
                for policy, point in (("endpoint", path[-1]), ("validation_contract",
                                      select_and_contract(path, validation, saved["calibration"]))):
                    metrics[method + "__" + policy] = dict(point=point,
                        train_mae=prediction_loss(train, point["supports"], point["r"], point["x"]),
                        validation_mae=prediction_loss(validation, point["supports"], point["r"], point["x"]),
                        test_mae=prediction_loss(test, point["supports"], point["r"], point["x"]),
                        **clade_score(point, len(labels), true))
            row.update(status="completed", baseline=baseline, control=control, rebuild=rebuilt,
                       metrics=metrics, input_sha256={split: sha(payload[split]) for split in ("train", "validation", "test")},
                       parent_result_sha256=sha(directory / "result.json"),
                       parent_payload_sha256=sha(directory / "payload.json"), truth_used_by_search=False)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        write(output / "real.json", rows)
        print(case, row["status"], row.get("error", ""), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/local_tree_rebuild_20261030")
    parser.add_argument("--parent", type=Path, default=ROOT / "outputs/wzzt_paper_20261029_full")
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), Path(__file__).with_name("model.py"), Path(core.__file__),
             ROOT / "research_experiments/wzzt_paper/common.py",
             ROOT / "research_experiments/rnj/audit_source_ac96_benchmark.py"]
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    python=sys.version, executable=sys.executable, platform=platform.platform(),
                    numpy=np.__version__, scipy=scipy.__version__, solver="SciPy bundled HiGHS",
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    sources={str(p.relative_to(ROOT)): sha(p) for p in files},
                    protocol=dict(synthetic_cases=12, synthetic_seconds_per_MILP=5.,
                                  real_cases=0 if args.synthetic_only else 4, N=48, voltage_sigma=.0005,
                                  pq_sigma=.005, bundle=0, max_blocks=5, max_solves=12,
                                  per_solve_seconds=3., extra_seconds_per_method=args.seconds,
                                  coefficient_bound=2., cuts="subset", real_start="saved ridge_union contracted family",
                                  real_comparator="one-candidate crossing-removal rule in saved ridge candidate pool",
                                  acceptance="audited strict improvement in training L1",
                                  deployment="existing 3% validation selection and 0.16 depth contraction",
                                  scope="small diagnostic pilot, no hyperparameter tuning or old-batch resumption"),
                    command="python -m research_experiments.local_rebuild.run " + " ".join(sys.argv[1:]))
    write(args.output / "manifest.json", manifest)
    started = perf_counter()
    synthetic_rows = synthetic(args.output)
    real_rows = [] if args.synthetic_only else real_pilot(args.output, args.parent, args.seconds)
    completion = dict(seconds=perf_counter() - started, synthetic_returned=len(synthetic_rows),
                      synthetic_completed=sum(r["status"] == "completed" for r in synthetic_rows),
                      real_returned=len(real_rows), real_completed=sum(r["status"] == "completed" for r in real_rows))
    write(args.output / "completion.json", completion)
    print(completion, flush=True)


if __name__ == "__main__":
    main()
