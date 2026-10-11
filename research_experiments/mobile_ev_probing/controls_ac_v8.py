"""Fixed-action AC sensitivity controls with matched original physical states.

Voltage-budget inflation does not increase the realized voltage noise. The
separate P/Q-error control adds new bounded measurement errors and declares
their corresponding boxes. Neither control reruns action selection.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ac_shared_v8 as ac


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def interval_valid(result):
    if result.get("lower") is None:
        return False
    lo, hi = F(result["exact_lower"]), F(result["exact_upper"])
    return (0 <= lo <= hi and F(result["lower"]) <= lo and F(result["upper"]) >= hi and
            F(result["gap"]) >= hi-lo and result.get("exact_primal_verified") is True and
            result.get("exact_box_dual_verified") is True)


def run(formal, output):
    formal, output = Path(formal), Path(output)
    data = json.loads((formal/"results.json").read_text(encoding="utf-8"))
    expected = {(h, "flat", "symmetric", 8201) for h in ac.v6.CANDIDATES}
    selected = [e for e in data["episodes"] if (e["truth"], e["history"], e["profile"], e["seed"]) in expected]
    if len(selected) != 3:
        raise ValueError("three predetermined formal episodes are required")
    output.mkdir(parents=True, exist_ok=True)
    if (output/"protocol.json").exists():
        raise FileExistsError("use a fresh controls output directory")
    protocol = dict(
        cases=sorted(expected), formal_results_sha256=sha(formal/"results.json"),
        formal_core_sha256=data["protocol"]["source_sha256"],
        control_source_sha256=sha(__file__), core_source_sha256=sha(ac.__file__),
        dependency_sha256={Path(module.__file__).name:sha(module.__file__) for module in (ac.v7, ac.v6)},
        python=platform.python_version(), platform=platform.platform(), numpy=np.__version__, scipy=ac.scipy.__version__,
        voltage_error_budgets=[1e-4, 5e-4, 2e-3],
        voltage_control="same recorded physical data and tiny realized voltage noise; only declared error budget enlarged",
        pq_error_bound=.001, pq_noise_fraction=.8, pq_noise_seeds=[9101,9102,9103],
        pq_control="new bounded P/Q meter errors on original physical observations; original voltage error retained",
        policies=["selected_stationary", "selected_mobile", "fixed_site1_endpoint", "fixed_mobile12"],
        action_selection="frozen original formal selections; never rerun under modified measurements/budgets",
        duplicate_actions="computed once per episode/control; policy aliases retained explicitly",
        statistical_scope="three predetermined synthetic truth cases; no population inference",
    )
    if protocol["formal_core_sha256"] != protocol["core_source_sha256"]:
        raise RuntimeError("AC core changed since formal experiment")
    dump(output/"protocol.json", protocol)
    records, input_hashes, original_hashes, summaries, issues = [], {}, {}, [], []
    for case_index, episode in enumerate(sorted(selected, key=lambda e:e["truth"])):
        h = episode["truth"]
        prefix = f"{h}_flat_symmetric_8201"
        design = episode["design"]
        policies = dict(selected_stationary=(design.get("selection") or {}).get("selected"),
                        selected_mobile=(design.get("mobile_selection") or {}).get("selected"),
                        fixed_site1_endpoint="site1_endpoint", fixed_mobile12="mobile12")
        # Same fresh P/Q error realization is reused across all actions of this
        # truth case, so the action comparison remains paired.
        rng = np.random.default_rng(9101+case_index)
        noises = rng.uniform(-.0008, .0008, (2, 36, 3))
        np.savez_compressed(output/(prefix+"_pq_errors.npz"), P_error_realized=noises[0], Q_error_realized=noises[1])
        input_hashes[prefix+"_pq_errors.npz"] = sha(output/(prefix+"_pq_errors.npz"))
        by_action = {}
        for action in dict.fromkeys(a for a in policies.values() if a):
            obsfile, truthfile = formal/(prefix+"_"+action+".npz"), formal/(prefix+"_"+action+"_truth.json")
            original_hashes[obsfile.name], original_hashes[truthfile.name] = sha(obsfile), sha(truthfile)
            with np.load(obsfile) as handle:
                base = {k:handle[k] for k in handle.files}
            truth = json.loads(truthfile.read_text(encoding="utf-8"))
            action_stats = {}
            for control in ("epsilon_0.0001", "epsilon_0.0005", "epsilon_0.002", "pq_error_0.001"):
                obs = {k:v.copy() for k,v in base.items()}
                if control.startswith("epsilon"):
                    epsilon = float(control.split("_")[1])
                    obs["drop_error_bound"] = np.full_like(obs["Y_drop"], epsilon)
                    physical_checks = dict(original_y_unchanged=np.array_equal(obs["Y_drop"], base["Y_drop"]),
                                           original_pq_unchanged=np.array_equal(obs["P"], base["P"]) and np.array_equal(obs["Q"], base["Q"]),
                                           actual_noise_within_declared=bool(np.all(abs(np.asarray(truth["actual_noise"]))<=epsilon)))
                else:
                    obs["P"] += noises[0]; obs["Q"] += noises[1]
                    obs["P_error"] = np.full_like(obs["P"], .001)
                    obs["Q_error"] = np.full_like(obs["Q"], .001)
                    physical_checks = dict(original_y_unchanged=np.array_equal(obs["Y_drop"], base["Y_drop"]),
                                           p_error_within_declared=bool(np.all(abs(obs["P"]-base["P"])<=obs["P_error"])),
                                           q_error_within_declared=bool(np.all(abs(obs["Q"]-base["Q"])<=obs["Q_error"])),
                                           original_source_caps_unchanged=np.array_equal(obs["source_p_abs_cap"], base["source_p_abs_cap"]) and np.array_equal(obs["source_q_abs_cap"],base["source_q_abs_cap"]))
                filename = prefix+"_"+action+"_"+control+".npz"
                np.savez_compressed(output/filename, **obs)
                input_hashes[filename] = sha(output/filename)
                wrong, truth_retained, results = 0, True, {}
                for candidate in ac.v6.CANDIDATES:
                    result = ac.solve_outer(obs, candidate)
                    results[candidate] = result
                    if not interval_valid(result):
                        issues.append(prefix+action+control+candidate+": invalid certificate")
                    if candidate == h:
                        truth_retained = result["status"] != "excluded"
                    else:
                        wrong += result["status"] == "excluded"
                if not truth_retained or not all(physical_checks.values()):
                    issues.append(prefix+action+control+": truth/event check failed")
                records.append(dict(case=prefix, truth=h, action=action, control=control,
                                    observations_file=filename, evaluation_truth_reference=truthfile.name,
                                    physical_checks=physical_checks, results=results))
                action_stats[control] = dict(wrong_excluded=wrong, truth_retained=truth_retained)
            # Increasing the same voltage budget can only enlarge the outer
            # feasible set; compare certified intervals, not raw lower bounds.
            action_records = [r for r in records if r["case"] == prefix and r["action"] == action and r["control"].startswith("epsilon")]
            for left, right in zip(action_records, action_records[1:]):
                for candidate in ac.v6.CANDIDATES:
                    left_upper, right_lower = left["results"][candidate]["upper"], right["results"][candidate]["lower"]
                    if left_upper is not None and right_lower is not None and left_upper+1e-12 < right_lower:
                        issues.append(prefix+action+candidate+": error-budget relaxation order violated")
            by_action[action] = action_stats
        summaries.append(dict(case=prefix, policies=policies, by_action=by_action,
                              missing_selections=[p for p,a in policies.items() if a is None]))
    result = dict(protocol=protocol, passed=not issues, issues=issues, summaries=summaries, records=records,
                  original_input_sha256=original_hashes, control_input_sha256=input_hashes,
                  distinct_lp_count=3*len(records), policy_aliases_are_not_independent_trials=True,
                  max_gap=max((r["results"][h]["gap"] for r in records for h in ac.v6.CANDIDATES
                               if r["results"][h]["gap"] is not None), default=None),
                  source_unchanged=sha(__file__)==protocol["control_source_sha256"] and sha(ac.__file__)==protocol["core_source_sha256"])
    dump(output/"results.json", result)
    print(json.dumps({k:result[k] for k in ("passed", "issues", "distinct_lp_count", "max_gap", "summaries")}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("formal", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.formal, args.output)
    raise SystemExit(0 if result["passed"] else 1)
