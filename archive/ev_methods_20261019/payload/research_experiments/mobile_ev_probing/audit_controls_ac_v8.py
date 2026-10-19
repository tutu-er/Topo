"""Independent file-level verification of fixed-action AC sensitivity controls."""
from collections import defaultdict
from fractions import Fraction as F
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(formal, controls):
    formal, controls = Path(formal), Path(controls)
    data = json.loads((controls/"results.json").read_text(encoding="utf-8"))
    protocol = json.loads((controls/"protocol.json").read_text(encoding="utf-8"))
    issues, groups = [], defaultdict(lambda:defaultdict(lambda:dict(successes=0, cases=0)))

    def check(value, message):
        if not value:
            issues.append(message)

    check(data["protocol"] == protocol, "protocol mismatch")
    check(sha(formal/"results.json") == protocol["formal_results_sha256"], "formal aggregate hash mismatch")
    check(sha(Path(__file__).with_name("controls_ac_v8.py")) == protocol["control_source_sha256"], "control source changed")
    check(sha(Path(__file__).with_name("ac_shared_v8.py")) == protocol["core_source_sha256"], "core source changed")
    for filename, digest in data["original_input_sha256"].items():
        check(sha(formal/filename) == digest, "original input hash mismatch: "+filename)
    for filename, digest in data["control_input_sha256"].items():
        check(sha(controls/filename) == digest, "control input hash mismatch: "+filename)
    indexed = {}
    false_exclusions, maximum_gap = 0, 0.
    for record in data["records"]:
        key = (record["case"], record["action"], record["control"])
        check(key not in indexed, "duplicate control record: "+str(key)); indexed[key] = record
        base_path = formal/(record["case"]+"_"+record["action"]+".npz")
        with np.load(base_path) as handle:
            base = {k:handle[k] for k in handle.files}
        with np.load(controls/record["observations_file"]) as handle:
            current = {k:handle[k] for k in handle.files}
        if record["control"].startswith("epsilon"):
            epsilon = float(record["control"].split("_")[1])
            check(all(np.array_equal(base[k], current[k]) for k in base if k != "drop_error_bound"), str(key)+": changed physical data")
            check(np.all(current["drop_error_bound"] == epsilon), str(key)+": wrong voltage error budget")
            check(epsilon in protocol["voltage_error_budgets"], str(key)+": unplanned voltage control")
        else:
            with np.load(controls/(record["case"]+"_pq_errors.npz")) as errors:
                for name in ("P", "Q"):
                    delta = current[name]-base[name]
                    check(np.max(abs(delta-errors[name+"_error_realized"])) < 1e-15,
                          str(key)+": P/Q error realization mismatch")
                    check(np.all(abs(delta) <= .001) and np.all(current[name+"_error"] == .001),
                          str(key)+": P/Q error not covered")
            check(all(np.array_equal(base[k], current[k]) for k in base if k not in ("P", "Q")),
                  str(key)+": P/Q control changed other observations or budgets")
        for candidate, result in record["results"].items():
            lo, hi = F(result["exact_lower"]), F(result["exact_upper"])
            check(0 <= lo <= hi and F(result["lower"]) <= lo and F(result["upper"]) >= hi,
                  str(key)+candidate+": certificate interval invalid")
            check(F(result["gap"]) >= hi-lo and result["exact_primal_verified"] and result["exact_box_dual_verified"],
                  str(key)+candidate+": certificate metadata invalid")
            check(result["status"] == ("excluded" if lo > 0 else "unresolved"), str(key)+candidate+": status mismatch")
            maximum_gap = max(maximum_gap, result["gap"])
            false_exclusions += candidate == record["truth"] and result["status"] == "excluded"
    formal_data = json.loads((formal/"results.json").read_text(encoding="utf-8"))
    for summary in data["summaries"]:
        episode = next(e for e in formal_data["episodes"] if summary["case"] == f"{e['truth']}_{e['history']}_{e['profile']}_{e['seed']}")
        check(summary["policies"]["selected_stationary"] == episode["design"]["selection"]["selected"], "stationary selector changed")
        check(summary["policies"]["selected_mobile"] == episode["design"]["mobile_selection"]["selected"], "mobile selector changed")
        for policy, action in summary["policies"].items():
            for control in ("epsilon_0.0001", "epsilon_0.0005", "epsilon_0.002", "pq_error_0.001"):
                record = indexed[(summary["case"], action, control)]
                wrong = sum(h != record["truth"] and r["status"] == "excluded" for h,r in record["results"].items())
                groups[control][policy]["cases"] += 1
                groups[control][policy]["successes"] += wrong == 2
    check(false_exclusions == 0, "a true candidate was excluded")
    check(len(indexed)*3 == data["distinct_lp_count"] == 96, "distinct LP count mismatch")
    result = dict(passed=not issues, issues=issues, records=len(indexed), distinct_LPs=len(indexed)*3,
                  true_candidate_exclusions=false_exclusions, maximum_gap=maximum_gap,
                  policy_counts=groups, source_sha256=sha(__file__), results_sha256=sha(controls/"results.json"),
                  semantics="voltage budget inflation preserves original noise; P/Q control adds matched bounded meter errors; action selections frozen",
                  scope="file/input and recorded-interval audit; no additional LP replay")
    (controls/"input_audit.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("formal", type=Path)
    parser.add_argument("controls", type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.formal, args.controls)["passed"] else 1)
