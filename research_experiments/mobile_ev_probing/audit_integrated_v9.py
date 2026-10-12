"""Independent artifact/paired-data audit; no complete LP re-execution.

Physical voltages are independently recomputed from nodal admittances.
Saved rational interval ordering is checked, not unstored dual multipliers.
Use --replays 3 for three deterministic fresh LP problem reconstructions.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def walk(value, prefix=""):
    if isinstance(value, dict):
        yield prefix, value
        for key, item in value.items():
            yield from walk(item, prefix + "/" + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk(item, prefix + "/" + str(index))


def metrics(supports, truth_supports, n):
    a = {frozenset(s) for s in supports if 1 < len(s) < n}
    b = {frozenset(s) for s in truth_supports if 1 < len(s) < n}
    common = len(a & b)
    return dict(exact=a == b, precision=common / len(a) if a else 0.,
                recall=common / len(b) if b else 1.,
                f1=2 * common / (len(a) + len(b)) if a or b else 1.)


def independent_nodal_ac(metadata, theta, p, q, source_p):
    """Separately assembled complex nodal current balance."""
    m = len(metadata["supports"])
    theta, p, q, sp = map(np.asarray, (theta, p, q, source_p))
    yy = np.zeros((m + 1, m + 1), complex)
    for e, parent in enumerate(metadata["parents"]):
        left, right = 0 if parent < 0 else parent + 1, e + 1
        admittance = 2 / (theta[e] * (1 + .5j))
        yy[left, left] += admittance
        yy[right, right] += admittance
        yy[left, right] -= admittance
        yy[right, left] -= admittance
    loads = np.zeros((len(p), m), complex)
    loads[:, 0] = sp * (1 + .3j)
    terminals = list(metadata["terminal_edges"])
    loads[:, terminals] += p + 1j * q
    yn, root = yy[1:, 1:], yy[1:, 0]
    voltage = np.ones((len(p), m), complex)
    for _ in range(2000):
        new = np.linalg.solve(yn, (-root[None, :] - np.conj(loads / voltage)).T).T
        if np.max(abs(new - voltage)) < 2e-14:
            voltage = new
            break
        voltage = new
    else:
        raise RuntimeError("Independent nodal solver did not converge")
    residual = float(np.max(abs(voltage @ yn.T + root + np.conj(loads / voltage))))
    return 1 - abs(voltage[:, terminals])**2, voltage, residual


class Audit:
    def __init__(self):
        self.checks, self.lp_records, self.exclusions = 0, 0, 0
        self.issues, self.cases = [], []
        self.max_gap, self.max_kcl = 0., 0.
        self.max_accepted_wzzt_gap = 0.

    def check(self, condition, description):
        self.checks += 1
        if not bool(condition):
            self.issues.append(description)

    def near(self, actual, expected, description, atol=1e-12):
        self.check(np.allclose(actual, expected, rtol=0, atol=atol), description)

    def lp(self, record, name):
        self.lp_records += 1
        self.check(record.get("certificate_valid_for_AC") and not record.get("linear_only"), name + ": AC scope")
        self.check(record.get("outer_feasibility_is_not_AC_feasibility") is True, name + ": outer/AC semantics")
        if record["status"] == "solver_failure_unresolved":
            self.check(record.get("lower") is None, name + ": failed solver cannot exclude")
            return
        lo, hi = F(record["exact_lower"]), F(record["exact_upper"])
        self.check(0 <= lo <= hi, name + ": exact interval")
        self.check(F(float(record["lower"])) <= lo, name + ": lower serialized outward")
        self.check(F(float(record["upper"])) >= hi, name + ": upper serialized outward")
        self.check(F(float(record["gap"])) >= hi - lo, name + ": gap serialized outward")
        self.check((record["status"] == "excluded") == (lo > 0), name + ": exclusion rule")
        self.check(record.get("exact_primal_verified") and record.get("exact_box_dual_verified"), name + ": verification flags")
        self.exclusions += record["status"] == "excluded"
        self.max_gap = max(self.max_gap, float(record["gap"]))

    def bank(self, bank, name):
        n, rows = bank["n_terminal"], bank["candidates"]
        self.check(len({row["id"] for row in rows}) == len(rows), name + ": unique IDs")
        self.check(bank.get("ranking_is_ac_evidence") is False, name + ": ranking not evidence")
        self.check(bank.get("selection_uses_truth") is False, name + ": no truth flag")
        self.check(bank.get("certified_supports") == [], name + ": no certified atoms")
        self.check(bank.get("rnj_supports_fixed_in_wzzt") is False, name + ": RNJ not fixed")
        self.check(bank["retained_count"] == len(rows), name + ": retained count")
        self.check(bank["generated_count"] >= bank.get("refreshed_proposal_count", len(rows)), name + ": fresh-bank count")
        wzzt = bank["diagnostics"]["wzzt"]
        self.check(wzzt.get("called") is True, name + ": WZZT invoked")
        if "mandatory_supports" in wzzt:
            self.check(wzzt["mandatory_supports"] == [[i] for i in range(n)], name + ": singleton restart")
            self.check(wzzt.get("global_topology_optimality") is False, name + ": greedy scope")
        for attempt in wzzt.get("solver_attempts", []):
            if attempt["accepted"]:
                primal, dual, gap = (attempt.get(k) for k in ("objective", "dual_bound", "mip_gap"))
                absolute_ok = primal is not None and dual is not None and np.isfinite(primal) and np.isfinite(dual) and abs(primal-dual) <= 1e-10
                relative_ok = gap is not None and np.isfinite(gap) and 0 <= gap <= 1e-10
                self.check(attempt["status"] == 0 and attempt["success"] and
                           (attempt["binary_variable_count"] == 0 or absolute_ok or relative_ok),
                           name + ": accepted stage own-domain numerical optimality")
                if gap is not None and np.isfinite(gap):
                    self.max_accepted_wzzt_gap = max(self.max_accepted_wzzt_gap, gap)
        for row in rows:
            supports = list(map(set, row["supports"]))
            self.check(all(1 < len(s) < n and s <= set(range(n)) for s in supports), name + ": support indices")
            self.check(all(not (a & b) or a <= b or b <= a for a in supports for b in supports), name + ": laminarity")
            self.check(row.get("replaceable") and row.get("certified_supports") == [] and row.get("ac_evidence") is None,
                       name + ": uncertified replaceable proposal")
            canonical = tuple(sorted((tuple(sorted(s)) for s in row["supports"]), key=lambda s: (-len(s), s)))
            expected_id = "passive_" + hashlib.sha256(repr((n, canonical)).encode("ascii")).hexdigest()[:12]
            self.check(row["id"] == expected_id, name + ": content-addressed candidate identity")

    def physical(self, obs, truth, name):
        p, q, y = (np.asarray(truth[key], float) for key in ("P", "Q", "Y"))
        n, h = p.shape[1], truth["history_count"]
        self.check(len(p) == h + 24, name + ": 24 future slots")
        self.check(np.array_equal(obs["batch_id"], np.r_[np.zeros(h), np.ones(24)]), name + ": batch labels")
        vm, vhat = np.asarray(truth["voltage_terminal"]), np.asarray(truth["voltage_measured"])
        root_hat = np.asarray(truth["root_voltage_measured"])
        gains, jitter = np.asarray(truth["meter_gains"]), np.asarray(truth["meter_jitter"])
        settings = truth["meter_settings"]
        self.check(np.all(abs(gains - 1) <= settings["beta"] + 2e-16), name + ": gain bounds")
        self.check(np.all(abs(jitter) <= settings["jitter"]), name + ": jitter bounds")
        self.near(vhat, gains[None, 1:] * vm + jitter[:, 1:], name + ": terminal meter equation")
        self.near(root_hat, gains[0] + jitter[:, 0], name + ": root meter equation")
        self.near(obs["Y_drop"], root_hat[:, None]**2 - vhat**2, name + ": measured drops")
        self.near(y, 1 - vm**2, name + ": true drops")
        for measured, actual, error in (("P", p, "P_error"), ("Q", q, "Q_error"), ("Y_drop", y, "drop_error_bound")):
            self.check(np.all(abs(obs[measured] - actual) <= obs[error]), name + ": " + error + " covers truth")
        source = np.asarray(truth["source_p"])
        self.check(np.all(abs(source) <= obs["source_p_abs_cap"]), name + ": source P cap")
        self.check(np.all(abs(.3 * source) <= obs["source_q_abs_cap"]), name + ": source Q cap")
        self.check(np.all(abs(1.15 * source - obs["source_nominal_eff"]) <= obs["source_error_eff"]), name + ": source shared budget")
        metadata, theta = truth["candidate"], np.asarray(truth["theta"])
        bounds = [(.002, .014) if len(s) == n else ((.03, .10) if len(s) == 1 else (1e-6, .04)) for s in metadata["supports"]]
        self.check(all(lo <= v <= hi for v, (lo, hi) in zip(theta, bounds, strict=True)), name + ": theta domain")
        iy, voltage, residual = independent_nodal_ac(metadata, theta, p, q, source)
        self.near(iy, y, name + ": independent nodal AC", 2e-11)
        self.check(residual <= 1e-8, name + ": independent current balance")
        self.max_kcl = max(self.max_kcl, residual)
        self.check(np.min(abs(voltage)) >= .9 and np.max(abs(voltage)) <= 1.1, name + ": voltage domain")
        self.near(truth["min_voltage"], np.min(abs(voltage)), name + ": saved voltage min", 1e-11)
        self.near(truth["max_voltage"], np.max(abs(voltage)), name + ": saved voltage max", 1e-11)
        self.check(truth["equation_residual"] <= 1e-11, name + ": simulator residual")
        device = truth["device"]
        power, connection = np.asarray(device["power_kw"]), np.asarray(device["connection"])
        self.check(power.shape == connection.shape == (24,), name + ": EV shape")
        self.check(np.all((connection >= -1) & (connection < n)), name + ": connection indices")
        self.check(np.all((power == 0) | ((power >= 4.2) & (power <= 6))), name + ": charging power bounds")
        self.check(np.all(power[connection == -1] == 0), name + ": disconnected power zero")
        energy = float(np.sum(power) * .25)
        self.near(device["energy_kwh"], energy, name + ": reported grid energy")
        self.near(energy, 0 if device["kind"] == "none" else 27, name + ": same charging energy")
        travel = np.zeros(24)
        if device["kind"] == "mobile":
            self.check(np.array_equal(np.flatnonzero(connection == -1), [11, 12]), name + ": travel slots")
            travel[11:13] = 1
        self.near(device["travel_kwh"], np.sum(travel), name + ": travel energy")
        soc = .4 + np.cumsum(.95 * power * .25 - travel) / 60
        self.near(device["soc"], soc, name + ": independent SOC")
        self.check(np.all((soc >= .1) & (soc <= .9)), name + ": SOC bounds")
        increments = np.zeros((24, n))
        for slot, port in enumerate(connection):
            if port >= 0:
                increments[slot, port] = power[slot] / 20
        return dict(P=p, Q=q, source=source, increments=increments, history=h, gains=gains, jitter=jitter,
                    P_fraction=(obs["P"] - p) / (.02 * abs(p) + .0005),
                    Q_fraction=(obs["Q"] - q) / (.025 * np.hypot(p, q) + .0005))


def audit(output, replay_count=0, allow_incomplete=False):
    output = Path(output)
    protocol, checker = read(output / "protocol.json"), Audit()
    source_files = {}
    for label, expected in protocol["source_sha256"].items():
        path = Path(label)
        choices = [path] if path.is_absolute() else [ROOT / path, HERE / path]
        located = next((p for p in choices if p.is_file()), None)
        checker.check(located is not None and sha(located) == expected, "source hash: " + label)
        source_files[label] = None if located is None else sha(located)
    case_files = sorted(output.glob("*/result.json"))
    checker.check(0 < len(case_files) <= len(protocol["cases"]) if allow_incomplete else
                  len(case_files) == len(protocol["cases"]), "every required case completed")
    planned = {tuple(config) for config in protocol["cases"]}
    replays = []
    for case_file in case_files:
        folder, result = case_file.parent, read(case_file)
        case = result["case"]
        checker.check(tuple(result["config"]) in planned, case + ": planned case")
        checker.check(folder.name == case, case + ": folder identity")
        arrays, truths, physical = {}, {}, {}
        for filename, expected in result["inputs"].items():
            path = folder / filename
            checker.check(sha(path) == expected, case + ": observation hash " + filename)
            with np.load(path, allow_pickle=False) as values:
                arrays[path.stem] = {key: values[key].copy() for key in values.files}
        for filename, expected in result["truth_files"].items():
            path = folder / filename
            checker.check(sha(path) == expected, case + ": truth hash " + filename)
            truths[path.stem.removesuffix("_truth")] = read(path)
        checker.check(set(arrays) == set(truths), case + ": observation/truth pairing")
        for key in arrays:
            physical[key] = checker.physical(arrays[key], truths[key], case + "/" + key)
        baseline_key = next(key for key, tr in truths.items() if tr["device"]["kind"] == "none")
        baseline, base_obs = physical[baseline_key], arrays[baseline_key]
        for key, data in physical.items():
            h = data["history"]
            for field in base_obs:
                checker.check(np.array_equal(arrays[key][field][:h], base_obs[field][:h]), case + "/" + key + ": paired history " + field)
            checker.near(data["P"][h:] - data["increments"], baseline["P"][h:], case + "/" + key + ": paired background")
            for field in ("Q", "source", "gains", "jitter", "P_fraction", "Q_fraction"):
                checker.near(data[field], baseline[field], case + "/" + key + ": paired " + field, 2e-12)
        banks, seen_banks = [], set()
        for path, item in walk(result):
            if "candidates" in item and "n_terminal" in item and "diagnostics" in item:
                encoded = json.dumps(item, sort_keys=True)
                if encoded not in seen_banks:
                    checker.bank(item, case + path)
                    banks.append(item)
                    seen_banks.add(encoded)
            if "exact_lower" in item or (item.get("status") == "solver_failure_unresolved" and "certificate_valid_for_AC" in item):
                checker.lp(item, case + path)
            if {"records", "retained", "model"} <= item.keys():
                retained = [key for key, record in item["records"].items() if record["status"] != "excluded"]
                checker.check(item["retained"] == retained, case + path + ": retain unless valid exclusion")
        initial = result["proposals"]
        initial_ids = {row["id"] for row in initial["candidates"]}
        checker.check(set(result["historical_evidence"]["records"]) == initial_ids, case + ": historical bank coverage")
        for filename, field in (("passive.json", "proposals"), ("historical_evidence.json", "historical_evidence"), ("design.json", "design")):
            checker.check(read(folder / filename) == result[field], case + ": saved intermediate " + filename)
        truth_supports = truths[baseline_key]["candidate"]["supports"]
        n = initial["n_terminal"]
        all_rows = {row["id"]: row for bank in banks for row in bank["candidates"]}
        expected_metrics = {key: metrics(row["supports"], truth_supports, n) for key, row in all_rows.items()}
        checker.check(expected_metrics == result["candidate_truth_metrics"], case + ": independent clade metrics")
        checker.check(result["truth_in_bank"] == any(expected_metrics[key]["exact"] for key in initial_ids), case + ": initial truth coverage")
        checker.check(result["passive_best_metrics"] == expected_metrics[initial["candidates"][0]["id"]], case + ": initial rank-one metric")
        for key, models in result["true_model_evaluation_only"].items():
            for model, record in models.items():
                checker.check(record["status"] != "excluded", case + "/" + key + "/" + model + ": true model retained")
        for key, bank in result.get("post_banks", {}).items():
            bank_ids = {row["id"] for row in bank["candidates"]}
            checker.check(initial_ids <= bank_ids, case + "/" + key + ": old candidates preserved in common ablation bank")
            checker.check(set(bank["initial_retained_preserved"]) == initial_ids, case + "/" + key + ": preserved IDs")
            checker.check(not bank["excluded_history_ids"], case + "/" + key + ": neutral history does not delete broader-domain candidates")
            checker.check(read(folder / (key + "_post_passive.json")) == bank, case + "/" + key + ": saved post bank")
            for model, record in result["records"][key].items():
                checker.check(set(record["records"]) == bank_ids, case + "/" + key + "/" + model + ": exact bank coverage")
                for candidate_id in bank_ids:
                    if expected_metrics[candidate_id]["exact"]:
                        checker.check(candidate_id in record["retained"], case + "/" + key + "/" + model + ": exact true-family bank member retained")
        policy_counts = {}
        for label, summary in result["policies"].items():
            key = summary["action"]
            bank = result.get("post_banks", {}).get(key, initial)
            for model in ("shared_gain", "independent_box"):
                retained = result["records"][key][model]["retained"]
                checker.check(summary[model]["retained_ids"] == retained and summary[model]["retained_count"] == len(retained), case + "/" + label + ": summary " + model)
                checker.check(summary[model]["excluded_count"] == len(bank["candidates"]) - len(retained), case + "/" + label + ": exclusion count " + model)
                if "initial_bank_retained_ids" in summary[model]:
                    common_survivors = sorted(initial_ids & set(retained))
                    checker.check(summary[model]["initial_bank_retained_ids"] == common_survivors, case + "/" + label + ": common-bank IDs " + model)
                    checker.check(summary[model]["initial_bank_retained_count"] == len(common_survivors), case + "/" + label + ": common-bank retained count " + model)
                    checker.check(summary[model]["initial_bank_excluded_count"] == len(initial_ids) - len(common_survivors), case + "/" + label + ": common-bank exclusion count " + model)
            if "truth_in_updated_bank" in summary:
                checker.check(summary["truth_in_updated_bank"] == any(expected_metrics[row["id"]]["exact"] for row in bank["candidates"]), case + "/" + label + ": refreshed truth coverage")
                checker.check(summary["post_passive_best_metrics"] == expected_metrics[bank["candidates"][0]["id"]], case + "/" + label + ": refreshed rank-one F1")
            policy_counts[label] = dict(initial_common_bank_survivors=len(initial_ids & set(result["records"][key]["shared_gain"]["retained"])),
                updated_bank_survivors=len(result["records"][key]["shared_gain"]["retained"]),
                updated_bank_size=len(bank["candidates"]))
        design = result["design"]
        if design["status"] == "scored":
            for kind, chosen in design["selected"].items():
                eligible = [row for row in design["scores"] if row["kind"] == kind]
                checker.check(chosen == max(eligible, key=lambda row: row["score"]), case + ": action maximum " + kind)
            checker.check(set(design["references"]) <= set(result["historical_evidence"]["retained"]), case + ": references from survivors")
        checker.cases.append(dict(case=case, truth_in_initial_bank=result["truth_in_bank"],
            initial_rank_one_f1=result["passive_best_metrics"]["f1"], physical_actions=len(arrays),
            banks_checked=len(banks), policy_counts=policy_counts,
            references_without_numeric_witness=sum(not ref.get("historical_numeric_witness", False) for ref in design.get("references", {}).values())))
        # Reconstruct three problems from the first completed case: physical
        # truth, a saved excluded candidate (if any), and another bank member.
        if replay_count and not replays:
            sys.path.insert(0, str(HERE))
            import ac_general_v9 as ac
            key = baseline_key
            settings = truths[key]["meter_settings"]
            kwargs = dict(shared_voltage_gain={k: settings[k] for k in ("gain_bounds", "root_jitter", "terminal_jitter", "vmax")})
            specs = [(truths[key]["candidate"]["name"], truths[key]["candidate"]["supports"], result["true_model_evaluation_only"][key]["shared_gain"])]
            bank = result.get("post_banks", {}).get(key, initial)
            rows = sorted(bank["candidates"], key=lambda row: result["records"][key]["shared_gain"]["records"][row["id"]]["status"] != "excluded")
            specs += [(row["id"], row["supports"], result["records"][key]["shared_gain"]["records"][row["id"]]) for row in rows]
            for candidate_id, supports, old in specs[:replay_count]:
                candidate = ac.RadialCandidate.from_supports(candidate_id, n, supports)
                box = tuple((.002, .014) if len(s) == n else ((.03, .10) if len(s) == 1 else (1e-6, .04)) for s in candidate.supports)
                fresh = ac.solve_outer(arrays[key], candidate, box=box, **kwargs)
                checker.lp(fresh, case + "/fresh_replay/" + candidate_id)
                overlapping = (F(old["exact_lower"]) <= F(fresh["exact_upper"]) and F(fresh["exact_lower"]) <= F(old["exact_upper"]))
                checker.check(overlapping, case + ": replay certificate intervals overlap " + candidate_id)
                checker.check(fresh["status"] == old["status"], case + ": replay status " + candidate_id)
                replays.append(dict(case=case, action=key, candidate=candidate_id, model="shared_gain", intervals_overlap=overlapping, fresh=fresh))
    checker.check(len(replays) == replay_count, "requested LP replay count completed")
    aggregate = output / "results.json"
    if aggregate.exists():
        data = read(aggregate)
        checker.check(data["protocol"] == protocol, "aggregate protocol matches frozen file")
        aggregated = {r["case"]: r for r in data["results"]}
        checker.check(set(aggregated) == {read(p)["case"] for p in case_files}, "aggregate complete case set")
        for path in case_files:
            item = read(path)
            checker.check(aggregated[item["case"]] == item, item["case"] + ": aggregate equals per-case artifact")
    report = dict(passed=not checker.issues, checks=checker.checks, issues=checker.issues,
        completed_cases=len(case_files), planned_cases=len(protocol["cases"]), source_hashes=source_files,
        complete=len(case_files) == len(protocol["cases"]), allow_incomplete=allow_incomplete,
        lp_records_checked=checker.lp_records, excluded_records=checker.exclusions,
        maximum_serialized_gap=checker.max_gap, maximum_independent_KCL_residual=checker.max_kcl,
        maximum_accepted_WZZT_relative_gap=checker.max_accepted_wzzt_gap,
        cases=checker.cases, lp_replays=replays, auditor_sha256=sha(__file__),
        protocol_sha256=sha(output / "protocol.json"),
        limitations=["Saved rational intervals are checked; unstored original dual multipliers are not reconstructed.",
                     "WZZT stage acceptance uses the existing 1e-10 numerical gap contract, not rational global topology optimality.",
                     "Candidate coverage is finite and post-hoc truth coverage is reported separately.",
                     "Raw updated-bank survivor counts cannot be compared alone across different candidate banks.",
                     "Synthetic error coverage is not field-calibrated confidence."])
    destination = "artifact_audit_partial.json" if allow_incomplete else "artifact_audit.json"
    (output / destination).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--replays", type=int, default=0)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result = audit(args.output, args.replays, args.allow_incomplete)
    print(json.dumps({key: result[key] for key in ("passed", "checks", "issues", "completed_cases")}, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)
