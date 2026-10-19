"""Independent replay of the saved known-line case33 factorial experiment.

Run from this directory with the canonical research checkout on PYTHONPATH.
Approximate outputs below are reconstructed from physical branch tables; they
do not call FixedResponse or use its candidate matrices or loss coefficients.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from research_experiments.mobile_ev_mle.reference_benchmark import representation


CANONICAL = Path("C:/Users/23761/.codex/worktrees/36a1/Topo").resolve()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(ok, message):
    if not ok:
        raise AssertionError(message)


class PhysicalReplay:
    def __init__(self, network):
        self.net = network
        self.ids = network["observation_bus_ids"]
        self.base = network["base_kw"]
        self.root = network["root_bus"]
        self.n = len(self.ids)
        self.adj = {b["bus_id"]: [] for b in network["buses"]}
        for edge in network["branches"]:
            a, b = edge["from_bus"], edge["to_bus"]
            z = complex(edge["r_ohm"], edge["x_ohm"])
            z /= network["base_kv"] ** 2 / network["base_mva"]
            self.adj[a].append((b, z))
            self.adj[b].append((a, z))
        self.order = [self.root]
        self.parent = {self.root: None}
        self.z = {}
        for a in self.order:
            for b, impedance in self.adj[a]:
                if b == self.parent[a]:
                    continue
                require(b not in self.parent, "physical graph is not a tree")
                self.parent[b] = a
                self.z[b] = impedance
                self.order.append(b)

    def output(self, z, backend):
        z = np.asarray(z, float)
        if backend == "ac":
            p = representation.physical_tree_ac(
                self.net["branches"], self.ids, self.root,
                self.net["base_kv"], self.base,
                z[:self.n] * self.base, z[self.n:] * self.base,
                source_bus=2, source_p_kw=0., source_q_kvar=0., root_v=1.)
            return np.r_[p["v"][0], p["master_p"][0], p["master_q"][0]]
        require(backend in ("linear", "i2"), "unknown backend")
        demand = {b: 0j for b in self.order}
        for j, b in enumerate(self.ids):
            demand[b] = complex(z[j], z[self.n+j])
        subtree = dict(demand)
        for b in reversed(self.order[1:]):
            subtree[self.parent[b]] += subtree[b]
        # Approximate current uses lossless downstream demand and root V=1.
        ell = {b: abs(subtree[b]) ** 2 if backend == "i2" else 0.
               for b in self.order[1:]}
        # Explicit bottom-up accumulation includes the loss on the edge itself.
        downstream_loss = {b: self.z[b] * ell[b] for b in self.order[1:]}
        downstream_loss[self.root] = 0j
        for b in reversed(self.order[1:]):
            downstream_loss[self.parent[b]] += downstream_loss[b]
        u = {self.root: 1.}
        for b in self.order[1:]:
            sending = subtree[b] + downstream_loss[b]
            u[b] = (u[self.parent[b]]
                    - 2 * (self.z[b].conjugate() * sending).real
                    + abs(self.z[b]) ** 2 * ell[b])
            require(u[b] > 0., "nonpositive approximate squared voltage")
        master = subtree[self.root] + downstream_loss[self.root]
        return np.r_[[np.sqrt(u[b]) for b in self.ids], master.real, master.imag]


def objective(replay, backend, z, zobs, yobs, sz, sy):
    ry = (replay.output(z, backend) - yobs) / sy
    rz = (z-zobs) / sz
    parts = dict(pq=.5*float(rz@rz), v=.5*float(ry[:-2]@ry[:-2]),
                 root_p=.5*float(ry[-2]**2), root_q=.5*float(ry[-1]**2))
    return sum(parts.values()), parts


def gradient(replay, backend, z, zobs, yobs, sz, sy):
    # Differentiate independent physical output in standardized correction w.
    # h=0.01 means one hundredth of one instrument standard deviation.
    h = .01
    j = np.empty((len(yobs), len(z)))
    for c in range(len(z)):
        delta = np.zeros_like(z)
        delta[c] = h * sz[c]
        j[:, c] = ((replay.output(z+delta, backend)
                    - replay.output(z-delta, backend)) / (2*h)) / sy
    residual = (replay.output(z, backend)-yobs)/sy
    return (z-zobs)/sz + j.T@residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--results", dest="output", type=Path, required=True)
    args = parser.parse_args()
    began = perf_counter()
    require(CANONICAL in Path(representation.__file__).resolve().parents,
            "audit must import the canonical checkout, not the D: mainline")
    output = args.output.resolve()
    cfg = read(output / "configuration.json")
    execution = read(output / "execution.json")
    network = read(output / "case33.json")["network"]
    states = read(output / "states.json")
    replay = PhysicalReplay(network)
    require(replay.ids == list(range(2, 34)), "all 32 nonroot meters required")
    require(len(network["branches"]) == 32, "32 original physical lines required")
    nominal = {b["bus_id"]: b for b in network["buses"]}
    p0 = np.array([nominal[b]["pd_kw"] for b in replay.ids])
    q0 = np.array([nominal[b]["qd_kvar"] for b in replay.ids])
    require(abs(sum(p0)-3715.) < 1e-10, "case33 nominal P changed")
    require(abs(sum(q0)-2300.) < 1e-10, "case33 nominal Q changed")
    require(cfg["fitted_line_parameters"] is False, "this audit fixes true RX")
    require(cfg["source_P_Q_fixed"] == [0., 0.], "source must be fixed absent")
    max_state_error = 0.
    for st in states:
        definition = st["state"]
        pv = np.zeros(replay.n)
        for bus, weight in zip(cfg["pv_buses"], cfg["pv_weights"]):
            pv[replay.ids.index(bus)] = (cfg["pv_capacity_fraction"] * sum(p0)
                                        * weight * definition["pv_factor"])
        expected = np.r_[definition["p_load"]*p0-pv, definition["q_load"]*q0]/replay.base
        require(np.max(abs(expected-np.array(st["z_true"]))) < 1e-14,
                "saved true PQ differs from stated load/PV arithmetic")
        require(np.max(abs(pv-np.array(st["pv_kw"]))) < 1e-10,
                "PV allocation mismatch")
        err = float(np.max(abs(replay.output(expected, "ac")-np.array(st["y_true"]))))
        max_state_error = max(max_state_error, err)
        require(err < 1e-10, "independent truth replay mismatch")
    by_name = {st["name"]: st for st in states}
    samples = []
    for st in cfg["states"]:
        name = st["name"]
        for sigma in cfg["power_sigmas"]:
            for seed in execution["actual_seeds"]:
                samples.append(read(output/f"{name}_{sigma}_{seed}.json"))
        samples.append(read(output/f"{name}_clean.json"))
    require(len(execution["actual_seeds"]) == 8, "main audit expects all eight seeds")
    require(len(samples) == 51, "48 noisy plus three clean samples required")
    expected_methods = {a+"/"+b for a in ("linear", "i2", "ac") for b in ("FIX", "EIV")}
    records = []
    maxima = dict(self_score_absolute=0., AC_back_score_absolute=0.,
                  score_parts_absolute=0., gradient_inf=0.,
                  reference_gradient_inf=0., gradient_record_difference=0.,
                  score_decomposition_absolute=0.)
    failures = []
    reference_unqualified = []
    manifest = []
    for sample in samples:
        require(set(sample["methods"]) == expected_methods, "six method slots missing")
        zobs, yobs, sz, sy = [np.array(sample[k], float)
                              for k in ("zobs", "yobs", "sigma_z", "sigma_y")]
        st = by_name[sample["state"]]
        if sample["clean"]:
            require(np.array_equal(zobs, np.array(st["z_true"])), "clean PQ is not clean")
            require(np.array_equal(yobs, np.array(st["y_true"])), "clean y is not clean")
        else:
            for observed, truth, sigma, noise in (
                (zobs, st["z_true"], sz, sample["standardized_noise_z"]),
                (yobs, st["y_true"], sy, sample["standardized_noise_y"])):
                require(np.max(abs(observed-(np.array(truth)+sigma*np.array(noise)))) < 1e-14,
                        "noise reconstruction mismatch")
        if not sample.get("reference_qualified", False):
            reference_unqualified.append((sample["state"], sample["power_sigma"], sample["seed"], sample["clean"]))
        for key, item in sample["methods"].items():
            record = dict(state=sample["state"], power_sigma=sample["power_sigma"],
                          seed=sample["seed"], clean=sample["clean"], method=key)
            if not item["success"]:
                failures.append(record)
                records.append(dict(record, success=False))
                continue
            backend, mode = key.split("/")
            z = np.array(item["z"], float)
            if mode == "FIX":
                require(np.array_equal(z, zobs), "FIX changed noisy PQ")
            self_score, parts = objective(replay, backend, z, zobs, yobs, sz, sy)
            ac_score, _ = objective(replay, "ac", z, zobs, yobs, sz, sy)
            self_error = abs(self_score-item["score"])
            ac_error = abs(ac_score-item["AC_back_score"])
            part_error = max(abs(parts[k]-item["parts"][k]) for k in parts)
            maxima["self_score_absolute"] = max(maxima["self_score_absolute"], self_error)
            maxima["AC_back_score_absolute"] = max(maxima["AC_back_score_absolute"], ac_error)
            maxima["score_parts_absolute"] = max(maxima["score_parts_absolute"], part_error)
            require(self_error < 1e-7*(1+self_score), "independent self score mismatch")
            require(ac_error < 1e-7*(1+ac_score), "independent AC replay score mismatch")
            require(part_error < 1e-7*(1+self_score), "independent score channels mismatch")
            refscore = sample["methods"]["ac/"+mode]["score"]
            decomp = ((self_score-ac_score)+(ac_score-refscore))-(self_score-refscore)
            maxima["score_decomposition_absolute"] = max(maxima["score_decomposition_absolute"], abs(decomp))
            record.update(success=True, self_score=self_score, AC_back_score=ac_score,
                          self_score_error=self_error, AC_back_score_error=ac_error)
            if mode == "EIV":
                gmax = float(np.max(abs(gradient(replay, backend, z, zobs, yobs, sz, sy))))
                maxima["gradient_inf"] = max(maxima["gradient_inf"], gmax)
                maxima["gradient_record_difference"] = max(maxima["gradient_record_difference"], abs(gmax-item["gradient_inf"]))
                record["independent_gradient_inf"] = gmax
                record["reported_gradient_inf"] = item["gradient_inf"]
                if backend == "ac":
                    maxima["reference_gradient_inf"] = max(maxima["reference_gradient_inf"], gmax)
                    require(gmax < cfg["reference_gradient_tolerance"]+2e-7,
                            "independent AC reference gradient exceeds tolerance")
                fixed = sample["methods"][backend+"/FIX"]["score"]
                require(self_score <= fixed+1e-7*(1+fixed), "EIV is worse than feasible zero correction")
            records.append(record)
    checks = read(output / "derivative_checks.json")
    require(len(checks) == 9, "all three state by three backend derivative checks required")
    require(all(c["max_relative_error"] <= 2e-5 and c["directions"] == 6 for c in checks),
            "saved analytic derivative checks exceeded tolerance")
    summary = read(output / "summary.json")
    require(summary["noisy_samples"] == 48 and summary["clean_samples"] == 3,
            "summary sample denominators differ")
    require(summary["method_slots"] == 306, "summary method denominator must be 306")
    for filename in ("configuration.json", "execution.json", "case33.json", "states.json", "summary.json"):
        blob = (output/filename).read_bytes()
        manifest.append(dict(path=filename, sha256=hashlib.sha256(blob).hexdigest()))
    result = dict(audit_version="physical_tree_independent_factorial_audit_v1",
        passed=not failures and not reference_unqualified,
        canonical_import=str(Path(representation.__file__).resolve()),
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        samples=51, noisy_samples=48, clean_samples=3, method_slots=len(records),
        failed_slots=failures, unqualified_references=reference_unqualified,
        state_truth_max_error=max_state_error, maxima=maxima,
        approximation_replay="independent physical-tree downstream-loss accumulation; no FixedResponse or candidate matrices",
        gradient_replay="independent centered differences of physical outputs in standardized-PQ coordinates; step=0.01",
        global_MLE_certified=False, inputs=manifest, records=records,
        seconds=perf_counter()-began)
    (output/"audit.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("records", "inputs")}, indent=2))
    require(result["passed"], "failures or unqualified references retained")


if __name__ == "__main__":
    main()
