"""Original device graph with several exact posterior-moment constraints.

Spatial/temporal auxiliary variables factor the same retained Gaussian
quadratic without materializing dense time-by-resource factor matrices.
"""
from __future__ import annotations

import os
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from scipy.special import logsumexp

from ... import io
from ...dr.pipeline import load_resources
from ...dr.device_graph import build_device_graph, _attribute, _values
from .history_scan import ROOT, load_request, require, sha, utc
from .prospective import ForecastLaw


class GaussianGate:
    def __init__(self, difference, law, sigma2, node, qratio, base_kw, *, competitor, h):
        self.competitor, self.h = int(competitor), float(h)
        dh = np.asarray(difference, float); self.law = law
        n = dh.shape[0]; scale = np.sqrt(sigma2); self.coefficient = .125
        factor = (dh[:, :n]*law.sd[:n]+dh[:, n:]*law.sd[n:])/scale
        self.V, singular, _ = np.linalg.svd(np.sqrt(self.coefficient)*factor, full_matrices=True)
        self.eigen = 1+2*law.values[:, None]*singular[None, :]**2
        self.half_logdet = float(.5*np.log(self.eigen).sum())
        self.mean = law.mean@dh.T/scale
        self.effect = (dh[:, node]+dh[:, n+np.asarray(node)]*np.asarray(qratio)[None])/base_kw/scale
        self.active = np.flatnonzero(np.any(self.effect != 0, axis=0))
        self.mean_spatial = np.sqrt(self.coefficient)*self.mean@self.V
        self.resource_spatial = np.sqrt(self.coefficient)*self.effect.T@self.V
        self.U = law.U
        self.trace_C = float(len(law.mean)*np.sum(factor**2))
        node_C = factor@factor.T
        self.trace_C2 = float(np.sum(law.values**2)*np.sum(node_C**2))
        self.max_eigen_C = float(law.values.max()*np.linalg.eigvalsh(node_C).max())

    def value(self, delivery):
        spatial = self.mean_spatial+np.asarray(delivery).T@self.resource_spatial
        z = (self.U.T@spatial)/np.sqrt(self.eigen)
        return float(np.sum(z*z)+self.half_logdet)

    def capacity_bounds(self, low, high, history_log_odds, beta):
        positive, negative = np.maximum(self.effect, 0.), np.minimum(self.effect, 0.)
        mlo = self.mean+np.asarray(low).T@positive.T+np.asarray(high).T@negative.T
        mhi = self.mean+np.asarray(high).T@positive.T+np.asarray(low).T@negative.T
        norm_upper = float(np.maximum(np.abs(mlo), np.abs(mhi)).ravel()@np.maximum(np.abs(mlo), np.abs(mhi)).ravel())
        # The Gaussian inverse factor is contractive; box contains original devices.
        moment_upper = self.coefficient*norm_upper+self.half_logdet
        # Necessary pair event for the ORIGINAL full-posterior chance constraint.
        # L=L_h+Q/2+noise; E L=L_h+E Q/2, Var L=E Q+Var(Q)/4.
        q_mean_upper = norm_upper+self.trace_C
        mean_upper = float(history_log_odds+q_mean_upper/2)
        variance_upper = float(q_mean_upper+self.trace_C2/2+self.max_eigen_C*norm_upper)
        threshold = float(np.log(beta/(1-beta)))
        success_upper = 1.
        if mean_upper < threshold:
            success_upper = variance_upper/(variance_upper+(threshold-mean_upper)**2)
        return dict(moment_upper=moment_upper, required=self.h,
            sufficient_constraint_box_infeasible=moment_upper+1e-9 < self.h,
            pair_success_upper=success_upper, mean_log_odds_upper=mean_upper,
            variance_log_odds_upper=variance_upper,
            original_chance_infeasible=success_upper < .9,
            certificate_domain="all original controls contained in the delivery-capacity box, declared fixed Gaussian law")

    def add_to_model(self, model, delivery, resources, prefix):
        import gurobipy as gp
        T, n = self.mean_spatial.shape
        coefficients = self.resource_spatial[resources]
        low, high = delivery.LB, delivery.UB
        positive, negative = np.maximum(coefficients, 0.), np.minimum(coefficients, 0.)
        wl = self.mean_spatial+low.T@positive+high.T@negative
        wh = self.mean_spatial+high.T@positive+low.T@negative
        w = model.addMVar((T, n), lb=wl, ub=wh, name=prefix+"_spatial")
        for k in range(n):
            model.addConstr(w[:, k] == self.mean_spatial[:, k]+coefficients[:, k]@delivery)
        up, un = np.maximum(self.U.T, 0.), np.minimum(self.U.T, 0.)
        divisor = np.sqrt(self.eigen)
        zl, zh = (up@wl+un@wh)/divisor, (up@wh+un@wl)/divisor
        z = model.addMVar((T, n), lb=zl, ub=zh, name=prefix+"_temporal")
        for k in range(n):
            model.addConstr(z[:, k] == (self.U.T@w[:, k])/divisor[:, k])
        flat = z.reshape(-1)
        model.addConstr(flat@flat+self.half_logdet >= self.h+1e-7, name=prefix+"_posterior_moment")
        return w, z


def select_inputs(run):
    rows = io.read(run/"development/results.json")
    beta = io.read(run/"development/threshold_lock.json")["beta_star"]
    selected = io.read(run/"development/protocol.json")["selected"]
    primary, seen = [], set()
    for item in selected:
        valid = [r for r in rows if r["id"] == item["id"] and r["status"] == "complete" and r["target_correct"]
                 and r["truth_retained_at_beta_star"] and r["gates"][str(beta)]["gap_evidence"]
                 and r["gates"][str(beta)]["active_resource_count"] > 0]
        if valid and item["case"] not in seen:
            primary.append(dict(item, role="primary")); seen.add(item["case"])
    controls = []
    for item in selected:
        if item["id"] in {x["id"] for x in primary} or len(controls) >= 2:
            continue
        valid = [r for r in rows if r["id"] == item["id"] and r["status"] == "complete" and r["target_correct"]
                 and r["truth_retained_at_beta_star"] and r["gates"][str(beta)]["zero_certified"]]
        if valid:
            controls.append(dict(item, role="zero_control"))
    return primary[:4]+controls, beta


def setup(run):
    out = run/"planning"; require(not out.exists(), "Existing pricing preserved")
    out.mkdir()
    selected, beta = select_inputs(run)
    source = {}
    for path in [*sorted(Path(__file__).parent.glob("*.py")),
                 *sorted((ROOT/"rnj_wzzt_dr/dr").glob("*.py"))]:
        relative = path.relative_to(ROOT)
        target = out/"source_snapshot"/relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        source[relative.as_posix()] = io.ref(path)
    protocol = dict(created_utc=utc(), selected=selected, beta_star=beta, eta=.9,
        selection_rule="first ranked qualified residual-gap input per network, plus first two qualified zero-bound controls",
        offline_truth_role="sample eligibility only; quote builder does not receive a truth topology or physical branch parameters",
        target_rule="both predeclared historical top two, all frozen competitors retained",
        sources=source, seconds_per_target=1800., seconds_per_input=3600., threads=1, mip_gap=.01,
        independent_model_confirmation_paths=2048, physical_paths=32,
        physical_stream=dict(namespace="posterior_dr_confirm_20261112", study_seed=2026111201, replicate_start=20000, replicate_stop=20032),
        no_actual_future_opened=True, objective="unchanged original nominal device fee",
        inner_domain="fixed uniform active failure allocations and uniform tail; not the full original chance domain")
    io.write(out/"protocol.json", protocol)
    print("Frozen pricing/test input IDs: "+", ".join(x["id"] for x in selected), flush=True)
    return out, protocol


def build_request(run, identity, target, beta):
    request = next(r for r in io.read(run/"requests.json") if r["id"] == identity)
    native, train, valid = load_request(request)
    scored = io.read(run/"scores"/(identity+".json"))
    row = io.read(run/"development/targets"/(identity+f"__target{target}.json"))
    require("target_correct" not in row, "Truth label entered the quote builder")
    forecast = np.column_stack((np.vstack((train.p, valid.p)), np.vstack((train.q, valid.q))))
    law = ForecastLaw(forecast)
    H = np.asarray([p["H"] for p in scored["parameters"]])
    original, problem, base_kw = load_resources(request["case"], labels=train.labels)
    allocation = row["gates"][str(beta)]["allocation"]
    gates = [GaussianGate(H[target]-H[x["competitor"]], law, scored["sigma2"], original.node,
                         original.qratio, base_kw, competitor=x["competitor"], h=x["h"])
             for x in allocation["thresholds"]]
    zero = np.zeros_like(original.delta_lower_kw)
    by_j = {p["competitor"]: p for p in row["gates"][str(beta)]["bound"]["pairs"]}
    capacity = []
    for gate in gates:
        require(abs(gate.value(zero)-by_j[gate.competitor]["T"]) <= 1e-8*(1+by_j[gate.competitor]["T"]), "Control factor disagrees with zero moment")
        log_odds = scored["models"][target]["log_probability"]-scored["models"][gate.competitor]["log_probability"]
        capacity.append(dict(competitor=gate.competitor,
            **gate.capacity_bounds(original.delta_lower_kw, original.delta_upper_kw, log_odds, beta)))
    receipt = dict(id=identity, target=target, beta=beta, eta=.9,
        score_sha256=sha(run/"scores"/(identity+".json")),
        source_result=request["result"], source_observed=request["observed"],
        parameter_sha256=scored["parameter_sha256"], sigma2=scored["sigma2"],
        forecast_rule="repeat known historical 96 observed rows", forecast_sha256=io.contract_hash(forecast),
        allocation=allocation, capacity=capacity, zero_bound=row["gates"][str(beta)]["bound"],
        original_resources=dict(case=request["case"], base_kw=base_kw, ids=original.ids),
        receipt_excludes_truth_parameters=True)
    return original, problem, gates, receipt


def solve(problem, gates, destination, seconds, gap=.01):
    import gurobipy as gp
    began = time.perf_counter()
    resources = np.array(sorted({int(r) for gate in gates for r in gate.active}), dtype=int)
    warm = problem.expand(problem.zero)
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0); env.start()
        with gp.Model("posterior_full_bank_original_devices", env=env) as model:
            settings = dict(OutputFlag=1, LogToConsole=0, LogFile=str(destination/"gurobi.log"), Threads=1,
                Seed=0, NonConvex=2, NumericFocus=2, MIPFocus=1, MIPGap=gap, TimeLimit=seconds,
                FeasibilityTol=1e-8, IntFeasTol=1e-8, OptimalityTol=1e-8, DualReductions=0)
            for key, value in settings.items():
                model.setParam(key, value)
            x, u, cost = build_device_graph(model, problem, warm)
            for r in set(range(problem.device.R))-set(resources.tolist()):
                for h in range(problem.H):
                    model.addConstr(x[r, h].expr == 0.)
            lo, hi = problem.device.delta_lower_kw[resources], problem.device.delta_upper_kw[resources]
            delivery = model.addMVar(lo.shape, lb=lo, ub=hi, name="original_delivery")
            for index in np.ndindex(lo.shape):
                model.addConstr(delivery[index] == u[resources[index[0]], index[1]].expr)
            model.update()
            auxiliaries = [g.add_to_model(model, delivery, resources, f"competitor_{g.competitor}") for g in gates]
            model.setObjective(cost, gp.GRB.MINIMIZE); model.update()
            model.Params.TimeLimit = max(.01, seconds-(time.perf_counter()-began))
            model.write(str(destination/"model.mps.gz")); model.optimize()
            result = dict(status="solver_finished", native_status=int(model.Status), has_solution=bool(model.SolCount),
                qualified=False, objective=None, lower_bound=_attribute(model, "ObjBound"),
                bound_scope="fixed-allocation posterior-moment sufficient domain only",
                native_gap=_attribute(model, "MIPGap") if model.SolCount else None,
                variables=model.NumVars, binaries=model.NumBinVars, active_resources=len(resources),
                constraints=len(gates), elapsed_seconds=time.perf_counter()-began)
            if model.SolCount:
                requests = _values(x)
                values = np.clip(requests.ravel()[problem.active], problem.lower, problem.upper)
                point = problem.nominal(values)
                moments = [g.value(point["mean_u"]) for g in gates]
                audit = problem.audit_delivery(point["mean_u"])
                margins = [value-g.h for value, g in zip(moments, gates, strict=True)]
                if audit["passed"] and min(margins, default=0.) >= 0.:
                    result.update(qualified=True, objective=float(point["fee"]), device=audit,
                                  moments=moments, margins=margins)
                    np.savez_compressed(destination/"quote.npz", x=point["x"], mean_u=point["mean_u"])
                else:
                    result.update(replay_failed=True, device=audit, moments=moments, margins=margins)
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(); run = args.run.resolve()
    out, protocol = setup(run)
    results = []
    for item in protocol["selected"]:
        for target in item["targets"]:
            destination = out/"targets"/(item["id"]+f"__target{target}"); destination.mkdir(parents=True)
            began = time.perf_counter()
            try:
                original, problem, gates, receipt = build_request(run, item["id"], target, protocol["beta_star"])
                io.write(destination/"request.json", receipt)
                row = dict(id=item["id"], target=target, role=item["role"], status="preflight_complete", qualified=False)
                dev = io.read(run/"development/targets"/(item["id"]+f"__target{target}.json"))
                if dev["gates"][str(protocol["beta_star"])]["zero_certified"]:
                    point = problem.nominal(problem.zero)
                    np.savez_compressed(destination/"quote.npz", x=point["x"], mean_u=point["mean_u"])
                    row.update(status="zero_certified", qualified=True, objective=0., lower_bound=0.,
                               certified_original_minimum=True, bound_scope="original nonnegative fee domain")
                elif any(c["original_chance_infeasible"] for c in receipt["capacity"]):
                    row.update(status="original_chance_infeasible", probability_certificate=receipt["capacity"],
                               infeasible_scope="original nominal delivery capacity box, declared Gaussian law")
                elif any(c["sufficient_constraint_box_infeasible"] for c in receipt["capacity"]):
                    row.update(status="fixed_allocation_certificate_infeasible", probability_certificate=receipt["capacity"],
                               original_chance_feasibility="not determined by this certificate")
                elif not args.preflight_only:
                    row.update(solve(problem, gates, destination, max(.01, 1800-(time.perf_counter()-began))))
                row["elapsed_seconds"] = time.perf_counter()-began
            except Exception as exc:
                row = dict(id=item["id"], target=target, role=item["role"], status="failed", qualified=False,
                           error=repr(exc), traceback=traceback.format_exc())
            io.write(destination/"summary.json", row); results.append(row)
            print(json.dumps({k: row[k] for k in ("id", "target", "status", "qualified", "objective") if k in row}), flush=True)
    io.write(out/"results.json", results)
    io.write(out/("preflight_complete.json" if args.preflight_only else "complete.json"),
        dict(status="complete", created_utc=utc(), protocol_sha256=sha(out/"protocol.json"),
             results_sha256=sha(out/"results.json"), qualified=sum(r["qualified"] for r in results),
             preflight_only=args.preflight_only))


if __name__ == "__main__":
    main()
