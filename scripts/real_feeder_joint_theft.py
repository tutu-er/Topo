"""Independent 65037 experiment with known root voltage; never edits mainline.

Public topology/cables and a public load snapshot drive our own four-wire AC
simulation. Time series and theft are synthetic. The inverse model uses the
mean squared phase-neutral voltage and total three-phase P/Q, whose scalar
wzzT representation is a first-order approximation, not full three-phase SE.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter
import hashlib
import json

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from theft_wzzt.graph.rooted_neighbor_joining import rooted_neighbor_joining

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_external/lvnetworkdataset_20260920"
SCALE = 1e4


@dataclass
class Feeder:
    graph: nx.Graph
    order: list[int]
    parent: dict[int, int]
    meters: list[int]
    snapshot: np.ndarray
    z: dict[int, np.ndarray]
    metadata: dict


def read_feeder() -> Feeder:
    raw = json.loads((DATA / "networks/65037/net.json").read_text())["network"]
    cables = json.loads((DATA / "cables/cableparameters.json").read_text())
    loads = json.loads((DATA / "networks/65037/imbalance.json").read_text())
    graph = nx.Graph()
    for edge in raw["branch"].values():
        # Dataset cable values are ohm/km; branch lengths interpreted as metres.
        cable = cables[edge["type"]]
        impedance = (np.array(cable["real"]) + 1j * np.array(cable["image"])) * edge["length"] / 1000
        graph.add_edge(edge["nodefrom"], edge["nodeto"], z=impedance,
                       cable=edge["type"], length_m=edge["length"])
    assert nx.is_tree(graph) and sorted(graph) == list(range(53))
    assert [int(k) for k, v in raw["bus"].items() if v["slack"]] == [0]
    parent = dict(nx.bfs_predecessors(graph, 0))
    order = list(nx.bfs_tree(graph, 0))
    snapshot = np.zeros((53, 3), complex)
    meters, missing = [], []
    for k, bus in raw["bus"].items():
        if bus["name"] in loads:
            load = loads[bus["name"]]
            # The author's loads.py defines negative injection as import.
            snapshot[int(k)] = -(np.array(load["p"]) + 1j * np.array(load["q"])) / 1000
            meters.append(int(k))
        elif bus["load"]:
            missing.append(int(k))
    z = {child: graph.edges[par, child]["z"] for child, par in parent.items()}
    metadata = dict(buses=len(graph), physical_branches=len(parent), meters=meters,
                    missing_load_records_assumed_zero=missing, root=0,
                    lengths_assumption="metres; README only specifies impedance ohm/km",
                    snapshot_total_kw=float(snapshot.real.sum()),
                    commit=json.loads((DATA / "manifest.json").read_text())["commit"])
    return Feeder(graph, order, parent, meters, snapshot, z, metadata)


def ac_flow(feeder: Feeder, loads_kw: np.ndarray, root_v: np.ndarray):
    """Radial four-wire constant-power BFS; root-neutral grounded, no shunts.

    Includes neutral impedance and mutual terms. No downstream earth bonds are
    specified in the public data, so none are added. Currents are in amperes.
    """
    s = np.asarray(loads_kw, complex) * 1000
    t = len(s)
    root = np.column_stack([root_v * np.exp(1j * angle) for angle in (0, -2*np.pi/3, 2*np.pi/3)]
                           + [np.zeros(t)])
    v = np.broadcast_to(root[:, None, :], (t, len(feeder.graph), 4)).copy()
    for iteration in range(300):
        iph = np.conj(s / (v[:, :, :3] - v[:, :, 3:]))
        injection = np.concatenate([iph, -iph.sum(axis=2, keepdims=True)], axis=2)
        flow = injection.copy()
        for child in reversed(feeder.order[1:]):
            flow[:, feeder.parent[child]] += flow[:, child]
        updated = v.copy(); updated[:, 0] = root
        for child in feeder.order[1:]:
            updated[:, child] = updated[:, feeder.parent[child]] - flow[:, child] @ feeder.z[child].T
        error = float(np.max(abs(updated - v)))
        v = updated
        if error < 1e-10:
            break
    else:
        raise RuntimeError("AC power flow failed to converge")
    # Recompute currents at final V so power and KCL audits are independent of
    # the stopping iterate. Branch currents are stored at their child index.
    iph = np.conj(s / (v[:, :, :3] - v[:, :, 3:]))
    injection = np.concatenate([iph, -iph.sum(axis=2, keepdims=True)], axis=2)
    flow = injection.copy()
    for child in reversed(feeder.order[1:]):
        flow[:, feeder.parent[child]] += flow[:, child]
    head = np.sum(root * np.conj(flow[:, 0]), axis=1) / 1000
    loss = np.zeros(t, complex); ohm = 0.0
    for child in feeder.order[1:]:
        dz = flow[:, child] @ feeder.z[child].T
        loss += np.sum(dz * np.conj(flow[:, child]), axis=1) / 1000
        ohm = max(ohm, float(np.max(abs(v[:, feeder.parent[child]] - v[:, child] - dz))))
    balance = float(np.max(abs(head - s.sum(axis=(1, 2))/1000 - loss)))
    audit = dict(iterations=iteration+1, ohm_residual_v=ohm, power_balance_residual_kva=balance,
                 min_phase_neutral_v=float(np.min(abs(v[:, :, :3] - v[:, :, 3:]))),
                 max_phase_neutral_v=float(np.max(abs(v[:, :, :3] - v[:, :, 3:]))),
                 max_neutral_v=float(np.max(abs(v[:, :, 3]))), mean_loss_kw=float(np.mean(loss.real)))
    assert ohm < 1e-7 and balance < 1e-7
    return v, head, loss, audit


def sample(feeder, seed, count, balanced, condition="clean", source_bus=5,
           pq_noise=.005, v_noise=.0002, master_noise=.002):
    rng = np.random.default_rng(seed)
    base = feeder.snapshot.copy()
    if balanced:
        base[:] = base.sum(axis=1, keepdims=True) / 3
    # Synthetic independently excited snapshots, not measured time series.
    factor = rng.uniform(.55, 1.15, (count, len(feeder.meters), 1))
    loads = np.zeros((count, 53, 3), complex)
    loads[:, feeder.meters] = factor * base[feeder.meters]
    reactive = rng.uniform(-.08, .15, factor.shape) * abs(base[feeder.meters].real)
    loads[:, feeder.meters] += 1j * reactive
    root_v = 230 + .8 * np.sin(np.linspace(0, 4*np.pi, count)) + rng.normal(0, .1, count)
    amplitude = np.zeros(count)
    if condition == "persistent":
        amplitude[:] = 15
    elif condition == "intermittent":
        amplitude[rng.choice(count, count//2, replace=False)] = 30
    elif condition != "clean":
        raise ValueError(condition)
    actual = loads.copy()
    actual[:, source_bus] += amplitude[:, None] * (1+.4j) / 3
    volt, head, loss, audit = ac_flow(feeder, actual, root_v)
    p = loads[:, feeder.meters].real.sum(axis=2)
    q = loads[:, feeder.meters].imag.sum(axis=2)
    p *= 1 + rng.normal(0, pq_noise, p.shape)
    q *= 1 + rng.normal(0, pq_noise, q.shape)
    mags = abs(volt[:, feeder.meters, :3] - volt[:, feeder.meters, 3:])
    mags *= 1 + rng.normal(0, v_noise, mags.shape)
    # Keep absolute reference: no temporal centering and no free intercept.
    y = (root_v[:, None]**2 - np.mean(mags**2, axis=2)) * 3/2000
    p0 = head.real * (1+rng.normal(0, master_noise, count))
    return dict(p=p, q=q, y=y, root_v=root_v, p0=p0, amplitude=amplitude,
                loss=loss.real, audit=audit)


def truth_atoms(feeder):
    """Evaluation only. Contract repeated downstream-meter masks by summing Z1."""
    directed = nx.bfs_tree(feeder.graph, 0)
    grouped = {}; buses = {}
    for child in feeder.order[1:]:
        descendants = nx.descendants(directed, child) | {child}
        support = frozenset(i for i, bus in enumerate(feeder.meters) if bus in descendants)
        if not support:
            continue
        edge_z = feeder.z[child]
        z1 = edge_z[0, 0] - edge_z[0, 1]
        grouped[support] = grouped.get(support, 0j) + z1
        buses.setdefault(support, []).append(child)
    return grouped, buses


def supports_from_measurements(*datasets):
    """Finite pool from observed P,Q,V only, with multiple RNJ tolerances.

    Unconstrained OLS only seeds RNJ candidates; theft selection is binary MILP.
    No topology/impedance/source truth is accepted by this function.
    """
    n = datasets[0]["p"].shape[1]
    pool = {frozenset([i]) for i in range(n)}
    for data in datasets:
        b = np.linalg.lstsq(np.column_stack([data["p"], data["q"]]), data["y"], rcond=None)[0]
        r = (b[:n]+b[:n].T)/2; x = (b[n:]+b[n:].T)/2
        for alpha in (1., .75, 0.):
            score = alpha*r/max(np.linalg.norm(r), 1e-12) + (1-alpha)*x/max(np.linalg.norm(x), 1e-12)
            depths = np.maximum(np.diag(score), 0)
            scale = max(np.median(depths), 1e-6)
            for fraction in (0., .02, .08):
                tree = rooted_neighbor_joining(score, depths, list(range(n)), -1000, fraction*scale)
                graph = nx.Graph(); graph.add_weighted_edges_from(tree.edges)
                directed = nx.bfs_tree(graph, tree.root)
                for node in directed:
                    if node == tree.root:
                        continue
                    downstream = (nx.descendants(directed, node) | {node}) & set(range(n))
                    if downstream:
                        pool.add(frozenset(downstream))
    return tuple(sorted(pool, key=lambda s: (len(s), sorted(s))))


def predict(p, q, supports, r, x, amplitude=None, source=None):
    z = np.array([[j in support for j in range(p.shape[1])] for support in supports], float)
    fp, fq = p @ z.T, q @ z.T
    if source is not None:
        b = np.array([supports[source] <= s for s in supports])
        fp += amplitude[:, None]*b
        fq += .4*amplitude[:, None]*b
    return (fp*np.asarray(r) + fq*np.asarray(x)) @ z


def fit(data, supports, amplitude=None, *, fixed_supports=None, seconds=60., penalty=.001):
    """Joint finite-pool laminar topology, R/X, stationary single-source MILP.

    If fixed_supports is supplied, only source and R/X are re-estimated. All
    models use identical broad weight bounds, objective, and raw observations.
    No free intercept, no truth-derived bounds, no source truth supplied.
    """
    p, q, y = (data[k] for k in ("p", "q", "y"))
    t, n = p.shape; e = len(supports)
    z = np.array([[j in support for j in range(n)] for support in supports], float)
    a = np.array([[h <= support for h in supports] for support in supports], float)
    r = np.arange(e); x = r+e; u = x+e; s = u+e; vr = s+e; vx = vr+e
    residual = np.arange(6*e, 6*e+t*n); size = int(residual[-1]+1)
    lb = np.zeros(size); ub = np.full(size, np.inf)
    ub[np.r_[r, x, vr, vx]] = 1.
    ub[np.r_[u, s]] = 1.
    integrality = np.zeros(size, int); integrality[np.r_[u, s]] = 1
    objective = np.zeros(size); objective[residual] = SCALE/y.size
    for k, support in enumerate(supports):
        if len(support) == 1:
            lb[u[k]] = 1
        else:
            objective[u[k]] = SCALE*penalty
        if fixed_supports is not None:
            lb[u[k]] = ub[u[k]] = int(k in fixed_supports)
    joint = amplitude is not None
    if amplitude is None:
        amplitude = np.zeros(t); ub[s] = 0
    ri, ci, values, lows, highs = [], [], [], [], []
    def add(coeff, lo=-np.inf, hi=np.inf):
        row = len(lows)
        for k, value in coeff.items():
            if value:
                ri.append(row); ci.append(int(k)); values.append(float(value))
        lows.append(lo); highs.append(hi)
    # Laminar subsets are a valid rooted reduced forest attached to reference.
    for i, j in combinations(range(e), 2):
        if supports[i] & supports[j] and not (supports[i] <= supports[j] or supports[j] <= supports[i]):
            add({u[i]: 1, u[j]: 1}, hi=1)
    add({k: 1 for k in s}, int(joint), int(joint))
    for k in range(e):
        add({s[k]: 1, u[k]: -1}, hi=0)
        for weight in (r[k], x[k]):
            add({weight: 1, u[k]: -1}, hi=0)
        b = {s[h]: a[k, h] for h in range(e) if a[k, h]}
        for weight, product in ((r[k], vr[k]), (x[k], vx[k])):
            add({product: 1, **{j: -v for j, v in b.items()}}, hi=0)
            add({product: 1, weight: -1}, hi=0)
            add({product: 1, weight: -1, **{j: -v for j, v in b.items()}}, lo=-1)
    fp, fq = p @ z.T, q @ z.T
    for k in range(t):
        for j in range(n):
            coef = {}
            for edge in np.flatnonzero(z[:, j]):
                coef[r[edge]] = fp[k, edge]; coef[x[edge]] = fq[k, edge]
                coef[vr[edge]] = amplitude[k]; coef[vx[edge]] = .4*amplitude[k]
            slack = residual[k*n+j]
            add({**coef, slack: -1}, hi=y[k, j])
            add({**coef, slack: 1}, lo=y[k, j])
    matrix = coo_matrix((values, (ri, ci)), shape=(len(lows), size)).tocsc()
    started = perf_counter()
    result = milp(objective, integrality=integrality, bounds=Bounds(lb, ub),
                  constraints=LinearConstraint(matrix, lows, highs),
                  options=dict(time_limit=seconds, mip_rel_gap=1e-6))
    output = dict(status="optimal" if result.status == 0 else "incomplete", solver_status=int(result.status),
                  message=result.message, seconds=perf_counter()-started,
                  mip_gap=float(result.mip_gap) if getattr(result, "mip_gap", None) is not None else None,
                  candidate_count=e, topology_penalty=penalty)
    if result.x is None:
        return output
    solution = result.x
    chosen = np.flatnonzero(solution[s] > .5)
    source = int(chosen[0]) if len(chosen) else None
    active = list(map(int, np.flatnonzero(solution[u] > .5)))
    fitted = predict(p, q, supports, solution[r], solution[x], amplitude, source)
    direct = float(np.mean(abs(fitted-y)))
    penalized = direct + penalty*sum(len(supports[k]) > 1 for k in active)
    activity = matrix @ solution
    feasible = max(float(np.max(np.maximum(np.asarray(lows)-activity, 0))),
                   float(np.max(np.maximum(activity-np.asarray(highs), 0))),
                   float(np.max(np.maximum(lb-solution, 0))),
                   float(np.max(np.maximum(solution-ub, 0))),
                   float(np.max(abs(solution[np.r_[u, s]]-np.rint(solution[np.r_[u, s]])))))
    b = a @ np.rint(solution[s])
    product = float(max(np.max(abs(solution[vr]-solution[r]*b)), np.max(abs(solution[vx]-solution[x]*b))))
    objective_error = abs(penalized-float(result.fun)/SCALE)
    assert feasible < 3e-6 and product < 3e-6 and objective_error < 3e-5
    output.update(source=source, source_support=None if source is None else sorted(supports[source]),
                  active=active, r=solution[r].tolist(), x=solution[x].tolist(), train_mae=direct,
                  audit=dict(feasibility=feasible, product_error=product, objective_error=objective_error))
    return output


def estimate_loss(data, supports, fitted):
    z = np.array([[j in support for j in range(data["p"].shape[1])] for support in supports], float)
    flow2 = (data["p"] @ z.T)**2 + (data["q"] @ z.T)**2
    return (flow2 @ np.asarray(fitted["r"])) * 1000/(3*data["root_v"]**2)


def evaluate(feeder, data, supports, fitted, holdout, source_bus):
    if "r" not in fitted:
        return
    truth, _ = truth_atoms(feeder)
    n = len(feeder.meters)
    z = np.array([[j in s for j in range(n)] for s in supports], float)
    matrices = []
    for component in ("real", "imag"):
        mat = np.zeros((n, n))
        for support, value in truth.items():
            row = np.array([j in support for j in range(n)], float)
            mat += getattr(value, component)*np.outer(row, row)
        matrices.append(mat)
    for key, mat in zip(("r", "x"), matrices):
        fitted[key+"_relative_error"] = float(np.linalg.norm(z.T@(np.asarray(fitted[key])[:, None]*z)-mat)/np.linalg.norm(mat))
    clean_prediction = predict(holdout["p"], holdout["q"], supports, fitted["r"], fitted["x"])
    fitted["heldout_clean_mae_v_approx"] = float(np.mean(abs(clean_prediction-holdout["y"]))*2000/(6*230))
    true_clades = {s for s in truth if len(s)>1}
    selected = {supports[k] for k in fitted["active"] if len(supports[k])>1}
    fitted["clade_precision"] = len(true_clades & selected)/max(len(selected),1)
    fitted["clade_recall"] = len(true_clades & selected)/max(len(true_clades),1)
    directed = nx.bfs_tree(feeder.graph, 0)
    descendants = nx.descendants(directed, source_bus) | {source_bus}
    source_support = frozenset(i for i, bus in enumerate(feeder.meters) if bus in descendants)
    fitted["source_region_correct"] = (fitted["source_support"] is not None
                                          and frozenset(fitted["source_support"]) == source_support)
    fitted["source_truth_in_pool"] = source_support in supports
    fitted["true_clade_pool_recall"] = len(true_clades & set(supports))/max(len(true_clades),1)


def main(output, seeds=2, count=64, seconds=60):
    output = output.resolve()
    assert output.is_relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    feeder = read_feeder()
    protocol = dict(dataset=feeder.metadata, seeds=seeds, train_samples=count, history_samples=192,
                    holdout_samples=96, source_bus=5, conditions=["clean", "persistent", "intermittent"],
                    voltage_reference="known 230 V with observed variation; no centering or intercept",
                    physics=["balanced four-wire AC", "original unbalanced snapshot scaled; four-wire AC"],
                    amplitude="positive feeder-minus-meters balance minus history-wzzT estimated technical loss",
                    known_theft_pf_q_over_p=.4, topology_penalty=.001,
                    noise=dict(pq_relative=.005, phase_voltage_relative=.0002, head_p_relative=.002),
                    limitations="Synthetic excitation and theft; single stationary source conditional on presence. Not calibrated detection or a full unbalanced inverse model.",
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, "Protocol mismatch; use a new output directory"
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    for seed in range(seeds):
        for balanced in (True, False):
            prefix = f"seed{seed}_{'balanced' if balanced else 'unbalanced'}"
            history = sample(feeder, 74000+seed, 192, balanced)
            holdout = sample(feeder, 78000+seed, 96, balanced)
            pool_history = supports_from_measurements(history)
            path = output / (prefix+"_history.json")
            if path.exists():
                historical = json.loads(path.read_text())
            else:
                historical = fit(history, pool_history, seconds=seconds)
                path.write_text(json.dumps(historical, indent=2), encoding="utf-8")
            if "r" not in historical:
                print(prefix+" history failed", flush=True); continue
            for condition in protocol["conditions"]:
                path = output / (prefix+"_"+condition+".json")
                if path.exists():
                    continue
                data = sample(feeder, 76000+seed, count, balanced, condition)
                supports = supports_from_measurements(history, data)
                loss_hat = estimate_loss(data, pool_history, historical)
                amplitude = np.maximum(data["p0"]-data["p"].sum(axis=1)-loss_hat, 0)
                results = {}
                results["topology_only"] = fit(data, supports, seconds=seconds)
                fixed = results["topology_only"].get("active")
                if fixed is not None:
                    results["sequential"] = fit(data, supports, amplitude, fixed_supports=fixed, seconds=seconds)
                results["joint"] = fit(data, supports, amplitude, seconds=seconds)
                for fitted in results.values():
                    evaluate(feeder, data, supports, fitted, holdout, protocol["source_bus"])
                row = dict(seed=seed, balanced=balanced, condition=condition,
                           supports=[sorted(s) for s in supports], ac_audit=data["audit"],
                           amplitude_mae_kw=float(np.mean(abs(amplitude-data["amplitude"]))),
                           amplitude_mean_kw=float(np.mean(amplitude)),
                           true_amplitude_mean_kw=float(np.mean(data["amplitude"])),
                           loss_estimate_mae_kw=float(np.mean(abs(loss_hat-data["loss"]))),
                           fits=results)
                path.write_text(json.dumps(row, indent=2), encoding="utf-8")
                print(json.dumps(dict(case=prefix+"_"+condition, fits={k:dict(status=v['status'], seconds=round(v['seconds'],2), source=v.get('source_support'), r_error=v.get('r_relative_error'), x_error=v.get('x_relative_error')) for k,v in results.items()})), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    main(args.output, args.seeds, args.samples, args.seconds)
