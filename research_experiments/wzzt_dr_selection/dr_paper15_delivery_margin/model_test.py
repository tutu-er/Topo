"""Frozen paper15 contract: two one-dimensional delivery deviations."""
import argparse
import hashlib
import json
import os
import pickle
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import brentq

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK.parents[1]))
from research_experiments.wzzt_dr_selection.dr_heterogeneous_pilot.devices import Devices
from research_experiments.wzzt_dr_selection.dr_heterogeneous_resources.build_allocations import profiles
from research_experiments.wzzt_dr_selection.dr_joint_loss_methods.kernel import ray_probability

CASE = "paper15_b2_v0005"
TARGET = .90 / .99


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def locked(link):
    data = Path(link["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != link["sha256"]:
        raise ValueError("Frozen input hash mismatch")
    return data


def delayed(devices, contract, minutes):
    """Hold within a slot, delay, average, then execute causal device control."""
    lag = minutes / (60 * devices.dt)
    whole = int(np.floor(lag))
    fraction = lag - whole
    request = np.zeros_like(contract)
    for mask, start, end in [(devices.kind != 2, 0, 96),
                             (devices.kind == 2, 0, 32), (devices.kind == 2, 72, 96)]:
        for offset, weight in [(whole, 1 - fraction), (whole + 1, fraction)]:
            if weight and start + offset < end:
                request[mask, start+offset:end] += weight * contract[mask, start:end-offset]
    return devices.project(request)


class Background:
    def __init__(self, packet):
        self.packet = packet
        joint = np.vstack([packet["L_C"], packet["L_F"]])
        left, singular, _ = np.linalg.svd(joint, full_matrices=False)
        if singular[2] >= 1e-12 * singular[0]:
            raise ValueError("Background has more than two active directions")
        self.relative_third = float(singular[2] / singular[0])
        self.factor = left[:, :2] * singular[:2]
        size = len(packet["mu_C"])
        self.weights = np.r_[np.ones(size), -1.03*np.ones(size)] / size

    def intercept(self, delivered):
        return np.concatenate([self.packet["mu_"+k] +
                               (delivered.T @ self.packet["effect_"+k].T).ravel()
                               for k in ("C", "F")])

    def probability(self, mu, nodes, offset):
        # Each radial interval is integrated by the existing signed-L1 kernel.
        # Tiles avoid allocating a nodes by (2*96*15) matrix at high resolutions.
        theta = 2*np.pi*(np.arange(nodes)+offset)/nodes
        direction = np.column_stack((np.cos(theta), np.sin(theta)))
        total = 0.
        for first in range(0, nodes, 32):
            slopes = direction[first:first+32] @ self.factor.T
            total += sum(ray_probability(mu, b, self.weights, self.packet["threshold"], 2)
                         for b in slopes)
        return total/nodes


def solve_axis(name, trajectory, step, limit, reference, background, devices):
    rows = {}
    def evaluate(amount, nodes, offset):
        key = (round(float(amount), 12), nodes, offset)
        if key not in rows:
            delivered = trajectory(amount)
            physical = devices.audit(delivered)
            if not physical["passed"]:
                raise ValueError("Infeasible delivery")
            probability = background.probability(background.intercept(delivered), nodes, offset)
            rows[key] = dict(amount=float(amount), angles=nodes, offset=offset, p=probability,
                             energy_kwh=physical["absolute_DR_kwh"],
                             shape_difference_kw=float(np.abs(delivered-reference).sum()/96))
        return rows[key]["p"]

    # Search in order; bisect only inside the first encountered failing cell.
    first = None
    for j in range(int(round(limit/step))+1):
        if evaluate(j*step, 256, .5) < TARGET:
            first = j*step
            break
    if first is None or first == 0:
        raise ValueError("No interior crossing in the prespecified range")
    fine_step = step/4
    for j in range(int(round((first+4*step)/fine_step))+1):
        x = j*fine_step
        if evaluate(x, 512, .25) < TARGET:
            break
    else:
        raise ValueError("No crossing in refined scan")
    roots = []
    for count, offset in [(1024, .5), (2048, .25), (4096, .5)]:
        lower, upper = max(0, x-2*step), min(limit, x+2*step)
        f = lambda v: evaluate(v, count, offset)-TARGET
        root = brentq(f, lower, upper, xtol=fine_step/100)
        roots.append(dict(angles=count, offset=offset, crossing=float(root)))
    # A visibly separated pair of limits with two independent angular grids.
    left = np.floor(min(v["crossing"] for v in roots)/fine_step)*fine_step
    right = np.ceil(max(v["crossing"] for v in roots)/fine_step)*fine_step
    for _ in range(6):
        bracket = [[evaluate(left, n, o), evaluate(right, n, o)]
                   for n,o in [(4096,.5), (8192,.25)]]
        if min(p[0] for p in bracket) >= TARGET and max(p[1] for p in bracket) < TARGET:
            break
        left = max(0, left-fine_step)
        right = min(limit, right+fine_step)
    else:
        raise ArithmeticError("The two numerical grids do not bracket the crossing")
    mid = roots[-1]["crossing"]
    print(name, mid, left, right, bracket, flush=True)
    return dict(scan_step=step, finer_prefix_step=fine_step, roots=roots, interval=[left,right],
                probabilities_at_interval=bracket, at_crossing=devices.audit(trajectory(mid)),
                evaluations=list(rows.values()))


def main(archive, output):
    folder = archive / CASE
    source = read(archive/"protocol.json")
    packet_link = read(folder/"inputs.json")["problem.pkl"]
    price_link = read(archive/"price_freeze.json")["cases"][CASE]["plans"]
    packet = pickle.loads(locked(packet_link))
    frozen = pickle.loads(locked(price_link))["candidate"]
    allocation = next(x for x in json.loads(locked(source["allocations"]))["cases"] if x["case"] == "paper15")
    dev = Devices(allocation, json.loads(locked(source["library"])), profiles)
    contract = frozen["mean_u"]
    assert list(dev.labels) == list(packet["labels"])
    assert np.max(np.abs(delayed(dev, contract, 0)-contract)) < 1e-10
    nominal = dev.audit(contract)
    assert nominal["passed"]
    bg = Background(packet)
    config = dict(plan_hash=price_link["sha256"], problem_hash=packet_link["sha256"],
                  hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                      Path(__file__), TASK/"dr_joint_loss_methods/kernel.py",
                      TASK/"dr_heterogeneous_pilot/devices.py", TASK/"dr_heterogeneous_resources/build_allocations.py"]},
                  resource_hashes={k:source[k]["sha256"] for k in ["library","allocations"]},
                  python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  threads=os.environ.get("OPENBLAS_NUM_THREADS"), target=TARGET, rng_used=False,
                  fee=frozen["cost"], nominal=nominal, threshold=packet["threshold"],
                  relative_third_singular=bg.relative_third)
    config["nominal_p"] = [bg.probability(bg.intercept(contract), n, o) for n,o in [(2048,.25),(8192,.5)]]
    print("contract", config["nominal_p"], nominal, flush=True)
    config["shortfall"] = solve_axis("shortfall",lambda s:(1-s)*contract,.01,1.,contract,bg,dev)
    config["delay"] = solve_axis("delay",lambda t:delayed(dev,contract,t),1.,1440.,contract,bg,dev)
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    main(args.archive,args.output)
