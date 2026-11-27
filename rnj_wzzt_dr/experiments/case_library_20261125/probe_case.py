"""A simulated probing day on a development twin: Algorithm 1 on the 8-day history, then Algorithm 2
(identification_v8/probe.py) on day 9. Development cases only (load_dev_case checks the development manifest; the
sealed set is never opened).

    python -m rnj_wzzt_dr.experiments.case_library_20261125.probe_case --names dev_small_003_head --variant T1 [--quick]

The world: day 9's true quarter-hourly loads and unmetered load, the AC power flow (physics.power_flow), and the same
meters as the history (their fixed errors kept, new noise). Resources per meter follow scenario R1 of the earlier probing
study (hourly_probing_20261118/config.py):
  - a quarter of the households are enrolled;
  - a quarter of those have a 5 kW / 13.5 kWh battery (state of charge 0.2-0.8, efficiency 0.95, starting at 0.5);
  - a fifth have a 7 kW EV charger, plugged in from 18:00 to 08:00, half of which can pause.
Blocks are served by the batteries first.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
from dataclasses import replace

import numpy as np

from rnj_wzzt_dr.identification_v8.identify import Settings, identify
from rnj_wzzt_dr.identification_v8.labels import DECIDED, grouping
from rnj_wzzt_dr.identification_v8.probe import ProbeSettings, probe_day

from . import build, physics
from .dev_cases import f1, load_dev_case

ENROLLED, BATTERY, EV = .25, (.25, 5., 13.5), (.20, 7.)     # share enrolled; (share, kW, kWh); (share, kW)
SOC, ETA, EV_HOURS, EV_PAUSE = (.2, .8), .95, (18, 8), .5
QUICK = dict(short_seconds=5, long_seconds=5, start_seconds=2, polish_seconds=2, node_seconds=5, rounds=1)


class World:
    """Day 9 of a case, with blocks."""

    def __init__(self, rec, meter_seed=0):
        net, n = rec["network"], rec["n"]
        self.network, self.labels, self.base_kw = net, rec["labels"], net["base_kw"]
        day = slice((build.DAYS-1)*96, build.DAYS*96)
        self.p, self.q, self.v0 = (rec["true"][k][day] for k in ("P", "Q", "V0"))
        src = rec["source"]
        self.extra = None if src is None else (src["bus"], src["P"][day], src["Q"][day])
        self.meters = physics.Meters(n, [*rec["key"], int(meter_seed)])
        physics.read(rec["true"], self.meters, 1)               # the history's noise draws: the day's noise is new
        homes = {b["bus_id"]: b["customers"] for b in net["buses"]}
        enrolled = ENROLLED*np.array([homes[label] for label in self.labels], float)
        self.bat_kw, self.bat_kwh = enrolled*BATTERY[0]*BATTERY[1], enrolled*BATTERY[0]*BATTERY[2]
        self.ev_kw = enrolled*EV[0]*EV[1]
        self.soc = np.full(n, .5)

    def battery(self, soc):
        """Battery headroom (kW): up (charging) and down (discharging) for one hour."""
        up = np.minimum(self.bat_kw, np.maximum((SOC[1]-soc)*self.bat_kwh/ETA, 0.))
        down = np.minimum(self.bat_kw, np.maximum((soc-SOC[0])*self.bat_kwh*ETA, 0.))
        return up, down

    def after(self, soc, u):
        """State of charge after one hour of block u (pu); the batteries serve it first."""
        kw = u*self.base_kw
        up, down = self.battery(soc)
        b = np.where(kw >= 0, np.minimum(kw, up), -np.minimum(-kw, down))
        return np.clip(soc+b*np.where(b >= 0, ETA, 1/ETA)/np.maximum(self.bat_kwh, 1e-9), 0., 1.)

    def headroom(self, hour, pending=None):
        soc = self.soc if pending is None else self.after(self.soc, pending)
        up, down = self.battery(soc)
        ev = self.ev_kw*(hour >= EV_HOURS[0] or hour < EV_HOURS[1])
        return (up+ev)/self.base_kw, (down+EV_PAUSE*ev)/self.base_kw

    def run_hour(self, hour, u):
        s = slice(4*hour, 4*hour+4)
        extra = None if self.extra is None else (self.extra[0], self.extra[1][s], self.extra[2][s])
        p = self.p[s]+u[None, :]                                # the meters see the households' net power
        order, v, s_pcc, _ = physics.power_flow(self.network, self.labels, p, self.q[s], self.v0[s], extra)
        at = {b: k for k, b in enumerate(order)}
        volt = v[:, [at[label] for label in self.labels]]
        r = self.meters.read(p.mean(0), self.q[s].mean(0), volt.mean(0), self.v0[s].mean(), s_pcc.mean())
        self.soc = self.after(self.soc, u)
        return dict(P=r["P"], Q=r["Q"], Y=r["V0"]**2-r["V"]**2, V0sq=r["V0"]**2, Ppcc=r["Ppcc"], Qpcc=r["Qpcc"])


def run_case(name, variant, st, ps, say=print):
    rec, inp = load_dev_case(name)
    data = {k: inp.hist[k] for k in ("P", "Q", "Y", "V0sq", "Ppcc", "Qpcc")}
    first = identify(**data, st=st)
    day = probe_day(data, first, World(rec), st, replace(ps, variant=variant), say)
    n = rec["n"]
    truth = {g for g in (grouping(C, n) for C in rec["truth"]) if g}
    tree = {tuple(g) for g in day["tree"]}
    targets = []
    for M in day["targets"]:                                    # a meter grouping, or "place"
        label = day["gates"][-1]["targets"][str(M)][0]
        grouping_ = M != "place"
        targets.append(dict(target=str(M), label=label, inside=grouping_ and M in tree, true=grouping_ and M in truth,
                            wrong=grouping_ and label == DECIDED and (M in tree) != (M in truth)))
    return dict(name=name, n=n, load=rec["source"] and rec["source"]["where"], variant=variant,
                f1_before=f1({tuple(g) for g in first["tree"]}, truth)["f1"], f1_after=f1(tree, truth)["f1"],
                cost=day["cost"], blocks=sum(g["block"] is not None for g in day["gates"]), gates=len(day["gates"]),
                stop=day["stop"], targets=targets, seconds=[g["seconds"] for g in day["gates"]], day=day["gates"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--names", required=True, help="development cases, comma-separated")
    ap.add_argument("--variant", default="T0", help="T0, T1, T2, or several, comma-separated")
    ap.add_argument("--quick", action="store_true", help="short time limits (smoke test)")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--psi", type=float, default=ProbeSettings.psi, help="CNY per half-BIC unit (for the knee grid)")
    ap.add_argument("--out", default=None, help="jsonl file to append to")
    args = ap.parse_args()
    st = Settings(**QUICK) if args.quick else Settings()
    for name in args.names.split(","):
        for variant in args.variant.split(","):
            row = run_case(name, variant, st, ProbeSettings(hours=args.hours, psi=args.psi))
            print(f"{name} {variant}: F1 {row['f1_before']:.3f} -> {row['f1_after']:.3f}, {row['blocks']} blocks, "
                  f"{row['cost']:.1f} CNY, stop: {row['stop']}; targets {[(t['target'], t['label']) for t in row['targets']]}",
                  flush=True)
            if args.out:
                with open(args.out, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, default=str)+"\n")


if __name__ == "__main__":
    main()
