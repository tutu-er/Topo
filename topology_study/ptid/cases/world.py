"""A simulated probing day on a development case: Algorithm 2's world (ptid/method/probe.py).

Day 9's true quarter-hourly loads and unmetered load, the AC power flow (physics.power_flow), and the same meters as the
history (their fixed errors kept, new noise). Resources per meter follow scenario R1 of the earlier probing study:
  - a quarter of the households are enrolled;
  - a quarter of those have a 5 kW / 13.5 kWh battery (state of charge 0.2-0.8, efficiency 0.95, starting at 0.5);
  - a fifth have a 7 kW EV charger, plugged in from 18:00 to 08:00, half of which can pause.
Blocks are served by the batteries first. (A copy of probe_case.World of the case library, unchanged.)
"""
import numpy as np

from . import physics
from .library import DAYS

ENROLLED, BATTERY, EV = .25, (.25, 5., 13.5), (.20, 7.)     # share enrolled; (share, kW, kWh); (share, kW)
SOC, ETA, EV_HOURS, EV_PAUSE = (.2, .8), .95, (18, 8), .5


class World:
    """Day 9 of a case, with blocks."""

    def __init__(self, rec, meter_seed=0):
        net, n = rec["network"], rec["n"]
        self.network, self.labels, self.base_kw = net, rec["labels"], net["base_kw"]
        day = slice((DAYS-1)*96, DAYS*96)
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
