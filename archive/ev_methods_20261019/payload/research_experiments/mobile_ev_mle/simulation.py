"""Synthetic steady-state meters; evaluation truth never passed to selectors."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .candidates import RadialCandidate
from .likelihood import Observations, MeasurementModel, ac_predict

BASE_KW = 20.
BASE_KV = .4
ZBASE_OHM = BASE_KV**2/(BASE_KW/1000.)


@dataclass
class SyntheticPath:
    candidate: object
    r: np.ndarray
    x: np.ndarray
    p: np.ndarray
    q: np.ndarray
    source_p: np.ndarray
    source_q: np.ndarray
    noise: dict
    bias: dict
    history_count: int
    seed: int
    case: str


def model(n=3, misspecified_bias=False):
    bias_p = np.linspace(-.00015, .00015, n)
    bias_q = np.linspace(.0001, -.0001, n)
    bias_v = np.linspace(-.00004, .00004, n)
    return MeasurementModel(p_sigma=.0006, q_sigma=.0004, v_sigma=.00012,
        master_p_sigma=.001, master_q_sigma=.001,
        bias_p=np.zeros(n) if misspecified_bias else bias_p,
        bias_q=np.zeros(n) if misspecified_bias else bias_q,
        bias_v=np.zeros(n) if misspecified_bias else bias_v)


def generate(seed, case="strong", n=3, history_count=12, future_count=8,
             truth_family=None):
    rng = np.random.default_rng(seed)
    pair = tuple(rng.choice(n, size=2, replace=False))
    pair = tuple(sorted(pair))
    family = (pair,)
    if truth_family is not None:
        family = truth_family
    if n > 3 and truth_family is None:
        family = ((0, 1), (0, 1, 2))
    candidate = RadialCandidate.from_supports("evaluation_truth", n, family)
    r = np.array([.005 if len(s) == n else .028+.003*s[0] if len(s) == 1
                  else .00003 if case == "weak" else .012
                  for s in candidate.supports])
    x = np.array([.0035 if len(s) == n else .011+.002*s[0] if len(s) == 1
                  else .00002 if case == "weak" else .007
                  for s in candidate.supports])
    t = history_count+future_count
    common = rng.uniform(-.012, .012, (t, 1))
    p = .12+common+rng.uniform(-.004, .004, (t, n))
    q = .04+.25*common+rng.uniform(-.004, .004, (t, n))
    # Independent small reactive variation is deliberately present: no known X/R.
    if n > 3:
        p += rng.uniform(-.015, .015, (t, n))
        q += rng.uniform(-.008, .008, (t, n))
    sp = .10+rng.uniform(-.035, .035, t)
    sq = .035+rng.uniform(-.015, .015, t)
    bias = {"p": np.linspace(-.00015, .00015, n),
            "q": np.linspace(.0001, -.0001, n),
            "v": np.linspace(-.00004, .00004, n)}
    if case == "bias_misspec":
        bias["v"] = np.linspace(-.00065, .00065, n)
    noise = {"p": rng.normal(0, .0006, (t, n)),
             "q": rng.normal(0, .0004, (t, n)),
             "v": rng.normal(0, .00012, (t, n)),
             "master_p": rng.normal(0, .001, t),
             "master_q": rng.normal(0, .001, t)}
    return SyntheticPath(candidate, r, x, p, q, sp, sq, noise, bias,
                         history_count, seed, case)


def observe(path, rows, action=None, *, noise=True):
    p = path.p[rows].copy()
    q = path.q[rows].copy()
    if action is not None:
        p += action.p
        q += action.q
    pred = ac_predict(path.candidate, path.r, path.x, p, q,
                      path.source_p[rows], path.source_q[rows], root_v=1.)
    obs = Observations(
        p=p+path.bias["p"]+(path.noise["p"][rows] if noise else 0),
        q=q+path.bias["q"]+(path.noise["q"][rows] if noise else 0),
        v=pred["v"]+path.bias["v"]+(path.noise["v"][rows] if noise else 0),
        root_v=np.ones(len(p)),
        master_p=pred["master_p"]+(path.noise["master_p"][rows] if noise else 0),
        master_q=pred["master_q"]+(path.noise["master_q"][rows] if noise else 0))
    return obs, pred


def truth_metadata(path):
    return {"seed": path.seed, "case": path.case,
        "supports": [list(s) for s in path.candidate.supports if 1 < len(s) < path.candidate.n],
        "r_pu": path.r, "x_pu": path.x, "r_ohm": path.r*ZBASE_OHM,
        "x_ohm": path.x*ZBASE_OHM, "p": path.p, "q": path.q,
        "source_p": path.source_p, "source_q": path.source_q,
        "bias": path.bias, "evaluation_only": True}

