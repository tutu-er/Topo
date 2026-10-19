"""Observation-only passive proposals with a declared local repair bank."""
from __future__ import annotations
from itertools import combinations
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "research_experiments" / "mobile_ev_probing"
for path in (ROOT, ROOT / "rnj_wzzt_core", OLD):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from ac_general_v9 import RadialCandidate
from research_experiments.mobile_ev_probing.passive_bridge_v9 import (
    build_passive_candidates, normalize_family,
)


def propose(observations, *, run_wzzt=True, wzzt_time_limit=.3, max_candidates=8):
    """No truth input. Complete only for n=3 reduced root/star/cherry class."""
    p, q, v = observations.p, observations.q, observations.v
    root = np.broadcast_to(np.asarray(observations.root_v), (len(p),))
    raw = build_passive_candidates(p, q, root[:, None]**2-v**2,
        kappa=.5, max_candidates=max_candidates, run_wzzt=run_wzzt,
        wzzt_time_limit=wzzt_time_limit, neighbor_seed_count=3)
    n = p.shape[1]
    families, origins = [], {}
    def add(family, origin):
        family = normalize_family(family, n)
        if family not in origins:
            families.append(family)
            origins[family] = []
        origins[family].append(origin)
    for row in raw["candidates"]:
        add(row["supports"], "passive")
    if n == 3:
        add((), "declared_local_repair_star")
        for pair in combinations(range(n), 2):
            add((pair,), "declared_local_repair_all_cherries")
    else:
        # Never add evaluation truth; expand passive repairs within budget.
        add((), "local_repair_star")
        for pair in combinations(range(n), 2):
            if len(families) >= max_candidates:
                break
            add((pair,), "local_repair_pair")
        families = families[:max_candidates]
    candidates = [RadialCandidate.from_supports("star" if not f else
        "clade_"+"_".join("".join(map(str, s)) for s in f), n, f)
        for f in families]
    return candidates, {"passive": raw,
        "bank": [{"name": c.name, "supports": list(map(list, f)),
                  "origins": origins[f]} for c, f in zip(candidates, families)],
        "uses_truth": False, "n3_complete_reduced_class": n == 3,
        "coverage": "reduced star/cherry class" if n == 3 else
                    "finite replaceable passive/local repair bank"}
