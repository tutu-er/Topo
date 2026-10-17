"""History-only WZZT/RNJ proposals for the full-pipeline EV experiment.

This adapter calls the existing constrained sensitivity/RNJ and greedy WZZT
implementations.  Their scores and bootstrap frequencies are proposal tools,
not additional likelihood terms or structural correctness probabilities.
Every proposed clade can subsequently be deleted or replaced.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "rnj_wzzt_core") not in sys.path:
    sys.path.insert(0, str(ROOT / "rnj_wzzt_core"))

from research_experiments.mobile_ev_mle.likelihood import RadialCandidate
from research_experiments.mobile_ev_probing import passive_bridge_v9 as bridge
from rnj_wzzt.graph.bootstrap import _moving_block_bootstrap_copy


DEFAULTS = dict(max_candidates=8, bootstrap_replicates=3, block_length=4,
    kappa=.5, tolerance_factors=(0., .08, .16), coefficient_bound=.2,
    wzzt_time_limit=.5, wzzt_max_atoms=None, run_wzzt=True,
    repair_seed_count=2, max_repairs=8)


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return int(value)


def _config(config):
    supplied = {} if config is None else dict(config)
    if set(supplied) - set(DEFAULTS):
        raise ValueError("unknown passive options: " + repr(sorted(set(supplied)-set(DEFAULTS))))
    cfg = dict(DEFAULTS, **supplied)
    for key, minimum in (("max_candidates", 1), ("bootstrap_replicates", 0),
            ("block_length", 1), ("repair_seed_count", 0), ("max_repairs", 0)):
        cfg[key] = _integer(cfg[key], key, minimum)
    for key in ("kappa", "coefficient_bound", "wzzt_time_limit"):
        cfg[key] = float(cfg[key])
        if not np.isfinite(cfg[key]) or cfg[key] < 0 or (key != "kappa" and cfg[key] == 0):
            raise ValueError(key + " must have a finite admissible value")
    cfg["tolerance_factors"] = tuple(float(x) for x in cfg["tolerance_factors"])
    if not cfg["tolerance_factors"] or any(not np.isfinite(x) or x < 0 for x in cfg["tolerance_factors"]):
        raise ValueError("nonempty nonnegative tolerance_factors required")
    if not isinstance(cfg["run_wzzt"], (bool, np.bool_)):
        raise ValueError("run_wzzt must be boolean")
    cfg["run_wzzt"] = bool(cfg["run_wzzt"])
    if cfg["wzzt_max_atoms"] is not None:
        cfg["wzzt_max_atoms"] = _integer(cfg["wzzt_max_atoms"], "wzzt_max_atoms", 1)
    return cfg


def _calibrated(observations, meter):
    p = np.asarray(observations.p, float)
    if p.ndim != 2 or min(p.shape) < 2 or not np.isfinite(p).all():
        raise ValueError("at least two arrived rows and terminals required")
    if getattr(meter, "fit_bias", False):
        raise ValueError("passive input requires fixed independently calibrated bias")
    rows = {}
    for key in ("p", "q", "v"):
        value = np.asarray(getattr(observations, key), float)
        if value.shape != p.shape or not np.isfinite(value).all():
            raise ValueError("arrived P/Q/V arrays must share a finite (time,terminal) shape")
        bias = np.broadcast_to(np.asarray(getattr(meter, "bias_"+key, 0.), float), p.shape)
        if not np.isfinite(bias).all():
            raise ValueError("finite calibrated bias required")
        rows[key] = value-bias
    root = np.broadcast_to(np.asarray(observations.root_v, float), (len(p),))
    if np.any(root <= 0) or np.any(rows["v"] <= 0):
        raise ValueError("positive calibrated voltage magnitudes required")
    return rows["p"], rows["q"], root[:, None]**2-rows["v"]**2


def _family(raw, n):
    return bridge.normalize_family(raw, n)


def _repair_families(seed_families, observed_families, n, limit):
    """Bounded local edits; do not enumerate the full radial topology class.

    New clades come from actual RNJ/WZZT/bootstrap disagreement.  A final
    sibling merge may use two existing child clades.  Even unanimously
    proposed clades may be removed, so there are no locked 'trusted blocks'.
    """
    observed = list(dict.fromkeys(_family(f, n) for f in observed_families))
    counts = Counter(s for f in observed for s in f)
    support_pool = sorted(counts, key=lambda s: (counts[s], len(s), s))
    output = []
    seen = set(observed)

    def add(family, operation):
        if len(output) >= limit:
            return
        try:
            f = _family(family, n)
        except ValueError:
            return
        if f not in seen:
            seen.add(f); output.append((f, operation))

    # Round-robin edit classes ensure a short budget is not consumed by
    # every deletion before any historical alternative can be considered.
    for seed in seed_families:
        for old in sorted(seed, key=lambda s: (counts[s], len(s), s))[:1]:
            add([s for s in seed if s != old], "delete_clade")
    for seed in seed_families:
        for new in support_pool:
            if new in seed:
                continue
            crossing = [s for s in seed if set(s)&set(new) and not
                (set(s)<=set(new) or set(new)<=set(s))]
            if len(crossing) == 1:
                add([s for s in seed if s != crossing[0]]+[new], "replace_disputed_clade")
            elif not crossing:
                add([*seed, new], "merge_observed_clade")
            if len(output) >= limit:
                return output
    for seed in seed_families:
        # At most two data-selected child unions per seed.  Existing clades
        # define their parents; no all-route/all-topology search is performed.
        full = frozenset(range(n))
        nodes = {full, *(frozenset(s) for s in seed), *(frozenset((i,)) for i in range(n))}
        parent = {s: min((a for a in nodes if s<a), key=len) for s in nodes if s != full}
        made = 0
        for left, right in combinations(sorted(nodes, key=lambda s:(len(s),tuple(sorted(s)))), 2):
            if left&right or parent.get(left) != parent.get(right) or len(left|right) == n:
                continue
            new = tuple(sorted(left|right))
            # Only expand a child union already seen in historical proposals.
            if new in counts and new not in seed:
                add([*seed, new], "merge_observed_siblings"); made += 1
            if made == 2 or len(output) >= limit:
                break
    return output


def propose(observations, meter, config=None, seed=0):
    """Return ``(RadialCandidate list, JSON-safe provenance record)``.

    Only arrived observations, externally calibrated meter settings, public
    proposal options and a resampling seed are accepted.  No prior candidate
    is certified.  WZZT's optional source-amplitude input is deliberately empty:
    it does not estimate the unmetered source in this proposal stage.
    """
    began = perf_counter(); cfg = _config(config)
    seed = _integer(seed, "seed")
    p, q, y = _calibrated(observations, meter); count, n = p.shape
    rng = np.random.default_rng(seed)
    pool = {}; failures = []; boot_records = []; native = {}

    def add(family, origin):
        f = _family(family, n)
        if f not in pool:
            pool[f] = dict(origins=[], **bridge.contrast_rank(p+cfg["kappa"]*q,
                y, f, cfg["coefficient_bound"]))
        if origin not in pool[f]["origins"]:
            pool[f]["origins"].append(origin)
        return f

    def run(pp, qq, yy, use_wzzt):
        return bridge.build_passive_candidates(pp, qq, yy,
            source_p_nominal=None, source_q_nominal=None,
            kappa=cfg["kappa"], tolerance_factors=cfg["tolerance_factors"],
            max_candidates=max(8, 4*n+2*len(cfg["tolerance_factors"])),
            wzzt_time_limit=cfg["wzzt_time_limit"], wzzt_max_atoms=cfg["wzzt_max_atoms"],
            coefficient_bound=cfg["coefficient_bound"], neighbor_seed_count=0,
            run_wzzt=bool(use_wzzt))

    raw = run(p, q, y, cfg["run_wzzt"])
    native = dict(rnj=[], wzzt=[], diagnostics=raw["diagnostics"], seconds=raw["seconds"])
    for row in raw["candidates"]:
        f = add(row["supports"], "native")
        for origin in row["proposal_origins"]:
            add(f, origin)
            kind = "rnj" if origin.startswith("rnj:") else "wzzt" if origin.startswith("wzzt:") else None
            if kind:
                native[kind].append(dict(supports=[list(s) for s in f], origin=origin))
    for key in ("rnj",):
        failures.extend(dict(stage="native_"+key, details=d) for d in raw["diagnostics"][key]
            if not d.get("success", False))
    wzzt = raw["diagnostics"]["wzzt"]
    if wzzt.get("error"):
        failures.append(dict(stage="native_wzzt", details=wzzt["error"]))
    elif cfg["run_wzzt"] and any(text in str(wzzt.get("stop_reason", "")) for text in
            ("not_proven", "failed")):
        failures.append(dict(stage="native_wzzt", details=wzzt["stop_reason"]))

    scenario = dict(name="arrived_history", P_terminal=pd.DataFrame(p),
        Q_terminal=pd.DataFrame(q), drop_target=pd.DataFrame(y))
    for index in range(cfg["bootstrap_replicates"]):
        started = perf_counter()
        try:
            sample = _moving_block_bootstrap_copy([scenario], rng, cfg["block_length"])[0]
            result = run(*(sample[k].to_numpy() for k in ("P_terminal", "Q_terminal", "drop_target")), False)
            families = []
            for row in result["candidates"]:
                if not any(o.startswith("rnj:") for o in row["proposal_origins"]):
                    continue
                f = add(row["supports"], "bootstrap_rnj:"+str(index))
                families.append([list(s) for s in f])
            boot_records.append(dict(index=index, families=families,
                diagnostics=result["diagnostics"]["rnj"], seconds=perf_counter()-started))
            failures.extend(dict(stage="bootstrap_rnj_mode", index=index, details=d)
                for d in result["diagnostics"]["rnj"] if not d.get("success", False))
            if not families:
                failures.append(dict(stage="bootstrap_rnj", index=index, details="no RNJ proposal"))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            details=type(exc).__name__+": "+str(exc)
            failures.append(dict(stage="bootstrap_rnj", index=index, details=details))
            boot_records.append(dict(index=index, families=[], failure=details, seconds=perf_counter()-started))

    rank = lambda f: (pool[f]["linear_contrast_mae"], len(f), f)
    raw_families = sorted(pool, key=rank)
    repairs = _repair_families(raw_families[:cfg["repair_seed_count"]], raw_families, n, cfg["max_repairs"])
    for family, operation in repairs:
        add(family, "local_repair:"+operation)
    failures.extend(dict(stage="contrast_rank",supports=[list(s) for s in f],
        details="proposal-ranking fit did not report success; candidate retained")
        for f,row in pool.items() if not row["rank_fit_success"])

    # Preserve an original RNJ and final accepted WZZT when both exist; all
    # further positions use the same arrived-data proposal ranking.
    selected = []
    rnj_families = {_family(row["supports"], n) for row in native["rnj"]}
    if rnj_families:
        selected.append(min(rnj_families, key=rank))
    if native["wzzt"]:
        final = max(native["wzzt"], key=lambda row:int(row["origin"].rsplit(":",1)[1]))
        f = _family(final["supports"], n)
        if f not in selected and len(selected)<cfg["max_candidates"]:
            selected.append(f)
    for f in sorted(pool, key=rank):
        if len(selected) >= cfg["max_candidates"]:
            break
        if f not in selected:
            selected.append(f)
    candidates = []
    bank = []
    for f in selected:
        name = "passive_"+sha256(repr((n,f)).encode("ascii")).hexdigest()[:12]
        candidate=RadialCandidate.from_supports(name,n,f); candidates.append(candidate)
        bank.append(dict(name=name,supports=[list(s) for s in f],
            full_supports=[list(s) for s in candidate.supports], **pool[f],
            replaceable=True, certified_supports=[], ac_evidence=None))
    common = set(raw_families[0]) if raw_families else set()
    union = set()
    for f in raw_families:
        common.intersection_update(f); union.update(f)
    frequency = Counter()
    for record in boot_records:
        frequency.update({tuple(s) for f in record["families"] for s in f})
    denominator = sum(bool(r["families"]) for r in boot_records)
    record = dict(schema="full_pipeline_passive_v1", config=cfg, seed=seed,
        n_terminal=n,n_arrived=count, bank=bank,native=native,bootstrap=boot_records,
        bootstrap_support_frequency=[dict(support=list(s),count=k,
            successful_replicates=denominator,frequency=k/denominator if denominator else None)
            for s,k in sorted(frequency.items())],
        bootstrap_frequency_meaning="support occurs in any RNJ tolerance/mode within a block-resampled history; not correctness probability",
        common_supports=[list(s) for s in sorted(common)],
        disputed_supports=[list(s) for s in sorted(union-common)],
        disputed_nodes=sorted({j for s in union-common for j in s}),
        repairs=[dict(supports=[list(s) for s in f],operation=op) for f,op in repairs],
        generated_count=len(pool),retained_count=len(bank),
        truncated_count=max(0,len(pool)-len(bank)),
        discarded=[dict(supports=[list(s) for s in f],**pool[f]) for f in sorted(pool,key=rank) if f not in selected],
        failures=failures,seconds=perf_counter()-began,
        source_inputs_empty=True,source_estimated_by_wzzt=False,
        source_model="native WZZT optional source input is None; unmetered source remains a proposal-model limitation",
        proposal_kappa_only=True,shared_AC_ratio_constraint=False,
        proposal_coefficient_bound_only=True,shared_AC_parameter_bound=False,
        extra_likelihood=False,ranking_is_AC_evidence=False,
        truth_seen=False,future_seen=False,certified_supports=[],
        coverage="finite history-generated replaceable proposal bank; no complete topology enumeration",
        budget=dict(wzzt_seconds_per_solve=cfg["wzzt_time_limit"],
            wzzt_whole_call_wall_time_guaranteed=False,
            bootstrap_replicates_requested=cfg["bootstrap_replicates"],
            bootstrap_replicates_with_proposals=denominator))
    return candidates,record


generate_candidates = propose

__all__ = ["propose", "generate_candidates", "DEFAULTS"]
