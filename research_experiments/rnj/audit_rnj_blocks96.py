"""Read-only audit primitives for the new 96-row RNJ block experiment.

No estimator, runtime package, model builder or solver is imported.  Counts
and scores are reconstructed from saved evidence.  A summary-only screen can
certify its selection arithmetic, but cannot certify unrecorded bootstrap
trees or resampling.  Missing evidence is reported separately from failure.
The runner-specific orchestration can call these functions for any feeder or
terminal count; physical truth is used only by ``physical_metrics``.
"""
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_experiments.rnj.audit_source_ac96_benchmark import (
    Audit, clade_metrics, optimality_certificate, physical_truth, relative_error,
)
from research_experiments.rnj import audit_source_ac288 as evidence
from research_experiments.rnj.audit_rx75_source_comparison import (
    independent_calibration, independent_tree_resolution, resolution_atoms,
)


FINAL_STATES = {"complete", "incomplete", "failed"}
SCREEN_POLICY = {"bootstrap_replicates": 100, "block_length": 4,
                 "confidence_threshold": .75, "maximum_candidate_count": 2,
                 "tolerance_factor": .16, "maximum_subclade_frequency": .15,
                 "fine_tolerance_ratio": .5, "minimum_block_size": 2}


def _sets(values):
    return {frozenset(map(int, value)) for value in values}


def _frequency(records):
    result = {}
    for row in records:
        key = frozenset(map(int, row["clade"]))
        if key in result:
            raise ValueError("duplicate clade in frequency records")
        result[key] = float(row["frequency"])
    return result


def terminal_peer_groups(tree, terminal_labels):
    """Recover COMPLETE direct-leaf peer groups from the actual rooted tree.

    An internal child, including a unary latent child with no nontrivial
    clade, disqualifies its parent.  Inclusion-minimal clades alone cannot
    establish this property.  Root-level direct leaves are not atomic groups.
    """
    root, labels = int(tree["root"]), set(map(int, terminal_labels))
    if not labels or len(labels) != len(terminal_labels):
        raise ValueError("distinct nonempty terminal labels required")
    adjacency = {root: set()}
    for edge in tree["edges"]:
        a, b = map(int, edge[:2])
        if a == b or b in adjacency.get(a, set()):
            raise ValueError("duplicated or self tree edge")
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    parents, order, children = {root: None}, [root], {a: [] for a in adjacency}
    for node in order:
        for child in sorted(adjacency[node]):
            if child == parents[node]:
                continue
            if child in parents:
                raise ValueError("saved RNJ tree contains a cycle")
            parents[child] = node
            children[node].append(child)
            order.append(child)
    if set(order) != set(adjacency) or not labels <= set(order):
        raise ValueError("saved RNJ tree disconnected or missing terminals")
    if any(children[label] for label in labels):
        raise ValueError("observed terminals must be leaves")
    return {frozenset(direct) for parent, direct in children.items()
            if parent != root and parent not in labels
            and 1 < len(direct) < len(labels) and set(direct) <= labels}


def selected_partition(selected_labels, labels, *, compressed=True):
    """Size-two blocks are allowed; every unselected terminal stays singleton."""
    labels = list(map(int, labels))
    if len(set(labels)) != len(labels):
        raise ValueError("terminal label order must be unique")
    position, groups, occupied = {value: i for i, value in enumerate(labels)}, [], set()
    for clade in selected_labels:
        group = tuple(sorted(position[int(value)] for value in clade))
        if len(group) < 2 or len(set(group)) != len(group) or occupied.intersection(group):
            raise ValueError("atomic blocks must be disjoint distinct-member sets of size >=2")
        occupied.update(group)
        groups.append(group)
    if not compressed:
        return tuple((i,) for i in range(len(labels)))
    groups.extend((i,) for i in range(len(labels)) if i not in occupied)
    return evidence.canonical_partition(groups, len(labels))


def audit_selector(screen, labels, audit, prefix="screen", *, policy=None):
    """Recompute selection, explicitly distinguishing summary and trace evidence.

    Optional ``diagnostics.full_tree`` and ``full_fine_clades`` independently
    verify the raw complete-peer and full-data finer-tree veto.  Optional
    ``bootstrap_records`` use one row per attempt with ``identifiable``,
    ``peers`` (or ``tree``), ``clades`` and ``fine_clades``.  Peer frequency
    always divides by ALL attempts, refinement by identifiable attempts only.
    """
    wanted = SCREEN_POLICY | (policy or {})
    params, diag = screen["parameters"], screen["diagnostics"]
    for name, value in wanted.items():
        actual = params.get(name, diag.get(name))
        audit.require(f"{prefix}: declared {name}", actual == value)
    b = int(wanted["bootstrap_replicates"])
    valid, invalid = diag["valid_bootstrap_replicates"], diag["unidentifiable_bootstrap_replicates"]
    valid_counts = (isinstance(valid, int) and not isinstance(valid, bool)
                    and isinstance(invalid, int) and not isinstance(invalid, bool)
                    and 0 <= valid <= b and 0 <= invalid <= b and valid + invalid == b)
    audit.require(f"{prefix}: all attempted bootstrap denominators retained", valid_counts)
    if not valid_counts:
        raise ValueError("invalid bootstrap denominators")
    universe = set(map(int, labels))
    peers = _sets(diag["full_peers"])
    full = _sets(screen["full_clades"])
    audit.require(f"{prefix}: full peers are proper original-terminal clades",
                  all(1 < len(c) < len(labels) and c <= universe for c in peers) and peers <= full)
    unavailable = []
    if "full_tree" in diag:
        audit.require(f"{prefix}: complete peer sets recovered from actual tree",
                      terminal_peer_groups(diag["full_tree"], labels) == peers)
        audit.require(f"{prefix}: full clades recovered from actual tree",
                      independent_tree_resolution(diag["full_tree"], labels, 0.)["family"] == full)
    else:
        unavailable.append("complete direct-leaf peer definition: full RNJ tree not saved")
    confidence = _frequency(screen["confidence_records"])
    sub = _frequency(diag["subclade_frequency_records"])
    for name, values, denominator in (("peer", confidence, b), ("refinement", sub, valid)):
        audit.require(f"{prefix}: {name} frequencies retain correct denominator", all(
            np.isfinite(f) and 0 <= f <= 1 and
            (abs(denominator*f-round(denominator*f)) <= 1e-8 if denominator else f == 1.)
            for f in values.values()))
    audit.require(f"{prefix}: all full peers have saved peer and refinement counts",
                  peers <= set(confidence) and set(sub) == peers)
    vetoed = _sets(diag["full_fine_subclades"])
    audit.require(f"{prefix}: full finer-tree veto names full peers only", vetoed <= peers)
    if "full_fine_clades" in diag:
        fine = _sets(diag["full_fine_clades"])
        if "full_fine_tree" in diag:
            audit.require(f"{prefix}: finer clades recovered from actual tree",
                          independent_tree_resolution(diag["full_fine_tree"], labels, 0.)["family"] == fine)
        expected_veto = {c for c in peers if any(1 < len(child) and child < c for child in fine)}
        audit.require(f"{prefix}: full finer-tree veto independently reconstructed", vetoed == expected_veto)
    else:
        unavailable.append("full-data finer-tree veto: fine clades not saved")
    trace = diag.get("bootstrap_records")
    if trace is None:
        unavailable.append("raw bootstrap peer/refinement recount and row synchronization: trace not saved")
    else:
        audit.require(f"{prefix}: bootstrap trace includes all attempts", len(trace) == b)
        counts, refinements, actual_valid = Counter(), Counter(), 0
        for index, record in enumerate(trace):
            flag = record.get("identifiable")
            audit.require(f"{prefix}: bootstrap[{index}] identifiable explicitly boolean", isinstance(flag, bool))
            if "valid_for_frequency_counts" in record:
                audit.require(f"{prefix}: bootstrap[{index}] validity and counting agree",
                              record["valid_for_frequency_counts"] is flag)
            if flag is not True:
                continue
            actual_valid += 1
            sampled_peers = (terminal_peer_groups(record["tree"], labels)
                             if "tree" in record else _sets(record["peers"]))
            counts.update(sampled_peers)
            children = _sets(record.get("coarse_clades", record.get("clades", []))) | _sets(record["fine_clades"])
            refinements.update(c for c in peers if any(1 < len(child) and child < c for child in children))
        audit.require(f"{prefix}: raw identifiable bootstrap count reproduced", actual_valid == valid)
        expected_conf = {c: counts[c]/b for c in set(counts) | peers}
        expected_sub = {c: refinements[c]/valid if valid else 1. for c in peers}
        audit.require(f"{prefix}: peer frequencies independently recounted", confidence == expected_conf)
        audit.require(f"{prefix}: refinement frequencies independently recounted", sub == expected_sub)
    identifiable = diag["source_adjusted_identifiable"]
    audit.require(f"{prefix}: full Source-adjusted identifiability explicitly boolean", isinstance(identifiable, bool))
    eligible = {c for c in peers if len(c) >= wanted["minimum_block_size"]
                and confidence.get(c, 0.) >= wanted["confidence_threshold"]
                and sub.get(c, 1.) <= wanted["maximum_subclade_frequency"]
                and c not in vetoed and identifiable is True and valid > 0}
    audit.require(f"{prefix}: saved eligible set follows predeclared rule", _sets(diag["eligible"]) == eligible)
    selected, occupied = [], set()
    for clade in sorted(eligible, key=lambda c: (-confidence[c], -len(c), tuple(sorted(c)))):
        if len(selected) >= wanted["maximum_candidate_count"]:
            break
        if occupied.isdisjoint(clade):
            selected.append(sorted(clade))
            occupied.update(clade)
    audit.require(f"{prefix}: stable disjoint deterministic selection", screen["selected_support_labels"] == selected)
    positions = {int(label): i for i, label in enumerate(labels)}
    indices = [[positions[value] for value in clade] for clade in selected]
    audit.require(f"{prefix}: labels map to original-terminal positions", screen["selected_support_indices"] == indices)
    return {"selected_labels": selected, "selected_indices": indices,
            "eligible": [sorted(c) for c in sorted(eligible, key=lambda c: tuple(sorted(c)))],
            "unavailable_checks": unavailable, "bootstrap_trace_recounted": trace is not None}


def source_adjusted_ranks(p, q, source_p, source_q):
    """Independent SVD arithmetic, with scale fixed by the ORIGINAL design.

    With no nonzero Source the runtime's eligibility flag is true even when
    D=[P Q] is rank deficient. With Source it requires projected rank 2*n.
    This is a numerical eligibility check, not topology confidence.
    """
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    u = np.column_stack((np.asarray(source_p).reshape(-1), np.asarray(source_q).reshape(-1)))
    if p.ndim != 2 or p.shape != q.shape or u.shape[0] != p.shape[0] or not np.isfinite([p, q]).all() or not np.isfinite(u).all():
        raise ValueError("matching finite observed and Source arrays required")
    design = np.column_stack((p, q))
    active, rank = bool(np.any(u != 0.)), 0
    projected = design
    if active:
        basis, singular, _ = np.linalg.svd(u, full_matrices=False)
        rank = int(np.count_nonzero(singular > np.finfo(float).eps*max(u.shape)*singular[0]))
        basis = basis[:, :rank]
        projected = design-basis@(basis.T@design)
    cutoff = np.finfo(float).eps*max(design.shape)*np.linalg.norm(design, ord=2)
    projected_rank = int(np.count_nonzero(np.linalg.svd(projected, compute_uv=False) > cutoff))
    return {"source_design_rank": rank, "projected_design_rank": projected_rank,
            "source_adjusted_identifiable": not active or projected_rank == design.shape[1]}


def audit_resampling_trace(screen, arrays, audit, prefix="bootstrap"):
    """Rebuild circular moving-block rows and Source-adjusted ranks, without fit.

    The one saved index stream addresses P, Q, Y and Source together.  This
    verifies the recorded row contract and rank arithmetic; without saved
    sampled design arrays it does not independently observe arrays consumed
    inside the regression call.
    """
    params, diag = screen["parameters"], screen["diagnostics"]
    trace = diag.get("bootstrap_records")
    if trace is None:
        return {"checked": False, "reason": "bootstrap trace not saved"}
    p, q = np.asarray(arrays["P"]), np.asarray(arrays["Q"])
    y = np.asarray(arrays["Y"])
    source_p, source_q = np.asarray(arrays["P_source"]).reshape(-1), np.asarray(arrays["Q_source"]).reshape(-1)
    count = len(p)
    if count != 96 or p.shape != q.shape or y.shape != p.shape or source_p.shape != (96,) or source_q.shape != (96,):
        raise ValueError("one matched 96-row scenario is required")
    full = source_adjusted_ranks(p, q, source_p, source_q)
    audit.require(f"{prefix}: full-data eligibility independently reproduced",
                  diag["source_adjusted_identifiable"] is full["source_adjusted_identifiable"])
    rng = np.random.default_rng(params["seed"])
    length = min(int(params["block_length"]), count)
    checked = 0
    for number, record in enumerate(trace):
        starts = rng.integers(0, count, size=(count+length-1)//length)
        expected = np.concatenate([(start+np.arange(length)) % count for start in starts])[:count]
        indices = record.get("resample_indices")
        valid = (isinstance(indices, list) and len(indices) == 1
                 and np.asarray(indices[0]).shape == (96,)
                 and np.asarray(indices[0]).dtype.kind in "iu"
                 and np.array_equal(indices[0], expected))
        audit.require(f"{prefix}: replicate[{number}] original circular moving-block indices", valid)
        audit.require(f"{prefix}: replicate[{number}] sequence retained", record.get("replicate") == number)
        if not valid:
            continue
        ranks = source_adjusted_ranks(p[expected], q[expected], source_p[expected], source_q[expected])
        for field in ("source_design_rank", "projected_design_rank"):
            audit.require(f"{prefix}: replicate[{number}] {field} independently reproduced", record.get(field) == ranks[field])
        audit.require(f"{prefix}: replicate[{number}] eligibility follows same Source-adjusted rows",
                      record.get("identifiable") is ranks["source_adjusted_identifiable"])
        checked += 1
    return {"checked": True, "replicates_rebuilt": checked, "full_data_ranks": full,
            "scope": "saved common indices and independently reconstructed observed/Source rank arithmetic; no estimator rerun"}


def audit_initialization(model, selected_labels, labels, audit, prefix="initial", *, compressed,
                         ordinary_supports=None):
    """Ordinary initial supports and indivisible atomic groups are separate."""
    n, positions = len(labels), {int(value): i for i, value in enumerate(labels)}
    selected = _sets([[positions[int(value)] for value in group] for group in selected_labels])
    ordinary = _sets(model.get("ordinary_initial_supports", []) if ordinary_supports is None else ordinary_supports)
    initial = _sets(model["initial_supports"])
    leaves = {frozenset((i,)) for i in range(n)}
    audit.require(f"{prefix}: every original singleton leaf retained", leaves <= initial)
    audit.require(f"{prefix}: ordinary initial supports reported separately", ordinary <= initial)
    audit.require(f"{prefix}: initial family contains only declared leaves ordinary and selected supports",
                  initial == leaves | ordinary | selected)
    audit.require(f"{prefix}: selected atomic blocks are present independently of compression", selected <= initial)
    expected = selected_partition(selected_labels, labels, compressed=compressed)
    actual = evidence.canonical_partition(model["terminal_groups"], n)
    audit.require(f"{prefix}: actual membership partition matches group toggle", actual == expected)
    requested = model.get("requested_terminal_groups", [])
    audit.require(f"{prefix}: requested indivisible groups match toggle",
                  _sets(requested) == (selected if compressed else set()))
    return {"selected_count": len(selected), "ordinary_nontrivial_count": sum(1 < len(s) < n for s in ordinary),
            "partition": [list(group) for group in actual], "partition_size": len(actual)}


class _IndexedRows(list):
    """Index only fully aggregated, exactly zero-cleared COO column supports."""
    def __init__(self, rows):
        super().__init__(rows)
        self.by_columns = {}
        self.columns, self.values = [], []
        for index, row in enumerate(self):
            columns = tuple(sorted(row))
            values = np.asarray([row[column] for column in columns], dtype=float)
            self.columns.append(columns)
            self.values.append(values)
            self.by_columns.setdefault(columns, []).append(index)


def _coo_rows(saved):
    rows = [{} for _ in saved["constraint_lower"]]
    for row, column, value in zip(saved["constraint_row_index"], saved["constraint_col_index"], saved["constraint_data"], strict=True):
        target = rows[int(row)]
        target[int(column)] = target.get(int(column), 0.) + float(value)
    return _IndexedRows([{column: value for column, value in row.items() if value != 0.} for row in rows])


def _scalar_close(actual, expected):
    if np.isinf(expected):
        return actual == expected
    return abs(actual-expected) <= 1e-12+1e-10*abs(expected)


def _has_row(rows, saved, entries, *, lower=-np.inf, upper=np.inf):
    entries = {int(k): float(v) for k, v in entries.items() if v != 0.}
    columns = tuple(sorted(entries))
    values = np.asarray([entries[column] for column in columns], dtype=float)
    if not isinstance(rows, _IndexedRows):
        rows = _IndexedRows(rows)
    return any(_scalar_close(saved["constraint_lower"][i], lower)
               and _scalar_close(saved["constraint_upper"][i], upper)
               and np.allclose(rows.values[i], values, atol=1e-12, rtol=1e-10)
               for i in rows.by_columns.get(columns, ()))


def audit_floor_domain(event, saved, n, partition, calibration, audit, prefix):
    """Same original floor-row audit, indexed without scanning residual rows."""
    slices, rows = event["variable_slices"], _coo_rows(saved)
    r, x = slices.get("old_r", slices.get("r")), slices.get("old_x", slices.get("x"))
    if r is None or x is None:
        raise ValueError("raw atom-weight slices missing")
    k = int(r[1])-int(r[0])
    audit.require(f"{prefix}: old weights include original singleton prefix", k >= n and int(x[1])-int(x[0]) == k)
    a, b, tau = (calibration[name] for name in ("a", "b", "tau"))
    audit.require(f"{prefix}: positive fixed geometry fits coefficient capacity", a > 0 and b > 0 and tau > 0
                  and np.isfinite([a, b, tau]).all()
                  and a*saved["upper_bounds"][r[0]]+b*saved["upper_bounds"][x[0]]+1e-12 >= tau)
    active = slices.get("edge_active")
    audit.require(f"{prefix}: one activation per old non-singleton only", (active is None and k == n)
                  or (active is not None and int(active[1])-int(active[0]) == k-n))
    if active is not None:
        for offset, atom in enumerate(range(n, k)):
            ri, xi, ei = int(r[0])+atom, int(x[0])+atom, int(active[0])+offset
            ur, ux = saved["upper_bounds"][ri], saved["upper_bounds"][xi]
            audit.require(f"{prefix}: old edge[{atom}] activation binary domain", saved["integrality"][ei] == 1
                          and saved["lower_bounds"][ei] == 0 and saved["upper_bounds"][ei] == 1)
            audit.require(f"{prefix}: old edge[{atom}] exact zero-or-floor rows",
                          _has_row(rows, saved, {ri: 1., ei: -ur}, upper=0.)
                          and _has_row(rows, saved, {xi: 1., ei: -ux}, upper=0.)
                          and _has_row(rows, saved, {ri: a, xi: b, ei: -tau}, lower=0.))
            containing = [index for columns, indices in rows.by_columns.items() if ei in columns for index in indices]
            audit.require(f"{prefix}: activation does not constrain Source path", len(containing) == 3
                          and all(set(rows.columns[i]) <= {ri, xi, ei} for i in containing))
    if "new_r" in slices:
        ri, xi, z = int(slices["new_r"][0]), int(slices["new_x"][0]), slices["z"]
        audit.require(f"{prefix}: candidate coefficient reaches fixed resolution", _has_row(rows, saved, {ri: a, xi: b}, lower=tau))
        audit.require(f"{prefix}: membership dimension is partition size", int(z[1])-int(z[0]) == len(partition))
        cardinality = {int(z[0])+i: float(len(group)) for i, group in enumerate(partition)}
        audit.require(f"{prefix}: original-terminal candidate cardinality >=2", _has_row(rows, saved, cardinality, lower=2.))
        if "primal" in saved:
            value = saved["primal"]
            count = sum(len(group) for i, group in enumerate(partition) if value[int(z[0])+i] > .5)
            audit.require(f"{prefix}: incumbent contains >=2 original terminals", count >= 2)
            audit.require(f"{prefix}: incumbent new coefficient satisfies fixed floor", a*value[ri]+b*value[xi]+1e-6 >= tau)


def audit_membership_domain(event, saved, n, partition, supports, audit, prefix="raw", *, source_active,
                            samples=96, require_terminal_cardinality=True):
    """Inspect saved COO aliasing, dimensions and original-terminal counting.

    ``supports`` is the OLD family at this solve, in variable-weight order.
    Full Source h includes all K old atoms and the candidate atom, regardless
    of the compressed membership dimension. Fixed-family solves have K h's.
    """
    groups = evidence.canonical_partition(partition, n)
    supports = tuple(tuple(map(int, s)) for s in supports)
    m, k = len(groups), len(supports)
    slices, rows = event["variable_slices"], _coo_rows(saved)
    length = lambda name: int(slices[name][1])-int(slices[name][0]) if name in slices else None
    extension = "z" in slices
    for name in ("old_r", "old_x") if extension else ("r", "x"):
        audit.require(f"{prefix}: {name} covers original atom family", length(name) == k)
    audit.require(f"{prefix}: observations remain original 96-by-n dimension",
                  length("absolute_residual") == samples*n)
    if extension:
        zstart, zstop = map(int, slices["z"])
        valid_slice = 0 <= zstart < zstop <= len(saved["integrality"])
        audit.require(f"{prefix}: membership z count equals partition count", valid_slice and zstop-zstart == m)
        if not valid_slice:
            return
        audit.require(f"{prefix}: all and only membership variables in z are binary",
                      np.all(saved["integrality"][zstart:zstop] == 1)
                      and np.all(saved["lower_bounds"][zstart:zstop] == 0)
                      and np.all(saved["upper_bounds"][zstart:zstop] == 1))
        for name in ("y", "u_r", "u_x"):
            audit.require(f"{prefix}: {name} uses group-pair dimension", length(name) == m*(m+1)//2)
        for atom, support in enumerate(supports):
            old = set(support)
            entries = {zstart+i: len(group)-2*len(set(group) & old) for i, group in enumerate(groups)}
            audit.require(f"{prefix}: duplicate[{atom}] Hamming row counts ORIGINAL terminals",
                          _has_row(rows, saved, entries, lower=1.-len(old)))
        known = set(supports)
        minimum_groups = 2. if all(group in known for group in groups) else 1.
        audit.require(f"{prefix}: valid group-count strengthening kept distinct from terminal count",
                      _has_row(rows, saved, {zstart+i: 1. for i in range(m)}, lower=minimum_groups))
        if require_terminal_cardinality:
            audit.require(f"{prefix}: candidate cardinality uses ORIGINAL terminal weights",
                          _has_row(rows, saved, {zstart+i: float(len(group)) for i, group in enumerate(groups)}, lower=2.))
    hcount = k + int(extension)
    if source_active:
        for name in ("source_h", "source_w_r", "source_w_x"):
            audit.require(f"{prefix}: {name} covers full K+candidate Source path", length(name) == hcount)
        if extension:
            for name in ("source_new_v_r", "source_new_v_x"):
                audit.require(f"{prefix}: {name} uses membership group dimension", length(name) == m)
        if "source_h" in slices:
            lo, hi = map(int, slices["source_h"])
            audit.require(f"{prefix}: complete Source h domain is binary zero-one",
                          np.all(saved["integrality"][lo:hi] == 1)
                          and np.all(saved["lower_bounds"][lo:hi] == 0)
                          and np.all(saved["upper_bounds"][lo:hi] == 1))
        audit_source_linear_domain(event, saved, supports, audit, prefix+": source")
    else:
        audit.require(f"{prefix}: all-zero Source falls back to ordinary model",
                      not any(name.startswith("source_") for name in slices))
    evidence.audit_raw_primal(event, saved, audit, prefix)


def audit_source_linear_domain(event, saved, supports, audit, prefix="source_domain"):
    """Check original full-path rows and both exact bounded product chains."""
    slices, rows = event["variable_slices"], _coo_rows(saved)
    extension, k = "z" in slices, len(supports)
    count = k+int(extension)
    hstart, hstop = map(int, slices["source_h"])
    if hstop-hstart != count:
        audit.require(f"{prefix}: complete path slice required for row audit", False)
        return
    audit.require(f"{prefix}: nonempty constraint uses the complete Source h",
                  _has_row(rows, saved, {hstart+i: 1. for i in range(count)}, lower=1.))
    sets = [set(s) for s in supports]
    for i, a in enumerate(sets):
        for j, b in enumerate(sets[i+1:], start=i+1):
            if a < b:
                expected, upper = {hstart+i: 1., hstart+j: -1.}, 0.
            elif b < a:
                expected, upper = {hstart+j: 1., hstart+i: -1.}, 0.
            elif a.isdisjoint(b):
                expected, upper = {hstart+i: 1., hstart+j: 1.}, 1.
            else:
                audit.require(f"{prefix}: distinct laminar original Source family", False)
                continue
            audit.require(f"{prefix}: original path relation[{i},{j}] retained",
                          _has_row(rows, saved, expected, upper=upper))
    def products(weight, indicator, product, bound, name):
        audit.require(name, _has_row(rows, saved, {product: 1., indicator: -bound}, upper=0.)
                      and _has_row(rows, saved, {product: 1., weight: -1.}, upper=0.)
                      and _has_row(rows, saved, {product: 1., weight: -1., indicator: -bound}, lower=-bound))
    for channel in ("r", "x"):
        old_name = ("old_" if extension else "")+channel
        old = list(range(*map(int, slices[old_name])))
        weights = old+([int(slices["new_"+channel][0])] if extension else [])
        start, stop = map(int, slices["source_w_"+channel])
        if stop-start != count:
            audit.require(f"{prefix}: complete {channel} Source products required", False)
            continue
        for atom, weight in enumerate(weights):
            products(weight, hstart+atom, start+atom, float(saved["upper_bounds"][weight]),
                     f"{prefix}: {channel} weight-times-h[{atom}] exact original rows")
        if extension:
            zstart, zstop = map(int, slices["z"])
            vstart, vstop = map(int, slices["source_new_v_"+channel])
            if vstop-vstart != zstop-zstart:
                audit.require(f"{prefix}: complete {channel} new Source membership products required", False)
                continue
            bound = float(saved["upper_bounds"][weights[-1]])
            for group, z in enumerate(range(zstart, zstop)):
                products(stop-1, z, vstart+group, bound,
                         f"{prefix}: {channel} new Source-times-z[{group}] exact original rows")
    if extension:
        relation = slices.get("relations")
        if relation is None or int(relation[1])-int(relation[0]) != 3*k:
            audit.require(f"{prefix}: all old/new Source relation variables retained", False)
            return
        for atom in range(k):
            subset = int(relation[0])+3*atom
            audit.require(f"{prefix}: candidate/old full-path relation[{atom}] retained",
                          _has_row(rows, saved, {hstart+k: 1., hstart+atom: -1., subset: 1.}, upper=1.)
                          and _has_row(rows, saved, {hstart+atom: 1., hstart+k: -1., subset+1: 1.}, upper=1.)
                          and _has_row(rows, saved, {hstart+k: 1., hstart+atom: 1., subset+2: 1.}, upper=2.))


def audit_residual_rows(event, saved, arrays, supports, partition, audit, prefix="residual"):
    """Check every original output equation, including Source counted ONCE.

    Membership products use sums of ordinary P/Q within an input group. The
    known Source amplitude multiplies only its own response product; it is
    never multiplied by a group size or replicated over member terminals.
    """
    p, q, y = (np.asarray(arrays[name], dtype=float) for name in ("P", "Q", "Y"))
    samples, n = p.shape
    groups = evidence.canonical_partition(partition, n)
    slices, rows = event["variable_slices"], _coo_rows(saved)
    extension, source = "z" in slices, "source_h" in slices
    rstart, xstart = (int(slices[("old_" if extension else "")+name][0]) for name in ("r", "x"))
    absolute, stop = map(int, slices["absolute_residual"])
    audit.require(f"{prefix}: full original residual dimension", stop-absolute == samples*n)
    objective = np.asarray(saved["objective_coefficients"])
    expected_objective = np.zeros_like(objective)
    expected_objective[absolute:stop] = 1./(samples*n)
    audit.close(f"{prefix}: MAE denominator is samples times ORIGINAL terminals", objective, expected_objective, atol=1e-15, rtol=1e-12)
    by_error = {column: [] for column in range(absolute, stop)}
    for index, row in enumerate(rows):
        for column in row:
            if absolute <= column < stop:
                by_error[column].append(index)
    group_of = {terminal: i for i, group in enumerate(groups) for terminal in group}
    pairs = {(i, j): k for k, (i, j) in enumerate((i, j) for i in range(len(groups)) for j in range(i, len(groups)))}
    source_p, source_q = np.asarray(arrays["P_source"]).reshape(-1), np.asarray(arrays["Q_source"]).reshape(-1)
    mismatches = []
    for t in range(samples):
        for output in range(n):
            prediction = {}
            for atom, support in enumerate(supports):
                if output in support:
                    prediction[rstart+atom] = float(p[t, support].sum())
                    prediction[xstart+atom] = float(q[t, support].sum())
                    if source:
                        prediction[int(slices["source_w_r"][0])+atom] = float(source_p[t])
                        prediction[int(slices["source_w_x"][0])+atom] = float(source_q[t])
            if extension:
                outgroup = group_of[output]
                for index, group in enumerate(groups):
                    pair = pairs[min(outgroup, index), max(outgroup, index)]
                    prediction[int(slices["u_r"][0])+pair] = float(p[t, group].sum())
                    prediction[int(slices["u_x"][0])+pair] = float(q[t, group].sum())
                if source:
                    prediction[int(slices["source_new_v_r"][0])+outgroup] = float(source_p[t])
                    prediction[int(slices["source_new_v_x"][0])+outgroup] = float(source_q[t])
            prediction = {column: value for column, value in prediction.items() if value != 0.}
            error, target = absolute+t*n+output, float(y[t, output])
            indices = by_error.get(error, [])
            correct = len(indices) == 2
            for sign, lo, hi in ((1., target, np.inf), (-1., -np.inf, target)):
                expected = prediction | {error: sign}
                columns = tuple(sorted(expected))
                values = np.asarray([expected[column] for column in columns])
                correct &= any(rows.columns[i] == columns
                               and _scalar_close(saved["constraint_lower"][i], lo)
                               and _scalar_close(saved["constraint_upper"][i], hi)
                               and np.allclose(rows.values[i], values, atol=1e-12, rtol=1e-10)
                               for i in indices)
            if not correct:
                mismatches.append([t, output])
    audit.require(f"{prefix}: every ordinary and Source residual coefficient uses original output rows exactly once",
                  not mismatches, observation_count=samples*n, first_mismatches=mismatches[:10])


def audit_training96(arrays, variant_directory, labels, audit, prefix="inputs", *, data_use=None):
    """Match actual saved 96 rows against observed/source CSV time indices."""
    directory = Path(variant_directory)
    read = lambda path: pd.read_csv(path, index_col=0, float_precision="round_trip")
    frames = {name: read(directory/"observed"/filename) for name, filename in
              {"P": "P_terminal.csv", "Q": "Q_terminal.csv", "Y": "drop_target.csv",
               "V": "V_terminal.csv", "root": "root_voltage.csv"}.items()}
    source = read(directory/"source/amplitudes.csv")
    index = frames["P"].index
    audit.require(f"{prefix}: exactly 96 distinct observed time labels", len(index) == 96 and index.is_unique)
    for name, frame in frames.items():
        audit.require(f"{prefix}: {name} time order equals original observed P", frame.index.equals(index))
    audit.require(f"{prefix}: Source time order equals original observations", source.index.equals(index))
    for name in ("P", "Q", "Y", "V"):
        audit.require(f"{prefix}: {name} terminal label order preserved", list(map(int, frames[name].columns)) == list(map(int, labels)))
    for name in ("P", "Q", "Y"):
        audit.close(f"{prefix}: full observed {name} array preserved", arrays[name], frames[name].to_numpy(), atol=0., rtol=0.)
    for name in ("P", "Q"):
        audit.close(f"{prefix}: full known Source {name} amplitudes preserved", np.asarray(arrays[name+"_source"]).reshape(-1),
                    source[name+"_pu"].to_numpy(), atol=0., rtol=0.)
    audit.require(f"{prefix}: Source table contains amplitudes only",
                  {"P_pu", "Q_pu"} <= set(source.columns)
                  and set(source.columns) <= {"P_pu", "Q_pu", "P_kw", "Q_kvar"})
    audit.close(f"{prefix}: squared-voltage target uses measured root", frames["Y"].to_numpy(),
                frames["root"].iloc[:, 0].to_numpy()[:, None]**2-frames["V"].to_numpy()**2, atol=5e-15, rtol=1e-12)
    if "row_position" in arrays:
        audit.require(f"{prefix}: all 96 original row positions retained", np.array_equal(arrays["row_position"], np.arange(96)))
    if "row_scenario" in arrays:
        audit.require(f"{prefix}: exactly one day/scenario retained", np.asarray(arrays["row_scenario"]).shape == (96,)
                      and len(set(np.asarray(arrays["row_scenario"]).tolist())) == 1)
    audit.require(f"{prefix}: no validation arrays contain samples", not any("validation" in name.lower() and np.asarray(value).size
                  for name, value in arrays.items()))
    if data_use is not None:
        audit.require(f"{prefix}: 96 rows all used for fit", data_use.get("training_samples") == 96)
        audit.require(f"{prefix}: validation count zero", data_use.get("validation_samples") == 0)
    return {"samples": len(index), "terminal_count": len(labels), "time_labels": index.tolist()}


def audit_fixed_support_pair(left_event, left_saved, right_event, right_saved, audit, prefix="fixed_pair"):
    """Check group on/off fixed-family domain equivalence without re-solving.

    Nonunique optimal solutions may have different R/X or h. Identical saved
    model coefficients, bounds, integrality and row multisets, plus certified
    equal optimum, establish this control; coefficient equality is not assumed.
    """
    audit.require(f"{prefix}: neither fixed-support model introduces membership z",
                  "z" not in left_event["variable_slices"] and "z" not in right_event["variable_slices"])
    audit.require(f"{prefix}: variable slices remain identical", left_event["variable_slices"] == right_event["variable_slices"])
    for name in ("objective_coefficients", "lower_bounds", "upper_bounds", "integrality"):
        audit.require(f"{prefix}: original {name} identical", np.array_equal(left_saved[name], right_saved[name]))
    def canonical(saved):
        return Counter((tuple(sorted(row.items())), float(saved["constraint_lower"][i]),
                        float(saved["constraint_upper"][i])) for i, row in enumerate(_coo_rows(saved)))
    audit.require(f"{prefix}: original linear constraints identical after row permutation", canonical(left_saved) == canonical(right_saved))
    for name, event, saved in (("ungrouped", left_event, left_saved), ("grouped", right_event, right_saved)):
        audit.require(f"{prefix}: {name} fixed solve certified", optimality_certificate(event))
        evidence.audit_raw_primal(event, saved, audit, f"{prefix}: {name}")
    audit.close(f"{prefix}: same certified fixed-family optimum", left_event.get("objective"), right_event.get("objective"), atol=1e-9)


def physical_metrics(model, arrays, network_directory, *, calibration):
    """Posthoc physical metrics; never use truth to select a path or blocks."""
    labels = list(map(int, model["terminal_labels"]))
    truth = physical_truth(Path(network_directory), labels)
    if model.get("support_indices") is not None:
        reconstructed = evidence.atom_reconstruction(model, labels)
        family, active = reconstructed["family"], reconstructed["active"]
        _, keep = resolution_atoms(model["support_indices"], model["r_values"], model["x_values"], calibration)
        resolved = {frozenset(labels[i] for i in support) for support, retained in zip(model["support_indices"], keep, strict=True) if retained}
    else:
        tree = independent_tree_resolution(model["tree"], labels, calibration["tau"])
        family = active = tree["family"]
        resolved = tree["resolution"]
    metrics = {}
    for name, clades in (("family", family), ("active", active), ("resolution", resolved)):
        metrics.update({name+"_"+key: value for key, value in clade_metrics(clades, truth["clades"], labels).items()})
    metrics.update(r_relative_error=relative_error(arrays["R"], truth["R"]),
                   x_relative_error=relative_error(arrays["X"], truth["X"]), truth_used_for_selection=False)
    return metrics


def planned_outcomes(jobs, results):
    """Keep every predeclared job; missing and failed metrics stay None."""
    ids = [str(job["job_id"]) for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("planned job IDs must be unique")
    if set(results)-set(ids):
        raise ValueError("unplanned result must not enter the comparison denominator")
    rows = []
    for job in jobs:
        key, result = str(job["job_id"]), results.get(str(job["job_id"]))
        state = result.get("status") if result else "pending"
        rows.append(dict(job) | {"status": state, "is_final": state in FINAL_STATES,
                                "stop_reason": result.get("stop_reason") if result else None,
                                "metrics": result.get("metrics") if result else None,
                                "elapsed_seconds": result.get("elapsed_seconds") if result else None})
    return {"planned_denominator": len(jobs), "status_counts": dict(Counter(row["status"] for row in rows)), "jobs": rows}


def audit_path96(path, arrays, labels, initial_supports, partition, r_bound, x_bound, audit, prefix):
    """Reconstruct every accepted point; select the saved last point only."""
    if not path:
        raise ValueError("available model requires a nonempty certified path")
    initial, groups = _sets(initial_supports), evidence.canonical_partition(partition, len(labels))
    source_active = bool(np.any(arrays["P_source"] != 0.) or np.any(arrays["Q_source"] != 0.))
    values, previous = [], None
    for index, point in enumerate(path):
        tag = f"{prefix}: path[{index}]"
        value = evidence.atom_reconstruction(point, labels)
        audit.require(f"{tag}: fixed initial family retained", initial <= set(value["sets"]))
        audit.require(f"{tag}: later supports are unions of complete membership groups", all(
            not (s & set(group)) or set(group) <= s for s in value["sets"] if s not in initial for group in groups))
        if index == 0:
            audit.require(f"{tag}: path0 exactly fits declared initial family", set(value["sets"]) == initial)
        audit.require(f"{tag}: original singleton prefix preserved", point["support_indices"][:len(labels)] == [[i] for i in range(len(labels))])
        audit.require(f"{tag}: original positional supports and labels agree", point.get("support_labels") == [[labels[i] for i in s] for s in value["supports"]])
        for name, bound in (("r", r_bound), ("x", x_bound)):
            audit.require(f"{tag}: {name} coefficient box", np.all(value[name] >= -1e-8) and np.all(value[name] <= bound+1e-8))
        gr, gx = evidence.audit_source_path(point, value, source_active, audit, tag)
        prediction, loss = evidence.prediction_and_loss(arrays, value["R"], value["X"], gr, gx)
        audit.close(f"{tag}: exact 96-row full training MAE", point["train_mae"], loss, atol=1e-9)
        audit.require(f"{tag}: no validation or 1SE selection", point.get("validation_mae") is None and point.get("validation_se") is None)
        diagnostic = point.get("solver", point.get("diagnostics", {}))
        audit.require(f"{tag}: accepted point has strict numerical certificate", optimality_certificate(diagnostic))
        audit.close(f"{tag}: accepted solver objective", diagnostic.get("objective"), loss, atol=2e-9)
        if previous is not None:
            gain = previous-loss
            tolerance = 1e-10+1e-8*max(abs(previous), np.finfo(float).eps)
            audit.require(f"{tag}: accepted improvement exceeds fixed tolerance", gain > tolerance-1e-10)
            audit.close(f"{tag}: accepted improvement reproduced", point["accepted_gain"], gain, atol=2e-9)
        previous = loss
        values.append(value | {"loss": loss, "prediction": prediction, "source_response_r": gr, "source_response_x": gx})
    return values


def _hash_check(root, relative, checksum, audit, name):
    root, target = Path(root).resolve(), (Path(root)/relative).resolve()
    audit.require(name, target.is_relative_to(root) and target.is_file() and evidence.file_hash(target) == checksum)


def audit_model96(folder, model, arrays, job, input_directory, network, audit, *, expected_core=None):
    """Audit the runner's saved model without importing that runner."""
    folder, key = Path(folder), job["job_id"]
    labels = list(map(int, model["terminal_labels"]))
    n = len(labels)
    if expected_core is not None:
        audit.require(f"{key}: actual fit parameters follow frozen dimension-specific policy",
                      model.get("core_parameters") == expected_core | {"max_atoms": 2*n-1})
    audit_training96(arrays, input_directory, labels, audit, key,
                     data_use=evidence.read_json(folder/"data_use.json"))
    calibration = independent_calibration(arrays["calibration_R"], arrays["calibration_X"])
    saved_calibration = evidence.read_json(folder/"rx75_calibration.json")
    for name, value in calibration.items():
        audit.close(f"{key}: observed-only calibration {name}", saved_calibration.get(name), value, atol=1e-12)
        audit.close(f"{key}: model fixed calibration {name}", model["rx75_calibration"].get(name), value, atol=1e-12)
    audit.require(f"{key}: calibration truth-free pre-tree origin", saved_calibration.get("truth_used") is False
                  and saved_calibration.get("matrix_origin") == "pre_tree_ordered_FWL_regression")
    audit.require(f"{key}: all 96 rows fit without validation", model.get("training_rows") == 96
                  and model.get("validation_supplied") is False and model.get("validation_mae") is None
                  and model.get("validation_se") is None)
    source = np.column_stack((arrays["P_source"], arrays["Q_source"]))
    g = np.linalg.lstsq(source, arrays["Y"]-arrays["P"]@arrays["calibration_R"].T-arrays["Q"]@arrays["calibration_X"].T, rcond=None)[0]
    audit.close(f"{key}: calibration Source nuisance minimum-norm response", arrays["calibration_source_g"], g, atol=1e-8)
    source_active = bool(np.any(source != 0.))
    selected, screen_detail, resampling = [], None, None
    if job["method"] != "WZZT":
        screen = evidence.read_json(folder/"rnj_screen.json")
        audit.require(f"{key}: screening has 96 training rows and no validation", screen.get("training_sample_count") == 96 and screen.get("validation_used") is False)
        screen_detail = audit_selector(screen, labels, audit, key+": screen", policy=job["screen_parameters"])
        resampling = audit_resampling_trace(screen, arrays, audit, key+": bootstrap")
        selected = screen_detail["selected_labels"]
        full = screen["full_clades"]
    else:
        full = []
        audit.require(f"{key}: pure WZZT has no RNJ screening", model.get("bootstrap") is None)
    positions = {label: i for i, label in enumerate(labels)}
    ordinary = [[positions[int(value)] for value in clade] for clade in full] if job["method"].endswith("_tree_init") else []
    initialized = audit_initialization(model, selected, labels, audit, key+": initialization",
                                      compressed=job["method"] in {"RNJ+WZZT", "RNJ+WZZT_tree_init"}, ordinary_supports=ordinary)
    path = model["path"]
    values = audit_path96(path, arrays, labels, model["initial_supports"], model["terminal_groups"],
                          model["r_upper_bound"], model["x_upper_bound"], audit, key)
    audit.require(f"{key}: last certified point selected without truth", model["selected_path_index"] == len(path)-1
                  and model.get("last_path_index", len(path)-1) == len(path)-1)
    chosen = values[-1]
    for name in ("R", "X"):
        audit.close(f"{key}: final {name} independently reconstructed", arrays[name], chosen[name], atol=1e-10)
    for name in ("support_indices", "r_values", "x_values", "source_h"):
        audit.require(f"{key}: selected {name} remains original last point", model.get(name) == path[-1].get(name))
    for name, field in (("source_g_r", "source_response_r"), ("source_g_x", "source_response_x")):
        audit.require(f"{key}: final {name} equals last Source response", model.get(name) == path[-1].get(field))
    floor_records, unavailable = [], []
    for index, point in enumerate(path):
        strengths, _ = resolution_atoms(point["support_indices"], point["r_values"], point["x_values"], calibration)
        audit.require(f"{key}: path[{index}] nonleaf is zero or reaches fixed resolution", all(
            len(s) == 1 or (abs(r) <= 1e-8 and abs(x) <= 1e-8) or w+1e-10 >= calibration["tau"]
            for s, r, x, w in zip(point["support_indices"], point["r_values"], point["x_values"], strengths, strict=True)))
    evidence.audit_solver_events(folder, model, job, audit, key)
    evidence.audit_stopping(model, audit, key)
    for event in model.get("solver_events", []):
        if not event.get("has_incumbent"):
            continue
        with np.load(folder/event["primal_file"], allow_pickle=False) as archive:
            raw = {name: archive[name] for name in archive.files}
        tag = f"{key}: raw[{event['sequence']}]"
        audit_floor_domain(event, raw, n, model["terminal_groups"], calibration, audit, tag)
        supports = event.get("old_support_indices")
        if supports is None:
            unavailable.append(f"solve[{event['sequence']}]: original family annotations absent")
            continue
        audit.require(f"{tag}: annotation identifies actual atom and builder counts",
                      event.get("old_atom_count") == len(supports)
                      and event.get("is_extension") is ("z" in event["variable_slices"]))
        actual_partition = event.get("terminal_partition")
        if "z" in event["variable_slices"]:
            audit.require(f"{tag}: annotated membership partition equals current method partition",
                          actual_partition is not None and evidence.canonical_partition(actual_partition, n)
                          == evidence.canonical_partition(model["terminal_groups"], n))
            actual_partition = actual_partition if actual_partition is not None else model["terminal_groups"]
            zstart = int(event["variable_slices"]["z"][0])
            alias = [None]*n
            for group_index, group in enumerate(evidence.canonical_partition(actual_partition, n)):
                for terminal in group: alias[terminal] = zstart+group_index
            audit.require(f"{tag}: terminal membership aliases preserved", event.get("terminal_z_columns") == alias)
        else:
            actual_partition = [[i] for i in range(n)]
        audit_membership_domain(event, raw, n, actual_partition, supports, audit, tag, source_active=source_active)
        audit_residual_rows(event, raw, arrays, supports, actual_partition, audit, tag+": original equations")
        floor_records.append({"sequence": event["sequence"], "is_extension": "z" in event["variable_slices"],
                              "old_atom_count": len(supports), "z_count": len(actual_partition) if "z" in event["variable_slices"] else 0,
                              "source_h_count": len(supports)+int("z" in event["variable_slices"]) if source_active else 0})
    snapshots = []
    for target in sorted((folder/"progress").glob("*.json")):
        snapshot = evidence.read_json(target)
        audit_path96(snapshot["path"], arrays, labels, snapshot["initial_supports"], snapshot["terminal_groups"],
                     snapshot["r_upper_bound"], snapshot["x_upper_bound"], audit, f"{key}: {target.name}")
        for name, value in calibration.items():
            audit.close(f"{key}: {target.name} fixed restart calibration {name}", snapshot["rx75_calibration"].get(name), value, atol=1e-12)
        audit.require(f"{key}: {target.name} last-point selection", snapshot["selected_path_index"] == len(snapshot["path"])-1)
        snapshots.append(snapshot)
    audit.require(f"{key}: passive progress retained", bool(snapshots))
    if snapshots:
        audit.require(f"{key}: final path is last passive checkpoint", snapshots[-1]["path"] == path)
    pending, restarts = None, 0
    for index, snapshot in enumerate(snapshots):
        tag = f"{key}: checkpoint[{index}]"
        if index == 0:
            audit.require(f"{tag}: bound-two initialization retained", snapshot["event"] == "initialized"
                          and snapshot["bound_expansions"] == 0 and snapshot["r_upper_bound"] == 2.
                          and snapshot["x_upper_bound"] == 2. and len(snapshot["path"]) == 1)
        if snapshot["event"] == "bounds_expansion_requested":
            pending = snapshot
        elif snapshot["event"] == "bounds_restarted":
            restarts += 1
            audit.require(f"{tag}: pre-expansion path and attempt retained", pending is not None)
            if pending is not None:
                extension = pending["attempted_extensions"][-1]
                audit.require(f"{tag}: triggering extension certified", optimality_certificate(extension["diagnostics"]))
                for name in ("r", "x"):
                    old = float(pending[f"{name}_upper_bound"])
                    touched = np.max(extension[f"{name}_values"], initial=0.) >= .995*old
                    audit.close(f"{tag}: only touched {name} bound doubles", snapshot[f"{name}_upper_bound"], old*(2. if touched else 1.), atol=1e-12)
                audit.require(f"{tag}: bound expansion count increments", snapshot["bound_expansions"] == pending["bound_expansions"]+1)
            audit.require(f"{tag}: restart refits initial family", len(snapshot["path"]) == 1)
            pending = None
    audit.require(f"{key}: every completed restart retained", restarts == model.get("bound_expansions", 0))
    pointer_path = folder/"last_certified_model.json"
    audit.require(f"{key}: certified model JSON/NPZ pointer retained", pointer_path.is_file())
    if pointer_path.is_file():
        pointer = evidence.read_json(pointer_path)
        audit.require(f"{key}: pointer binds exactly model JSON and NPZ", len(pointer["files"]) == 2
                      and {Path(p).name for p in pointer["files"]} == {"model.json", "model.npz"})
        for relative, checksum in pointer["files"].items():
            _hash_check(folder, relative, checksum, audit, f"{key}: atomic certified pointer {relative}")
            target = (folder/relative).resolve()
            if target.is_relative_to(folder.resolve()) and target.is_file():
                if target.name == "model.json":
                    audit.require(f"{key}: final path matches atomic certified model", evidence.read_json(target)["path"] == path)
                elif target.name == "model.npz":
                    with np.load(target, allow_pickle=False) as paired:
                        audit.require(f"{key}: atomic arrays have same inventory", set(paired.files) == set(arrays))
                        for name in set(paired.files) & set(arrays):
                            audit.close(f"{key}: atomic array {name}", arrays[name], paired[name], atol=0., rtol=0.)
    metric = physical_metrics(model, arrays, network, calibration=calibration)
    _, keep = resolution_atoms(model["support_indices"], model["r_values"], model["x_values"], calibration)
    clades = {"family": chosen["family"], "active": chosen["active"],
              "resolution": {frozenset(labels[i] for i in support) for support, retained in zip(model["support_indices"], keep, strict=True)
                             if retained and 1 < len(support) < n}}
    for kind, expected in clades.items():
        audit.require(f"{key}: saved {kind} clades independently reconstructed", _sets(model[kind+"_clades"]) == expected)
    return {"metrics_recomputed": metric, "calibration_recomputed": calibration,
            "initialization": initialized, "screen": screen_detail, "resampling": resampling,
            "raw_domain_records": floor_records, "unavailable_checks": unavailable}


def _path0_raw(folder):
    """Return only a completed model's first fixed-family solve, never extension."""
    folder = Path(folder)
    if not (folder/"result.json").is_file():
        return None
    result = evidence.read_json(folder/"result.json")
    if result.get("status") not in FINAL_STATES or not result.get("has_model"):
        return None
    model = evidence.read_json(folder/"model.json")
    if not model.get("path"):
        return None
    family = model["path"][0]["support_indices"]
    for event in model.get("solver_events", []):
        if "z" in event["variable_slices"] or event.get("old_support_indices") != family:
            continue
        if not event.get("has_incumbent"):
            continue
        with np.load(folder/event["primal_file"], allow_pickle=False) as archive:
            saved = {name: archive[name] for name in archive.files}
        return event, saved, model
    return None


def audit_path0_controls(root, jobs, audit):
    """Compare fixed path0 only; grouped and ungrouped SEARCH domains differ."""
    by_condition = {(job["case"], job["variant"], job["method"]): job for job in jobs}
    rows = []
    for job in jobs:
        if job["method"] != "RNJ+WZZT":
            continue
        other = by_condition.get((job["case"], job["variant"], "RNJ+WZZT_ungrouped"))
        if other is None:
            continue
        row = {"grouped_job": job["job_id"], "ungrouped_job": other["job_id"], "checked": False,
               "scope": "same selected supports fixed path0 only; later grouped search domain can differ"}
        left, right = _path0_raw(Path(root)/"jobs"/job["job_id"]), _path0_raw(Path(root)/"jobs"/other["job_id"])
        if left is not None and right is not None:
            le, ls, lm = left
            re, rs, rm = right
            audit.require(f"{job['job_id']}: paired controls keep exactly same initial family", lm["initial_supports"] == rm["initial_supports"])
            audit_fixed_support_pair(le, ls, re, rs, audit, job["job_id"]+": grouped/ungrouped path0")
            row.update(checked=True, grouped_path0_sequence=le["sequence"], ungrouped_path0_sequence=re["sequence"],
                       grouped_path0_objective=le["objective"], ungrouped_path0_objective=re["objective"])
        else:
            row["reason"] = "both completed path0 raw models not yet available"
        rows.append(row)
    return rows


def audit_reused_baselines(root, protocol, audit):
    """Verify copied historical fits and recompute their physical scores only."""
    root = Path(root)
    jobs, binding = protocol.get("baseline_jobs", []), protocol.get("baseline_evidence")
    if not jobs:
        return [], {}
    if binding is None:
        raise ValueError("reused baselines require an explicit provenance binding")
    audit.require("all declared reused baseline jobs retained", len(jobs) == binding["baseline_denominator"])
    _hash_check(root, binding["baseline_provenance_path"], binding["baseline_provenance_sha256"], audit, "reused baseline total provenance hash")
    for relative, checksum in binding["inventory"].items():
        _hash_check(root, relative, checksum, audit, "reused baseline artifact "+relative)
    original = Path(binding["original_root"])
    audit.require("historical producer protocol hash", evidence.file_hash(original/"protocol.json") == binding["original_protocol_sha256"])
    old_protocol = evidence.read_json(original/"protocol.json")
    for relative, checksum in old_protocol["source_sha256"].items():
        _hash_check(original/"source_snapshot", relative, checksum, audit, "historical producer source "+relative)
    for name, field in (("environment.json", "original_environment_sha256"),
                        ("source_snapshot_manifest.json", "original_source_snapshot_manifest_sha256")):
        if field in binding:
            audit.require("historical "+name+" provenance", evidence.file_hash(original/name) == binding[field])
    audit.require("historical fit data/baseline scope fixed", old_protocol["samples_per_fit"] == 96
                  and old_protocol["validation_samples"] == 0 and old_protocol["rx75_weight"] == .75
                  and old_protocol["tree_tolerance_factor"] == .16)
    rows, calibrations = [], {}
    by_condition = {(job["case"], job["variant"], job["method"]): job for job in jobs}
    for job in jobs:
        key = job["job_id"]
        folder = root/job["directory"]
        row = dict(job) | {"role": "baseline_reuse", "audited": False, "status": "pending", "has_model": False,
                           "metrics_recomputed": None}
        rows.append(row)
        try:
            if not (folder/"result.json").is_file():
                continue
            result = evidence.read_json(folder/"result.json")
            row.update(status=result.get("status"), has_model=result.get("has_model"), stop_reason=result.get("stop_reason"),
                       compute_seconds=result.get("compute_seconds", result.get("elapsed_seconds")))
            if row["status"] not in FINAL_STATES:
                continue
            row["audited"] = True
            provenance = evidence.read_json(folder/"reuse_provenance.json")
            producer = Path(provenance["original_directory"])
            old_result = evidence.read_json(producer/"result.json")
            old_model = evidence.read_json(producer/"model.json")
            audit.require(f"{key}: reused result belongs to original producer", result.get("protocol_sha256") == binding["original_protocol_sha256"]
                          and result.get("producer_protocol_sha256") == binding["original_protocol_sha256"])
            for relative, checksum in provenance["original_file_sha256"].items():
                _hash_check(producer, relative, checksum, audit, f"{key}: original saved evidence {relative}")
            audit.require(f"{key}: NPZ is an exact historical byte copy", evidence.file_hash(folder/"model.npz")
                          == provenance["reused_npz_sha256"] == provenance["original_file_sha256"]["model.npz"])
            audit.require(f"{key}: baseline is reused without source refit or reinference", provenance.get("estimation_performed") is False
                          and provenance.get("source_response_refitted") is False and provenance.get("raw_tree_reinferred") is False)
            audit.require(f"{key}: historical fitting time preserved", row["compute_seconds"] == old_result.get("compute_seconds", old_result["elapsed_seconds"])
                          and result.get("elapsed_seconds") == old_result["elapsed_seconds"])
            audit.require(f"{key}: postprocessing time recorded separately", np.isfinite(result["postprocess_seconds"]) and result["postprocess_seconds"] >= 0.)
            if not row["has_model"]:
                audit.require(f"{key}: missing historical model retains null scores", row["status"] != "complete"
                              and all(value is None for value in (result.get("metrics") or {}).values()))
                continue
            model = evidence.read_json(folder/"model.json")
            with np.load(folder/"model.npz", allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            labels = list(map(int, model["terminal_labels"]))
            inputs = root/"inputs"/job["case"]/job["variant"]
            audit_training96(arrays, inputs, labels, audit, key+": reused inputs")
            for relative, checksum in provenance["input_sha256"].items():
                _hash_check(inputs, relative, checksum, audit, f"{key}: same reused/new worker input {relative}")
            audit.require(f"{key}: historical raw tree preserved", model["tree"] == old_model["tree"])
            audit.require(f"{key}: baseline matrix origin is unrestricted Source-FWL regression", model.get("matrix_origin") == "source_adjusted_ordered_regression"
                          and model.get("support_indices") is None and model.get("source_h") is None)
            u = np.column_stack((arrays["P_source"], arrays["Q_source"]))
            residual = arrays["Y"]-arrays["P"]@arrays["R"].T-arrays["Q"]@arrays["X"].T
            g = np.vstack((model["source_g_r"], model["source_g_x"]))
            audit.close(f"{key}: unrestricted nuisance G and minimum-norm gauge", g, np.linalg.lstsq(u, residual, rcond=None)[0], atol=1e-8)
            prediction = arrays["P"]@arrays["R"].T+arrays["Q"]@arrays["X"].T+u@g
            audit.close(f"{key}: original 96-row training loss", model["train_mae"], float(np.abs(arrays["Y"]-prediction).mean()), atol=1e-12)
            diagnostics = model["regression_diagnostics"]
            audit.require(f"{key}: Source location and path were not used", diagnostics.get("source_location_used") is False
                          and diagnostics.get("source_path_constraints") is False and diagnostics.get("intercept_added") is False)
            nj_job = by_condition[(job["case"], job["variant"], "NJ")]
            nj_folder = root/nj_job["directory"]
            with np.load(nj_folder/"model.npz", allow_pickle=False) as nj:
                calibration = independent_calibration(nj["R"], nj["X"])
                calibrations[(job["case"], job["variant"])] = (nj["R"].copy(), nj["X"].copy())
                audit.close(f"{key}: same shared NJ/RNJ dense R", arrays["R"], nj["R"], atol=0., rtol=0.)
                audit.close(f"{key}: same shared NJ/RNJ dense X", arrays["X"], nj["X"], atol=0., rtol=0.)
            for name, value in calibration.items():
                audit.close(f"{key}: reused fixed calibration {name}", model["rx75_calibration"][name], value, atol=1e-12)
            metric = physical_metrics(model, arrays, root/"evaluation_network"/job["case"], calibration=calibration)
            for name, value in metric.items():
                if name == "truth_used_for_selection":
                    continue
                audit.close(f"{key}: reused physical score {name}", result["metrics"].get(name), value, atol=1e-10)
            audit.close(f"{key}: reused primary resolution F1", result["metrics"].get("clade_f1"), metric["resolution_clade_f1"], atol=1e-10)
            row.update(metrics_recomputed=metric, calibration_recomputed=calibration, estimation_performed=False,
                       postprocess_seconds=result["postprocess_seconds"], matrix_origin=model["matrix_origin"])
        except Exception as error:
            audit.require(f"{key}: reused evidence independently readable", False, error=f"{type(error).__name__}: {error}")
            row["audit_error"] = f"{type(error).__name__}: {error}"
    return rows, calibrations


def audit_experiment(run_directory, dataset_directory=None, *, only_jobs=None):
    """Read final jobs in a declared catalog; pending jobs keep the denominator."""
    root, audit, rows = Path(run_directory).resolve(), Audit(), []
    protocol = evidence.read_json(root/"protocol.json")
    jobs = protocol["jobs"]
    keys = [job["job_id"] for job in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate planned job IDs")
    if only_jobs is not None and not set(only_jobs) <= set(keys):
        raise ValueError("unknown selected audit job")
    selected = set(keys if only_jobs is None else only_jobs)
    signature = evidence.file_hash(root/"protocol.json")
    audit.require("all declared new jobs retain comparison denominator", len(jobs) == protocol["expected_jobs"])
    for field, value in {"training_samples": 96, "validation_samples": 0, "validation_supplied": False,
                         "scenario_count": 1, "samples_per_scenario": 96, "truth_used_for_selection": False,
                         "resolution_absolute_tolerance": 1e-10}.items():
        audit.require(f"protocol {field}", protocol.get(field) == value)
    audit.require("predeclared relaxed block policy", protocol.get("screen_parameters") == SCREEN_POLICY | {"seed": 20260903})
    settings = protocol.get("measurement_settings", {})
    for name in ("v_noise_rel", "root_meter_noise_rel"):
        audit.require(f"original measurement precision {name}", settings.get(name) == .0002)
    audit.require("frozen dataset manifest", evidence.file_hash(root/"dataset_manifest.json") == protocol["dataset_manifest_sha256"])
    manifest = evidence.read_json(root/"dataset_manifest.json")
    for relative, checksum in protocol.get("source_sha256", {}).items():
        _hash_check(root/"source_snapshot", relative, checksum, audit, "frozen producer "+relative)
    runtime = Path(protocol["runtime_directory"])
    audit.require("frozen runtime manifest", evidence.file_hash(runtime/"runtime_manifest.json") == protocol["runtime_manifest_sha256"])
    runtime_manifest = evidence.read_json(runtime/"runtime_manifest.json")
    for field, folder in (("runtime_sha256", runtime/"rnj_wzzt"), ("source_sha256", runtime/"source_snapshot/rnj_wzzt")):
        for relative, checksum in runtime_manifest[field].items():
            _hash_check(folder, relative, checksum, audit, "frozen runtime "+field+" "+relative)
    if dataset_directory is not None:
        dataset = Path(dataset_directory).resolve()
        audit.require("original data manifest unchanged", evidence.file_hash(dataset/"manifest.json") == protocol["dataset_manifest_sha256"])
        for relative, checksum in manifest["file_sha256"].items():
            _hash_check(dataset, relative, checksum, audit, "original data "+relative)
    for job in jobs:
        key = job["job_id"]
        folder = root/"jobs"/key
        row = dict(job) | {"audited": False, "status": "running" if (folder/"execution.json").exists() else "pending",
                           "has_model": False, "metrics_recomputed": None}
        rows.append(row)
        if not (folder/"result.json").exists():
            continue
        try:
            result = evidence.read_json(folder/"result.json")
            row.update(status=result.get("status"), has_model=result.get("has_model"), stop_reason=result.get("stop_reason"),
                       compute_seconds=result.get("compute_seconds", result.get("elapsed_seconds")))
            if key not in selected or row["status"] not in FINAL_STATES:
                continue
            row["audited"] = True
            audit.require(f"{key}: final result binds frozen protocol", result.get("protocol_sha256") == signature)
            audit.require(f"{key}: actual full method time retained", row["compute_seconds"] is not None
                          and np.isfinite(row["compute_seconds"]) and row["compute_seconds"] >= 0)
            audit.require(f"{key}: budget fixed before fitting", job.get("per_solve_seconds") == 6000. and job.get("method_wall_seconds") == 30000.)
            input_directory = root/job.get("input_relative_directory", job.get("input_directory", f"inputs/{job['case']}/{job['variant']}"))
            network = root/job.get("evaluation_network_directory", f"evaluation_network/{job['case']}")
            inventory = evidence.read_json(folder/"inputs_hash.json")["file_sha256"]
            audit.require(f"{key}: complete observed/source-only input whitelist", set(inventory) == evidence.INPUT_NAMES
                          and {p.relative_to(input_directory).as_posix() for p in input_directory.rglob("*") if p.is_file()} == evidence.INPUT_NAMES)
            for relative, checksum in inventory.items():
                _hash_check(input_directory, relative, checksum, audit, f"{key}: original worker input {relative}")
                audit.require(f"{key}: input matches original generation manifest {relative}",
                              manifest["file_sha256"].get(f"{job['case']}/{job['variant']}/{relative}") == checksum)
            network_prefix = f"{job['case']}/network/"
            for relative, checksum in manifest["file_sha256"].items():
                if relative.startswith(network_prefix):
                    _hash_check(network, relative[len(network_prefix):], checksum, audit, f"{key}: physical network {relative}")
            if not row["has_model"]:
                audit.require(f"{key}: unavailable model retains failure with null metrics", row["status"] != "complete"
                              and all(value is None for value in (result.get("metrics") or {}).values()))
                continue
            model = evidence.read_json(folder/"model.json")
            with np.load(folder/"model.npz", allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            evidence.audit_runtime_import(evidence.read_json(folder/"runtime_import.json"), runtime, audit, key,
                                          completed=result.get("status") == "complete")
            details = audit_model96(folder, model, arrays, job, input_directory, network, audit,
                                    expected_core=protocol.get("core_parameters_base"))
            row.update(details)
            for kind in ("family", "active", "resolution"):
                for metric in ("clade_f1", "precision", "recall"):
                    name = kind+"_"+metric
                    audit.close(f"{key}: independent physical metric {name}", result.get("metrics", {}).get(name), details["metrics_recomputed"][name], atol=1e-10)
            for name in ("r_relative_error", "x_relative_error"):
                audit.close(f"{key}: independent physical matrix metric {name}", result.get("metrics", {}).get(name), details["metrics_recomputed"][name], atol=1e-10)
            audit.close(f"{key}: primary topology score uses fixed resolution", result.get("metrics", {}).get("clade_f1"), details["metrics_recomputed"]["resolution_clade_f1"], atol=1e-10)
        except Exception as error:
            audit.require(f"{key}: saved evidence independently readable", False, error=f"{type(error).__name__}: {error}")
            row["audit_error"] = f"{type(error).__name__}: {error}"
    paired_controls = audit_path0_controls(root, [job for job in jobs if job["job_id"] in selected], audit)
    baseline_rows, references = audit_reused_baselines(root, protocol, audit)
    for row in rows:
        condition = row["case"], row["variant"]
        if row.get("calibration_recomputed") is None or condition not in references:
            continue
        with np.load(root/"jobs"/row["job_id"]/"model.npz", allow_pickle=False) as arrays:
            for name, reference in zip(("R", "X"), references[condition], strict=True):
                audit.close(f"{row['job_id']}: new calibration equals reused NJ {name}", arrays["calibration_"+name], reference, atol=1e-9, rtol=1e-8)
    combined = rows+baseline_rows
    return audit.result() | {"expected_new_jobs": len(jobs), "expected_baseline_jobs": len(baseline_rows),
                             "expected_jobs": len(combined), "selected_new_jobs": len(selected), "rows": combined,
                             "status_counts": dict(Counter(row["status"] for row in combined)),
                             "new_status_counts": dict(Counter(row["status"] for row in rows)),
                             "baseline_status_counts": dict(Counter(row["status"] for row in baseline_rows)),
                             "audited_final_jobs": sum(row["audited"] for row in combined),
                             "audited_final_new_jobs": sum(row["audited"] for row in rows),
                             "audited_reused_baselines": sum(row["audited"] for row in baseline_rows),
                             "run_complete": all(row["status"] in FINAL_STATES for row in combined),
                             "fixed_path0_compression_controls": paired_controls,
                             "protocol_sha256": signature, "truth_used_for_selection": False,
                             "baseline_audit_scope": "historical fits reused without refit; 32-baseline and 64-new-job denominators reported separately and together"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--only-job", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("audit output already exists; use a fresh path")
    report = audit_experiment(args.run_dir, args.dataset_dir, only_jobs=args.only_job)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
