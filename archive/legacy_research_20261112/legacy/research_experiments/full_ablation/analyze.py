"""Independent, post-freeze scoring of the R/T/E full-factorial experiment.

Only this analysis process opens truth and test files. Saved coefficients are
audited with an independent NumPy response calculation and SciPy/HiGHS family
LP. Pending, failed and invalid runs retain their planned denominators.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import combinations, permutations, product
import json
from math import factorial
from pathlib import Path
import shutil
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linprog, OptimizeWarning
from scipy.sparse import csr_matrix, eye, hstack, vstack

from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth


ROOT = Path(__file__).resolve().parents[2]
FACTORS = ("R", "T", "E")
FIELDS = ("seed", "tail", "exit")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def markdown(frame):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}" if np.isfinite(value) else "unavailable"
        return str(value).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |",
                      "|" + "---|" * len(frame.columns),
                      *["| " + " | ".join(cell(v) for v in row) + " |"
                        for row in frame.itertuples(index=False, name=None)]])


def method_map(protocol):
    methods = protocol["methods"]
    assert len(methods) == 8, "The primary factorial must contain exactly eight configurations."
    mapping = {tuple(int(bool(m[k])) for k in FIELDS): m["id"] for m in methods}
    assert set(mapping) == set(product((0, 1), repeat=3)) and len(set(mapping.values())) == 8
    return mapping


def load_split(path, N=None):
    """Read the original three-day arrays without using estimator preprocessing."""
    with np.load(Path(path), allow_pickle=False) as data:
        indices = np.arange(96) if N is None else np.arange(0, 96, 288 // int(N))
        labels = data["labels"].astype(int).tolist()
        arrays = [np.asarray(data[k][:, indices], float).reshape(-1, len(labels)) for k in ("P", "Q", "V")]
        root = np.asarray(data["root"][:, indices], float).reshape(-1)
    p, q, v = arrays
    y = root[:, None] ** 2 - v ** 2
    assert p.shape == q.shape == y.shape and all(np.isfinite(a).all() for a in (p, q, y))
    return dict(p=p, q=q, y=y, labels=labels, n=len(labels))


def family(point, n):
    sets = tuple(tuple(int(i) for i in C) for C in point["supports"])
    assert len(sets) == len(set(sets)) and all(C and C == tuple(sorted(set(C))) for C in sets)
    assert all(0 <= i < n for C in sets for i in C)
    for C, D in combinations(map(set, sets), 2):
        assert not C & D or C <= D or D <= C, "Nonlaminar saved support family."
    r, x = np.asarray(point["r"], float), np.asarray(point["x"], float)
    assert len(r) == len(x) == len(sets) and np.isfinite(r).all() and np.isfinite(x).all()
    assert np.min(np.r_[r, x], initial=0.) >= -1e-9 and np.max(np.r_[r, x], initial=0.) <= 2. + 1e-9
    return sets


def matrices(point, n):
    z = np.asarray([[i in C for i in range(n)] for C in point["supports"]], float)
    return (z.T * np.asarray(point["r"])) @ z, (z.T * np.asarray(point["x"])) @ z


def point_loss(split, point):
    R, X = matrices(point, split["n"])
    return float(np.mean(np.abs(split["y"] - split["p"] @ R.T - split["q"] @ X.T)))


def independent_profile(split, supports):
    """Continuous L1 optimum for one fixed family, independent of native LP code."""
    p, q, y, n = (split[k] for k in ("p", "q", "y", "n"))
    z = np.asarray([[i in C for i in range(n)] for C in supports], float)
    features = csr_matrix(np.column_stack([((a @ row)[:, None] * row).ravel() for a in (p, q) for row in z]))
    m, k = features.shape
    I = eye(m, format="csr")
    A = vstack((hstack((features, -I)), hstack((-features, -I))), format="csr")
    b = np.r_[y.ravel(), -y.ravel()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        result = linprog(np.r_[np.zeros(k), np.ones(m) / m], A_ub=A, b_ub=b,
                         bounds=[(0, 2)] * k + [(0, None)] * m, method="highs", options={"threads": 1})
    if not result.success:
        raise RuntimeError(f"Independent family LP: {result.message}")
    violation = float(np.max(A @ result.x - b, initial=0.))
    assert violation <= 1e-7
    return dict(objective=float(result.fun), violation=violation, status=int(result.status))


def score(point, splits, truth, labels):
    n = len(labels)
    predicted = {frozenset(labels[i] for i in C) for C, r, x in zip(point["supports"], point["r"], point["x"])
                 if 1 < len(C) < n and max(r, x) > 1e-9}
    target = truth["clades"]
    tp, fp, fn = len(predicted & target), len(predicted - target), len(target - predicted)
    R, X = matrices(point, n)
    return dict(clade_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
                precision=tp / (tp + fp) if tp + fp else float(not target),
                recall=tp / (tp + fn) if tp + fn else 1., exact=predicted == target,
                matched_clades=tp, predicted_clades=len(predicted), true_clades=len(target), internal=len(predicted),
                R_relative=float(np.linalg.norm(R - truth["R"]) / np.linalg.norm(truth["R"])),
                X_relative=float(np.linalg.norm(X - truth["X"]) / np.linalg.norm(truth["X"])),
                **{name + "_mae": point_loss(data, point) for name, data in splits.items()})


def solver_records(result):
    trajectory = result.get("trajectory", {})
    for event in trajectory.get("attempts", trajectory.get("events", [])):
        for record in event.get("trials", [event]):
            if record.get("status") is not None:
                yield dict(stage=event.get("kind", "unknown"), **record)
    event = result.get("exit_event", {})
    if event and event.get("status") is not None:
        yield dict(stage="final_exit", **event)


def analyze_job(args):
    output, job = args
    output = Path(output)
    protocol, catalog = read(output / "protocol.json"), read(output / "catalog.json")
    meta = catalog[job]
    common_path = output / "common" / job / "common.json"
    common = read(common_path)
    assert {name: sha(meta[name]) for name in ("train", "validation", "test")} == protocol["input_hashes"][job]
    splits = {name: load_split(meta[name], meta["N"] if name == "train" else None)
              for name in ("train", "validation", "test")}
    labels = splits["train"]["labels"]
    assert common["labels"] == labels and all(d["labels"] == labels for d in splits.values())
    truth = physical_truth(Path(meta["truth_network"]), labels)
    header = dict(job=job, **{k: meta[k] for k in ("panel", "case", "bundle", "N", "voltage_sigma", "condition")})
    profiles, rows, events, errors = {}, [], [], []
    checks, raw_hashes = 0, {}
    for config in protocol["methods"]:
        method = config["id"]
        file = output / "jobs" / job / method / "result.json"
        error = file.with_name("error.json")
        row = dict(**header, method=method, **{k: bool(config[k]) for k in FIELDS}, status="pending", available=False)
        if not file.exists():
            if error.exists():
                row["status"] = "failed"
                errors.append(dict(job=job, method=method, kind="worker_error", detail=read(error)))
            rows.append(row)
            continue
        raw_hashes[method] = sha(file)
        try:
            result = read(file)
            assert result["job"] == job and result["method"] == method
            assert result["input_hashes"] == {k: protocol["input_hashes"][job][k] for k in ("train", "validation")}
            assert result.get("truth_used", False) is False and result.get("test_opened", False) is False
            assert result.get("protocol_sha256", sha(output / "protocol.json")) == sha(output / "protocol.json")
            assert not result.get("LP_failures", []) and not result.get("post_LP_failures", [])
            if result.get("common_sha256") is not None:
                assert result["common_sha256"] == sha(common_path)
            timing = result.get("timing", result)
            fresh, shared, total = (float(timing[k]) for k in ("fresh_seconds", "common_seconds", "total_seconds"))
            assert fresh >= 0 and shared >= 0 and abs(fresh + shared - total) < 1e-7
            selected = result.get("selected", result.get("choice"))
            assert selected is not None and selected.get("source", "own_path") in {"own_path", "singleton", "baseline", "path"}
            point = selected["point"]
            supports = family(point, len(labels))
            train = point_loss(splits["train"], point)
            assert abs(train - point["objective"]) <= 1e-8
            key = tuple(sorted(supports))
            if key not in profiles:
                profiles[key] = independent_profile(splits["train"], key)
            assert abs(profiles[key]["objective"] - train) <= 2e-7
            for saved in result.get("trajectory", {}).get("path", []):
                family(saved, len(labels))
                assert abs(point_loss(splits["train"], saved) - saved["objective"]) <= 1e-8
                checks += 3
            metrics = score(point, splits, truth, labels)
            assert abs(metrics["validation_mae"] - float(selected["validation"])) <= 1e-10
            calls = list(solver_records(result))
            events.extend(dict(**header, method=method, **record) for record in calls)
            zeros = [e for e in result.get("trajectory", {}).get("zero_events", []) if e.get("accepted")]
            for e in zeros:
                assert e["after"] <= e["before"] + e.get("loss_tolerance", 1e-10) + 1e-12
            row.update(status="verified", available=True, selected_source=selected.get("source", "own_path"),
                       **metrics, fresh_seconds=fresh, common_seconds=shared, total_seconds=total,
                       cap_exceeded=bool(result.get("total_cap_exceeded", False)),
                       MIPs=len(calls), gap_met=sum(e.get("status") == 2 for e in calls),
                       MIP_timeouts=sum(e.get("status") == 9 for e in calls),
                       zero_release_events=len(zeros), zero_released_sets=sum(len(e["removed"]) for e in zeros),
                       stop_reason=result.get("trajectory", {}).get("stop_reason"),
                       solver_gaps_all_met=all(e.get("status") == 2 for e in calls))
            checks += 12
        except Exception as exc:
            row.update(status="invalid", error=f"{type(exc).__name__}: {exc}")
            errors.append(dict(job=job, method=method, kind="verification_error", detail=row["error"]))
        rows.append(row)
    return dict(job=job, rows=rows, solver=events, errors=errors, checks=checks, independent_LPs=len(profiles),
                protocol_sha256=sha(output / "protocol.json"), analyzer_sha256=sha(Path(__file__)), raw_hashes=raw_hashes)


def aggregate(frame, keys):
    rows = []
    for key, group in frame.groupby(keys, sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        ok = group[group.status == "verified"]
        exact = int(ok.exact.sum()) if len(ok) else 0
        rows.append(dict(zip(keys, key)) | dict(planned=len(group), returned=int(group.status.isin(("verified", "invalid")).sum()),
                    verified=len(ok), pending=int(group.status.eq("pending").sum()), failed=int(group.status.eq("failed").sum()),
                    invalid=int(group.status.eq("invalid").sum()), F1=float(ok.clade_f1.mean()) if len(ok) else np.nan,
                    exact=exact, recovery_rate_completed=exact / len(ok) if len(ok) else np.nan,
                    exact_fraction_planned=exact / len(group), complete=len(ok) == len(group),
                    total_seconds=float(ok.total_seconds.sum()) if len(ok) else np.nan,
                    median_seconds=float(ok.total_seconds.median()) if len(ok) else np.nan,
                    MIPs=int(ok.MIPs.sum()) if len(ok) else 0, MIP_timeouts=int(ok.MIP_timeouts.sum()) if len(ok) else 0,
                    cap_exceeded=int(ok.cap_exceeded.sum()) if len(ok) else 0))
    return pd.DataFrame(rows)


def factorial_tables(frame, protocol):
    mapping = method_map(protocol)
    good = frame[frame.status == "verified"].set_index(["job", "method"])
    metadata = frame.drop_duplicates("job").set_index("job")
    marginals, interactions, shapley, orders = [], [], [], []
    for job in protocol["jobs"]:
        available = {bits: good.loc[(job, method)] for bits, method in mapping.items() if (job, method) in good.index}
        header = dict(job=job, panel=metadata.loc[job, "panel"], case=metadata.loc[job, "case"])
        for factor, letter in enumerate(FACTORS):
            for bits in product((0, 1), repeat=3):
                if bits[factor]:
                    continue
                right = tuple(1 if i == factor else v for i, v in enumerate(bits))
                if bits not in available or right not in available:
                    continue
                a, b = available[bits], available[right]
                marginals.append(dict(**header, factor=letter, before_method=mapping[bits], after_method=mapping[right],
                    context="".join(f"{FACTORS[i]}{bits[i]}" for i in range(3) if i != factor),
                    delta_F1=float(b.clade_f1 - a.clade_f1), delta_exact=int(b.exact) - int(a.exact),
                    delta_seconds=float(b.total_seconds - a.total_seconds), new_exact=bool(b.exact and not a.exact),
                    lost_exact=bool(a.exact and not b.exact)))
        for i, j in combinations(range(3), 2):
            k = next(v for v in range(3) if v not in (i, j))
            for other in (0, 1):
                corners = [tuple((a if z == i else b if z == j else other) for z in range(3))
                           for a, b in ((0, 0), (1, 0), (0, 1), (1, 1))]
                if not all(bits in available for bits in corners):
                    continue
                data = [available[bits] for bits in corners]
                interactions.append(dict(**header, factors=FACTORS[i] + FACTORS[j], context=f"{FACTORS[k]}{other}",
                    interaction_F1=float(data[3].clade_f1 - data[1].clade_f1 - data[2].clade_f1 + data[0].clade_f1),
                    interaction_exact=int(data[3].exact) - int(data[1].exact) - int(data[2].exact) + int(data[0].exact),
                    interaction_seconds=float(data[3].total_seconds - data[1].total_seconds - data[2].total_seconds + data[0].total_seconds)))
        if len(available) != 8:
            continue
        job_shapley = []
        for factor, letter in enumerate(FACTORS):
            values = dict(F1=0., exact=0., seconds=0.)
            for bits in product((0, 1), repeat=3):
                if bits[factor]:
                    continue
                right = tuple(1 if i == factor else v for i, v in enumerate(bits))
                a, b = available[bits], available[right]
                k = sum(bits)
                weight = factorial(k) * factorial(2-k) / factorial(3)
                values["F1"] += weight * float(b.clade_f1-a.clade_f1)
                values["exact"] += weight * (int(b.exact)-int(a.exact))
                values["seconds"] += weight * float(b.total_seconds-a.total_seconds)
            record = dict(**header, factor=letter, **{"shapley_" + k: v for k, v in values.items()})
            shapley.append(record)
            job_shapley.append(record)
        base, full = available[(0, 0, 0)], available[(1, 1, 1)]
        assert abs(sum(r["shapley_F1"] for r in job_shapley) - (full.clade_f1-base.clade_f1)) < 1e-10
        assert abs(sum(r["shapley_exact"] for r in job_shapley) - (int(full.exact)-int(base.exact))) < 1e-10
        assert abs(sum(r["shapley_seconds"] for r in job_shapley) - (full.total_seconds-base.total_seconds)) < 1e-6
        for order in permutations(range(3)):
            bits = [0, 0, 0]
            name = "→".join(FACTORS[i] for i in order)
            for step in range(4):
                row = available[tuple(bits)]
                orders.append(dict(**header, order=name, step=step, added="base" if not step else FACTORS[order[step-1]],
                                   method=mapping[tuple(bits)], F1=float(row.clade_f1), exact=int(row.exact), seconds=float(row.total_seconds)))
                if step < 3:
                    bits[order[step]] = 1
    marginals, interactions, shapley, orders = (pd.DataFrame(v) for v in (marginals, interactions, shapley, orders))
    marginal_summary = []
    if len(marginals):
        for (letter, context), group in marginals.groupby(["factor", "context"], sort=False):
            marginal_summary.append(dict(factor=letter, context=context, paired_inputs=len(group), planned_pairs=len(protocol["jobs"]),
                delta_F1=float(group.delta_F1.mean()), delta_exact=int(group.delta_exact.sum()),
                delta_recovery_rate=float(group.delta_exact.mean()), delta_seconds=float(group.delta_seconds.sum()),
                wins=int((group.delta_F1 > 1e-10).sum()), ties=int((abs(group.delta_F1) <= 1e-10).sum()),
                losses=int((group.delta_F1 < -1e-10).sum()), new_exact=int(group.new_exact.sum()), lost_exact=int(group.lost_exact.sum())))
    interaction_summary = interactions.groupby(["factors", "context"]).agg(paired_inputs=("job", "size"),
        interaction_F1=("interaction_F1", "mean"), interaction_exact=("interaction_exact", "sum"),
        interaction_seconds=("interaction_seconds", "sum")).reset_index() if len(interactions) else pd.DataFrame()
    shapley_summary = shapley.groupby("factor").agg(paired_inputs=("job", "size"), shapley_F1=("shapley_F1", "mean"),
        shapley_exact=("shapley_exact", "sum"), shapley_recovery_rate=("shapley_exact", "mean"),
        shapley_seconds=("shapley_seconds", "sum")).reset_index() if len(shapley) else pd.DataFrame()
    order_summary = orders.groupby(["order", "step", "added", "method"], sort=False).agg(paired_inputs=("job", "size"),
        F1=("F1", "mean"), exact=("exact", "sum"), recovery_rate=("exact", "mean"), seconds=("seconds", "sum")).reset_index() if len(orders) else pd.DataFrame()
    return dict(marginal_inputs=marginals, marginal_effects=pd.DataFrame(marginal_summary), interaction_inputs=interactions,
                interactions=interaction_summary, shapley_inputs=shapley, shapley=shapley_summary,
                addition_order_inputs=orders, addition_orders=order_summary)


def analyze(output, *, workers=4, render=True):
    started = perf_counter()
    output = Path(output).resolve()
    protocol = read(output / "protocol.json")
    method_map(protocol)
    for name, digest in protocol.get("source_hashes", {}).items():
        assert sha(output / "source" / name) == digest, name
    dest = output / "analysis"
    dest.mkdir(parents=True, exist_ok=True)
    results, todo = {}, []
    analyzer_hash = sha(Path(__file__))
    for job in protocol["jobs"]:
        path = dest / "jobs" / job / "verified.json"
        raw = {m["id"]: sha(output / "jobs" / job / m["id"] / "result.json") for m in protocol["methods"]
               if (output / "jobs" / job / m["id"] / "result.json").exists()}
        cached = read(path) if path.exists() else None
        # Failed/pending states can change without an existing result hash changing.
        if cached and cached.get("analyzer_sha256") == analyzer_hash and cached.get("protocol_sha256") == sha(output / "protocol.json") and cached.get("raw_hashes") == raw and all(r["status"] == "verified" for r in cached["rows"]):
            results[job] = cached
        else:
            todo.append((str(output), job))
    if workers == 1:
        generated = map(analyze_job, todo)
        for record in generated:
            results[record["job"]] = record
            write(dest / "jobs" / record["job"] / "verified.json", record)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for record in executor.map(analyze_job, todo):
                results[record["job"]] = record
                write(dest / "jobs" / record["job"] / "verified.json", record)
                if len(results) % 10 == 0:
                    print(f"Audited {len(results)}/{len(protocol['jobs'])} inputs", flush=True)
    records = [results[job] for job in protocol["jobs"]]
    frame = pd.DataFrame([row for r in records for row in r["rows"]])
    planned = len(protocol["jobs"]) * 8
    assert len(frame) == planned
    # Preserve stable metric columns even when no job has returned yet.
    for name in ("clade_f1", "exact", "total_seconds", "median_seconds", "MIPs", "MIP_timeouts", "cap_exceeded"):
        if name not in frame:
            frame[name] = np.nan
    tables = dict(per_model=frame, summary=aggregate(frame, ["method", *FIELDS]),
                  cases=aggregate(frame, ["case", "method", *FIELDS]), panels=aggregate(frame, ["panel", "method", *FIELDS]),
                  conditions=aggregate(frame, ["panel", "voltage_sigma", "N", "method", *FIELDS]),
                  stress=aggregate(frame[frame.panel == "stress"], ["condition", "method", *FIELDS]))
    tables.update(factorial_tables(frame, protocol))
    solver = [e for record in records for e in record["solver"]]
    errors = [e for record in records for e in record["errors"]]
    tables["failures"] = pd.DataFrame(errors)
    tables["solver"] = pd.DataFrame([dict(job=e["job"], method=e["method"], stage=e["stage"], status=e.get("status"),
        seconds=e.get("seconds"), relative_gap=e.get("relative_gap", e.get("gap")),
        bound=e.get("dual_bound", e.get("bound", e.get("final_bound"))),
        incumbent=e.get("objective", e.get("final_objective"))) for e in solver])
    for name, table in tables.items():
        table.to_csv(dest / (name + ".csv"), index=False, lineterminator="\n")
    returned = int(frame.status.isin(("verified", "invalid")).sum())
    verified = int(frame.status.eq("verified").sum())
    complete = verified == planned
    budget_exceeded = int(frame.loc[frame.status.eq("verified"), "cap_exceeded"].sum())
    receipt = dict(passed=complete and not errors and not budget_exceeded, complete=complete, planned=planned, returned=returned, verified=verified,
                   failed=int(frame.status.eq("failed").sum()), invalid=int(frame.status.eq("invalid").sum()),
                   pending=int(frame.status.eq("pending").sum()), independent_LPs=sum(r["independent_LPs"] for r in records),
                   checks=sum(r["checks"] for r in records), MIPs=len(solver),
                   gap_met=sum(e.get("status") == 2 for e in solver), MIP_timeouts=sum(e.get("status") == 9 for e in solver),
                   budget_exceeded=budget_exceeded, analyzer_sha256=analyzer_hash, protocol_sha256=sha(output / "protocol.json"),
                   no_complete_tree_RNJ_output_fallback=True, source_hashes_verified=True,
                   all_planned_denominators_preserved=True, elapsed_seconds=perf_counter() - started,
                   order_interpretation="module addition across configurations, not temporal execution-stage permutations")
    write(dest / "verification.json", receipt)
    delivery = ROOT / "docs" / output.name
    delivery.mkdir(parents=True, exist_ok=True)
    for file in dest.iterdir():
        if file.is_file():
            shutil.copy2(file, delivery / file.name)
    if render:
        from .figures import render_figures
        render_figures(delivery, tables, protocol)
    summary = tables["summary"]
    text = "# R/T/E完整因子消融\n\n"
    text += f"核验完成{verified}/{planned}；返回{returned}/{planned}，失败{receipt['failed']}，待完成{receipt['pending']}，核验无效{receipt['invalid']}。\n\n"
    text += "R为结构RNJ指导（预选、候选池与热启动），T为≤5直属块联合末端，E为一次最终结构退出。及时清零固定开启；所有方法共享损失尺度f0、几何标定、计入成本的公共前端、惩罚与输出规则。基座是‘WZZT（公共标定固定）’，并非早期无惩罚纯WZZT。最终只比较自身路径与单点输出，不加入完整RNJ树备用。\n\n"
    text += "## 配置成绩与时间\n\n" + markdown(summary) + "\n\n"
    text += "未完成情况下，F1与完成恢复率仅描述已核验子集；exact_fraction_planned是已确认完整恢复数除以计划输入数，非未完成算法的最终恢复率。累计耗时是逐输入算法墙钟时间之和，不是并行批次等待时间。\n\n"
    text += "## 条件边际贡献\n\n" + markdown(tables["marginal_effects"]) + "\n\n"
    text += "每一行比较其余两个模块开关固定时，加入一个模块的成对差异。模块效应可能为负，加入顺序不同也可能出现不同收益。\n\n"
    text += "## 所有加入顺序与平均贡献\n\n" + markdown(tables["shapley"]) + "\n\n"
    text += "Shapley列为六种模块加入顺序的平均边际贡献；三个贡献之和等于完整组合减去基座的成绩差。它是本批固定配置结果的分解，不是拓扑正确性的概率。这里的‘顺序’指从基座逐项加入模块的实验比较顺序，RNJ预选、末端重构、最终退出在单次算法里的执行次序仍按定义保持。仅全部八配置均核验的同一输入进入顺序和Shapley配对，配对分母另列。\n\n"
    text += "## 交互作用\n\n" + markdown(tables["interactions"]) + "\n\n"
    text += "交互项为f(AB)-f(A)-f(B)+f(∅)，按第三模块固定的两个条件分别计算。正值表示这两个模块结合的增益超过单独增益相加；其符号依指标及条件而定。\n\n"
    text += "MILP状态只对应各候选域子问题。输入哈希、权重边界、层级相容、原始量测损失和独立固定家族L1最优值已核验；完整恢复只按真实非平凡后代集合评分。原始真拓扑与测试数据只在本后分析中打开，没有参与训练和验证选模。\n"
    (delivery / "RESULTS.md").write_text(text, encoding="utf-8", newline="\n")
    write(delivery / "manifest.json", {"files": {f.name: dict(bytes=f.stat().st_size, sha256=sha(f))
          for f in sorted(delivery.iterdir()) if f.is_file() and f.name != "manifest.json"}})
    print(markdown(summary), flush=True)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    analyze(args.output, workers=args.workers, render=not args.no_figures)
