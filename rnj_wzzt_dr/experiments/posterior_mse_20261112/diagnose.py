"""Postprocess sealed weights into numerical details and proposed decision sets.

No observation loading, fitting, new probability scale, or operational changes.
Raw MSE, shared scales and fitted support coefficients stay in private outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from ... import io
from .deliver import write_csv


def credible_set(models, level):
    ordered = sorted(models, key=lambda m: (m["validation_mse"], m["index"]))
    mass = 0.; kept = []
    for model in ordered:
        kept.append(model); mass += model["probability"]
        if mass >= level:
            break
    boundary = kept[-1]["log_probability"]
    kept.extend(m for m in ordered[len(kept):] if m["log_probability"] == boundary)
    return dict(indices=[m["index"] for m in kept],
                topology_ids=[m["topology_id"] for m in kept],
                mass=sum(m["probability"] for m in kept))


def describe(values):
    values = np.asarray(values, dtype=float)
    return dict(min=float(values.min()), median=float(np.median(values)), max=float(values.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve(); private = args.private.resolve(); public = args.public.resolve()
    if private.exists() or public.exists():
        raise FileExistsError("Preserve prior diagnostics; choose new destinations")
    protocol = io.read(run/"protocol.json"); seal = io.read(run/"selection_seal.json")
    evaluations = {r["id"]: r for r in io.read(run/"evaluation.json")}
    requests = {r["id"]: r for r in protocol["requests"]}
    rows, candidates, supports, blocks = [], [], [], []
    public.mkdir(parents=True); private.mkdir(parents=True)
    for identity, expected in seal["scores"].items():
        path = run/"scores"/(identity+".json")
        if io.ref(path)["sha256"] != expected:
            raise ValueError("Sealed weights changed")
        d = io.read(path)
        if d["status"] != "complete":
            raise ValueError("Diagnostic input has an unresolved score")
        ms = sorted(d["models"], key=lambda m: (m["validation_mse"], m["index"]))
        first, second = ms[:2]; loss = first["validation_mse"]
        sets = {str(q): credible_set(ms, q) for q in (.9, .95, .99)}
        if abs(sum(m["probability"] for m in ms)-1) > 1e-12:
            raise AssertionError("Probabilities do not sum to one")
        for m in ms:
            score = (m["validation_mse"]-loss)/d["s"]
            log_odds = first["log_probability"]-m["log_probability"]
            if abs(score-log_odds) > 1e-8*max(1., abs(score)):
                raise AssertionError("MSE difference and posterior odds disagree")
            candidates.append(dict(id=identity, index=m["index"], topology_id=m["topology_id"],
                member_indices=json.dumps(m["member_indices"]), training_mse=m["train_mse"],
                validation_mse=m["validation_mse"], relative_gap=(m["validation_mse"]-loss)/loss,
                difference_over_s=score, probability=m["probability"], log_probability=m["log_probability"]))
        active = sorted({tuple(C) for m in ms for C in m["active_clades"]})
        uncertain = 0
        for C in active:
            included = [m for m in ms if C in map(tuple, m["active_clades"])]
            probability = sum(m["probability"] for m in included)
            uncertain += .1 < probability < .9
            blocks.append(dict(id=identity, support=json.dumps(C), probability=probability,
                log_probability=float(logsumexp([m["log_probability"] for m in included]))))
        gap = second["validation_mse"]-loss
        ev = evaluations[identity]
        a, b = set(map(tuple, first["active_clades"])), set(map(tuple, second["active_clades"]))
        row = dict(id=identity, cohort=d["cohort"], method=d["method"], case=ev["case"],
            seed_index=ev["seed_index"], voltage_sigma=ev["voltage_sigma"], N=d["cells"],
            topologies=len(ms), raw_points=d["raw_points"], m1=loss,
            m2=second["validation_mse"], sigma2=d["sigma2"], s=d["s"], gap=gap,
            relative_gap=gap/loss, difference_over_s=gap/d["s"],
            p1=first["probability"], p2=second["probability"],
            tail_after_top2=sum(m["probability"] for m in ms[2:]),
            log_remaining_mass=float(logsumexp([m["log_probability"] for m in ms[1:]])),
            credible90=len(sets["0.9"]["indices"]), credible95=len(sets["0.95"]["indices"]),
            credible99=len(sets["0.99"]["indices"]), uncertain_supports_10_90=uncertain,
            near_pair_003=d["near_pair_003"], correct=ev["map_correct"], truth_covered=ev["truth_covered"],
            only_first=json.dumps(sorted(a-b)), only_second=json.dumps(sorted(b-a)))
        rows.append(row)
        if d["near_pair_003"]:
            reference = requests[identity]["result"]
            if io.ref(reference["path"])["sha256"] != reference["sha256"]:
                raise ValueError("Original frozen fit changed")
            native = io.read(reference["path"])["result"]
            for name, model in (("first", first), ("second", second)):
                point = native["frozen_candidates"][model["index"]]["point"]
                for C, r, x in zip(point["supports"], point["r"], point["x"], strict=True):
                    labels = tuple(sorted(d["labels"][i] for i in C))
                    if labels in a^b:
                        supports.append(dict(id=identity, candidate=name, index=model["index"],
                            support=json.dumps(labels), r_response=r, x_response=x,
                            probability=model["probability"],
                            coefficient_meaning="squared-voltage response block, not raw physical line impedance"))
    summaries = {}
    for name in sorted({r["cohort"]+"/"+r["method"] for r in rows}):
        selected = [r for r in rows if r["cohort"]+"/"+r["method"] == name]
        summaries[name] = dict(count=len(selected),
            numerical={k: describe([r[k] for r in selected]) for k in
                ("N", "topologies", "raw_points", "m1", "sigma2", "s", "relative_gap", "difference_over_s", "p1", "p2", "tail_after_top2")},
            candidate_set_counts={k: dict(Counter(str(r[k]) for r in selected))
                                  for k in ("credible90", "credible95", "credible99")},
            displayed_one=sum(r["p1"] == 1 for r in selected))
    write_csv(private/"libraries.csv", rows)
    write_csv(private/"candidates.csv", candidates)
    write_csv(private/"disputed_support_coefficients.csv", supports)
    write_csv(private/"support_probabilities.csv", blocks)
    io.write(private/"summary.json", summaries)
    safe_keys = [k for k in rows[0] if k not in {"m1", "m2", "sigma2", "s", "gap", "only_first", "only_second"}]
    write_csv(public/"libraries_normalized.csv", [{k: r[k] for k in safe_keys} for r in rows])
    write_csv(public/"near_pairs_normalized.csv", [{k: r[k] for k in safe_keys} for r in rows if r["near_pair_003"]])
    published = {name: dict(count=value["count"], displayed_one=value["displayed_one"],
        candidate_set_counts=value["candidate_set_counts"],
        numerical={k: v for k, v in value["numerical"].items() if k not in {"m1", "sigma2", "s"}})
        for name, value in summaries.items()}
    io.write(public/"summary_normalized.json", published)
    report(private/"NUMERICS.md", rows, summaries, include_raw=True)
    report(public/"NUMERICS.md", rows, summaries, include_raw=False)
    receipt = dict(status="complete", source=io.ref(__file__), libraries=len(rows),
        protocol_sha256=io.ref(run/"protocol.json")["sha256"],
        original_selection_seal_sha256=io.ref(run/"selection_seal.json")["sha256"],
        posterior_values_changed=False, model_defaults_changed=False,
        displayed_one=sum(r["p1"] == 1 for r in rows),
        underflowed_model_probabilities=sum(c["probability"] == 0 for c in candidates),
        models=len(candidates), max_tail_after_top2=max(r["tail_after_top2"] for r in rows),
        credible90_counts=dict(Counter(str(r["credible90"]) for r in rows)),
        credible99_counts=dict(Counter(str(r["credible99"]) for r in rows)),
        public_files={p.name: io.ref(p)["sha256"] for p in public.iterdir() if p.is_file()})
    io.write(public/"RECEIPT.json", receipt)
    io.write(private/"RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def report(path, rows, summaries, include_raw):
    lines = ["# 后验数值细节与累计候选集", "", "只读取已封存的100个库；概率、拟合和默认规则均保持原样。以下候选集是输出判据的反事实核算。", "",
        "| 来源 | 不同拓扑数：最小/中位/最大 | 第二名相对MSE差中位数 | 显示最高权重=1 | 90%累计集：单/双 | 99%累计集：单/双 |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, g in summaries.items():
        n = g["numerical"]; k = n["topologies"]; c = g["candidate_set_counts"]
        lines.append(f"| {name} | {k['min']:g}/{k['median']:g}/{k['max']:g} | {100*n['relative_gap']['median']:.4f}% | {g['displayed_one']}/{g['count']} | {c['credible90'].get('1',0)}/{c['credible90'].get('2',0)} | {c['credible99'].get('1',0)}/{c['credible99'].get('2',0)} |")
    lines += ["", "主组N=3168–5184，严格96点补充组N=352–576。相同的相对MSE差在大N下会产生更集中的权重。", "",
        f"本批前两名以外的总质量最大为{100*max(r['tail_after_top2'] for r in rows):.8f}%；其余常小到机器显示为0。最高权重显示为1的有{sum(r['p1']==1 for r in rows)}/100，不应解释为逻辑确定；原log权重保留。", "",
        "## 原.003门槛内的逐项比较", ""]
    if include_raw:
        lines += ["| 输入/方法 | N | MSE1 | MSE2 | s | 相对差 | 差/s | p1 | p2 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    else:
        lines += ["| 输入/方法 | N | 相对MSE差 | 差/s=log赔率 | p1 | p2 | 90%累计集大小 | 库含真 |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        if not r["near_pair_003"]: continue
        name = r["id"].removeprefix("full48__").replace("day96__", "")
        if include_raw:
            lines.append(f"| {name} | {r['N']} | {r['m1']:.12e} | {r['m2']:.12e} | {r['s']:.12e} | {100*r['relative_gap']:.8f}% | {r['difference_over_s']:.9f} | {r['p1']:.9f} | {r['p2']:.9f} |")
        else:
            lines.append(f"| {name} | {r['N']} | {100*r['relative_gap']:.8f}% | {r['difference_over_s']:.9f} | {r['p1']:.9f} | {r['p2']:.9f} | {r['credible90']} | {r['truth_covered']} |")
    if include_raw:
        lines += ["", "## 共同尺度范围", "", "| 来源 | sigma² 最小/中位/最大 | s 最小/中位/最大 |", "|---|---|---|"]
        for name, g in summaries.items():
            a, b = g["numerical"]["sigma2"], g["numerical"]["s"]
            lines.append(f"| {name} | {a['min']:.6e}/{a['median']:.6e}/{a['max']:.6e} | {b['min']:.6e}/{b['median']:.6e}/{b['max']:.6e} |")
        lines += ["", "逐库/逐候选原MSE及尺度见 libraries.csv / candidates.csv；争议支持的响应系数见 disputed_support_coefficients.csv；所有内部支持的边际权重见 support_probabilities.csv。响应系数不是原始物理线路阻抗。"]
    else:
        lines += ["", "原始MSE、共同方差、尺度与响应系数留在本地outputs的数值明细目录；本公开表保留无量纲量和概率。"]
    lines += ["", "90%累计集按权重从大到小取最短前缀，达到.9即止；边界相同权重全部保留。同理定义95%和99%集。这些集条件于现有库和已声明的误差法则，不能发现库外真拓扑。", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
