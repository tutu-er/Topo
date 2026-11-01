"""Bounded same-budget audit of refitted family/validation exclusion margins.

Reuses the eight saved main-case observed packets. No new physical future,
topology search, response sampling or price deployment is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys
import traceback

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def bound(reference):
    path = Path(reference["path"])
    if sha(path) != reference["sha256"]:
        raise RuntimeError("input hash drift")
    return path


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".partial")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def family(point):
    # Retain the full requested joint family, including numerically zero blocks.
    return point.get("full_fit", {}).get("supports", point.get("native_family", point["supports"]))


def run(source, output):
    from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import normalize
    from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import source_hashes
    from .profile_margin import refit_pair
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    c = read(Path(source)/"main/protocol.json")
    inputs = dict(history=c["history_packet"], historical_fit=c["historical_baseline_result"],
                  original_protocol=ref(Path(source)/"main/protocol.json"),
                  existing_observations=ref(Path(source)/"main/test/observations.json"),
                  existing_scored=ref(Path(source)/"main/test/scored.json"))
    sources = source_hashes()
    sources.update({str(p.resolve()):sha(p) for p in Path(__file__).parent.glob("*.py")})
    protocol = dict(status="frozen", created_utc=datetime.now(timezone.utc).isoformat(),
        id="same_budget_full_refit_validation_margin_audit_v1", inputs=inputs, source_sha256=sources,
        policies=["zero", "rotation", "socp", "misocp"], paths=[1100,1101], expected_records=8,
        range_probes=[dict(policy="zero",path=1100),dict(policy="socp",path=1100)],
        face_epsilon=1e-8, face_seconds_per_min_or_max=8.,
        unchanged_validation_band=.03, unchanged_weight_bounds=[0.,2.], unchanged_prune_tolerance=1e-9,
        new_AC=0,new_future=0,new_RNJ_W_search=0,
        scope="mechanism reanalysis of previously seen observations, not independent validation of a new pricing optimizer")
    write(output/"protocol.json", protocol)
    with bound(inputs["history"]).open("rb") as stream:
        history = pickle.load(stream)["history"]
    old = read(bound(inputs["historical_fit"]))
    comp = c["competition"]
    coarse = family(old["selection_candidates"][comp["selected_index"]])
    fine = family(old["selection_candidates"][comp["alternative_index"]])
    labels = c["labels"]
    added = sorted(set(tuple(s) for s in fine)-set(tuple(s) for s in coarse))
    if len(added) != 1:
        raise ValueError("this bounded pilot requires the one historical disputed addition")
    target = tuple(added[0])
    scored = read(bound(inputs["existing_scored"]))
    rows = []
    for source_row in scored["rows"]:
        row = dict(policy=source_row["policy"], path=source_row["path"], status="started")
        try:
            with bound(source_row["packet"]).open("rb") as stream:
                packet = pickle.load(stream)
            train, _, _ = normalize(history["training"]+packet["training"], "training")
            valid, _, _ = normalize(history["validation"]+packet["validation"], "validation")
            assert len(train[0]) == len(valid[0]) == 480
            probes = (source_row["path"] == 1100 and source_row["policy"] in ("zero", "socp"))
            fitted = refit_pair(train, valid, coarse, fine, ranges=probes,
                face_epsilon=protocol["face_epsilon"], face_seconds=protocol["face_seconds_per_min_or_max"])
            actual = read(bound(source_row["result"]))
            def has(point):
                return target in {tuple(s) for s in point["supports"]}
            yes = [p["validation_mae"] for p in actual["selection_candidates"] if has(p)]
            no = [p["validation_mae"] for p in actual["selection_candidates"] if not has(p)]
            raw = actual["models"]["multistart_original_rule"]["selected_uncontracted"]
            cal = actual["calibrations"]["raw"]
            theta_slack = None
            if target in {tuple(s) for s in fitted["fine_supports"]}:
                index = [tuple(s) for s in fitted["fine_supports"]].index(target)
                theta_slack = cal["a"]*fitted["fine_weights"]["r"][index]+cal["b"]*fitted["fine_weights"]["x"][index]-.16*cal["median_root_depth"]+1e-10
            private = dict(fitted)
            write(output/f"private_{source_row['policy']}_{source_row['path']}.json", private)
            public = {k:v for k,v in fitted.items() if k not in ("fine_weights", "training_LP_solves")}
            row.update(status="complete", **public,
                target_output_slack=theta_slack,
                actual_candidate_exit_margin=min(no)-1.03*min(yes)-1e-12 if yes and no else None,
                actual_candidate_comparison_available=bool(yes and no),
                actual_raw_target=has(raw), actual_final_target=has(actual["models"]["multistart_original_rule"]),
                training_LP_calls=len(fitted["training_LP_solves"]),
                original_full_errors=source_row["models"]["multistart_original_rule"]["full_errors"],
                original_actual_fee=source_row["actual_payment"])
        except Exception as error:
            row.update(status="failed", error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        rows.append(row)
        write(output/"progress.json", dict(status="running", rows=rows, planned=8))
        print(json.dumps({k:row.get(k) for k in ("policy","path","status","fitted_validation_exit_margin","safe_validation_exit_margin")}), flush=True)
    for row in rows:
        baseline = next(r for r in rows if r["policy"] == "zero" and r["path"] == row["path"])
        if row["status"] == baseline["status"] == "complete":
            row["margin_improvement_vs_zero"] = row["fitted_validation_exit_margin"]-baseline["fitted_validation_exit_margin"]
    summary = dict(status="complete", rows=rows, planned=8, complete=sum(r["status"]=="complete" for r in rows),
        target_labels=[labels[i] for i in target], coarse_requested_blocks=len(coarse), fine_requested_blocks=len(fine),
        training_LP_calls=sum(r.get("training_LP_calls",0) for r in rows),
        face_problem_count=sum(sum(len(f["problems"]) for f in r.get("optimal_face_ranges",{}).values()) for r in rows),
        new_AC=0,new_future=0,new_RNJ_W_search=0,
        reuses_old_evaluation_observations=True,new_price_optimizer_validated=False,
        profile_scope="all nonnegative R/X jointly reoptimized, no positive weight floor; validation never used for training")
    write(output/"summary.json", summary)
    write(output/"ended.json",dict(status="complete", summary=ref(output/"summary.json")))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    for path in (REPO,REPO/"rnj_wzzt_core"):
        if str(path) not in sys.path:sys.path.insert(0,str(path))
    run(args.source,args.output)


if __name__=="__main__":main()
