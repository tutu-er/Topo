"""Independent saved-score arithmetic audit; Python standard library only.

With no --output this is read-only and prints a compact result. An explicit
--output creates a new JSON file exclusively, never overwriting existing data.
No project collector, selector, fitter, simulator, or numerical package is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

FAMILIES = ("star", "clade_01", "clade_02", "clade_12")
BASE = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def number(value):
    return math.inf if value == "inf" else value


def encoded(value):
    return "inf" if value == math.inf else value


def json_safe(value):
    if isinstance(value,float) and not math.isfinite(value):
        return "inf" if value == math.inf else "-inf" if value == -math.inf else "nan"
    if isinstance(value,dict):
        return {key:json_safe(item) for key,item in value.items()}
    if isinstance(value,list):
        return [json_safe(item) for item in value]
    return value


def same(left,right):
    if isinstance(left,bool) or isinstance(right,bool):
        return type(left) is type(right) and left == right
    if isinstance(left,(int,float)) and isinstance(right,(int,float)):
        if math.isinf(left) or math.isinf(right):
            return left == right
        return math.isclose(left,right,rel_tol=1e-12,abs_tol=1e-10)
    return left == right


class Audit:
    def __init__(self):
        self.checks = 0
        self.differences = []

    def check(self,name,actual,expected):
        self.checks += 1
        if not same(actual,expected):
            self.differences.append(dict(check=name,actual=json_safe(actual),expected=json_safe(expected)))

    def fields(self,name,actual,expected):
        self.check(name+"/keys",sorted(actual),sorted(expected))
        for key,value in expected.items():
            self.check(name+"/"+key,actual.get(key),value)


def score_record(record,method):
    """Derive availability, score and M from candidate losses, not saved scores."""
    names = record["candidate_names"]
    family = record["family"]
    row = record["methods"][method]
    losses = row.get("losses",{})
    complete = (isinstance(losses,dict) and bool(losses) and set(losses) == set(names)
                and all(finite(value) and value >= 0 for value in losses.values()))
    omitted = bool(record.get("candidate_miss") or family not in names)
    if not complete:
        return dict(complete=False,omitted=omitted,score=math.inf,M=None)
    minimum = min(losses.values())
    score = math.inf if omitted else max(0.,losses[family]-minimum)
    alternatives = [value for candidate,value in losses.items() if candidate != family]
    margin = None if omitted or not alternatives else min(alternatives)-losses[family]
    return dict(complete=True,omitted=omitted,score=score,M=margin)


def statistics(values,total):
    values = sorted(float(value) for value in values if finite(value))
    n = len(values)
    middle = None if not n else values[n//2] if n % 2 else (values[n//2-1]+values[n//2])/2
    return dict(total=total,valid=n,missing_or_failed=total-n,
                mean=math.fsum(values)/n if n else None,median=middle,
                positive=sum(value > 0 for value in values),
                negative=sum(value < 0 for value in values),zero=sum(value == 0 for value in values),
                sum=math.fsum(values) if n else None)


def audit_denominators(audit,value,total,path):
    """Check every nested saved numeric summary's declared history denominator."""
    if isinstance(value,dict):
        if {"total","valid","missing_or_failed"} <= set(value):
            audit.check(path+"/total",value["total"],total)
            audit.check(path+"/valid+missing",value["valid"]+value["missing_or_failed"],total)
            if {"positive","negative","zero"} <= set(value):
                audit.check(path+"/sign-count",value["positive"]+value["negative"]+value["zero"],value["valid"])
        for key,item in value.items():
            audit_denominators(audit,item,total,path+"/"+str(key))


def phase_statistics(audit,report,phase,methods):
    records = report["records"]
    n = len(records)
    audit.check(phase+"/unique-seeds",len({r["seed"] for r in records}),n)
    audit.check(phase+"/seeds",report["seeds"],[r["seed"] for r in records])
    audit.check(phase+"/histories",report["summary"]["independent_histories"],n)
    audit.check(phase+"/family-counts",report["summary"]["family_counts"],dict(Counter(r["family"] for r in records)))
    audit.check(phase+"/candidate-omissions",report["summary"]["candidate_omissions"],
                sum(bool(r.get("candidate_miss") or r["family"] not in r["candidate_names"]) for r in records))
    audit_denominators(audit,report["summary"],n,phase+"/summary")
    computed = {}
    for record in records:
        for method in methods:
            value = score_record(record,method)
            computed[record["seed"],method] = value
            row = record["methods"][method]
            audit.check(phase+"/"+record["seed"]+"/"+method+"/status",
                        row["status"] == "complete",value["complete"])
            audit.check(phase+"/"+record["seed"]+"/"+method+"/saved-M",row.get("M"),value["M"])
            if row.get("saved_M") is not None:
                audit.check(phase+"/"+record["seed"]+"/"+method+"/original-M",row["saved_M"],value["M"])
    summary = {}
    for method in methods:
        values = [computed[r["seed"],method]["M"] for r in records]
        summary[method] = statistics(values,n)
        saved = report["summary"]["methods"][method]
        audit.fields(phase+"/"+method+"/M-summary",saved["M"],summary[method])
        complete = sum(computed[r["seed"],method]["complete"] for r in records)
        audit.check(phase+"/"+method+"/complete",saved["complete"],complete)
        audit.check(phase+"/"+method+"/failed",saved["failed"],n-complete)
        audit.check(phase+"/"+method+"/status-denominator",sum(saved["status_counts"].values()),n)
    for name,saved in report["summary"].get("paired_contrasts",{}).items():
        left,right = saved["left"],saved["right"]
        deltas = []
        for record in records:
            a,b = computed[record["seed"],left]["M"],computed[record["seed"],right]["M"]
            deltas.append(a-b if finite(a) and finite(b) else None)
        audit.fields(phase+"/paired/"+name,saved["delta_M"],statistics(deltas,n))
    return computed,summary


def decision_counts(items,method):
    rows = [item["methods"][method] for item in items]
    n = len(rows)
    rejected = sum(row["false_reject"] for row in rows)
    return dict(total=n,failed_keep_all=sum(row["failed_keep_all"] for row in rows),
                candidate_omissions=sum(item["candidate_miss"] for item in items),
                false_reject=rejected,false_reject_rate=rejected/n if n else None,
                unique_correct=sum(row["unique_correct"] for row in rows),
                unique_incorrect=sum(row["unique_incorrect"] for row in rows),
                unresolved=sum(row["unresolved"] for row in rows),
                valid_computations=sum(not row["failed_keep_all"] for row in rows))


def verify(threshold_path,evaluation_path,start_path):
    paths = dict(thresholds=Path(threshold_path),evaluation=Path(evaluation_path),evaluation_start=Path(start_path))
    loaded = {name:json.loads(path.read_bytes()) for name,path in paths.items()}
    cal,evaluation,start = loaded["thresholds"],loaded["evaluation"],loaded["evaluation_start"]
    audit = Audit()
    methods = cal["methods"]
    audit.check("method-count",len(methods),14)
    audit.check("method-uniqueness",len(set(methods)),14)
    audit.check("evaluation-methods",evaluation["methods"],methods)
    audit.check("calibration-histories",len(cal["records"]),36)
    audit.check("evaluation-histories",len(evaluation["records"]),12)
    audit.check("calibration-balanced",dict(Counter(r["family"] for r in cal["records"])),dict.fromkeys(FAMILIES,9))
    audit.check("evaluation-balanced",dict(Counter(r["family"] for r in evaluation["records"])),dict.fromkeys(FAMILIES,3))
    audit.check("alpha",cal["alpha"],.1)
    audit.check("evaluation-alpha",evaluation["alpha"],cal["alpha"])
    audit.check("calibration/evaluation-seeds-overlap",sorted(set(cal["seeds"]) & set(evaluation["seeds"])),[])
    for key in ("policy_id","config_sha256","source_manifest_sha256","source_sha256","thresholds_id"):
        audit.check("frozen/"+key,evaluation[key],cal[key])
    audit.check("start/policy",start["policy_id"],cal["policy_id"])
    audit.check("start/thresholds-id",start["thresholds_id"],cal["thresholds_id"])
    threshold_sha = sha(paths["thresholds"])
    audit.check("start/threshold-SHA",start["thresholds_sha256"],threshold_sha)
    audit.check("evaluation/threshold-SHA",evaluation["threshold_file_sha256"],threshold_sha)
    audit.check("start/planned-seeds",sorted(str(s) for s in start["planned_evaluation_seeds"]),sorted(evaluation["seeds"]))

    cal_values,cal_M = phase_statistics(audit,cal,"calibration",methods)
    eval_values,eval_M = phase_statistics(audit,evaluation,"evaluation",methods)
    thresholds = {}
    threshold_count = 0
    for method in methods:
        thresholds[method] = {}
        for family in FAMILIES:
            records = [r for r in cal["records"] if r["family"] == family]
            scores = sorted(cal_values[r["seed"],method]["score"] for r in records)
            n = len(scores)
            rank = math.ceil((n+1)*(1-cal["alpha"]))
            q = scores[rank-1] if rank <= n else math.inf
            thresholds[method][family] = q
            saved = cal["calibration"][method][family]
            prefix = "threshold/"+method+"/"+family
            audit.check(prefix+"/n",saved["n"],n)
            audit.check(prefix+"/rank",saved["rank"],rank)
            audit.check(prefix+"/rank-is-nine",rank,9)
            audit.check(prefix+"/value",number(saved["threshold"]),q)
            audit.check(prefix+"/sorted-scores",saved["sorted_scores"],[encoded(s) for s in scores])
            audit.check(prefix+"/missing-or-infinite",saved["missing_or_infinite"],sum(math.isinf(s) for s in scores))
            threshold_count += 1

    stored = {item["seed"]:item for item in evaluation["decisions"]}
    audit.check("decision-history-count",len(stored),len(evaluation["records"]))
    audit.check("decision-seeds",sorted(stored),sorted(evaluation["seeds"]))
    decisions = []
    decision_count = 0
    for record in evaluation["records"]:
        seed,family = record["seed"],record["family"]
        names = record["candidate_names"]
        local = dict(seed=seed,family=family,
                     candidate_miss=bool(record.get("candidate_miss") or family not in names),methods={})
        audit.check(seed+"/decision-family",stored[seed]["family"],family)
        audit.check(seed+"/decision-omission",stored[seed]["candidate_miss"],local["candidate_miss"])
        for method in methods:
            value = eval_values[seed,method]
            losses = record["methods"][method]["losses"]
            failed = not value["complete"]
            retained = list(names) if failed else [k for k in names
                if losses[k]-min(losses.values()) <= thresholds[method].get(k,math.inf)]
            rejected = local["candidate_miss"] or family not in retained
            derived = dict(retained=retained,failed_keep_all=failed,false_reject=bool(rejected),
                unique_correct=bool(not rejected and len(retained)==1),
                unique_incorrect=bool(rejected and len(retained)==1),
                unresolved=len(retained)!=1,retained_count=len(retained))
            saved = stored[seed]["methods"][method]
            for key,expected in derived.items():
                actual = saved.get(key)
                audit.check(seed+"/"+method+"/"+key,
                            sorted(actual) if key=="retained" else actual,
                            sorted(expected) if key=="retained" else expected)
            audit.check(seed+"/"+method+"/failure-reason-presence",bool(saved["failure_reasons"]),failed)
            local["methods"][method] = derived
            decision_count += 1
        decisions.append(local)
    outcomes = {}
    for method in methods:
        outcomes[method] = dict(overall=decision_counts(decisions,method),
            by_true_family={family:decision_counts([r for r in decisions if r["family"]==family],method)
                            for family in FAMILIES})
        audit.fields(method+"/evaluation-overall",evaluation["evaluation"][method]["overall"],outcomes[method]["overall"])
        for family in FAMILIES:
            audit.fields(method+"/evaluation/"+family,evaluation["evaluation"][method]["by_true_family"][family],
                         outcomes[method]["by_true_family"][family])
    audit.check("threshold-total",threshold_count,56)
    audit.check("decision-total",decision_count,168)
    return dict(success=not audit.differences,arithmetic_field_checks=audit.checks,
        threshold_groups_verified=threshold_count,retention_sets_verified=decision_count,
        M_summaries_verified=len(methods)*2,
        differences=audit.differences,difference_count=len(audit.differences),
        policy_id=cal["policy_id"],thresholds_id=cal["thresholds_id"],
        inputs={name:dict(path=str(path.resolve()),sha256=sha(path)) for name,path in paths.items()},
        script_sha256=sha(__file__),source="candidate losses and true-family fields of saved standard records",
        arithmetic_float_tolerance=dict(relative=1e-12,absolute=1e-10),
        failure_rule="unavailable complete candidate losses -> keep all",
        no_project_imports=True,new_fit_calls=0,new_selector_calls=0,new_simulation_calls=0,
        recomputed_thresholds={m:{f:encoded(q) for f,q in values.items()} for m,values in thresholds.items()},
        recomputed_evaluation=outcomes,recomputed_M=dict(calibration=cal_M,evaluation=eval_M))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds",type=Path,default=BASE/"thresholds.json")
    parser.add_argument("--evaluation",type=Path,default=BASE/"evaluation.json")
    parser.add_argument("--evaluation-start",type=Path,default=BASE/"evaluation_start.json")
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    result = verify(args.thresholds,args.evaluation,args.evaluation_start)
    if args.output is not None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x",encoding="utf-8") as stream:
            stream.write(json.dumps(result,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({key:result[key] for key in
          ("success","arithmetic_field_checks","threshold_groups_verified",
           "retention_sets_verified","M_summaries_verified","difference_count")}))
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
