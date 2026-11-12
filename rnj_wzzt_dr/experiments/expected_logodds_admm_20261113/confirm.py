"""Freeze selected policies, then measure evidence, posterior, rate and fee."""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from pathlib import Path
import shutil
import traceback
import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student

from ... import io
from ..unified_posterior_dr_20261112.history_scan import ROOT, require, sha, utc
from ..unified_posterior_dr_20261112.prospective import ForecastLaw, stream, binomial_interval
from ..unified_posterior_dr_20261112.confirmation import rotation, score_actual
from ..unified_posterior_dr_20261112.physical import open_runtime, generate_background, execute
from .run import REFERENCE, load_target
from .evidence import direct_gains

BETAS = (.8, .9, .95, .99)


def log_odds_samples(H, history, target, power, error, sigma2):
    answer = []
    for j in range(len(H)):
        prediction = power@(H[target]-H[j]).T
        increment = np.sum(prediction*prediction+2*error*prediction, axis=(1, 2))/(2*sigma2)
        answer.append(history[target]-history[j]+increment)
    return np.column_stack(answer)


def prepare(run):
    require((run/"planning_complete.json").exists(), "Finish bounded planning first")
    output = run/"confirmation"; require(not output.exists(), "Existing confirmation preserved")
    output.mkdir()
    protocol = io.read(run/"protocol.json"); plans = io.read(run/"planning_results.json")
    entries = []
    for row in plans:
        data = load_target(row, protocol["gain_required"])
        original = data["original"]
        policies = {"no_dr": np.zeros_like(original.delta_lower_kw)}
        entry = {k:row[k] for k in ("id", "case", "target", "competitors", "scores")}
        entry.update(fee_scale=data["fee_scale"], status=row["status"], policies={},
            target_topology_id=data["score"]["models"][row["target"]]["topology_id"])
        if row["status"] == "complete" and row["admm"]["qualified"]:
            path = run/"planning"/(row["id"]+f"__target{row['target']}")/"admm/quote.npz"
            with np.load(path, allow_pickle=False) as a:
                requests, policies["admm"] = a["x"], a["mean_u"]
            require(np.max(abs(original.nominal(requests)-policies["admm"])) < 1e-7, "Original nominal controller differs")
            gains = direct_gains(data["H"], row["target"], row["competitors"], data["forecast"],
                data["score"]["sigma2"], original, data["base"], policies["admm"])
            require(min(gains) >= protocol["gain_required"]-1e-8, "Independent policy gain failed")
            policies["rotation"], receipt = rotation(original, original.fees(policies["admm"]))
            entry.update(rotation_matching=receipt, policy_origin=row["admm"]["best"].get("origin", "ADMM_iterate"),
                quote=io.ref(path), analytic_gains=gains)
        for name in ("no_dr", "rotation", "admm"):
            if name not in policies:
                entry["policies"][name] = dict(available=False, reason="No qualified nonzero quote")
                continue
            require(original.audit(policies[name])["passed"], "Policy violates original resources")
            path = output/"policies"/(row["id"]+f"__target{row['target']}__{name}.npz")
            path.parent.mkdir(exist_ok=True); np.savez_compressed(path, mean_u=policies[name])
            entry["policies"][name] = dict(available=True, nominal=io.ref(path), fee=original.fees(policies[name]))
        entries.append(entry)
    sources = {}
    for folder in (Path(__file__).parent, ROOT/"rnj_wzzt_dr/dr",
                   ROOT/"rnj_wzzt_dr/experiments/unified_posterior_dr_20261112"):
        for p in folder.glob("*.py"):
            relative=p.relative_to(ROOT);dest=output/"source_snapshot"/relative
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);sources[relative.as_posix()]=io.ref(p)
    frozen = dict(created_utc=utc(), policies=entries, sources=sources, planning_protocol=io.ref(run/"protocol.json"),
        planning_results=io.ref(run/"planning_results.json"), gain_required=protocol["gain_required"],
        betas=BETAS, model=protocol["model_confirmation"], physical=protocol["physical_confirmation"],
        physical_inputs=protocol["selected"], all_future_unopened=True,
        model_law="same fixed background power law and iid residual sigma2; nominal policy mean only",
        physical_law="original full-horizon correlated multiplicative response, projection and relative P/Q/V/root meters",
        fee_rule="fixed original nominal contract payment; delivered-resource fee only a wear proxy",
        no_fits=True)
    io.write(output/"protocol.json",frozen)
    io.write(output/"policy_seal.json",dict(created_utc=utc(),protocol_sha256=sha(output/"protocol.json"),
        policy_files={p["nominal"]["path"]:p["nominal"]["sha256"] for e in entries for p in e["policies"].values() if p["available"]}))
    return output,frozen


def means_interval(values):
    values=np.asarray(values);mean=float(values.mean());se=float(values.std(ddof=1)/np.sqrt(len(values)))
    half=float(student.ppf(.975,len(values)-1))*se
    return dict(mean=mean,standard_error=se,mean_interval95=[mean-half,mean+half])


def model_confirmation(run,out,protocol):
    rows=[]
    for entry in protocol["policies"]:
        data=load_target(entry,protocol["gain_required"]);original=data["original"]
        H=data["H"];score=data["score"];history=np.array([m["log_probability"] for m in score["models"]])
        law=ForecastLaw(data["forecast"]);count=protocol["model"]["paths"]
        rng=stream(protocol["model"]["namespace"],entry["id"],entry["target"],protocol["model"]["study_seed"])
        power=law.sample(rng,count);error=np.sqrt(score["sigma2"])*rng.normal(size=(count,96,H.shape[1]))
        baseline=log_odds_samples(H,history,entry["target"],power,error,score["sigma2"])
        stored={}
        for name,policy in entry["policies"].items():
            if not policy["available"]:
                rows.append(dict(id=entry["id"],target=entry["target"],strategy=name,status="not_run_no_quote"));continue
            require(sha(policy["nominal"]["path"])==policy["nominal"]["sha256"],"Policy changed")
            with np.load(policy["nominal"]["path"],allow_pickle=False) as a:delivery=a["mean_u"]
            dp,dq=original.aggregate(delivery);shift=np.column_stack((dp,dq))/data["base"]
            odds=baseline if name=="no_dr" else log_odds_samples(H,history,entry["target"],power+shift[None],error,score["sigma2"])
            logp=-logsumexp(-odds,axis=1);weights=np.exp(logp)
            analytic=direct_gains(H,entry["target"],entry["competitors"],data["forecast"],score["sigma2"],original,data["base"],delivery)
            increments=[]
            for j,g in zip(entry["competitors"],analytic,strict=True):
                increments.append(dict(competitor=j,analytic_gain=float(g),**means_interval(odds[:,j]-baseline[:,j])))
            row=dict(id=entry["id"],target=entry["target"],strategy=name,status="complete",paths=count,
                posterior=means_interval(weights),target_MAP=int(np.sum(np.argmin(odds,axis=1)==entry["target"])),
                pass_rates={str(b):dict(successes=int(np.sum(weights>=b)),probability=float(np.mean(weights>=b)),
                    interval95=binomial_interval(int(np.sum(weights>=b)),count)) for b in BETAS},
                increments=increments,fee=policy["fee"],normalized_fee=policy["fee"]/entry["fee_scale"])
            rows.append(row);stored[name+"_target_log_probability"]=logp;stored[name+"_full_log_odds"]=odds
        np.savez_compressed(out/(entry["id"]+f"__target{entry['target']}_model.npz"),**stored)
        print(f"Independent model checked: {entry['id']} target {entry['target']}",flush=True)
    io.write(out/"model_results.json",rows)
    io.write(out/"model_complete.json",dict(status="complete",created_utc=utc(),rows=len(rows),
        source=io.ref(__file__),protocol_sha256=sha(out/"protocol.json"),results_sha256=sha(out/"model_results.json")))


def physical_confirmation(run,out,protocol):
    require((out/"model_complete.json").exists(),"Model confirmation must precede physical paths")
    if not any(e["policies"]["admm"]["available"] for e in protocol["policies"]):
        io.write(out/"actual_results.json",[])
        io.write(out/"actual_complete.json",dict(status="not_run_no_quote",created_utc=utc(),new_AC_days=0));return
    evaluation={r["id"]:r for r in io.read(REFERENCE/"evaluation.json")}
    rows=[];backgrounds=0;calls=0;modules={}
    for item in protocol["physical_inputs"]:
        entries=[r for r in protocol["policies"] if r["id"]==item["id"]]
        data=load_target(entries[0],protocol["gain_required"]);request=data["request"];original=data["original"]
        cohort=request["cohort"]
        if cohort not in modules:modules[cohort]=open_runtime(REFERENCE,cohort)
        module,batch,_=modules[cohort];old=io.read(batch/"generation.json")[request["job"]]["metadata"]
        parameters={k:old[k] for k in ("case","replicate","voltage_sigma","pq_sigma","root_sigma","study_seed","profile","level")}
        truth=evaluation[item["id"]];score=data["score"]
        nominal={}
        for entry in entries:
            for name,p in entry["policies"].items():
                if p["available"]:
                    with np.load(p["nominal"]["path"],allow_pickle=False) as a:nominal[entry["target"],name]=a["mean_u"]
        for replicate in range(protocol["physical"]["replicate_start"],protocol["physical"]["replicate_start"]+protocol["physical"]["paths"]):
            trial=[]
            try:
                parameters.update(replicate=replicate,study_seed=protocol["physical"]["study_seed"])
                background=generate_background(module,parameters);backgrounds+=1;calls+=1
                require(background["metadata"]["physical_branch_sha256"]==old["physical_branch_sha256"],"Physical branch changed")
                seed=int(stream("expected_logodds_admm_devices_v2",item["id"],replicate,protocol["physical"]["study_seed"]).integers(0,2**31-1))
                multiplier=original.draw(seed,1)[0];cache={}
                for entry in entries:
                    for name,p in entry["policies"].items():
                        basic=dict(id=item["id"],case=item["case"],target=entry["target"],strategy=name,replicate=replicate)
                        if not p["available"]:
                            trial.append(dict(**basic,status="not_run_no_quote"));continue
                        mean=nominal[entry["target"],name];key=io.contract_hash(mean)
                        if key not in cache:
                            observed,execution,latent=execute(module,background,original,mean,multiplier);calls+=execution["ac_day_calls"]
                            path=out/"actual_observations"/item["id"]/(str(replicate)+"__"+key[:16]+".npz");path.parent.mkdir(parents=True,exist_ok=True)
                            np.savez_compressed(path,P=observed.p,Q=observed.q,V=observed.voltage,root=observed.root,labels=observed.labels)
                            private=out/"actual_execution"/item["id"]/(str(replicate)+"__"+key[:16]+".npz");private.parent.mkdir(parents=True,exist_ok=True)
                            np.savez_compressed(private,**latent)
                            cache[key]=observed,execution,io.ref(path)
                        observed,execution,ref=cache[key]
                        decision=score_actual(score,observed,truth["truth_id"],entry["target_topology_id"])
                        final=decision.pop("log_probability");prefix=decision.pop("prefix_log_probability")
                        pf=out/"prefix"/(item["id"]+f"__{entry['target']}__{name}__{replicate}.npz");pf.parent.mkdir(exist_ok=True)
                        np.savez_compressed(pf,final_log_probability=final,prefix_log_probability=prefix)
                        predicted=set(map(tuple,decision.pop("selected_clades")));real=set(map(tuple,truth["truth_clades"]))
                        power=np.column_stack((observed.p,observed.q));loss=np.array([np.sum((observed.y-power@h.T)**2) for h in data["H"]])
                        history=np.array([p["history_sse"] for p in score["parameters"]]);L=(loss+history-loss[entry["target"]]-history[entry["target"]])/(2*score["sigma2"])
                        trial.append(dict(**basic,status="complete",target_correct=truth["truth_id"]==entry["target_topology_id"],
                            missing=len(real-predicted),extra=len(predicted-real),observation=ref,execution=execution,prefix=io.ref(pf),
                            final_major_log_odds=L[entry["competitors"]],**decision))
            except Exception as exc:
                trial=[dict(id=item["id"],case=item["case"],target=e["target"],strategy=name,replicate=replicate,
                    status="failed" if p["available"] else "not_run_no_quote",error=repr(exc),traceback=traceback.format_exc())
                    for e in entries for name,p in e["policies"].items()]
            rows.extend(trial);io.write(out/"actual_checkpoint.json",rows)
        print("Physical paths completed: "+item["id"],flush=True)
    io.write(out/"actual_results.json",rows)
    io.write(out/"actual_complete.json",dict(status="complete",created_utc=utc(),background_days=backgrounds,
        AC_day_calls=calls,rows=len(rows),results_sha256=sha(out/"actual_results.json"),protocol_sha256=sha(out/"protocol.json"),
        identifier_refits=0,parameter_refits=0))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",type=Path,required=True);a=p.parse_args()
    run=a.run.resolve();out,protocol=prepare(run)
    model_confirmation(run,out,protocol);physical_confirmation(run,out,protocol)


if __name__=="__main__":
    main()
