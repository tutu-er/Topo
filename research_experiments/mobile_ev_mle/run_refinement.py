"""Second-stage local AC, historical gate and low-variance EIG checks."""
from pathlib import Path
import hashlib
import json
from time import perf_counter
import numpy as np
from .audit_common import build_bank,unvectorize,array_hash
from .history_gate import historical_candidate_pvalues,compatible_history
from .likelihood import fit_candidate
from .symmetric_ac_local import local_symmetric_scores
from .bayes import posterior_weights
from .bayes_rb import rb_eig_scores
from .run_three_directions import fit_wider_family,best_name
from .run_study import dump


def main():
    here=Path(__file__).resolve().parent
    first=here/"outputs"/"three_directions"
    output=here/"outputs"/"refinement";output.mkdir(parents=True,exist_ok=True)
    bank=build_bank();began=perf_counter()
    names=[n for n in bank.actions if n!="none"]
    paths=[here/n for n in ("run_refinement.py","history_gate.py","symmetric_ac_local.py","bayes_rb.py",
                            "audit_common.py","power.py","bayes.py","likelihood.py","run_three_directions.py")]
    config=dict(source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        parent_config_sha256=hashlib.sha256((first/"config.json").read_bytes()).hexdigest(),
        alpha_history=.05,alpha_future=.05,historical_calibration_count=199,
        source_delta=.02,line_trust_fraction=.3,eig_counts=[2048,8192],integration_repeats=3,
        semantics="first-stage outcomes preserved; local forecast/QP is a new conditional criterion",
        bank_mean_hash=array_hash(bank.history_means,*bank.future_means.values()))
    dump(output/"config.json",config)
    records=[];gates=[]
    for path in sorted(first.glob("history_*.json")):
        old=json.loads(path.read_text());seed=old["seed"];yh=np.array(old["history_y"])
        gate=historical_candidate_pvalues(yh,bank.history_means,bank.labels,B=199,seed=seed+80000)
        gate.update(compatible_history(gate,.05));gate.update(seed=seed,finite_scope=old["true_atom_covered"])
        gates.append(gate)
        if old["stress"] is not None:continue
        hist=unvectorize(yh,bank.meter,12)
        fits={}
        for h,c in enumerate(bank.candidates):
            ids=np.flatnonzero(bank.labels==h)
            loss=np.sum((bank.history_means[ids]-yh)**2,axis=1)
            atom=bank.atoms[int(ids[np.argmin(loss)])]
            init={k:(atom[k][:12] if k in ("p","q","source_p","source_q") else atom[k])
                  for k in ("r","x","p","q","source_p","source_q")}
            fits[c.name]=fit_candidate(c,hist,bank.meter,initial=init,multistart=2,max_nfev=150,seed=seed+81000+h)
        # Sequentially complete direction1, direction2 gate, direction3.
        local={}
        for source_mode in ("fixed","local_refit"):
            local[source_mode]=local_symmetric_scores(bank.candidates,hist,bank.meter,
                list(bank.actions.values()),historical_fits=fits,source_mode=source_mode)
        post=posterior_weights(yh,bank.history_means,bank.labels,bank.prior)
        rb_runs={}
        if post.topology_entropy>1e-12:
            for count in (2048,8192):
                rb_runs[str(count)]=[rb_eig_scores(post,bank.future_means,samples_per_topology=count,
                    seed=seed+82000+j) for j in range(3)]
            rb_name=best_name(rb_runs["8192"][0],names,"eig")
        else:
            rb_name="constant_site_0"  # numerical information saturation, declared fallback
        local_choices={}
        for mode,score in local.items():
            valid={name:row for name,row in score["actions"].items() if row.get("success",False)}
            charged_valid=[n for n in names if n in valid]
            local_choices[mode]=best_name(valid,charged_valid,"score") if charged_valid else None
        choices={"none":"none","constant_fixed":"constant_site_0","bayes_rb":rb_name,
                 **{"symmetric_"+mode:name for mode,name in local_choices.items()}}
        noise=np.random.default_rng(seed+83000).normal(size=bank.future_means["none"].shape[1])
        # New observed-data wider-model checks, not finite-family alpha claims.
        continuous={}
        for name in sorted({n for n in choices.values() if n is not None}):
            true_atom=old["true_atom"]  # evaluation only, after choices are frozen
            yf=bank.future_means[name][true_atom]+noise
            continuous[name]=fit_wider_family(bank,yh,yf,name,seed+84000,150)
        record=dict(seed=seed,gate=gate,choices=choices,local=local,rb_runs=rb_runs,
                    continuous=continuous,history_fits={k:v.summary() for k,v in fits.items()},
                    future_noise_seed=seed+83000,future_noise_hash=array_hash(noise),
                    selector_uses_truth=False,selector_uses_future_noise=False)
        dump(output/f"history_{seed}.json",record);records.append(record)
        print(f"refinement {seed}: {choices}; historical_gate={gate['status']}",flush=True)
    summary=dict(config=config,gates=gates,history_count=len(records),seconds=perf_counter()-began,
                 continuous_fit_count=sum(len(rows) for r in records for rows in r["continuous"].values()),
                 continuous_failure_count=sum(not row["success"] for r in records for rows in r["continuous"].values() for row in rows),
                 expansion_justified=False)
    dump(output/"summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
