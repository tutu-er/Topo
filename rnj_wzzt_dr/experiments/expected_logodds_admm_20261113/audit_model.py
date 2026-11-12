"""Replay existing model streams; check by independent full residual likelihoods."""
from pathlib import Path
import argparse
import numpy as np
from scipy.special import logsumexp

from ... import io
from ..unified_posterior_dr_20261112.history_scan import require,utc
from ..unified_posterior_dr_20261112.prospective import ForecastLaw,stream
from .run import load_target


def check(run):
    out=run/"confirmation";protocol=io.read(out/"protocol.json")
    require((out/"actual_complete.json").exists(),"Audit completed streams only")
    rows={(r["id"],r["target"],r["strategy"]):r for r in io.read(out/"model_results.json")}
    compared=0;max_error=0.;max_odds_error=0.
    for entry in protocol["policies"]:
        data=load_target(entry,protocol["gain_required"]);count=protocol["model"]["paths"]
        rng=stream(protocol["model"]["namespace"],entry["id"],entry["target"],protocol["model"]["study_seed"])
        power=ForecastLaw(data["forecast"]).sample(rng,count)
        error=np.sqrt(data["score"]["sigma2"])*rng.normal(size=(count,96,data["H"].shape[1]))
        power,error=power[:16],error[:16]
        history=np.asarray([m["log_probability"] for m in data["score"]["models"]])
        path=out/(entry["id"]+f"__target{entry['target']}_model.npz")
        with np.load(path,allow_pickle=False) as samples:
            for name,policy in entry["policies"].items():
                if not policy["available"]:continue
                with np.load(policy["nominal"]["path"],allow_pickle=False) as a:delivery=a["mean_u"]
                dp,dq=data["original"].aggregate(delivery)
                controlled=power+np.column_stack((dp,dq))[None]/data["base"]
                y=controlled@data["H"][entry["target"]].T+error
                losses=np.column_stack([np.sum((y-controlled@h.T)**2,axis=(1,2)) for h in data["H"]])
                likelihood=history[None]-losses/(2*data["score"]["sigma2"])
                expected=likelihood-logsumexp(likelihood,axis=1)[:,None]
                saved=samples[name+"_target_log_probability"]
                e=float(np.max(abs(np.exp(expected[:,entry["target"]])-np.exp(saved[:16]))))
                odds=likelihood[:,entry["target"],None]-likelihood
                e2=float(np.max(abs(odds-samples[name+"_full_log_odds"][:16])))
                max_error=max(max_error,e);max_odds_error=max(max_odds_error,e2)
                require(e<1e-8 and e2<1e-6,"Independent residual likelihood differs")
                reported=rows[entry["id"],entry["target"],name]
                require(abs(np.exp(saved).mean()-reported["posterior"]["mean"])<1e-12,"Posterior mean statistic differs")
                for beta,rate in reported["pass_rates"].items():
                    require(int(np.sum(np.exp(saved)>=float(beta)))==rate["successes"],"Pass count differs")
                compared+=16
    receipt=dict(status="passed",created_utc=utc(),independent_residual_paths=compared,
        maximum_probability_error=max_error,maximum_log_odds_error=max_odds_error,
        all_saved_mean_and_threshold_statistics_checked=True,replayed_existing_streams=True,
        new_independent_model_paths=0,source=io.ref(__file__))
    io.write(out/"independent_model_audit.json",receipt)
    print({k:v for k,v in receipt.items() if k!="source"})
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",type=Path,required=True)
    check(parser.parse_args().run.resolve())
