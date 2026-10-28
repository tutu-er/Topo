from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from research_experiments.wzzt_systematic.common import read,write,observed

ROOT=Path(__file__).resolve().parents[2]


def noise_context(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    prior=ROOT/"outputs/wzzt_rnj_budget_20261028_full"
    protocol=read(prior/"protocol.json")
    frame=pd.read_csv(ROOT/"docs/wzzt_rnj_budget_20261028/per_job.csv")
    current=frame[frame.variant=="multistart_original_rule"].copy()
    baseline=frame[frame.variant=="baseline25"].set_index("job_id").clade_f1
    nj=pd.read_csv(ROOT/"docs/wzzt_retune_20261028/per_job.csv")
    nj=nj[nj.method=="NJ"].set_index("job_id").clade_f1
    rows=[]
    for job in protocol["jobs"]:
        payload=read(prior/"jobs"/job/"payload.json");obs=observed(payload["train"],payload["N"])
        truth=observed(Path(payload["train"]).with_name("train_truth.npz"),payload["N"])
        settings=protocol["conditions"][payload["condition"]]
        row=dict(job_id=job,case=payload["case"],condition=payload["condition"],N=payload["N"],samples_per_day=payload["N"]//3,
            pq_sigma_relative=settings["pq_noise"],v_sigma_relative=settings["v_noise"],root_sigma_relative=settings["v_noise"],
            load_factor=settings["load_factor"],low_xr=settings["low_xr"])
        for name,measured,true in zip(("P","Q","V","root"),obs,truth):
            e=(measured.to_numpy()-true.to_numpy())/np.maximum(abs(true.to_numpy()),1e-12)
            row[name+"_realized_relative_std"]=float(np.std(e))
        error_drop=(obs[3].to_numpy()[:,None]-obs[2].to_numpy())-(truth[3].to_numpy()[:,None]-truth[2].to_numpy())
        error_y=(obs[3].to_numpy()[:,None]**2-obs[2].to_numpy()**2)-(truth[3].to_numpy()[:,None]**2-truth[2].to_numpy()**2)
        row.update(drop_noise_std_pu=float(error_drop.std()),squared_drop_noise_std=float(error_y.std()),
            true_mean_drop_pct=float(np.mean((truth[3].to_numpy()[:,None]-truth[2].to_numpy())/truth[3].to_numpy()[:,None])*100))
        rows.append(row)
    noise=pd.DataFrame(rows)
    current=current[["job_id","clade_f1","F1_change","candidate_pool_true_recall","precision","recall"]].merge(noise,on="job_id",validate="one_to_one")
    current["baseline25_F1"]=current.job_id.map(baseline)
    current["NJ_F1"]=current.job_id.map(nj)
    current["delta_vs_NJ"]=current.clade_f1-current.NJ_F1
    current.to_csv(destination/"previous_all_noise.csv",index=False,encoding="utf-8-sig")
    current[current.clade_f1<1-1e-10].to_csv(destination/"previous_unidentified.csv",index=False,encoding="utf-8-sig")
    current[abs(current.F1_change)>1e-10].to_csv(destination/"previous_changes_vs_baseline25.csv",index=False,encoding="utf-8-sig")
    write(destination/"noise_definition.json",dict(distribution="independent zero-mean Gaussian meter errors; root has separate RNG",
        sigma_rule="relative_sigma * max(abs(instantaneous_true_channel), 1e-12)",
        sigma_is_max_error=False,root_and_terminal_sigma_are_same_declared_level=True,
        true_root_process_sigma=.003,root_process_is_measurement_noise=False,
        truth_used_only_for_realized_noise_diagnostics=True,
        previous_failed=15,previous_high_voltage_noise_failed=12,
        current_reference="multistart_original_rule",baselines=["baseline25","NJ"],
        data_sizes_are_total_over_three_days=True))
    print(current[current.clade_f1<1-1e-10].groupby(["condition","N"]).size().to_string(),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("destination");a=p.parse_args();noise_context(a.destination)
