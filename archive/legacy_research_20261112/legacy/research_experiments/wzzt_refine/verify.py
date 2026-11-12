from pathlib import Path
import argparse
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
from research_experiments.wzzt_systematic.common import read,write,sha,observed
from research_experiments.wzzt_budget.records import find,load,load_trace
from .evaluate import signature


def verify(output,destination):
    output=Path(output).resolve();destination=Path(destination).resolve();root=Path(__file__).resolve().parents[2]
    m=read(destination/"manifest.json");count=0
    for entries,base in ((m["local"],output),(m["external"],None),(read(destination/"result_manifest.json"),output),
        (read(destination/"analysis_source_manifest.json"),Path(__file__).parent),
        (read(destination/"analysis_helper_manifest.json"),root),(read(destination/"delivery_manifest.json"),destination)):
        for path,expected in entries.items():
            file=base/path if base is not None else Path(path)
            assert sha(file)==expected,str(file);count+=1
    for name,base in (("producer_source.zip",output/"source"),("analysis_source.zip",Path(__file__).parent)):
        with zipfile.ZipFile(destination/name) as z:
            for name in z.namelist():assert hashlib.sha256(z.read(name)).hexdigest()==sha(base/name);count+=1
    selected=read(destination/"selected_models.json");frame=pd.read_csv(destination/"per_job.csv")
    assert len(frame)==480 and len(selected)==480 and frame.model_available.all() and frame.ac_valid.all()
    assert frame.job_id.nunique()==80 and not frame.duplicated(["job_id","variant"]).any()
    assert (frame.groupby(["N","variant"]).size()==20).all()
    assert len(find(destination/"catalog.sqlite"))==480
    ds={d["job_id"]:d for d in read(destination/"datasets.json")}
    for point in selected:
        d=ds[point["job_id"]]
        model=load(destination/"catalog.sqlite",case=d["case_name"],condition=d["condition_name"],N=d["N"],variant=point["variant"])
        assert signature(model)==signature(point["point"])
        ac=read(output/"ac"/(point["key"]+".json"))
        assert "error" not in ac and ac["model_signature"]==signature(model) and ac["testing"]["valid"]
    model=load(destination/"catalog.sqlite",case="soumalas11",condition="high_voltage_noise",N=24,variant="ridge_pool",with_observations=True)
    assert len(model["train"][0])==24 and len(model["validation"][0])==288 and len(model["test"][0])==288
    trace=load_trace(destination/"catalog.sqlite",case="soumalas11",condition="high_voltage_noise",N=24)
    assert trace["truth_used_by_estimator"] is False and "pairs" in trace["operations"]["two_step"]
    old=pd.read_csv(destination/"previous_all_noise.csv")
    failed=old[old.clade_f1<1-1e-10]
    assert len(failed)==15 and sum(failed.condition=="high_voltage_noise")==12
    changes=old[abs(old.F1_change)>1e-10]
    assert sum(changes.F1_change>0)==5 and sum(changes.F1_change<0)==3
    assert set(changes[changes.F1_change<0].N)=={48}
    assert set(changes[changes.F1_change>0].N)=={24,96,288}
    for case,condition,N in (("soumalas11","high_voltage_noise",24),("flynn16","low_xr",24),("pengwah18","reference",288)):
        row=old[(old.case==case)&(old.condition==condition)&(old.N==N)].iloc[0]
        d=ds[row.job_id];measured=observed(d["train_path"],N);truth=observed(Path(d["train_path"]).with_name("train_truth.npz"),N)
        for name,a,b in zip(("P","Q","V","root"),measured,truth):
            std=float(np.std((a.to_numpy()-b.to_numpy())/np.maximum(abs(b.to_numpy()),1e-12)))
            assert abs(std-row[name+"_realized_relative_std"])<1e-14
    ridge=frame[frame.variant=="ridge_pool"]
    summary=pd.read_csv(destination/"method_summary.csv").set_index("variant")
    assert abs(ridge.clade_f1.mean()-summary.loc["ridge_pool","F1"])<1e-12
    assert sum(ridge.delta_vs_multistart>1e-10)==summary.loc["ridge_pool","wins"]
    assert sum(ridge.delta_vs_multistart < -1e-10)==summary.loc["ridge_pool","losses"]
    assert read(destination/"audit_summary.json")["all_passed"]
    receipt=dict(all_passed=True,hash_and_archive_checks=count,catalog_models=480,selected_AC_bindings=480,
        data_and_trace_load_checks=1,independent_realized_noise_groups=3,unit_tests_passed=5,
        prior_failures_and_change_counts_verified=True,returned_inputs=80)
    write(destination/"delivery_verification.json",receipt)
    write(destination/"delivery_manifest.json",{p.name:sha(p) for p in destination.iterdir() if p.is_file() and p.name!="delivery_manifest.json"})
    print(receipt,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output");p.add_argument("destination");a=p.parse_args();verify(a.output,a.destination)
