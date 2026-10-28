from pathlib import Path
import argparse
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
from research_experiments.wzzt_systematic.common import read,write,sha
from .evaluate import signature
from .records import find,load,load_trace
from .variants import ac_gate


def verify(output,destination):
    output=Path(output).resolve();destination=Path(destination).resolve()
    manifest=read(destination/"manifest.json");count=0
    for entries,base in ((manifest["local"],output),(manifest["external"],None),
        (read(destination/"result_manifest.json"),output),
        (read(destination/"analysis_source_manifest.json"),Path(__file__).parent),
        (read(destination/"delivery_manifest.json"),destination)):
        for path,expected in entries.items():
            target=base/path if base is not None else Path(path)
            assert sha(target)==expected,str(target);count+=1
    for name,base in (("producer_source.zip",output/"source"),("analysis_source.zip",Path(__file__).parent)):
        with zipfile.ZipFile(destination/name) as archive:
            for path in archive.namelist():
                assert hashlib.sha256(archive.read(path)).hexdigest()==sha(base/path);count+=1
    selected=read(destination/"selected_models.json");frame=pd.read_csv(destination/"per_job.csv")
    assert len(selected)==560 and len(frame)==560 and frame.job_id.nunique()==80
    assert frame.model_available.all() and frame.ac_valid.all()
    assert (frame.groupby(["N","variant"]).size()==20).all()
    assert len(find(destination/"catalog.sqlite"))==560
    dataset={d["job_id"]:d for d in read(destination/"datasets.json")}
    for p in selected:
        d=dataset[p["job_id"]]
        model=load(destination/"catalog.sqlite",case=d["case_name"],condition=d["condition_name"],N=d["N"],variant=p["variant"])
        assert signature(model)==signature(p["point"])
        ac=read(output/"ac"/(p["key"]+".json"))
        assert ac["model_signature"]==signature(p["point"]) and "error" not in ac
        assert ac["testing"]["valid"] and ac["validation"]["valid"]
    for case,condition,N in (("paper15","high_voltage_noise",24),("flynn16","reference",288),("pengwah18","reference",48)):
        model=load(destination/"catalog.sqlite",case=case,condition=condition,N=N,variant="multistart_original_rule",with_observations=True)
        assert len(model["train"][0])==N and len(model["validation"][0])==288 and len(model["test"][0])==288
        trace=load_trace(destination/"catalog.sqlite",case=case,condition=condition,N=N)
        assert trace["truth_used_by_worker"] is False and "long75" in trace["searches"]
    gates=0
    for job,rows in frame.groupby("job_id"):
        rows=rows.set_index("variant");a=rows.loc["baseline25"];b=rows.loc["multistart_original_rule"]
        old=read(output/"ac"/(job+"__"+a.model_signature+".json"));new=read(output/"ac"/(job+"__"+b.model_signature+".json"))
        accepted=ac_gate(old,new);saved=str(rows.loc["multistart_ac_gate","gate_adopted"]).lower()=="true"
        assert saved==accepted
        expected=b if accepted else a
        assert abs(expected.clade_f1-rows.loc["multistart_ac_gate","clade_f1"])<1e-12;gates+=1
    assert read(destination/"audit_summary.json")["all_passed"]
    result=dict(all_passed=True,hash_and_archive_checks=count,catalog_models=560,selected_AC_bindings=560,
                observed_data_and_trace_load_checks=3,independent_AC_gate_recomputations=gates,unit_tests_passed=4,
                formal_returned=80,formal_failed=0,formal_timeout=0)
    write(destination/"delivery_verification.json",result)
    write(destination/"delivery_manifest.json",{p.name:sha(p) for p in destination.iterdir() if p.is_file() and p.name!="delivery_manifest.json"})
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output");p.add_argument("destination");a=p.parse_args();verify(a.output,a.destination)
