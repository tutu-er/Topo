"""Export readable compact CSV evidence; keep verbose JSON locally."""
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
EVIDENCE=HERE/"evidence"


def cell(value):
    if isinstance(value,(dict,list)):
        return json.dumps(value,ensure_ascii=False,separators=(",",":"))
    return value


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        writer.writerows({k:cell(row.get(k)) for k in keys} for row in rows)
    with path.open(encoding="utf-8",newline="") as stream:
        assert len(list(csv.DictReader(stream)))==len(rows)


def export(stem):
    data=json.loads((EVIDENCE/(stem+"_records.json")).read_text(encoding="utf-8"))
    trials=data if isinstance(data,list) else [data]
    fits=[];design=[];light=[]
    for t in trials:
        meta={k:t[k] for k in ("role","case","seed","n")}
        for method,bank in [("historical",t["historical_fits"])]+[
            (r["method"],r.get("fits",[])) for r in t["records"]]:
            for f in bank:
                d=f["diagnostics"]
                fits.append(dict(meta,method=method,candidate=f["name"],supports=f["supports"],
                    success=f["success"],loss=f["loss"],attained_loglik=f["loglik"],
                    r_pu=f["r_pu"],x_pu=f["x_pu"],r_ohm=f["r_ohm"],x_ohm=f["x_ohm"],
                    initial_loss=d["initial_loss"],
                    shared_rank=d["local_shared_rank"],shared_dimension=d["shared_parameter_count"],
                    active_bound_count=d["active_bound_count"],restart_loss_spread=d["restart_loss_spread"],
                    solver_restarts=d["restarts"],ac_equation_residual=d["ac_equation_residual"],
                    power_balance_residual=d["power_balance_residual"],min_voltage=d["min_voltage"],
                    max_voltage=d["max_voltage"],seconds=d["seconds"],
                    global_optimum_certified=False,global_loss_lower_bound=None))
        for a in t["design"].get("scores",[]):
            for f in a.get("competitor_fits",[]):
                design.append(dict(meta,reference=t["design"].get("reference"),action=a["action"],
                    selected=a["action"]==t["design"].get("selected"),
                    action_valid=a["success"],action_score=a["score"],action_failures=a["failures"],
                    competitor=f["candidate"],**{k:v for k,v in f.items() if k!="candidate"}))
        small={k:v for k,v in t.items() if k not in ("historical_fits","records","design")}
        small["design"]={k:v for k,v in t["design"].items() if k!="scores"}
        small["records"]=[{k:v for k,v in r.items() if k!="fits"} for r in t["records"]]
        light.append(small)
    write_csv(EVIDENCE/(stem+"_fits.csv"),fits)
    write_csv(EVIDENCE/(stem+"_design.csv"),design)
    (EVIDENCE/(stem+"_trials.json")).write_text(
        json.dumps(light,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(stem,len(trials),"trials",len(fits),"candidate fits",len(design),"inner design fits")


if __name__=="__main__":
    for stem in ("formal","stress","scale"):
        export(stem)
