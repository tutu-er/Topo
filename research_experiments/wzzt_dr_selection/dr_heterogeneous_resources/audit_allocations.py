"""Independent arithmetic replay of exported resource allocations (no raw data)."""
import argparse
import hashlib
import json
from pathlib import Path


def run(root):
    root=Path(root);a=json.loads((root/"public/case_allocations.json").read_text(encoding="utf-8"))
    checks=[]
    def check(name,condition):checks.append(dict(name=name,passed=bool(condition)))
    reference=next(x for x in a["cases"] if x["case"]=="paper15")["sizing_summary"]["positive_net_mean_kw"]
    for c in a["cases"]:
        nodes=c["nodes"];tag=c["case"]
        check(tag+":unique_labels",len({n["node_label"] for n in nodes})==len(nodes))
        check(tag+":same_unit_fraction",abs(c["sizing_summary"]["total_kw"]-.2*c["sizing_summary"]["sum_nodal_positive_q90_kw"])<1e-8)
        check(tag+":enrollment_scale",abs(c["matched_enrollment_kw"]-2*c["sizing_summary"]["positive_net_mean_kw"]/reference)<1e-8)
        check(tag+":sum_inventory",abs(sum(n["inventory_kw"] for n in nodes)-c["sizing_summary"]["total_kw"])<1e-8)
        for n in nodes:
            uid=tag+":"+str(n["node_label"])
            check(uid+":sum",abs(n["inventory_kw"]-n["shiftable_kw"]-n["curtailable_kw"]-n["ev_charger_kw"]-n["battery_power_kw"])<1e-8)
            check(uid+":no_grid_invention",n["grid_kw"]==0)
            check(uid+":battery_energy",abs(n["battery_energy_kwh"]-2.5*n["battery_power_kw"])<1e-8)
            check(uid+":ev_energy",abs(n["ev_grid_task_kwh"]-20/7*n["ev_charger_kw"])<1e-8)
            check(uid+":curtail_share",abs(n["shiftable_kw"]-3*n["curtailable_kw"])<1e-8)
        # Independent closed form for the 09-17 interval: EV disconnected,
        # user availability .8. Other periods must be checked too.
        up=[];down=[]
        for t in range(96):
            h=t/4
            factor=.6 if h<6 else 1. if h<9 else .8 if h<17 else 1. if h<22 else .7
            connected=h<8 or h>=18
            up.append(sum(factor*n["shiftable_kw"]+n["battery_power_kw"]+(1-20/98)*n["ev_charger_kw"]*connected for n in nodes))
            down.append(sum(factor*(n["shiftable_kw"]+n["curtailable_kw"])+n["battery_power_kw"]+20/98*n["ev_charger_kw"]*connected for n in nodes))
        check(tag+":min_up",abs(min(up)-c["full_inventory_min_directional_kw"]["increase"])<1e-8)
        check(tag+":min_down",abs(min(down)-c["full_inventory_min_directional_kw"]["decrease"])<1e-8)
        check(tag+":power_diagnostic",c["power_inventory_diagnostic_pass"]==(min(min(up),min(down))>=2*c["matched_enrollment_kw"]))
    first=root.parent/"run_v1/public/case_allocations.json"
    if first.exists():
        check("serialization_fix_did_not_change_allocation",first.read_bytes()==(root/"public/case_allocations.json").read_bytes())
    result=dict(status="passed" if all(x["passed"] for x in checks) else "failed",checks=len(checks),failures=[x for x in checks if not x["passed"]],
        allocations_sha256=hashlib.sha256((root/"public/case_allocations.json").read_bytes()).hexdigest(),
        scope="independent arithmetic/unit replay of exported synthetic configuration, not independent data calibration or AC feasibility",
        new_optimizer_calls=0,new_identifier_calls=0,new_AC_calls=0)
    (root/"public/audit_summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result));return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=Path);run(p.parse_args().output)
