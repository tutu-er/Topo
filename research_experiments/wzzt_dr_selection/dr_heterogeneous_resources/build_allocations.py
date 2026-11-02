"""Freeze heterogeneous DR inventory from historical training P only.

No identifier, power flow, candidate results, test, validation or topology is
loaded. The configuration is separate from current experiment defaults.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_key(seed, case, label, purpose):
    return hashlib.sha256(f"{seed}|{case}|{label}|{purpose}".encode()).hexdigest()


def pick(values, seed, case, label, purpose):
    return values[int(digest_key(seed, case, label, purpose), 16) % len(values)]


def period_array(values, cfg):
    result = np.zeros(cfg["time"]["slots"])
    for value, (start, end) in zip(values, cfg["time"]["period_hours"], strict=True):
        result[round(start/cfg["time"]["dt_hours"]):round(end/cfg["time"]["dt_hours"])] = value
    return result


def select_catalog(catalog, source_root):
    # Project a whitelist before touching any result, test or graph path.
    rows = []
    for x in catalog:
        if x["N"] != 288 or x["condition"] != "reference" or x["voltage_sigma"] != .0002 or x["pq_sigma"] != .005:
            continue
        if x["panel"] not in ("main", "topology", "case33"):
            continue
        path = Path(x["train_path"])
        if path.name != "train_observed.npz":
            raise ValueError("only training observations permitted")
        mp = Path(source_root)/"networks"/x["case"]/"reference"/"metadata.json"
        m = read(mp)
        rows.append(dict(job_id=x["job_id"], case=x["case"], bundle=int(x["bundle"]), panel=x["panel"],
            train_path=str(path), train_sha256=x["train_sha256"],
            metadata_path=str(mp), metadata_sha256=sha(mp),
            base_kw=1000*float(m["base_mva"]), base_kv=float(m["base_kv"]), root_label=int(m["root_bus"]),
            labels=json.loads(x["labels_json"])))
    if len(rows) != 81 or len({r["case"] for r in rows}) != 17:
        raise ValueError("expected complete existing 17-network / 81-bundle reference catalog")
    if len({r["job_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate condition")
    return sorted(rows, key=lambda r: r["job_id"])


def prepare(root, catalog, source_root, library):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rows = select_catalog(read(catalog), source_root)
    for row in rows:
        if sha(row["train_path"]) != row["train_sha256"]:
            raise ValueError("archived input hash mismatch: "+row["job_id"])
    protocol = dict(id="heterogeneous_resource_allocation_v1", created_utc=datetime.now(timezone.utc).isoformat(),
        library_path=str(Path(library).resolve()), library_sha256=sha(library),
        builder_path=str(Path(__file__).resolve()), builder_sha256=sha(__file__),
        catalog_sha256=sha(catalog), sources=rows,
        allowed_observation_fields=["labels","P"],
        forbidden_value_reads=["Q","V","root trajectory","validation","test","result","network branches"],
        sizing_scope="existing synthetic calibration bank; not a new blind historical-performance experiment",
        policy="no rule/seed change after seeing sizing summaries; zero/insufficient inventories retained",
        environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()))
    write(root/"protocol.json", protocol)
    write(root/"prepared.json", dict(protocol_sha256=sha(root/"protocol.json"), observation_values_loaded=0))


def load_training(row):
    if sha(row["train_path"]) != row["train_sha256"]:
        raise ValueError("training bytes drifted")
    with np.load(row["train_path"], allow_pickle=False) as z:
        labels = z["labels"].astype(int).tolist()
        p = np.asarray(z["P"], float)*row["base_kw"]
    if labels != row["labels"] or len(set(labels)) != len(labels) or p.shape != (3,96,len(labels)) or not np.isfinite(p).all():
        raise ValueError("full training observation layout invalid")
    return p


def allocate(case, labels, historical_p_kw, cfg):
    arr = np.asarray(historical_p_kw)
    if arr.shape[-1] != len(labels) or not np.isfinite(arr).all() or len(set(labels)) != len(labels):
        raise ValueError("invalid training power or labels")
    a = cfg["allocation"]; seed = a["seed"]
    q90 = np.quantile(np.maximum(arr.reshape(-1,len(labels)),0), a["sizing_quantile"], axis=0)
    f = np.array([pick(a["node_size_factors"], seed, case, n, "size") for n in labels])
    total = float(a["inventory_fraction_of_sum_nodal_positive_net_P_quantiles"]*sum(q90))
    sizes = q90*f
    sizes = sizes*total/sum(sizes) if sum(sizes)>0 else np.zeros(len(labels))
    order = sorted(labels, key=lambda n:digest_key(seed,case,n,"mix_order"))
    types = list(a["mix_templates"])
    mixes = {n:types[i%len(types)] for i,n in enumerate(order)}
    nodes=[]
    for n, cap in zip(labels,sizes,strict=True):
        mix = a["mix_templates"][mixes[n]]
        user = float(cap*mix["user"])
        ev = float(cap*mix["ev"])
        bat = float(cap*mix["battery"])
        nodes.append(dict(node_label=int(n), inventory_kw=float(cap), mix=mixes[n],
            shiftable_kw=user*cfg["user"]["shiftable_share"],
            curtailable_kw=user*cfg["user"]["curtailable_share"], ev_charger_kw=ev,
            battery_power_kw=bat, battery_energy_kwh=bat*cfg["battery"]["energy_to_power_hours"],
            ev_battery_kwh=ev/cfg["ev"]["reference_charger_kw"]*cfg["ev"]["reference_battery_kwh"],
            ev_grid_task_kwh=ev/cfg["ev"]["reference_charger_kw"]*cfg["ev"]["reference_grid_energy_kwh_per_overnight_session"],
            grid_kw=0., user_q_over_p=pick(cfg["user"]["reactive_ratios"],seed,case,n,"Q_ratio"),
            cost_factor={r:pick(a["node_cost_factors"],seed,case,n,"cost_"+r) for r in ("user","ev","battery")},
            sd_factor={r:pick(a["node_sd_factors"],seed,case,n,"sd_"+r) for r in ("user","ev","battery")}))
    return nodes,dict(total_kw=total, positive_net_mean_kw=float(np.maximum(arr,0).sum(-1).mean()),
        net_mean_kw=float(arr.sum(-1).mean()),sum_nodal_positive_q90_kw=float(sum(q90)))


def profiles(node, cfg):
    """Full inventory envelopes. Enroll a share -> scale power AND energy.

    Arrays have [resource,slot] order: shiftable, curtailable, EV, battery.
    Net increments enter the old AC bridge only once; baselines are embedded.
    """
    T=cfg["time"]["slots"]; dt=cfg["time"]["dt_hours"]; hour=np.arange(T)*dt
    avail=period_array(cfg["user"]["availability_fraction_by_period"],cfg)
    base=np.zeros((4,T)); lo=np.zeros_like(base); hi=np.zeros_like(base)
    for i,key in enumerate(("shiftable_kw","curtailable_kw")):
        base[i]=node[key]*avail
        lo[i]=-base[i]
    hi[0]=base[0]
    ev=cfg["ev"]; connected=np.zeros(T,dtype=bool)
    windows=[]
    for start,end in ev["connected_hours"]:
        connected |= (hour>=start)&(hour<end)
        windows.append([round(start/dt),round(end/dt)])
    total_hours=dt*connected.sum()
    base[2,connected]=node["ev_grid_task_kwh"]/total_hours
    lo[2]=-base[2];hi[2,connected]=node["ev_charger_kw"]-base[2,connected]
    lo[3]=-node["battery_power_kw"];hi[3]=node["battery_power_kw"]
    costs=np.array([period_array(cfg["price"]["full_activation_price_by_period"][k],cfg)*node["cost_factor"][k]
        for k in ("user","user","ev","battery")])
    sd=np.array([np.full(T,cfg["response"]["relative_sd_by_type"][k]*node["sd_factor"][k])
        for k in ("user","user","ev","battery")])
    peak=((hour>=6)&(hour<9))|((hour>=17)&(hour<22))
    sd[:2,peak]*=cfg["response"]["user_peak_sd_multiplier"]
    sd[2,(hour>=6)&(hour<8)]*=cfg["response"]["ev_last_two_hours_before_08_departure_sd_multiplier"]
    sd[2,~connected]=0.
    return dict(baseline_kw=base,delta_lower_kw=lo,delta_upper_kw=hi,cost_full_activation=costs,
        relative_sd=sd,ev_windows=windows,
        q_over_p=np.array([node["user_q_over_p"],node["user_q_over_p"],0.,0.]))


def check_node(node, cfg):
    """Analytic nominal feasibility and nonzero energy-feasible witness.

    This is a configuration check, not Gaussian execution, AC or identification.
    """
    p=profiles(node,cfg);dt=cfg["time"]["dt_hours"];ev=cfg["ev"];b=cfg["battery"]
    checks={}
    checks["inventory_conserved"]=abs(node["inventory_kw"]-sum(node[k] for k in ("shiftable_kw","curtailable_kw","ev_charger_kw","battery_power_kw","grid_kw")))<1e-9
    checks["finite_nonnegative_parameters"]=bool(np.isfinite(p["baseline_kw"]).all() and (p["baseline_kw"]>=0).all() and (p["cost_full_activation"]>0).all() and (p["relative_sd"]>=0).all())
    checks["zero_increment_feasible"]=bool((p["delta_lower_kw"]<=0).all() and (p["delta_upper_kw"]>=0).all())
    checks["EV_no_V2G"]=bool((p["baseline_kw"][2]+p["delta_lower_kw"][2]>=-1e-10).all())
    checks["EV_task_scales_with_power"]=abs(dt*sum(p["baseline_kw"][2])-node["ev_grid_task_kwh"])<1e-9
    E=node["ev_battery_kwh"];eta=ev["charge_efficiency"]
    # Morning car already received the previous evening's declared partial task.
    evening_hours=sum(end-start for start,end in ev["connected_hours"] if start>=12)
    hours=sum(end-start for start,end in ev["connected_hours"])
    morning_initial=ev["arrival_soc"]*E+eta*node["ev_grid_task_kwh"]*evening_hours/hours
    soc_ok=True; ev_witness_ok=True
    ev_witness=np.zeros(96)
    for start,end in p["ev_windows"]:
        initial=morning_initial if start==0 else ev["arrival_soc"]*E
        path=initial+eta*dt*np.cumsum(p["baseline_kw"][2,start:end])
        soc_ok &= bool((path>=ev["soc_min"]*E-1e-10).all() and (path<=ev["soc_max"]*E+1e-10).all())
        if end-start>=2:
            # Shift one slot's small energy inside the same connection window.
            v=.5*min(-p["delta_lower_kw"][2,start],p["delta_upper_kw"][2,end-1])
            ev_witness[start]=-v;ev_witness[end-1]=v
        ev_witness_ok &= abs(dt*sum(ev_witness[start:end]))<1e-10
    checks["EV_baseline_SOC_and_segment_tasks_feasible"]=bool(soc_ok)
    checks["EV_nonzero_shift_feasible_or_zero_inventory"]=bool(ev_witness_ok and ((np.abs(ev_witness).sum()>0) or node["ev_charger_kw"]==0))
    eta_b=np.sqrt(b["roundtrip_efficiency"]);cap=node["battery_energy_kwh"]
    discharge=.1*node["battery_power_kw"];charge=discharge/b["roundtrip_efficiency"]
    e1=b["soc_initial"]*cap-dt*discharge/eta_b
    e2=e1+dt*eta_b*charge
    checks["battery_lossy_roundtrip_feasible"]=bool(charge<=node["battery_power_kw"]+1e-10 and e1>=b["soc_min"]*cap-1e-10 and abs(e2-b["soc_terminal"]*cap)<1e-9)
    shift=.1*node["shiftable_kw"]
    checks["user_shift_energy_feasible"]=bool(shift<=np.min(-p["delta_lower_kw"][0])+1e-10 and dt*shift<=cfg["user"]["shiftable_prefix_shift_hours"]*node["shiftable_kw"]+1e-10)
    checks["curtail_energy_feasible"]=bool(dt*.1*node["curtailable_kw"]<=cfg["user"]["curtailable_daily_reduction_hours"]*node["curtailable_kw"]+1e-10)
    return {name: bool(value) for name, value in checks.items()}


def run(root):
    root=Path(root);prot=read(root/"protocol.json")
    if (root/"ended.json").exists():
        raise FileExistsError("completed allocation retained")
    if sha(root/"protocol.json")!=read(root/"prepared.json")["protocol_sha256"]:
        raise ValueError("protocol drift")
    if sha(prot["builder_path"])!=prot["builder_sha256"] or sha(prot["library_path"])!=prot["library_sha256"]:
        raise ValueError("frozen code/config drift")
    cfg=read(prot["library_path"]);histories={};sources={}
    for row in prot["sources"]:
        if sha(row["metadata_path"])!=row["metadata_sha256"]:
            raise ValueError("base metadata drift")
        case=row["case"];histories.setdefault(case,[]).append(load_training(row));sources.setdefault(case,[]).append(row)
    allocated={};sizes={}
    for case,arrays in histories.items():
        ref=sources[case][0]
        if any(r["labels"]!=ref["labels"] or r["base_kw"]!=ref["base_kw"] for r in sources[case]):
            raise ValueError("labels/bases drift within case")
        allocated[case],sizes[case]=allocate(case,ref["labels"],np.concatenate(arrays),cfg)
    reference_mean=sizes["paper15"]["positive_net_mean_kw"]
    if reference_mean<=0:raise ValueError("paper15 scale undefined")
    result=[];rows=[];check_rows=[];counts=0;failed=0
    for case,nodes in allocated.items():
        stat=sizes[case];source=sources[case][0];budget=2*stat["positive_net_mean_kw"]/reference_mean
        upper=np.zeros(96);lower=np.zeros(96);case_checks=[]
        for node in nodes:
            pr=profiles(node,cfg);upper+=pr["delta_upper_kw"].sum(0);lower-=pr["delta_lower_kw"].sum(0)
            checks=check_node(node,cfg);counts+=len(checks);failed+=sum(not v for v in checks.values())
            case_checks.append(dict(node_label=node["node_label"],checks=checks))
            rows.append(dict(case=case,**{k:node[k] for k in ("node_label","mix","inventory_kw","shiftable_kw","curtailable_kw","ev_charger_kw","battery_power_kw","battery_energy_kwh","ev_battery_kwh","ev_grid_task_kwh","grid_kw","user_q_over_p")}))
        current=dict(case=case,panels=sorted({r["panel"] for r in sources[case]}),nodes=nodes,
            history_bundles=len(sources[case]),history_days=3*len(sources[case]),
            base_kw=source["base_kw"],base_kv=source["base_kv"],root_label=source["root_label"],
            matched_enrollment_kw=budget,sizing_summary=stat,
            total_by_type_kw={"user":sum(n["shiftable_kw"]+n["curtailable_kw"] for n in nodes),"ev":sum(n["ev_charger_kw"] for n in nodes),"battery":sum(n["battery_power_kw"] for n in nodes)},
            energy=dict(battery_nameplate_kwh=sum(n["battery_energy_kwh"] for n in nodes),
                battery_initial_deliverable_kwh=sum(n["battery_energy_kwh"] for n in nodes)*(cfg["battery"]["soc_initial"]-cfg["battery"]["soc_min"])*np.sqrt(cfg["battery"]["roundtrip_efficiency"]),
                ev_overnight_grid_task_kwh=sum(n["ev_grid_task_kwh"] for n in nodes),
                user_curtail_day_max_kwh=sum(n["curtailable_kw"] for n in nodes)*cfg["user"]["curtailable_daily_reduction_hours"]),
            full_inventory_min_directional_kw=dict(increase=float(min(upper)),decrease=float(min(lower))),
            full_inventory_max_directional_kw=dict(increase=float(max(upper)),decrease=float(max(lower))),
            inventory_to_matched_cap=stat["total_kw"]/budget if budget>0 else None,
            min_directional_to_matched_cap=float(min(min(lower),min(upper))/budget) if budget>0 else None,
            power_inventory_diagnostic_pass=bool(min(min(lower),min(upper))>=2*budget),
            limitations=["full-inventory power is not jointly executable all day; energy and enrolled shares still constrain schedules",
                "net-load scale is not measured gross device capacity; subload decomposition and fleet shares are synthetic",
                "no AC/95 percent target/final topology improvement check in this configuration batch"])
        result.append(current);check_rows.append(dict(case=case,nodes=case_checks))
        # Only synthetic parameters expanded here: no P/Q/V time series exported.
    public=root/"public";public.mkdir()
    write(public/"case_allocations.json",dict(id="heterogeneous_resources_v1",cases=result))
    with (public/"node_resources.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(root/"configuration_checks.json",check_rows)
    summary=dict(status="configured" if failed==0 else "configuration_failed",networks=len(result),nodes=len(rows),
        historical_training_files=len(prot["sources"]),training_days=3*len(prot["sources"]),
        checks=counts,failed_checks=failed,
        inventory_diagnostic_pass=sum(r["power_inventory_diagnostic_pass"] for r in result),
        inventory_diagnostic_total=len(result),
        tables=[{k:r[k] for k in ("case","history_bundles","base_kw","matched_enrollment_kw","sizing_summary","total_by_type_kw","energy","full_inventory_min_directional_kw","inventory_to_matched_cap","min_directional_to_matched_cap","power_inventory_diagnostic_pass")} for r in result],
        data_reads=dict(train_P=len(prot["sources"]),validation=0,test=0,V=0,Q=0,results=0,topology_branches=0),
        new_optimizer_calls=0,new_AC_calls=0,new_identifier_calls=0,
        runtime_wiring="configuration only; heterogeneous state-aware response/optimizer not yet connected",
        source_hashes=dict(library=prot["library_sha256"],builder=prot["builder_sha256"],protocol=sha(root/"protocol.json")))
    write(public/"summary.json",summary)
    # Public provenance only lists training paths and hashes, never power trajectories.
    write(public/"provenance.json",dict(protocol=prot,configuration_checks_sha256=sha(root/"configuration_checks.json")))
    write(root/"ended.json",dict(status=summary["status"],failed_checks=failed,public_files={p.name:sha(p) for p in public.iterdir()}))
    print(json.dumps({k:v for k,v in summary.items() if k not in ("tables","source_hashes")},ensure_ascii=False))
    return summary


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=["prepare","run"]);ap.add_argument("--output",required=True,type=Path)
    ap.add_argument("--catalog",type=Path);ap.add_argument("--source-root",type=Path)
    ap.add_argument("--library",type=Path,default=Path(__file__).with_name("library.json"))
    args=ap.parse_args()
    if args.action=="prepare":prepare(args.output,args.catalog,args.source_root,args.library)
    else:run(args.output)
