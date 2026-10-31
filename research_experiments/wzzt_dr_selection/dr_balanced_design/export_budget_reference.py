"""Export whitelisted evidence, excluding P/Q/V/AC arrays and fitted coefficients."""
from pathlib import Path
import argparse, hashlib, json, shutil
def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
def export(root,repo):
 root,repo=Path(root).resolve(),Path(repo).resolve()
 dest=repo/"research_experiments/wzzt_dr_selection/evidence/budget_reference_diagnostic_v1"
 code_dir=repo/"research_experiments/wzzt_dr_selection/dr_balanced_design"
 dest.mkdir(parents=True,exist_ok=True)
 for name in ("protocol.json","preparation_protocol.json","oracle_selected.json","oracle_selected.sha256.json","aggregate.json","interpretation_summary.json"):
  shutil.copyfile(root/name,dest/name)
 for src,target in (("audit_diagnostic.py","audit_budget_reference.py"),("audit_failure_fronts.py","audit_budget_failure_fronts.py"),("prepare_lp_method_probe.py","probe_budget_lp_methods.py")):
  shutil.copyfile(root.parent/"source"/src,code_dir/target)
 file=root/"audit_v3/receipt.json";audit=read(file)
 public={k:v for k,v in audit.items() if k not in ("checks","file_sha256","marker_sha256","score_rows","aggregate")}
 public.update(private_full_receipt_sha256=sha(file),omitted_detail="Individual checks/input paths retained in private full receipt.")
 write(dest/"independent_audit_summary.json",public)
 shutil.copyfile(root/"audit_v3/audit_protocol.json",dest/"audit_protocol.json")
 front=read(root/"supplement_failure_fronts_v1/report.json")
 public={k:v for k,v in front.items() if k not in ("rows","input_sha256")}
 fields=("case","phase","path","policy","original_status","status","models","primary_joint_failure_retained","original_actual_payment","original_fit_status","original_fit_stage","original_exception","original_lp_time_limit_seconds","nonoptimal_fixed_family_LPs","fixed_family_LP_calls_retained","successful_native_RNJ_replay_matches")
 public["records"]=[{k:r[k] for k in fields if k in r} for r in front["rows"]]
 public["private_full_report_sha256"]=sha(root/"supplement_failure_fronts_v1/report.json")
 for record in public["records"]:
  for point in record["models"].values():
   supports=point.pop("support_labels",None)
   if supports is not None:
    point["support_labels_sha256"]=hashlib.sha256(json.dumps(supports,sort_keys=True).encode()).hexdigest()
    point["support_labels_count"]=len(supports)
  for call in record.get("nonoptimal_fixed_family_LPs",[]):
   family=call.pop("family",None)
   if family is not None:
    call["family_sha256"]=hashlib.sha256(json.dumps(family,sort_keys=True).encode()).hexdigest()
    call["family_count"]=len(family)
 write(dest/"same_observation_geometry_summary.json",public)
 shutil.copyfile(root/"audit_failure_fronts_lock.json",dest/"geometry_audit_protocol.json")
 probe=read(root/"lp_method_probe_v1/ended.json");public={k:v for k,v in probe.items() if k!="records"}
 public["records"]=[{k:v for k,v in r.items() if k not in ("active_r","active_x","solution","coefficients","r","x","weights","features")} for r in probe["records"]]
 write(dest/"lp_method_probe_summary.json",public)
 shutil.copyfile(root/"lp_method_probe_v1/protocol.json",dest/"lp_method_probe_protocol.json")
 phases=dict(cases={},note="Metrics and failure records only; original records unchanged.")
 for case in ("flynn16","pengwah18"):
  phases["cases"][case]={}
  for phase in ("planning","test"):
   file=root/case/phase/"phase_ended.json";d=read(file)
   keys=("planned","recorded","evaluated","operation_failed_complete","elapsed_seconds","unique_new_fits","reused_fits","independent_response_paths","original_history_rows","merged_rows","historical_stop","truth_signature")
   item={k:d[k] for k in keys};item.update(phase_sha256=sha(file),LP_ledger_total=d["LP_ledger"]["total"],rows=[])
   fields=("record_id","case_key","phase","policy","path","status","actual_payment","committed_payment","unknown_payment_events","models","search","operation_checks_pass","quote_qualification_pass","physical_events_complete","training_rows","validation_rows","observed_cache_key","cache_reused","started_utc","ended_utc","evaluation_utc")
   for r in d["rows"]: item["rows"].append({k:r[k] for k in fields if k in r}|{"failure_message":r.get("message")})
   phases["cases"][case][phase]=item
 write(dest/"all_response_records.json",phases)
 manifest=dict(scope="Explicitly whitelisted code/configuration/aggregate; trajectories/coefficients private.",files={p.name:sha(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name!="export_manifest.json"},audit_sources={p.name:sha(p) for p in code_dir.iterdir() if p.name in ("audit_budget_reference.py","audit_budget_failure_fronts.py","probe_budget_lp_methods.py","export_budget_reference.py")})
 write(dest/"export_manifest.json",manifest)
 return dict(files=len(manifest["files"]),bytes=sum(p.stat().st_size for p in dest.iterdir() if p.is_file()))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--repo",type=Path,required=True);a=p.parse_args();print(json.dumps(export(a.root,a.repo)))
