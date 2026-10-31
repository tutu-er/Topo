"""Independent bounded audit. No imports from the original implementation."""
from pathlib import Path
import argparse, ast, collections, gzip, hashlib, json, math, platform, sys, traceback
import numpy as np

VIEWS=("training_3_days","all_available_6_days")
WINDOWS=("all_day",)+tuple(f"hours_{h:02d}_{h+4:02d}" for h in range(0,24,4))
CHANNELS=("P_kw","Q_kvar")
LEVELS=("1","2")
CASES={"paper15":15,"soumalas11":11,"flynn16":16,"pengwah18":18}
BASE=200.
SIGMA=.005
FLOOR=1e-9

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def bucket(v): return "unresolved" if v is None else ("at_or_below" if v else "above")
def count_template():
 flags=lambda:dict(at_or_below=0,above=0,unresolved=0)
 return dict(pair_count=0,both_PQ={k:flags() for k in LEVELS},channels={k:dict(statuses={},flags={level:flags() for level in LEVELS}) for k in CHANNELS})

def prepare(root):
 root=Path(root).resolve();out=Path(__file__).resolve().parent
 if (out/"config.json").exists(): raise FileExistsError("independent configuration already frozen")
 p=read(root/"protocol.json")
 if sha(root/"protocol.json")!=read(root/"protocol.sha256.json")["sha256"]: raise ValueError("original protocol hash")
 if not sha(root/"protocol.json").startswith("52e6522f"): raise ValueError("unexpected original protocol")
 rows=sorted(p["cases"],key=lambda x:x["job_id"])
 selected=[rows[0]["job_id"],rows[-1]["job_id"]]
 files={str(root/name):sha(root/name) for name in ("protocol.json","protocol.sha256.json","ended.json","aggregate_views.json")}
 files[str(Path(p["source_inventory"]["path"]))]=sha(p["source_inventory"]["path"])
 for name,expected in p["source_sha256"].items():
  if sha(name)!=expected: raise ValueError("original declared source drift: "+name)
  files[name]=expected
 for row in rows:
  directory=root/"cases"/row["job_id"]
  for name in ("case_summary.json","pair_diagnostics.jsonl.gz"): files[str(directory/name)]=sha(directory/name)
  for role,item in row["sources"].items():
   path=Path(item["path"])
   if role not in ("train","validation") or path.name!=role+"_observed.npz":raise ValueError("source role whitelist")
   if sha(path)!=item["sha256"] or item["sha256"]!=item["catalog_sha256"]: raise ValueError("observed hash")
   files[str(path)]=item["sha256"]
 config=dict(id="independent_history_input_audit_v1",root=str(root),script_sha256=sha(__file__),
  original_protocol_sha256=sha(root/"protocol.json"),input_sha256=files,
  sampling_rule="lexicographic job_id first and last; train/all views; all_day and 00-04; first archive-ordered pair; P and Q both directions",
  selected_job_ids=selected,planned_pair_sample_records=8,planned_scalar_channel_directions=32,
  sample_views=VIEWS,sample_windows=("all_day","hours_00_04"),pair_positions=[0,1],
  planned_raw_observed_value_files=4,planned_detail_files=40,planned_detail_records=60620,planned_case_windows=560,
  base_kw=BASE,sigma=SIGMA,numerical_floor_kw=FLOOR,
  source_rule="Only source text and declared metadata/hash; four selected NPZs lazily read labels/P/Q. All 40 gz diagnostic files reaggregated.",
  forbidden_value_reads=["V","root","test","truth","graph","household"],
  no_original_function_import=True,LP_calls=0,QP_calls=0,AC_calls=0,identifier_calls=0,
  environment=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,platform=platform.platform()))
 write(out/"config.json",config);write(out/"config.sha256.json",dict(sha256=sha(out/"config.json")))
 return dict(status="prepared",config_sha256=sha(out/"config.json"),selected_job_ids=selected,input_hashes=len(files))

class Audit:
 def __init__(self):self.total=0;self.failed=[];self.max_abs={};self.counters=collections.Counter()
 def check(self,label,condition,detail=None):
  self.total+=1
  if not bool(condition) and len(self.failed)<50:self.failed.append(dict(label=label,detail=detail))
 def equal(self,label,a,b):self.check(label,a==b,dict(actual=a,expected=b) if a!=b else None)
 def close(self,label,a,b,metric):
  if a is None or b is None:self.equal(label,a,b);return
  delta=abs(float(a)-float(b));self.max_abs[metric]=max(self.max_abs.get(metric,0),delta)
  self.check(label,math.isclose(float(a),float(b),rel_tol=3e-11,abs_tol=2e-12),dict(actual=a,expected=b,absolute_error=delta))

def independent_channel(x,y):
 x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);n=len(x)
 rx=float(np.linalg.norm(x)/math.sqrt(n));ry=float(np.linalg.norm(y)/math.sqrt(n))
 cx=float(np.linalg.norm(x-x.mean())/math.sqrt(n));cy=float(np.linalg.norm(y-y.mean())/math.sqrt(n))
 a=None;araw=None;res=None;noise=None;ratio=None;norm=None;one=None;two=None
 if min(rx,ry)<=FLOOR:status="near_zero_power_unresolved"
 else:
  araw=float(np.linalg.lstsq(y[:,None],x,rcond=None)[0][0]);a=max(0.,araw)
  res=float(np.linalg.norm(x-a*y)/math.sqrt(n));noise=SIGMA*math.hypot(rx,a*ry)
  ratio=res/noise if noise>FLOOR*SIGMA else None
  if araw<=0:status="nonpositive_best_proportion"
  elif min(cx,cy)<=FLOOR:status="constant_channel_not_standardized"
  else:
   norm=res/rx
   if ratio is None:status="noise_reference_unresolved"
   else:status="positive_proportion_evaluated";one=ratio<=1;two=ratio<=2
 return dict(status=status,count=n,target_rms=rx,predictor_rms=ry,target_centered_rms=cx,predictor_centered_rms=cy,
  alpha_unconstrained=araw,alpha_nonnegative=a,residual_rms=res,rms_normalized_residual=norm,
  noise_reference_rms=noise,residual_over_noise_reference=ratio,
  at_one_noise_reference=one,at_two_noise_reference=two,
  target_has_negative=bool(np.any(x<0)),predictor_has_negative=bool(np.any(y<0)))

def run(root):
 root=Path(root).resolve();out=Path(__file__).resolve().parent
 if (out/"started.json").exists() or (out/"receipt.json").exists():raise FileExistsError("independent attempt already retained")
 cfg=read(out/"config.json")
 if sha(out/"config.json")!=read(out/"config.sha256.json")["sha256"] or sha(__file__)!=cfg["script_sha256"]:raise ValueError("independent lock mismatch")
 for name,expected in cfg["input_sha256"].items():
  if sha(name)!=expected:raise ValueError("input changed: "+name)
 write(out/"started.json",dict(config_sha256=sha(out/"config.json"),raw_value_reads=0))
 a=Audit();p=read(root/"protocol.json");ended=read(root/"ended.json");original_views=read(root/"aggregate_views.json")["views"]
 rows=sorted(p["cases"],key=lambda r:r["job_id"])
 a.equal("panel40",len(rows),40)
 a.equal("panel_keys",{(r["case"],r["bundle"]) for r in rows},{(c,b) for c in CASES for b in range(10)})
 a.equal("original_success",ended["audited"],40);a.equal("original_unresolved",ended["failed_or_unresolved"],0)
 a.equal("expected_pairs",sum(CASES[r["case"]]*(CASES[r["case"]]-1)//2*14 for r in rows),60620)
 for r in rows:
  a.equal(r["job_id"]+".terminal_count",r["terminal_count"],CASES[r["case"]])
  a.equal(r["job_id"]+".normal_metadata",(r["N"],r["condition"],r["voltage_sigma"],r["pq_sigma"]),(288,"reference",.0002,.005))
  a.check(r["job_id"]+".id_format",r["job_id"]==f"main__{r['case']}__b{r['bundle']}__reference__v0002__N288")
 a.equal("controls retained",sum(r["retained_zero_gain_control"] for r in rows),2)
 a.equal("b7 retained",sum(r["previous_closed_b7_control"] for r in rows),1)
 module=Path(next(name for name in p["source_sha256"] if name.endswith("dr_day_modules\\history_audit.py")))
 tree=ast.parse(module.read_text(encoding="utf-8-sig"))
 loader=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="load_pq")
 fields={n.slice.value for n in ast.walk(loader) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=="archive" and isinstance(n.slice,ast.Constant)}
 a.equal("original_load_value_whitelist",fields,{"labels","P","Q"})
 loads=[n for n in ast.walk(loader) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="load"]
 a.check("allow_pickle_false",len(loads)==1 and any(k.arg=="allow_pickle" and isinstance(k.value,ast.Constant) and k.value.value is False for k in loads[0].keywords))
 bases={}
 for name in p["source_sha256"]:
  if name.endswith("paper_style_case_bank.py") or name.endswith("paper_style_terminal_lv.py"):
   tt=ast.parse(Path(name).read_text(encoding="utf-8-sig"));found=[]
   for node in ast.walk(tt):
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="base_mva" for t in node.targets) and isinstance(node.value,ast.Constant):found.append(node.value.value)
   bases[name]=found;a.check("producer_base."+Path(name).name,bool(found) and all(float(v)*1000==BASE for v in found),found)
 a.equal("protocol_base",p["physical_power_base_kw"],BASE)
 a.equal("protocol_sigma",p["parameters"]["sigma_PQ"],SIGMA)
 # Preselected raw reads: only four observed files, no original implementation imports.
 sample_raw={};value_accesses=[]
 for row in rows:
  if row["job_id"] not in cfg["selected_job_ids"]:continue
  role_data={}
  for role in ("train","validation"):
   path=Path(row["sources"][role]["path"])
   with np.load(path,allow_pickle=False) as archive:
    a.equal(str(path)+".schema",set(archive.files),{"labels","P","Q","V","root"})
    labels=np.asarray(archive["labels"]).astype(int).tolist();pp=np.asarray(archive["P"],dtype=float)*BASE;qq=np.asarray(archive["Q"],dtype=float)*BASE
    value_accesses.append(dict(path=str(path),fields=["labels","P","Q"]))
   a.equal(str(path)+".Pshape",tuple(pp.shape),(3,96,row["terminal_count"]))
   a.equal(str(path)+".Qshape",tuple(qq.shape),(3,96,row["terminal_count"]))
   a.check(str(path)+".finite",np.isfinite(pp).all() and np.isfinite(qq).all())
   role_data[role]=(labels,pp,qq)
  a.equal(row["job_id"]+".label_order",role_data["train"][0],role_data["validation"][0])
  for view in VIEWS:
   labels=role_data["train"][0]
   for window in cfg["sample_windows"]:
    pp=role_data["train"][1];qq=role_data["train"][2]
    if view=="all_available_6_days":pp=np.concatenate((pp,role_data["validation"][1]),axis=0);qq=np.concatenate((qq,role_data["validation"][2]),axis=0)
    pp=pp.reshape(-1,len(labels));qq=qq.reshape(-1,len(labels));mask=np.ones(len(pp),dtype=bool) if window=="all_day" else np.arange(len(pp))%96<16
    record=dict(history_view=view,window=window,labels=labels[:2],channels={})
    for channel,values in (("P_kw",pp),("Q_kvar",qq)):
     values=values[mask]
     record["channels"][channel]=dict(i_from_j=independent_channel(values[:,0],values[:,1]),j_from_i=independent_channel(values[:,1],values[:,0]))
    sample_raw[(row["job_id"],view,window,tuple(labels[:2]))]=record
 # Full descriptive reaggregation; scalar formulas rechecked independently.
 global_views={view+"/"+w:dict(case_count=0,pair_records=0,both_PQ_1=0,both_PQ_2=0,unresolved_2=0,case_with_PQ_2=0,P_only_2=0,Q_only_2=0) for view in VIEWS for w in WINDOWS}
 per_case=[];sample_records=[];first_truth_reads=0
 for row in rows:
  directory=root/"cases"/row["job_id"];summary=read(directory/"case_summary.json")
  a.equal(row["job_id"]+".summary_status",summary["status"],"audited")
  n=row["terminal_count"];labels=summary["terminal_labels"];expected_pairs={tuple((labels[i],labels[j])) for i in range(n) for j in range(i+1,n)}
  counts={(v,w):count_template() for v in VIEWS for w in WINDOWS};seen=set()
  with gzip.open(directory/"pair_diagnostics.jsonl.gz","rt",encoding="utf-8") as f:
   for index,line in enumerate(f):
    rec=json.loads(line);v,w,pair=rec["history_view"],rec["window"],tuple(rec["labels"])
    a.check(row["job_id"]+".record_key",(v,w) in counts and pair in expected_pairs and (v,w,pair) not in seen)
    seen.add((v,w,pair));acc=counts[(v,w)];acc["pair_count"]+=1;channel_flags={}
    expected_n=(288 if v=="training_3_days" else 576) if w=="all_day" else (48 if v=="training_3_days" else 96)
    for channel in CHANNELS:
     item=rec["channels"][channel];flags_by_level={level:[] for level in LEVELS}
     for direction,entry in item["directions"].items():
      label=row["job_id"]+"."+v+"."+w+"."+str(pair)+"."+channel+"."+direction
      a.equal(label+".count",entry["count"],expected_n)
      status=entry["status"];acc["channels"][channel]["statuses"][status]=acc["channels"][channel]["statuses"].get(status,0)+1
      rx,ry=entry["target_rms"],entry["predictor_rms"];aa=entry["alpha_nonnegative"]
      if aa is not None:
       noise=SIGMA*math.hypot(rx,aa*ry);a.close(label+".noise",entry["noise_reference_rms"],noise,"detail_noise")
       ratio=entry["residual_rms"]/noise if noise>FLOOR*SIGMA else None;a.close(label+".ratio",entry["residual_over_noise_reference"],ratio,"detail_ratio")
      else:ratio=None
      for level,key in (("1","at_one_noise_reference"),("2","at_two_noise_reference")):
       flag=(ratio<=float(level)) if status=="positive_proportion_evaluated" and ratio is not None else None
       a.equal(label+"."+key,entry[key],flag);flags_by_level[level].append(flag)
      if status=="positive_proportion_evaluated":a.close(label+".normalized",entry["rms_normalized_residual"],entry["residual_rms"]/rx,"detail_normalized")
     channel_flags[channel]={}
     for level,flags in flags_by_level.items():
      flag=all(flags) if all(x is not None for x in flags) else None
      a.equal(row["job_id"]+".channel_flag",item["descriptive_flags"][level],flag)
      channel_flags[channel][level]=flag;acc["channels"][channel]["flags"][level][bucket(flag)]+=1
    for level in LEVELS:
     flags=[channel_flags[c][level] for c in CHANNELS];flag=all(flags) if all(x is not None for x in flags) else None
     a.equal(row["job_id"]+".both_flag",rec["both_PQ_flags"][level],flag);acc["both_PQ"][level][bucket(flag)]+=1
    key=(row["job_id"],v,w,pair)
    if key in sample_raw:
     raw=sample_raw[key];sample_records.append(dict(job_id=row["job_id"],history_view=v,window=w,labels=list(pair),independent=raw))
     for channel in CHANNELS:
      for direction,entry in raw["channels"][channel].items():
       stored=rec["channels"][channel]["directions"][direction]
       for name,value in entry.items():
        if value is None or isinstance(value,(str,bool,int)):a.equal("sample."+str(key)+"."+channel+"."+direction+"."+name,stored[name],value)
        else:a.close("sample."+str(key)+"."+channel+"."+direction+"."+name,stored[name],value,"sample_"+name)
  a.equal(row["job_id"]+".detail_record_count",len(seen),len(expected_pairs)*14)
  old_windows={(r["history_view"],r["window"]):r for r in summary["windows"]}
  a.equal(row["job_id"]+".windows",len(old_windows),14)
  for key,acc in counts.items():
   v,w=key;old=old_windows[key]
   a.equal(row["job_id"]+"."+v+"."+w+".counts",acc,old["counts"])
   a.equal(row["job_id"]+"."+v+"."+w+".pairs",acc["pair_count"],len(expected_pairs))
   expected_n=(288 if v=="training_3_days" else 576) if w=="all_day" else (48 if v=="training_3_days" else 96)
   a.equal(row["job_id"]+"."+v+"."+w+".rows",old["observed_rows"],expected_n)
   gg=global_views[v+"/"+w];gg["case_count"]+=1;gg["pair_records"]+=acc["pair_count"]
   gg["both_PQ_1"]+=acc["both_PQ"]["1"]["at_or_below"];gg["both_PQ_2"]+=acc["both_PQ"]["2"]["at_or_below"]
   gg["unresolved_2"]+=acc["both_PQ"]["2"]["unresolved"];gg["case_with_PQ_2"]+=int(acc["both_PQ"]["2"]["at_or_below"]>0)
   gg["P_only_2"]+=acc["channels"]["P_kw"]["flags"]["2"]["at_or_below"];gg["Q_only_2"]+=acc["channels"]["Q_kvar"]["flags"]["2"]["at_or_below"]
  a.counters["detailed_pair_records"]+=len(seen);a.counters["case_windows"]+=14
  per_case.append(dict(job_id=row["job_id"],terminal_count=n,pair_records=len(seen),windows=14))
 a.equal("sample_pair_records",len(sample_records),8);a.equal("raw_files_read",len(value_accesses),4)
 a.equal("all_pair_records",a.counters["detailed_pair_records"],60620);a.equal("all_case_windows",a.counters["case_windows"],560)
 a.equal("global_view_aggregate",global_views,original_views)
 # Compare summaries embedded in ended.json to per-case markers.
 a.equal("ended_cases",len(ended["cases"]),40)
 for record in ended["cases"]:a.equal(record["job_id"]+".ended_vs_marker",record,read(root/"cases"/record["job_id"]/"case_summary.json"))
 for name,expected in cfg["input_sha256"].items():a.check("post_hash."+name,sha(name)==expected)
 receipt=dict(status="passed_with_label_caveat" if not a.failed else "failed",checks_total=a.total,checks_passed=a.total-len(a.failed),failed=a.failed,
  config_sha256=sha(out/"config.json"),script_sha256=sha(__file__),original_protocol_sha256=sha(root/"protocol.json"),
  counts=dict(a.counters,raw_observed_value_files_read=len(value_accesses),pair_samples=8,scalar_sample_directions=32,source_cases=40,
   LP_calls=0,QP_calls=0,AC_calls=0,identifier_calls=0,test_values_read=0,topology_truth_read=0,household_files_read=0),
  actual_npz_value_accesses=value_accesses,producer_bases_mva=bases,max_abs_errors=a.max_abs,
  independently_reaggregated_views=global_views,
  caveats=["P_only_2/Q_only_2 are marginal channel totals including both_PQ_2, not exclusive counts; parent will create separately named derivative",
   "Observed plug-in independent multiplicative-noise reference, not a test or confidence/identifiability certificate",
   "Pairwise nonproportional inputs do not rule out multivariate compensation or weak-block ambiguity",
   "Four-hour window flags do not establish ambiguity for the full six available historical days",
   "Original 40 generated histories are development data and paired by 10 bundles, not independent field episodes"],
  original_functions_imported=False,original_audit_rerun=False,repo_files_modified=False)
 write(out/"independent_samples.json",sample_records);write(out/"receipt.json",receipt)
 return {k:receipt[k] for k in ("status","checks_total","checks_passed","counts","max_abs_errors","caveats")}

if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("prepare","run"));parser.add_argument("--root",type=Path,required=True);args=parser.parse_args()
 try:result=prepare(args.root) if args.mode=="prepare" else run(args.root);print(json.dumps(result,allow_nan=False),flush=True)
 except Exception as exc:
  out=Path(__file__).resolve().parent
  write(out/"failure.json",dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
  raise
