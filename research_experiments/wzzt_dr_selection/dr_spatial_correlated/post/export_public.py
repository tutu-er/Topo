"""Explicit scalar/configuration allowlist; no input or control trajectories."""
from pathlib import Path
import json,hashlib,shutil,sys,argparse
ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,default=Path(__file__).resolve().parents[1])
STAGE=ap.parse_args().stage.resolve()
RUN=STAGE/'run_v2';OUT=STAGE/'public_release'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
OUT.mkdir(exist_ok=True)
for name in ['protocol.json','summary.json','planning_summary.json','problem_frozen.json']:
    shutil.copy2(RUN/name,OUT/name)
shutil.copy2(STAGE/'run_v1/protocol.json',OUT/'original_penalty_protocol.json')
shutil.copy2(STAGE/'run_v1/planning_summary.json',OUT/'original_penalty_planning_summary.json')
for name in ['audit_receipt.json','policy_actual_metrics.json','resource_excursion_summary.json']:
    shutil.copy2(STAGE/'audit'/name,OUT/name)
records=read(RUN/'scored_private.json')['rows']
keys=['record_id','day','profile_index','replicate','response_seed','law','policy','status','fee','selected','coarse_excluded','refined_excluded','D_micro','margin_micro','missing','extra','errors','raw_errors','error']
write(OUT/'per_day_scalar_results.json',dict(rows=[{k:r[k] for k in keys if k in r} for r in records],planned=144))
solves=[]
for directory,pattern in [(STAGE/'run_v1','solve_*.json'),(RUN,'refine_*.json')]:
    for file in directory.glob(pattern):
        r=read(file);item={k:v for k,v in r.items() if k!='best'}
        if r.get('best'):item['returned_point_summary']={k:r['best'][k] for k in ['cost','planning_probability']}
        item['source_name']=file.name;item['source_sha256']=sha(file);solves.append(item)
write(OUT/'solver_records.json',dict(solves=solves,scope='scalar solver trace; private controls omitted'))
for suffix in ['png','pdf','svg']:
    shutil.copy2(STAGE/'public_figures'/('spatial_correlated_aggregate.'+suffix),OUT/('spatial_correlated_aggregate.'+suffix))
    if suffix=='svg':
        svg=OUT/'spatial_correlated_aggregate.svg'
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
import gurobipy,numpy,scipy
write(OUT/'environment.json',dict(python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__,gurobi=gurobipy.gurobi.version()))
manifest=dict(files={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json'},
    private_references={name:dict(path=str(RUN/name),sha256=sha(RUN/name)) for name in ['price_freeze.json','problem_frozen.json','predictions_frozen.json','ended.json']},
    privacy='No household, P/Q/V, full R/X, random arrays, daily controls or quote trajectories. Scalar outcomes, allocation and capacity aggregates only.')
write(OUT/'manifest.json',manifest)
print(json.dumps(dict(files=len(manifest['files']),directory=str(OUT))))
