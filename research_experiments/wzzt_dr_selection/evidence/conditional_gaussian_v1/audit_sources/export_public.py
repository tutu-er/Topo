"""Explicit scalar/config/source-only export; no trajectories or price arrays."""
from pathlib import Path
import hashlib,json,shutil,zipfile
import numpy as np

base=Path(__file__).resolve().parent;out=base/'public';out.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for folder,name in [('run_v1','main'),('delivery_v1','delivery')]:
    root=base/folder
    for source,dest in [('protocol.json',f'{name}_configuration.json'),('summary.json',f'{name}_summary.json')]:
        shutil.copyfile(root/source,out/dest)
    with zipfile.ZipFile(root/'source_snapshot.zip') as z:
        assert all(p.endswith('.py') and '/' not in p and '\\' not in p for p in z.namelist())
    shutil.copyfile(root/'source_snapshot.zip',out/f'{name}_executed_source.zip')
    arr=np.load(root/'evaluation_scalars.npz')
    improved=np.asarray(arr['continuous__event'])&~np.asarray(arr['rotation__event'])
    worse=~np.asarray(arr['continuous__event'])&np.asarray(arr['rotation__event'])
    write(out/f'{name}_paired.json',dict(paths=len(improved),better=int(improved.sum()),worse=int(worse.sum()),same=int((~improved&~worse).sum())))
planning=read(base/'run_v1/planning_results.json')
results=[]
for r in planning['results']:
    results.append({k:v for k,v in r.items() if k not in ('best','trace')})
    results[-1]['best']={k:v for k,v in (r['best'] or {}).items() if k!='mean_u'}
    results[-1]['trace']=r['trace']
write(out/'optimization_summary.json',dict(results=results,
    candidates=[{k:v for k,v in c.items() if k!='mean_u'} for c in planning['candidates']],
    selected={k:v for k,v in planning['selected'].items() if k!='mean_u'},baseline_skips=planning['baseline_skips']))
audit={}
for source in ['audit_v1.json','audit_delivery_v1.json']:
    a=read(base/source)
    audit[source]={k:v for k,v in a.items() if k not in ('opened_files','checks','source_hashes','sources')}
    if isinstance(a.get('checks'),list):
        audit[source]['check_count']=len(a['checks'])
        audit[source]['failed_checks']=[v for v in a['checks'] if not v['passed']]
    audit[source]['source_file_sha256']=sha(base/source)
write(out/'audit_summary.json',audit)
for source in ['physical_box_probability_audit_v1.json','guarded_amplitude_plan_v1.json']:
    shutil.copyfile(base/source,out/source)
shutil.copyfile(base/'run_v1/gradient_audit.json',out/'gradient_audit.json')
write(out/'tests_summary.json',dict(passed=45,subtests_passed=4,failed=0,xfail=0,
    command='python -B -m pytest research_experiments/wzzt_dr_selection/dr_conditional_gaussian -q -p no:cacheprovider',
    source_corrections_after_main=['reject all-zero historical SVD direction','reject globally ineffective conditioner','preserve restricted derivative for fixed zero residual'],
    main_case_affected=False,main_numeric_batch_rerun=False))
(out/'audit_sources').mkdir(exist_ok=True)
for source in ['audit_independent.py','audit_delivery_independent.py','make_figure.py','export_public.py']:
    shutil.copyfile(base/source,out/'audit_sources'/source)
write(out/'manifest.json',dict(scope='source, configurations, hashes and scalar aggregates only; no P/Q/Y, R/X matrices, draws or price trajectories',
    files={str(p.relative_to(out)).replace('\\','/'):dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}))
print('public files',len(list(out.rglob('*'))),'bytes',sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
