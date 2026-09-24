import hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PACKAGE=ROOT/'theft_wzzt';sys.path.insert(0,str(PACKAGE))
OUT=PACKAGE/'outputs/theft_paper_v3'
protected=json.loads((PACKAGE/'outputs/theft_paper/protected_before.json').read_text())
result=dict(protected_count=len(protected),changed=[name for name,h in protected.items() if not (ROOT/name).is_file() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h],studies=[])
for name in ['theft_paper','theft_paper_v2','theft_paper_v3']:
 out=PACKAGE/'outputs'/name;protocol=json.loads((out/'frozen_protocol.json').read_text())
 changed=[f for f,h in protocol['source_sha256'].items() if hashlib.sha256((PACKAGE/f).read_bytes()).hexdigest()!=h]
 data=[json.loads(p.read_text()) for p in (out/'rows').glob('*.json')]
 key=lambda row:tuple((k,row[k]) for k in ['case','condition','role','replicate','bus','amp_kw','start'])
 expected={key(j) for j in protocol['jobs']};actual={key(d) for d in data}
 row=dict(study=name,source_changed=changed,expected=len(expected),rows=len(data),missing=len(expected-actual),unexpected=len(actual-expected),errors=[d for d in data if 'error' in d])
 assert not changed and not row['missing'] and not row['unexpected'] and not row['errors'] and len(data)==len(expected)
 result['studies'].append(row)
 for case in ['paper15','soumalas11','flynn16']:
  cal=json.loads((out/f'calibration_{case}.json').read_text())
  assert all(hashlib.sha256((PACKAGE/f).read_bytes()).hexdigest()==h for f,h in cal['inference_source_sha256'].items())
from experiments.run_theft import _m7_context
ctx=_m7_context();manifest=json.loads((PACKAGE/'outputs/theft/m7_run_manifest.json').read_text())
result['m7_signature_unchanged']=ctx['signature']==manifest['signature']
result['m7_context_signature']=ctx['signature']
result['original_pilot_files']=[str(p.relative_to(ROOT)) for p in (ROOT/'rnj_wzzt_core/tests/meter_theft_pilot').glob('*.py')]
assert len(result['original_pilot_files'])==4 and not result['changed'] and result['m7_signature_unchanged']
result['cli_examples']={}
for name,alarm,regions in [('null',False,None),('internal',True,[-21]),('terminal',True,[108])]:
 d=json.loads((OUT/f'example_{name}.json').read_text());assert d['decision']['alarm']==alarm and d['reported_regions']==regions
 result['cli_examples'][name]=dict(alarm=alarm,regions=regions)
log=(OUT/'all_tests.log').read_text(encoding='utf-8-sig');assert '32 passed' in log
result['tests']='32 passed'
report=ROOT/'docs/theft_paper_readiness_20260919.md'
links=re.findall(r'\]\(([^)]+)\)',report.read_text(encoding='utf-8'))
result['broken_local_links']=[x for x in links if not x.startswith('http') and not Path(x).is_file()]
assert not result['broken_local_links']
result['status']='verified'
(OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
