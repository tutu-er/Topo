"""Export bounded aggregate evidence; no household or P/Q/V trajectories."""
import argparse, hashlib, json
from pathlib import Path

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    rendered=json.dumps(v,indent=2,ensure_ascii=False)
    if len(rendered)>50000:
        compact=lambda x: json.dumps(x,ensure_ascii=False,separators=(',',':'))
        if isinstance(v,dict): rendered='{\n'+',\n'.join(compact(k)+':'+compact(x) for k,x in v.items())+'\n}'
        elif isinstance(v,list): rendered='[\n'+',\n'.join(compact(x) for x in v)+'\n]'
    p.write_text(rendered+'\n',encoding='utf-8')

parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
protocol=read(a.run/'protocol.json');summary=read(a.run/'summary.json');audit=read(a.run/'audit_final.json')
if summary['status']!='complete' or not audit['all_passed']:raise ValueError('Complete audited run required')
source_files=[a.run/n for n in ('protocol.json','summary.json','audit_first8.json','audit_final.json')]
locked={k:v for k,v in protocol.items() if k not in ('actions','environment')}
locked['actions']=[{'node':x['node'],'name':x['name'],'menus':x['menus'],
    'prediction_feasible':x['planning_audit']['feasible'],
    'conservative_payment_bound_nok':x['planning_audit']['conservative_payment_bound_nok'],
    'original_d_opt_score':x['planning_audit']['score']} for x in protocol['actions']]
locked['environment']=protocol['environment'];locked['private_full_protocol_sha256']=sha(a.run/'protocol.json')
write(a.output/'protocol.json',locked)
cells=[];fits={}
for row in summary['records']:
    cell={k:v for k,v in row.items() if k not in ('periods','physical_files')}
    cell['periods']={name:{'payment_nok':v['payment_nok'],'response_valid':v['response_audit']['valid'],
        'ac_usable':v['physical_checks']['usable_ac_solution'],'voltage_pass':v['physical_checks']['voltage_limits_pass']} for name,v in row.get('periods',{}).items()}
    cell['physical_files_sha256']=row.get('physical_files',{})
    cells.append(cell)
    if row['status']=='evaluated' and row['observation_key'] not in fits:
        f=read(Path(row['fit_result']))
        fits[row['observation_key']]={'result_sha256':row['fit_sha256'],'old_reused':row['old_cache_reused'],
            'elapsed_seconds':row['identifier_seconds'],'search_stop_statuses':f['search_stop_statuses'],
            'LP_count':len(f['LP_solves']),'models':{name:{k:v for k,v in m.items() if k in
              ('supports','support_labels','r','x','r_values','x_values','objective','train_mae','validation_mae','selection_validation_mae','origin','selection_index')}
              for name,m in f['models'].items()}}
write(a.output/'cell_results.json',cells)
write(a.output/'fit_summaries.json',fits)
write(a.output/'summary.json',{k:v for k,v in summary.items() if k!='records'})
compact={k:v for k,v in audit.items() if k not in ('fits','cells','physical_records')}
compact['fit_checks']=[{k:v for k,v in f.items() if k in ('key','first_task','result_sha256','issues','LP_count','old_cache_reused','pass_audit','models')} for f in audit.get('fits',[])]
write(a.output/'arithmetic_audit.json',compact)
for name in ('prior_selection_diagnosis.json','new_location_mechanisms_102_103.json','aggregate_counts.json'):
    p=a.run/name
    if p.exists():
        write(a.output/name,read(p));source_files.append(p)
manifest={'private_sources':{str(p):sha(p) for p in source_files},'exporter_sha256':sha(Path(__file__)),
    'outputs':{p.name:sha(p) for p in a.output.glob('*.json') if p.name!='evidence_manifest.json'},'raw_household_or_pqv_included':False}
write(a.output/'evidence_manifest.json',manifest)
print(json.dumps({'cells':len(cells),'fits':len(fits),'files':len(manifest['outputs'])}))
