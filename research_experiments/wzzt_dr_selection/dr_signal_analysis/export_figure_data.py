"""Keep the main plot data small; full old scalar records remain archived."""
from pathlib import Path
import argparse,json,hashlib

def export(source,output):
    source=Path(source);a=json.loads(source.read_text(encoding='utf-8'))
    keys=['status','requested_records','complete_records','planned_condition_days','valid_condition_days',
        'cases','plans','source_hashes','computation','threshold_warning']
    out={k:a[k] for k in keys}
    out['records']=[{k:v for k,v in r.items() if k!='score_scalar_ref'} for r in a['records']]
    chosen={('paper15_b2_v0005',1501,'w1_tau2pct'),('flynn16_b4_v0005',1512,'w2_tau1pct')}
    out['rule_records']=[{k:v for k,v in r.items() if not k.startswith('triangle_') and k!='observed_reference_residual_micro'}
        for r in a['rule_records'] if (r['condition'],r['path'],r['rule_id']) in chosen]
    for c in out['cases']:
        for p in c['policies']:
            p['relative_gain_of_case_means_pct']=100*p['distance_gain_micro']/p['distance_baseline_micro']
    out['private_full_scalar_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    Path(output).write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(records=len(out['records']),rule_examples=len(out['rule_records']),raw_array_exports=0)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();export(a.source,a.output)
