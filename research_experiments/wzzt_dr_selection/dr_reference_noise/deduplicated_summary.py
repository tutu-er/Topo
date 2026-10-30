"""Supplementary unique-model counts; preserves all original 72 mappings."""
import hashlib,json,os
from pathlib import Path
from collections import defaultdict,Counter
BASE=Path(os.environ.get('WZZT_REFERENCE_NOISE_SCRATCH',r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a'))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    truth=BASE/'wzzt_dr_reference_noise_generation_v1/evaluator_truth.json'
    fitted=BASE/'wzzt_dr_reference_noise_execution_v1/independent_structure_summary.json'
    cases={c['id']:c for c in read(truth)['cases']};rows=read(fitted)['rows'];unique={};multiplicity=Counter()
    for r in rows:
        t=cases[r['case']]['truth'];identity=json.dumps([t['R'],t['X']],separators=(',',':'))
        key=(r['noise'],r['policy'],r['method'],identity);multiplicity[key]+=1
        if key in unique:
            before=unique[key]
            for field in ('snapshot','status','payment','raw_error','raw_recovered','final_error','final_recovered','raw_missing','raw_false','final_missing','final_false'):
                if before.get(field)!=r.get(field):raise AssertionError('duplicate model has different result '+field)
        else:unique[key]=r
    groups=defaultdict(list)
    for k,r in unique.items():groups[k[:3]].append(r)
    table=[]
    for (noise,policy,method),values in groups.items():
        completed=[r for r in values if r['status']=='complete']
        if len(values)!=10:raise AssertionError('expected10 actual generating models per group')
        table.append(dict(noise=noise,policy=policy,method=method,unique_models=10,complete=len(completed),
            statuses=dict(Counter(r['status'] for r in values)),
            final_errors=sum(r['final_error'] for r in completed) if completed else None,
            final_full=sum(r['final_recovered'] for r in completed) if completed else None,
            raw_errors=sum(r['raw_error'] for r in completed) if completed else None,
            raw_full=sum(r['raw_recovered'] for r in completed) if completed else None))
    out=BASE/'wzzt_dr_reference_noise_execution_v1/deduplicated_summary.json'
    if out.exists():raise FileExistsError('supplement already retained')
    result=dict(status='passed',original_structure_mappings=len(rows),unique_structure_model_policy_method_entries=len(unique),
        identity='true R and X matrices plus noise/policy/method; evaluator only, not used in design',
        duplicate_aliases_verified_equal=True,table=table,source_sha256=sha(__file__),
        input_sha256={str(truth):sha(truth),str(fitted):sha(fitted)},new_LP=0,new_data=0,
        independent_random_trials=False,original_mapping_results_modified=False)
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('table','input_sha256')},indent=2))
if __name__=='__main__':main()
