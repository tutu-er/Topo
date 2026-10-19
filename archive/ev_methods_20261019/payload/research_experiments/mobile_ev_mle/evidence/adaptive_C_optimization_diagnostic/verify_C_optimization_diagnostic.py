"""Verify the saved compact diagnostic offline; standard library only."""
from pathlib import Path
import collections,gzip,hashlib,json,math
root=Path(__file__).resolve().parent
data=(root/'raw_extracts.json.gz').read_bytes()
inputs=json.loads((root/'input_hashes.json').read_text(encoding='utf-8'))
assert hashlib.sha256(data).hexdigest()==inputs['raw_extracts_compressed_sha256']
extracts=json.loads(gzip.decompress(data));assert len(extracts)==256
numbers=json.loads((root/'C_optimization_diagnostic_numbers.json').read_text(encoding='utf-8'))
classes=collections.Counter();stops=collections.Counter();solver=collections.Counter();paired=collections.Counter()
library_scores=0
for x in extracts:
    assert x['source_trial_sha256']==inputs['trial_sha256']['trial_'+x['history_id']+'.json']
    r=x['refit'];intervals=x['refit_selected_record_intervals'];lib=x['legacy_library'];d=x['diagnostic']
    lo=max(a['q_lower'] for a in intervals);hi=min(a['q_upper'] for a in intervals)
    assert lo==r['proxy_lower'] and hi==d['selected_action_q_upper'] and lo==d['selected_action_q_lower']
    assert d['global_absolute_gap']==max(0,r['proxy_upper']-r['proxy_lower'])==r['absolute_gap']
    assert d['global_relative_gap']==r['absolute_gap']/max(1,abs(r['proxy_lower']))==r['relative_gap']
    assert d['stopping_tolerance']==1e-4+1e-3*max(1,abs(r['proxy_lower']))
    assert r['proxy_gap_met']==(r['absolute_gap']<=d['stopping_tolerance'])
    library_scores+=len(lib['actions'])
    if lib['status']=='complete':
        assert len(lib['actions'])==11 and all(a['success'] for a in lib['actions'])
        chosen=min(lib['actions'],key=lambda a:(-a['q_lower'],a['action']));assert chosen['action']==lib['selected']
        assert chosen['q_lower']<=r['proxy_upper']
        cls='refit_LB_gt_library_UB' if lo>chosen['q_upper'] else 'library_LB_gt_refit_UB' if chosen['q_lower']>hi else 'overlap'
    else:cls='library_selection_failure'
    assert cls==d['interval_comparison'];classes[cls]+=1;stops[r['proxy_stop_reason']]+=1
    constant=next(a for a in lib['actions'] if a['action']=='constant_site_0')
    assert d['gain_vs_constant_lower']==lo-constant['q_upper'] and d['gain_vs_constant_upper']==hi-constant['q_lower']
    for s in r['solver_records']:solver[(s['kind'],s['status'],s['success'],s['incumbent'] is None,s['bound'] is None)]+=1
    paired[d['unique_correct_pair_category']]+=1
assert dict(classes)==numbers['interval_comparison_counts'] and dict(stops)==numbers['stop_reason_counts']
assert library_scores==2806 and dict(paired)==numbers['paired_unique_correct_counts']
expected={(x['kind'],x['status'],x['success'],x['incumbent_missing'],x['bound_missing']):x['count'] for x in numbers['solver_records']}
assert dict(solver)==expected
assert solver[('secant_upper',1,False,True,True)]==515
assert numbers['scale_and_sensitivity_counts']['gap_met_only_relative_part_not_absolute']==sum(x['refit']['proxy_gap_met'] and x['refit']['absolute_gap']>1e-4 for x in extracts)==126
print(json.dumps({'status':'PASS','history_count':256,'library_action_score_rows':2806,'all_interval_and_gap_counts_match':True,
                  'interval_counts':dict(classes),'all_secant_calls_time_limit_no_bound':515,'no_project_imports_or_optimization':True},ensure_ascii=False,indent=2))
