"""Offline verification of compact C evidence, using only the Python standard library.

Run: python verify_C_compact.py [archive_directory]
No original repository, optimization libraries, or network are required.
"""
from pathlib import Path
import collections, csv, gzip, hashlib, io, json, math, sys

root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
def sha(data): return hashlib.sha256(data).hexdigest()
def raw(name):
    data=(root/name).read_bytes()
    return gzip.decompress(data) if name.endswith('.gz') else data
def js(name): return json.loads(raw(name).decode('utf-8-sig'))
def rows(name): return list(csv.DictReader(io.StringIO(raw(name).decode('utf-8-sig'))))
def b(value): return value in ('True','true','1')
def mask(values): return sum(int(v)<<i for i,v in enumerate(values))
manifest=js('evidence_manifest.json')
for item in manifest['archive_files']:
    packed=(root/item['archive_path']).read_bytes()
    assert len(packed)==item['archive_bytes'] and sha(packed)==item['archive_sha256'],item['archive_path']
    data=gzip.decompress(packed) if item['compression']=='gzip' else packed
    assert len(data)==item['source_bytes'] and sha(data)==item['source_sha256'],item['archive_path']
    if 'data_row_count' in item:
        assert len(list(csv.DictReader(io.StringIO(data.decode('utf-8-sig')))))==item['data_row_count']
for item in manifest['derived_files']:
    data=(root/item['archive_path']).read_bytes()
    assert len(data)==item['bytes'] and sha(data)==item['sha256'],item['archive_path']

counts=js('compact_counts.json'); independent=js('formal_C_independent_numbers.json')
summary=js('originals/evaluation_v3_audit/audit_summary.json.gz')
gates=js('originals/calibration_v3/calibrated_gates.json')['calibration']
source=js('source_manifests.json')
assert len(source['frozen_core_sources'])==15 and len(source['postprocessing_sources'])==3
for phase,expected in [('calibration',257),('evaluation',513)]:
    source_manifest=source['raw_input_manifests'][phase]
    assert 1+len(source_manifest['trial_sha256'])+len(source_manifest['observation_archive_sha256'])==expected
    assert source_manifest==summary[phase+'_manifest']
assert all(x['match'] and x['observed_sha256']==x['frozen_sha256'] for x in source['frozen_core_sources']+source['postprocessing_sources'])

scores=rows('calibration_scores.csv'); assert len(scores)==128
assert sum(int(r['complete']) for r in scores)==127 and sum(int(r['compute_failure']) for r in scores)==1
for score,key in [('relative_score','relative'),('absolute_score','absolute')]:
    vals=sorted(float(r[score]) for r in scores)
    assert sum(math.isinf(v) and v>0 for v in vals)==1
    rank=math.ceil((len(vals)+1)*(1-gates[key]['alpha']))
    assert rank==123 and vals[rank-1]==gates[key]['threshold']
    assert gates[key]['rank_tail_bound']==6/129
metadata={r['history_id']:r for r in js('evaluation_metadata.json.gz')}
assert len(metadata)==256 and not(set(metadata)&{r['history_id'] for r in scores})
assert all(r['independent_history_unit'] and not r['selection_uses_truth'] and not r['selection_uses_future_noise'] for r in metadata.values())
fits=rows('originals/evaluation_v3_audit/fits.csv.gz'); actions=rows('originals/evaluation_v3_audit/actions.csv.gz')
masks=rows('history_method_masks.csv.gz'); canon=counts['candidate_bit_order']
assert canon==['star','clade_01','clade_02','clade_12']
assert len(fits)==7168 and len(actions)==len(masks)==1792
groups=collections.defaultdict(dict)
for row in fits:
    key=(row['history_id'],row['method'],row['round']); assert row['candidate'] not in groups[key]
    groups[key][row['candidate']]=row
actions={(r['history_id'],r['method'],r['round']):r for r in actions}
methods=list(counts['method_event_counts']); events=list(counts['method_event_counts'][methods[0]])
observed={m:dict.fromkeys(events,0) for m in methods}
strata={s:{'denominator':sum(r['stratum']==s for r in metadata.values()),'methods':{m:dict.fromkeys(events,0) for m in methods}} for s in counts['strata']}
union=set(); results={}; recomputed=0
for row in masks:
    key=(row['history_id'],row['method'],row['round']); group=groups[key]; action=actions[key]; meta=metadata[row['history_id']]
    assert len(group)==4 and row['candidate_order']=='|'.join(canon)
    assert row['stratum']==meta['stratum'] and row['true_candidate']==meta['true_candidate']
    losses=[float(group[n]['loss']) for n in canon]; true=canon.index(meta['true_candidate'])
    complete=all(b(group[n]['success']) and math.isfinite(loss) and loss>=0 for n,loss in zip(canon,losses))
    relative=[complete and loss-min(losses)>gates['relative']['threshold'] for loss in losses]
    absolute=[complete and loss>gates['absolute']['threshold'] for loss in losses]
    combined=[r or a for r,a in zip(relative,absolute)]; retained=[not x for x in combined]
    for field,values in [('relative_reject_mask',relative),('absolute_reject_mask',absolute),('combined_reject_mask',combined),('retained_mask',retained)]:
        assert int(row[field])==mask(values),(key,field)
    physical=b(action['actual_voltage_ok']); selection_failure=action['selection_status']=='selection_failure_fallback'
    eliminated=sum(combined[i] for i in range(4) if i!=true)
    correct=complete and physical and sum(retained)==1 and retained[true]
    relative_retained=[not x for x in relative]
    ev={'false_rejection':combined[true],'truth_omission':False,'truth_loss_event':combined[true],
        'unique_correct_retention':correct,'algorithm_unique_correct_retention':correct and not selection_failure,
        'relative_unique_correct_retention':complete and physical and sum(relative_retained)==1 and relative_retained[true],
        'wrong_unique_retention':sum(retained)==1 and not retained[true],'unresolved':sum(retained)>1,
        'empty_retained':sum(retained)==0,'selection_failure':selection_failure,'fit_failure':not complete,
        'physical_failure':not physical,'bank_adequacy_alarm':complete and min(losses)>gates['absolute']['threshold'],
        'any_wrong_candidate_eliminated':eliminated>0,'all_wrong_candidates_eliminated':eliminated==3}
    for event,value in ev.items():
        assert int(row[event])==int(value),(key,event)
        observed[row['method']][event]+=int(value); strata[meta['stratum']]['methods'][row['method']][event]+=int(value)
    if ev['false_rejection']: union.add(row['history_id'])
    assert int(row['complete'])==int(complete) and int(row['operational_voltage_ok'])==int(physical)
    assert int(row['wrong_candidates_eliminated'])==eliminated
    results[key[:2]]=ev; recomputed+=1
assert observed==counts['method_event_counts'] and strata==counts['strata']
for method in methods:
    for event,value in observed[method].items():
        assert value==summary['methods'][method]['events'][event]['count']
    for event,value in independent['method_event_counts'][method].items(): assert value==observed[method][event]
assert len(union)==14 and sorted(union)==counts['all_methods_false_rejection_union']['history_ids']
assert results[('730180','random_action')]['fit_failure'] and results[('730085','legacy_library_proxy')]['selection_failure']
weak=next(r for r in masks if r['history_id']=='730121' and r['method']=='random_action')
assert int(weak['relative_reject_mask'])==0 and int(weak['absolute_reject_mask'])==7 and int(weak['retained_mask'])==8
paired={}
for event in ['unique_correct_retention','false_rejection']:
    paired[event]={'event_0_to_1':sum(results[(tid,'refit_milp')][event] and not results[(tid,'constant_fixed')][event] for tid in metadata),
                   'event_1_to_0':sum(not results[(tid,'refit_milp')][event] and results[(tid,'constant_fixed')][event] for tid in metadata)}
assert paired['unique_correct_retention']=={'event_0_to_1':9,'event_1_to_0':0}
assert paired['false_rejection']=={'event_0_to_1':2,'event_1_to_0':0}
assert sum(r['ac_audit_status']=='verified' for r in fits)==7168
assert sum(b(r['global_optimum_certified']) or b(r['ac_audit_global_optimum_certified']) for r in fits)==0
crossings=js('originals/evaluation_v3_audit/raw_bound_crossings.json')
assert len(crossings['calibration'])==3 and len(crossings['evaluation'])==6
assert max(x['relative_crossing'] for x in crossings['evaluation'])==6.503972900110877e-7
print(json.dumps({'status':'PASS','offline':True,'project_imports':False,'optimization_run':False,
                  'archive_original_and_compressed_sha256_verified':len(manifest['archive_files']),
                  'derived_file_sha256_verified':len(manifest['derived_files']),
                  'calibration_total_complete_failure':[128,127,1], 'rank':123,
                  'calibration_positive_infinity_preserved':True,'evaluation_histories':256,
                  'candidate_fits':len(fits),'method_histories_recomputed':recomputed,
                  'all_masks_and_full_summary_counts_match':True,'union_false_rejection':14,
                  'refit_vs_constant_paired_event_transitions':paired,
                  'note':'source hashes were verified from originals at packaging; source content need not be present offline'},ensure_ascii=False,indent=2))
