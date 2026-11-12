"""Check the delivered records against frozen inputs, raw jobs and archives."""
from pathlib import Path
import argparse
import json
import zipfile
import hashlib
import numpy as np
import pandas as pd
from research_experiments.wzzt_systematic.common import read,write,sha
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
from .common import MAIN,TARGET,AC_METHODS
from .records import find,load,load_trace
from .evaluate import signature

ROOT=Path(__file__).resolve().parents[2]


def verify(output,destination):
    output=Path(output).resolve();destination=Path(destination).resolve();checks=[]
    def check(name,passed):checks.append(dict(check=name,passed=bool(passed)))
    def hashes(name,base,manifest):
        items=read(manifest)
        check(name,all((base/p).is_file() and sha(base/p)==h for p,h in items.items()))
        return items
    frozen=hashes('frozen producer, data and job payloads',output,output/'manifest.json')
    hashes('all saved raw result and AC bytes',output,destination/'result_manifest.json')
    analysis=hashes('current analysis source bytes',Path(__file__).parent,destination/'analysis_source_manifest.json')
    hashes('shared independent audit helper bytes',ROOT,destination/'analysis_helper_manifest.json')
    hashes('delivery bytes before verification',destination,destination/'delivery_manifest.json')
    for archive,items in (('producer_source.zip',{p.removeprefix('source/'):h for p,h in frozen.items() if p.startswith('source/')}),
                          ('analysis_source.zip',analysis)):
        with zipfile.ZipFile(destination/archive) as z:
            check(archive+' matches frozen source entries',all(p in z.namelist() and hashlib.sha256(z.read(p)).hexdigest()==h for p,h in items.items()))
    protocol=read(output/'protocol.json');completion=read(output/'completion.json');overview=read(destination/'overview.json')
    check('180 completed independent-input conditions',protocol['planned']==completion['returned']==overview['returned_inputs']==180 and not overview['incomplete_inputs'])
    for name in ('protocol.json','manifest.json','completion.json'):
        check('delivered '+name,sha(output/name)==sha(destination/name))
    frame=pd.read_csv(destination/'per_job.csv');rows=find(destination/'catalog.sqlite')
    check('complete 1620 main plus 144 targeted records',len(frame)==len(rows)==1764 and frame.model_available.all() and frame.worker_returned.all())
    check('main methods each on all 180 inputs',all(int(((frame.panel=='main')&(frame.variant==m)).sum())==180 for m in MAIN))
    check('six target methods each on fixed 24 inputs',all(int(((frame.panel=='targeted')&(frame.variant==m)).sum())==24 for m in TARGET))
    target=frame[frame.panel=='targeted']
    check('target subset noise and sizes',target.voltage_sigma.eq(.001).all() and target.N.isin([24,48]).all())
    check('three independent bundle IDs',set(frame.bundle)=={0,1,2} and len(protocol['bundles'])==3)
    index={(r['job_id'],r['variant']):r for r in rows}
    models=[json.loads(line) for line in (destination/'selected_models.jsonl').read_text(encoding='utf-8').splitlines()]
    check('all model signatures identical between JSONL and catalog',len(models)==1764 and all(signature(m['point'])==m['model_signature']==index[(m['job_id'],m['variant'])]['model_signature'] for m in models))
    raw_cache={};truths={};model_match=[];scores=[]
    for r in rows:
        if r['job_id'] not in raw_cache:raw_cache[r['job_id']]=read(r['result_path'])
        p=raw_cache[r['job_id']]['models'][r['variant']]
        model_match.append(signature(p)==r['model_signature'])
        labels=json.loads(r['labels_json'])
        if r['case_name'] not in truths:truths[r['case_name']]=physical_truth(Path(r['truth_network']),labels)['clades']
        truth=truths[r['case_name']]
        predicted={frozenset(labels[i] for i in s) for s in p['supports'] if 2<=len(s)<len(labels)}
        f1=2*len(predicted&truth)/(len(predicted)+len(truth))
        scores.append(abs(f1-r['clade_f1'])<=1e-12)
    check('catalog models identical to all raw selected models',all(model_match))
    check('all structural F1 recomputed from physical truth',all(scores))
    ac=read(destination/'ac_summary.json');planned=frame[frame.ac_planned]
    check('AC scope three predeclared methods only',set(planned.variant)==set(AC_METHODS) and len(planned)==540 and not frame[~frame.ac_planned].ac_evaluated.any())
    check('all planned AC records evaluated and physically valid',ac['planned_rows']==ac['evaluated_rows']==ac['valid_rows']==540 and ac['infrastructure_errors']==0)
    check('AC cache deduplication count',ac['unique_records']==len(list((output/'ac').glob('*.json')))==overview['AC_unique_models'])
    for N in (36,72):
        model=load(destination/'catalog.sqlite',case='flynn16',voltage_sigma=.0005,bundle=2,N=N,variant='ridge_union',with_observations=True)
        check(f'load new N={N} observations without losing bundle identity',len(model['train'][0])==N and len(model['validation'][0])==len(model['test'][0])==288 and model['record']['bundle']==2)
        check(f'load new N={N} reconstructed matrices',np.isfinite(model['R']).all() and model['R'].shape==model['X'].shape==(len(model['labels']),)*2)
    trace=load_trace(destination/'catalog.sqlite',case='flynn16',voltage_sigma=.0005,bundle=2,N=36)
    check('trace reader preserves observation-only provenance',trace['truth_used_by_estimator'] is False)
    check('independent data, LP and physical audits passed',read(destination/'audit_summary.json')['all_passed'])
    parity=read(destination/'legacy_parity.json');check('four historical parity checks',len(parity)==4 and all(r['passed'] for r in parity))
    summary=pd.read_csv(destination/'main_summary.csv').set_index('variant')
    check('reported means match complete paired panel',all(abs(frame[frame.variant==m].clade_f1.mean()-summary.loc[m,'F1'])<1e-12 for m in MAIN))
    result=dict(all_passed=all(r['passed'] for r in checks),checks=checks,inputs=180,model_records=len(rows),AC_evaluated=ac['evaluated_rows'])
    write(destination/'delivery_verification.json',result)
    write(destination/'delivery_manifest.json',{p.name:sha(p) for p in destination.iterdir() if p.is_file() and p.name!='delivery_manifest.json'})
    print(json.dumps(dict(all_passed=result['all_passed'],checks=len(checks),failed=[r['check'] for r in checks if not r['passed']]),ensure_ascii=False),flush=True)
    if not result['all_passed']:raise AssertionError('delivery verification failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('destination');a=p.parse_args();verify(a.output,a.destination)
