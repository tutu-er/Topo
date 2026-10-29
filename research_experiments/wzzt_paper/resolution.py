"""Exploratory filtering sensitivity; never selects a floor using test truth."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from research_experiments.wzzt_systematic.common import read,matrices,linear_loss,write
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth,clade_metrics
from .common import observed


def guard_filter(point,calibration,train,validation):
    """Protect the final filter using validation only; never exceeds floor .16."""
    from research_experiments.wzzt_refine.core import scored,select_pre
    candidates=[]
    for floor in (0.,.04,.08,.16):
        keep=[i for i,s in enumerate(point['supports']) if len(s)==1 or calibration['a']*point['r'][i]+calibration['b']*point['x'][i]+1e-10>=floor*calibration['median_root_depth']]
        model=dict(supports=[point['supports'][i] for i in keep],r=[point['r'][i] for i in keep],x=[point['x'][i] for i in keep],
                   final_floor=floor,origin='postfilter_validation_guard')
        R,X=matrices(train[0].shape[1],model['supports'],model['r'],model['x']);model['objective']=linear_loss(train,R,X)['mae']
        candidates.append(scored(model,validation))
    chosen=select_pre(candidates)
    # The original .16 family has the fewest supports in this nested set.
    # It is retained when eligible; otherwise its validation error is larger.
    if chosen['validation_mae']>candidates[-1]['validation_mae']+1e-12:raise AssertionError('validation protection failed')
    return chosen


def analyze(output,destination):
    output=Path(output).resolve();destination=Path(destination);records=[];truths={};guarded=[];guard_models=[]
    main=pd.read_csv(destination/'per_job.csv');original=main[main.variant=='ridge_union'].set_index('job_id').clade_f1
    for job in read(output/'protocol.json')['jobs']:
        folder=output/'jobs'/job;p=read(folder/'payload.json');file=folder/'result.json' if (folder/'result.json').exists() else folder/'checkpoint.json'
        header={k:p[k] for k in ('job_id','panel','case','condition','bundle','N','voltage_sigma')}
        r=read(file) if file.exists() else {}
        if 'ridge_union' not in r.get('models',{}):
            records.extend(dict(header,floor=f,model_available=False,clade_f1=0.) for f in (0.,.04,.08,.16,.24))
            guarded.append(dict(header,model_available=False,clade_f1=0.,delta_vs_ridge_union=-original[job]));continue
        points=r['base_points']+r['candidates']['ridge_union'];point=points[r['models']['ridge_union']['selection_index']]
        labels=r['labels'];cal=r['calibration'];train=observed(p['train'],p['N']);validation=observed(p['validation']);test=observed(p['test'])
        if p['truth_network'] not in truths:truths[p['truth_network']]=physical_truth(p['truth_network'],labels)['clades']
        for floor in (0.,.04,.08,.16,.24):
            keep=[i for i,s in enumerate(point['supports']) if len(s)==1 or cal['a']*point['r'][i]+cal['b']*point['x'][i]+1e-10>=floor*cal['median_root_depth']]
            supports=[point['supports'][i] for i in keep];rv=[point['r'][i] for i in keep];xv=[point['x'][i] for i in keep]
            R,X=matrices(len(labels),supports,rv,xv);predicted={frozenset(labels[i] for i in s) for s in supports if 2<=len(s)<len(labels)}
            scores=clade_metrics(predicted,truths[p['truth_network']],labels)
            if floor==.16 and abs(scores['clade_f1']-original[job])>1e-12:raise AssertionError('fixed 0.16 differs from original output')
            records.append(dict(header,floor=floor,model_available=True,**scores,validation_mae=linear_loss(validation,R,X)['mae'],
                test_mae=linear_loss(test,R,X)['mae'],supports=len(supports),fixed_preselected_model=True,truth_used_only_posthoc=True))
        model=guard_filter(point,cal,train,validation);R,X=matrices(len(labels),model['supports'],model['r'],model['x'])
        predicted={frozenset(labels[i] for i in s) for s in model['supports'] if 2<=len(s)<len(labels)}
        guarded.append(dict(header,model_available=True,**clade_metrics(predicted,truths[p['truth_network']],labels),
            validation_mae=model['validation_mae'],test_mae=linear_loss(test,R,X)['mae'],final_floor=model['final_floor'],
            delta_vs_ridge_union=clade_metrics(predicted,truths[p['truth_network']],labels)['clade_f1']-original[job],
            rule_uses_truth=False,analysis_type='exploratory; frozen after pilot and main-panel diagnosis'))
        guard_models.append(dict(job_id=job,labels=labels,point=model))
    frame=pd.DataFrame(records);frame.to_csv(destination/'postfilter_sensitivity.csv',index=False)
    frame.groupby(['panel','floor']).agg(rows=('job_id','count'),F1=('clade_f1','mean'),exact=('clade_f1',lambda s:int((s>=1-1e-12).sum())),
        validation_mae=('validation_mae','mean'),test_mae=('test_mae','mean')).to_csv(destination/'postfilter_summary.csv')
    pd.DataFrame(guarded).to_csv(destination/'postfilter_guard.csv',index=False)
    with (destination/'postfilter_guard_models.jsonl').open('w',encoding='utf-8') as stream:
        for row in guard_models:stream.write(__import__('json').dumps(row)+'\n')
    pd.DataFrame(guarded).groupby('panel').agg(rows=('job_id','count'),F1=('clade_f1','mean'),exact=('clade_f1',lambda s:int((s>=1-1e-12).sum())),
        delta_vs_ridge_union=('delta_vs_ridge_union','mean'),wins=('delta_vs_ridge_union',lambda s:int((s>1e-10).sum())),
        ties=('delta_vs_ridge_union',lambda s:int((abs(s)<=1e-10).sum())),losses=('delta_vs_ridge_union',lambda s:int((s < -1e-10).sum())),
        validation_mae=('validation_mae','mean'),test_mae=('test_mae','mean')).to_csv(destination/'postfilter_guard_summary.csv')
    from research_experiments.wzzt_paper.statistics import interval
    g=pd.DataFrame(guarded);prospective=g[((g.panel=='topology')&(g['case']!='random32_t0'))|((g.panel=='case33')&(g.bundle>0))]
    prospective.to_csv(destination/'postfilter_guard_holdout.csv',index=False);summary=[]
    for panel,d in prospective.groupby('panel'):
        block=d['case'] if panel=='topology' else d.bundle
        summary.append(dict(panel=panel,inputs=len(d),F1=d.clade_f1.mean(),**interval(d.delta_vs_ridge_union.groupby(block).mean()),
            wins=int((d.delta_vs_ridge_union>1e-10).sum()),ties=int((abs(d.delta_vs_ridge_union)<=1e-10).sum()),losses=int((d.delta_vs_ridge_union < -1e-10).sum())))
    pd.DataFrame(summary,columns=['panel','inputs','F1','mean','lower','upper','independent_blocks','positive_blocks','negative_blocks','sign_p','wins','ties','losses']).to_csv(destination/'postfilter_guard_holdout_summary.csv',index=False)
    print('postfilter sensitivity',len(frame),'rows; all original 0.16 outputs matched',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('destination');a=p.parse_args();analyze(a.output,a.destination)
