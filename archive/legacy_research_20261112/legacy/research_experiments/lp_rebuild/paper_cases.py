"""Replay frozen methods on the earlier four-paper-case comparison matrix."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from .run import ROOT,PARENT,read,write,sha,dispatch,tasks_for

FROZEN_PARENT=ROOT/'outputs/lp_rebuild_regularization_20261031_corrected'


def selected_jobs(protocol,catalog):
    return [job for job in protocol['evaluation'] if catalog[job]['panel'] in ('main','stress')
            and catalog[job]['case'] in ('paper15','soumalas11','flynn16','pengwah18')]


def freeze(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    prior=read(PARENT/'protocol.json');catalog=read(PARENT/'evaluation_catalog.json');jobs=selected_jobs(prior,catalog)
    assert len(jobs)==120 and sum(catalog[j]['panel']=='main' for j in jobs)==108
    source=FROZEN_PARENT/'source';source_protocol=read(FROZEN_PARENT/'protocol.json')
    for name,digest in source_protocol['source_hashes'].items():
        if sha(source/name)!=digest:raise ValueError('frozen inference source changed: '+name)
    shutil.copytree(source,output/'source')
    metadata={};reused={}
    for job in jobs:
        p=catalog[job];metadata[job]=p;target=output/'common'/job
        source_file=PARENT/'common'/job/'result.json';common=read(source_file)
        assert common['source_input_hashes']==dict(train=sha(p['train']),validation=sha(p['validation']))
        target.mkdir(parents=True);shutil.copy2(source_file,target/'result.json');reused[job]=sha(source_file)
        write(target/'payload.json',dict(job=job,N=p['N'],train=p['train'],validation=p['validation'],train_sha256=sha(p['train']),
            validation_sha256=sha(p['validation']),destination=str(target),result=str(target/'result.json')))
    write(output/'catalog.json',metadata)
    protocol=dict(source_protocol,created_utc=datetime.now(timezone.utc).isoformat(),development=[],evaluation=jobs,
        experiment='four fixed paper-style cases; exact earlier 108 main + 12 stress input matrix',
        parent_comparison_protocol_sha256=sha(PARENT/'protocol.json'),parent_parameter_sha256=sha(FROZEN_PARENT/'frozen.json'),
        tuning_performed=False,source_hashes=source_protocol['source_hashes'],frontend_reused=reused,
        evaluation_scope='known historical reference inputs, matched regression/reproduction comparison; not new blind data',
        main='4 cases x bundles2/3/4 x V sigma .0002/.0005/.001 x N24/48/96 =108',
        stress='4 cases x weak_excitation/root_noise/outliers x bundle2 x N48 =12',
        controller_source_sha256=sha(__file__))
    write(output/'protocol.json',protocol)
    original=FROZEN_PARENT/'frozen.json';shutil.copy2(original,output/'original_frozen.json')
    frozen=dict(read(original),protocol_sha256=sha(output/'protocol.json'),parent_frozen_sha256=sha(original),parameters_changed=False)
    write(output/'frozen.json',frozen)
    write(output/'replay_receipt.json',dict(parent=FROZEN_PARENT.as_posix(),unchanged_parameters=True,
        inference_source='exact prior frozen source; no algorithm modifications',original_frozen_sha256=sha(original),
        reused_frontends=120,inputs=120,planned_workers=120*len(frozen['evaluation_settings']),
        development_from='previous four-network development data, imported selection; no retuning in this replay'))
    print('FROZEN REPLAY',len(jobs),'inputs',120*len(frozen['evaluation_settings']),'workers',flush=True)


def analyze(output):
    from .analyze import evaluate
    output=Path(output).resolve();evaluate(output)
    destination=output/'analysis';catalog=read(output/'catalog.json');frame=pd.read_csv(destination/'per_model.csv')
    for name in ('panel','N','condition'):frame[name]=frame.job.map(lambda j:catalog[j][name])
    frame.to_csv(destination/'per_model.csv',index=False)
    summary=frame.groupby(['panel','view','method','config_id']).agg(inputs=('job','count'),available=('model_available','sum'),
        F1=('clade_f1','mean'),exact=('exact','sum'),train_MAE=('train_mae','mean'),validation_MAE=('validation_mae','mean'),
        test_MAE=('test_mae','mean'),R_error=('R_relative','mean'),X_error=('X_relative','mean'),internal=('internal','mean'),
        search_median=('search_seconds','median'),search_p90=('search_seconds',lambda s:s.quantile(.9)),
        total_estimate_median=('total_estimate_seconds','median')).reset_index()
    summary.to_csv(destination/'panel_summary.csv',index=False)
    pairs=[];blocks=[]
    for (panel,view),g in frame.groupby(['panel','view']):
        pivot=g.pivot(index='job',columns='method',values='clade_f1')
        for method in ('enum_control','best_floor','best_hybrid','native_corrected'):
            for reference in ('pool','native_joint','original_fullwarm','enum_control'):
                if method==reference:continue
                delta=pivot[method]-pivot[reference]
                pairs.append(dict(panel=panel,view=view,method=method,reference=reference,inputs=len(delta),mean_delta=delta.mean(),
                    wins=int((delta>1e-10).sum()),ties=int((delta.abs()<=1e-10).sum()),losses=int((delta< -1e-10).sum())))
                if panel=='main':
                    for bundle in (2,3,4):
                        ids=g[g.bundle==bundle].job.unique();d=delta.loc[ids]
                        blocks.append(dict(view=view,method=method,reference=reference,bundle=bundle,inputs=len(d),mean_delta=d.mean()))
    pd.DataFrame(pairs).to_csv(destination/'panel_paired.csv',index=False)
    pd.DataFrame(blocks).to_csv(destination/'paired_bundle.csv',index=False)
    main=frame[(frame.panel=='main')&(frame.view=='bank')]
    main.groupby(['case','method']).agg(inputs=('job','count'),F1=('clade_f1','mean'),exact=('exact','sum'),test_MAE=('test_mae','mean')).reset_index().to_csv(destination/'main_cases.csv',index=False)
    main.groupby(['voltage_sigma','N','method']).agg(inputs=('job','count'),F1=('clade_f1','mean'),exact=('exact','sum')).reset_index().to_csv(destination/'main_cells.csv',index=False)
    frame[(frame.panel=='stress')&(frame.view=='bank')].groupby(['condition','method']).agg(inputs=('job','count'),F1=('clade_f1','mean'),exact=('exact','sum'),test_MAE=('test_mae','mean')).reset_index().to_csv(destination/'stress_conditions.csv',index=False)
    frozen=read(output/'frozen.json');original=read(output/'original_frozen.json')
    assert frozen['methods']==original['methods'] and frozen['evaluation_settings']==original['evaluation_settings']
    write(destination/'paper_verification.json',dict(passed=True,inputs=120,main_inputs=108,stress_inputs=12,rows=len(frame),
        parameter_values_unchanged=True,original_frozen_sha256=sha(output/'original_frozen.json'),
        score_verification_sha256=sha(destination/'verification.json'),scored_rows_sha256=sha(destination/'per_model.csv'),
        controller_source_sha256=sha(__file__),statistical_scope='three main bundle blocks; stress same bundle2, paired conditions'))
    print(summary[(summary.panel=='main')&(summary.view=='bank')].to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','run','analyze'));p.add_argument('output');a=p.parse_args();output=Path(a.output).resolve()
    if a.command=='freeze':freeze(output)
    elif a.command=='analyze':analyze(output)
    else:
        protocol=read(output/'protocol.json');frozen=read(output/'frozen.json')
        dispatch(output,'evaluation',tasks_for(output,protocol['evaluation'],frozen['evaluation_settings'],'evaluation',sha(output/'frozen.json')))


if __name__=='__main__':main()
