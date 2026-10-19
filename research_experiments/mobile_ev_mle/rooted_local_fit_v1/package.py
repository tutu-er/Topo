"""Compact representatives and postprocessing provenance; full raws stay local."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
from .experiment import load,save,sha


def compact(record,method):
    keys=('loss','J','r','x','theta','qualified','status','physical','projected_gradient',
          'attempts','seconds','ac_calls','information','lines_updated','absolute_gap','lower_bound',
          'initial_theta','initial_qp','polish_trace')
    result={k:record[k] for k in keys if k in record}
    if method=='AC':result.update({k:record[k] for k in ('p','q','root_v','v') if k in record})
    if 'covariance' in record:
        result['constant_covariance']=record['covariance'][0]
        result['covariance_rule']='broadcast over time; all original meter sigmas are constant'
    return result


def package(folder,evidence):
    folder,evidence=Path(folder),Path(evidence)
    selected={f'{name}_36001_672_FULL':None for name in ('paper15','paper11')}
    for source in (folder/'inference').glob('*.json.gz'):
        record=load(source)
        failed=[key for key,value in record['fits'].items() if not value.get('qualified',False)]
        if failed:selected[source.name.removesuffix('.json.gz')]=failed
    manifest={}
    for key,labels in selected.items():
        inference=load(folder/'inference'/f'{key}.json.gz');record=load(folder/'records'/f'{key}.json.gz')
        if labels is None:labels=[label for label in inference['fits'] if label.startswith('base/')]
        obj=dict(spec=inference['spec'],input_sha256=inference['input_sha256'],
            original_inference_sha256=sha(folder/'inference'/f'{key}.json.gz'),
            selected_labels=labels,selection='predeclared first seed FULL base, plus every unqualified training fit',
            training={label:compact(inference['fits'][label],label.split('/')[1]) for label in labels},
            held_AC={label:compact(record['common_ac_holdout'][label],'AC') for label in labels if label in record['common_ac_holdout']})
        target=evidence/'representative_compact'/f'{key}.json.gz';save(target,obj)
        shutil.copyfile(folder/'inputs'/f'{key}.npz',target.with_suffix('').with_suffix('.npz'))
        manifest[key]=dict(compact_sha256=sha(target),labels=labels)
    provenance=dict(representatives=manifest,
        postprocessing_sha256={name:sha(Path(__file__).with_name(name)) for name in ('finish.py','package.py')},
        no_scientific_settings_changed=True)
    save(evidence/'package_manifest.json',provenance)
    print(dict(representative_conditions=len(selected),unqualified_training_fits=sum(len(x or []) for x in selected.values())))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--evidence',required=True)
    args=parser.parse_args();package(args.input,args.evidence)
