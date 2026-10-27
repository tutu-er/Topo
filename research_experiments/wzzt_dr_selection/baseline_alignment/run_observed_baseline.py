"""Apply the default full RNJ/WZZT stages to frozen v1 no-reward observations."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

from .mainline_adapter import fit_mainline_observed
from research_experiments.wzzt_dr_selection.end_to_end import wzzt_adapter as observed
from research_experiments.wzzt_dr_selection.end_to_end.runner import _exclusive_lock

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def load_inputs(source):
    summary=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    if summary['status']!='complete':raise ValueError('source run is not completed')
    files=[source/'history'/(d+'.pkl') for d in summary['wzzt_training_dates']]
    files.extend([source/'strategies/no_reward/action.pkl',source/'strategies/no_reward/validation.pkl'])
    raw=[]
    for path in files:
        with path.open('rb') as f:item=pickle.load(f)
        raw.append(item['simulation'] if 'simulation' in item else item)
    def convert(sim):
        s=sim['observed']
        return observed.make_observed_scenario(s['P_terminal'],s['Q_terminal'],s['V_terminal'],s['root_voltage'],s['name'])
    return [convert(s) for s in raw[:-1]],[convert(raw[-1])],raw[-2]['truth']['truth_supports'],{str(p):sha(p) for p in files}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--normal-reference-run',type=Path,required=True)
    a=p.parse_args(argv);output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    normal=a.normal_reference_run.resolve()
    if not (normal/'protocol.json').exists():raise ValueError('lock the unchanged normal reference CLI protocol before the external-data baseline')
    with _exclusive_lock(output,'mainline_no_reward_default'):
        if (output/'protocol.json').exists():raise FileExistsError('existing baseline evidence; no automatic duplicate solve')
        train,valid,truth,input_hashes=load_inputs(a.source_run.resolve())
        observed._write_json(output/'protocol.json',{'stage':'N1','strategy':'no_reward',
            'purpose':'Same frozen v1 observations with full current production RNJ/WZZT defaults; no price comparison',
            'source_run':str(a.source_run.resolve()),'source_file_sha256':input_hashes,
            'normal_reference_protocol_sha256':sha(normal/'protocol.json'),
            'concurrency':'N0 and N1 are independent baseline conditions; may overlap, no runtime speed comparison',
            'root_label':1,'training_time_rows':sum(len(s['P_terminal']) for s in train),
            'validation_time_rows':sum(len(s['P_terminal']) for s in valid),
            'data_difference_preserved':'96/24 hourly residential observations, not original reference288/288 mixed-resource observations',
            'explicit_algorithm_overrides':{},'truth_to_estimator':False})
        result=fit_mainline_observed(train,valid,1,output)
        evaluation=observed.evaluate_effective_supports(result,truth)
        observed._write_json(output/'evaluation.json',evaluation)
        summary={'stage':'N1','strategy':'no_reward','fit':observed.summarize_fit(result),'recovery':evaluation,
                 'rnj':result.get('rnj'),'source_input_hashes_unchanged':all(sha(Path(p))==h for p,h in input_hashes.items())}
        observed._write_json(output/'summary.json',summary)
        print(json.dumps({'status':result['status'],'has_model':result['has_model'],
            'nontrivial_recovery':evaluation.get('nontrivial_supports')},indent=2))
        return 0 if result['status']=='complete' else 2

if __name__=='__main__':raise SystemExit(main())
