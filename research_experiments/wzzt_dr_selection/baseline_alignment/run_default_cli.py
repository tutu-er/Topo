"""Run the normal paper15 search and compare the full RNJ model at validation.

Core CLI metrics and fitted.pkl retain WZZT-only results; final_selection.json
is the explicit additional final model. --wzzt-path-only reproduces the old audit.
"""
import argparse
import hashlib
import inspect
import json
import os
import pickle
from pathlib import Path
import platform
import runpy
import sys
import traceback
from time import perf_counter
from datetime import datetime, timezone

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wzzt-path-only',action='store_true',help='skip the added full-RNJ final comparison for historical replay')
    a=p.parse_args();repo=a.repo.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'rnj_wzzt_core'))
    import numpy as np
    import scipy
    from rnj_wzzt import pipeline
    from research_experiments.wzzt_dr_selection.end_to_end.wzzt_adapter import _plain,_write_json
    from research_experiments.wzzt_dr_selection.baseline_alignment.pooled_selection import fit_complete_rnj_candidate,select_with_complete_rnj
    if (out/'protocol.json').exists():raise RuntimeError('Existing run; inspect its checkpoint rather than rerunning')
    lock=out/'runner.lock'
    with lock.open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'script':__file__},f)
    start=perf_counter()
    def save(name,x):_write_json(out/name,_plain(x))
    def private(name,x):
        with (out/name).open('wb') as f:pickle.dump(x,f,protocol=4)
    defaults={k:(str(v.default) if isinstance(v.default,Path) else v.default) for k,v in inspect.signature(pipeline.run).parameters.items()}
    save('protocol.json',{'purpose':'Default paper15 RNJ/WZZT search with explicit full-RNJ final comparison; no DR',
        'compare_complete_rnj':not a.wzzt_path_only,
        'cli_overrides':{'cases':['paper15'],'output_dir':str(out/'cli')},
        'pipeline_defaults':defaults,'created_utc':datetime.now(timezone.utc).isoformat(),
        'environment':{'python':sys.version,'executable':sys.executable,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()},
        'source_sha256':{str(x.relative_to(repo)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted((repo/'rnj_wzzt_core/rnj_wzzt').rglob('*.py'))},
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'selection_source_sha256':{inspect.getfile(select_with_complete_rnj):hashlib.sha256(Path(inspect.getfile(select_with_complete_rnj)).read_bytes()).hexdigest()},
        'instrumentation':'core search unchanged; original CLI metrics remain WZZT-only; additional RNJ weight LP and common validation comparison saved separately',
        'condition_count':1,'price_comparison':False})
    prepare=pipeline._prepare_case;select=pipeline._select_case_rnj;fit=pipeline.fit_laminar_l1_sensitivity
    latest_selection = None
    def prepared(*args,**kwargs):
        result=prepare(*args,**kwargs);private('prepared_case.pkl',result)
        save('data_ready.json',{'kwargs':kwargs,'scenarios':result[1],'seconds':perf_counter()-start})
        print('Normal reference observations prepared',flush=True);return result
    def selected(*args,**kwargs):
        nonlocal latest_selection
        save('selection_started.json',{'kwargs':kwargs,'seconds':perf_counter()-start})
        result=select(*args,**kwargs);latest_selection=result;private('selection.pkl',result)
        save('selection.json',{'full_clades':[sorted(x) for x in result.full_clades],
            'selected':[sorted(x) for x in result.selected],
            'confidence':{','.join(map(str,sorted(k))):v for k,v in result.confidence.items()},
            'r2':result.r2_score,'condition':result.condition,'seconds':result.elapsed})
        print('Normal RNJ and100 bootstrap completed',flush=True);return result
    def fitted(*args,**kwargs):
        save('fit_arguments.json',{k:v for k,v in kwargs.items() if k!='validation_scenarios'})
        def progress(state):
            save('fit_progress.json',{**state,'elapsed_seconds':perf_counter()-start})
            print('WZZT '+str(state['event'])+' path='+str(len(state['path']))+' attempts='+str(len(state['attempted_extensions'])),flush=True)
        result=fit(*args,**kwargs,progress_callback=progress)
        private('fitted.pkl',result);save('fit.json',result)
        if not a.wzzt_path_only:
            candidate = None
            failure = None
            try:
                candidate = fit_complete_rnj_candidate(args[0], kwargs['validation_scenarios'],
                    latest_selection.full_clades, r_upper_bound=result.r_upper_bound,
                    x_upper_bound=result.x_upper_bound, time_limit=kwargs.get('time_limit'),
                    solver=kwargs.get('solver','highs'))
            except Exception as exc:
                failure = {'type':type(exc).__name__, 'message':str(exc)}
            comparison = select_with_complete_rnj(result.path,candidate,rnj_failure=failure)
            save('final_selection.json',{'selection':comparison,
                'complete_rnj_candidate':candidate,
                'wzzt_search_stop_reason':result.stop_reason,
                'wzzt_selected_path_index':result.selected_path_index,
                'note':'CLI metrics and fit.json are original WZZT-only; this file stores the final pooled choice'})
            print('Final selection '+comparison['selected_origin']+' status='+comparison['status'],flush=True)
        return result
    pipeline._prepare_case=prepared;pipeline._select_case_rnj=selected;pipeline.fit_laminar_l1_sensitivity=fitted
    try:
        sys.argv=[str(repo/'rnj_wzzt_core/run.py'),'--cases','paper15','--output',str(out/'cli')]
        runpy.run_path(sys.argv[0],run_name='__main__')
        save('completion.json',{'status':'finished','seconds':perf_counter()-start,'utc':datetime.now(timezone.utc).isoformat()})
        print('Normal CLI returned',flush=True)
    except Exception as exc:
        save('failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'seconds':perf_counter()-start});raise
    finally:
        pipeline._prepare_case=prepare;pipeline._select_case_rnj=select;pipeline.fit_laminar_l1_sensitivity=fit
        lock.unlink(missing_ok=True)

if __name__=='__main__':main()
