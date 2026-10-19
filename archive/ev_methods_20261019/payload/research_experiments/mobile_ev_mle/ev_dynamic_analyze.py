"""Read-only paired analysis and static figures for saved dynamic-EV results.

Input is a raw trial directory or a summarize/evaluate JSON from
ev_dynamic_collect. No strategy, simulator, fitter or optimizer is imported.
One base-history seed is one independent unit; methods are paired observations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from . import ev_dynamic_collect as collect

SCHEMA = 'ev_dynamic_paired_analysis_v1'
DOMAINS = ('charge_fixed','v2g_fixed','charge_mobile','v2g_mobile')
CONTRASTS = dict(collect.CONTRASTS,
    mobile_discharge=('design_v2g_mobile','design_charge_mobile'),
    highlow_vs_constant=('highlow','constant'),
    charge_fixed_vs_highlow=('design_charge_fixed','highlow'),
    quality_discharge=('quality_v2g_fixed','quality_charge_fixed'),
    quality_movement=('quality_v2g_mobile','quality_v2g_fixed'),
    quality_charge_movement=('quality_charge_mobile','quality_charge_fixed'))
ENERGY = ('grid_energy_kwh','charging_kwh','discharging_kwh','throughput_kwh',
          'driving_kwh','worker_minutes','moves','power_variance')
FREE_PLOTS = ('charge_fixed_vs_quality','v2g_fixed_vs_quality',
    'charge_mobile_vs_quality','v2g_mobile_vs_quality','discharge','mobile_discharge',
    'movement','charge_movement','timing','amplitude','forecast_policy')


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def classify(value,tolerance):
    if not finite(value):return 'unavailable'
    return 'positive' if value>tolerance else 'negative' if value < -tolerance else 'zero'


def stats(values,denominator,*,tolerance=0.):
    values=[float(x) for x in values if finite(x)]
    counts=Counter(classify(x,tolerance) for x in values)
    return dict(total=denominator,valid=len(values),missing_or_failed=denominator-len(values),
        mean=statistics.fmean(values) if values else None,
        median=statistics.median(values) if values else None,
        sum=math.fsum(values) if values else None,
        minimum=min(values) if values else None,maximum=max(values) if values else None,
        positive=counts['positive'],zero=counts['zero'],negative=counts['negative'],
        zero_tolerance=tolerance)


def _margin(row,family):
    if row.get('status')!='complete':return None
    losses=row.get('losses',{})
    if family not in losses or len(losses)<2:return None
    if not all(finite(x) and x>=0 for x in losses.values()):
        raise ValueError('invalid saved candidate loss')
    result=min(v for name,v in losses.items() if name!=family)-losses[family]
    if not finite(row.get('M')) or not math.isclose(result,row['M'],rel_tol=1e-12,abs_tol=1e-8):
        raise ValueError('saved M differs from candidate losses')
    return float(result)


def load_saved(*,directory=None,report=None,raw_directory=None):
    """Load a collector report; raw supplements never create new observations."""
    if (directory is None)==(report is None):
        raise ValueError('supply exactly one raw directory or collector report')
    if directory is not None:
        raw_directory=Path(directory)
        saved=collect.collect(raw_directory,mode='summarize')
        input_manifest=[dict(path=str(raw_directory/'freeze.json'),
            sha256=hashlib.sha256((raw_directory/'freeze.json').read_bytes()).hexdigest())]
    else:
        path=Path(report);saved=collect.read_json(path)
        input_manifest=[dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())]
        if saved.get('schema')!=collect.SCHEMA or saved.get('mode') not in ('summarize','evaluate'):
            raise ValueError('expected a summarize/evaluate collector report')
        if raw_directory is not None:
            raw_directory=Path(raw_directory)
            checked=collect.collect(raw_directory,mode='summarize')
            for key in ('policy_id','config_sha256','source_manifest_sha256','methods','records','trial_files'):
                if checked.get(key)!=saved.get(key):
                    raise ValueError('report/raw supplement mismatch: '+key)
    records=saved.get('records');methods=saved.get('methods')
    if not isinstance(records,list) or not records or not isinstance(methods,list) or not methods:
        raise ValueError('nonempty saved history records/methods required')
    seeds=[str(r['seed']) for r in records]
    if len(set(seeds))!=len(seeds):raise ValueError('duplicate independent history seed')
    if len(set(methods))!=len(methods):raise ValueError('duplicate method name')
    if any(set(r.get('methods',{}))!=set(methods) for r in records):
        raise ValueError('every history must retain all declared methods, including failures')
    raw={}
    if raw_directory is not None:
        for entry in saved['trial_files']:
            name=entry['name']
            if Path(name).name!=name:raise ValueError('trial manifest requires local filenames')
            path=raw_directory/name;data=path.read_bytes()
            if hashlib.sha256(data).hexdigest()!=entry['sha256']:
                raise ValueError('raw trial hash changed: '+name)
            value=collect.read_json(path)
            if value.get('policy_id')!=saved['policy_id']:
                raise ValueError('raw trial policy differs')
            raw[str(value['seed'])]=value
        if set(raw)!=set(seeds):raise ValueError('raw supplement seeds differ')
    input_manifest.extend(dict(path=str((Path(raw_directory)/e['name']).resolve())
        if raw_directory is not None else e['name'],sha256=e['sha256']) for e in saved.get('trial_files',[]))
    return saved,raw,input_manifest


def _selection(trial,method):
    if method=='design_mean_v2g_mobile':return trial.get('mean_selection',{}),'v2g_mobile','mean'
    key=method.removeprefix('design_')
    key={'quality_charge_fixed':'quality_fixed','quality_v2g_mobile':'quality_mobile'}.get(key,key)
    return trial.get('selection',{}),key,'ar'


def _origin_chain(selection,key):
    records=selection.get('methods',{});chain=[];seen=set()
    while key in records:
        if key in seen:raise ValueError('cyclic saved inheritance chain')
        seen.add(key);row=records[key];origin=row.get('selected_origin')
        chain.append(dict(method=key,origin=origin,
            recorded_new=row.get('selected_is_new_MILP_proposal') is True))
        if origin not in DOMAINS:break
        key=origin
    terminal=chain[-1]['origin'] if chain else None
    category=('model_distance_MILP' if terminal=='MILP' else
        'quality_price_MILP' if isinstance(terminal,str) and terminal.startswith('quality') else
        'deterministic_baseline' if terminal else 'not_applicable')
    return chain,category


def _rmse(a,b):
    def flatten(x):
        if isinstance(x,list):return [v for row in x for v in flatten(row)]
        if not finite(x):raise ValueError('nonfinite prediction input')
        return [float(x)]
    x,y=flatten(a),flatten(b)
    if len(x)!=len(y) or not x:raise ValueError('prediction/physical dimensions differ')
    return math.sqrt(math.fsum((v-w)**2 for v,w in zip(x,y))/len(x))


def _multiply(array,scale):
    return [_multiply(x,scale) for x in array] if isinstance(array,list) else array*scale


def _forecast_errors(trial):
    answer={};background=trial.get('background',{})
    for key,label in (('forecast','ar'),('mean_forecast','mean')):
        forecast=trial.get(key)
        if not forecast or not background:continue
        metadata=forecast.get('metadata',{});T=metadata.get('horizon');base=metadata.get('base_kw')
        if not isinstance(T,int) or T<1 or not finite(base) or base<=0:
            raise ValueError('forecast units/horizon metadata required for offline RMSE')
        values={}
        for field,truth,name in (('p','p_kw','P_RMSE_kw'),('q','q_kvar','Q_RMSE_kvar')):
            if field in forecast and truth in background:
                if len(forecast[field])!=T:raise ValueError('forecast horizon mismatch')
                values[name]=_rmse(_multiply(forecast[field],base),background[truth][-T:])
        root=forecast.get('root',forecast.get('root_v'))
        if root is not None and 'root_v' in background:
            values['root_RMSE_pu']=_rmse(root,background['root_v'][-T:])
        source=forecast.get('source_by_candidate',{}).get(trial.get('true_name'),{})
        for field,truth,name in (('source_p','source_p_kw','true_source_P_RMSE_kw'),
                ('source_q','source_q_kvar','true_source_Q_RMSE_kvar')):
            if field in source and truth in background:
                values[name]=_rmse(_multiply(source[field],base),background[truth][-T:])
        if label=='ar':
            for name,saved in trial.get('forecast_error',{}).items():
                if name in values and finite(saved) and not math.isclose(saved,values[name],rel_tol=1e-10,abs_tol=1e-10):
                    raise ValueError('saved/recomputed forecast RMSE mismatch')
        answer[label]=values
    return answer


def _raw_diagnostics(record,trial,tolerance):
    forecast=_forecast_errors(trial);methods={};solvers=[]
    for method in record['methods']:
        selection,key,policy=_selection(trial,method);row=selection.get('methods',{}).get(key,{})
        chain,terminal=_origin_chain(selection,key)
        score,initial=row.get('selected_score'),row.get('initial_best_score')
        gain=score-initial if finite(score) and finite(initial) else None
        methods[method]=dict(origin_chain=chain,lineage_terminal=terminal,
            selected_origin=row.get('selected_origin'),recorded_new=row.get('selected_is_new_MILP_proposal') is True,
            selected_predictive_score=score,initial_predictive_score=initial,predictive_gain=gain,
            material_predictive_gain=finite(gain) and gain>tolerance,
            recorded_new_with_material_gain=row.get('selected_is_new_MILP_proposal') is True
                and finite(gain) and gain>tolerance,
            proxy_gate=row.get('raw_proxy_gate'),recovered_failures=row.get('failures',[]))
    for selection_key,policy in (('selection','ar'),('mean_selection','mean')):
        for domain,row in trial.get(selection_key,{}).get('methods',{}).items():
            raw=row.get('raw')
            if not isinstance(raw,dict):continue
            lower,upper=raw.get('objective_lower'),raw.get('objective_upper')
            computed_gap=upper-lower if finite(lower) and finite(upper) else None
            saved_gap=raw.get('gap')
            if finite(computed_gap) and finite(saved_gap) and not math.isclose(computed_gap,saved_gap,rel_tol=1e-9,abs_tol=1e-7):
                raise ValueError('saved proxy gap differs from upper-lower')
            solvers.append(dict(seed=record['seed'],family=record['family'],policy=policy,domain=domain,
                proxy_completed=raw.get('completed'),proxy_unfinished=raw.get('unfinished'),
                lower=lower,upper=upper,gap=saved_gap,recomputed_gap=computed_gap,target_gap=raw.get('target_gap'),
                optimization_seconds=raw.get('seconds'),qp_calls=raw.get('qp_calls'),
                master_calls=sum(isinstance(it.get('master'),dict) for it in raw.get('iterations',[])),
                proxy_gate=row.get('raw_proxy_gate'),
                selected_origin=row.get('selected_origin'),selected_score=row.get('selected_score'),
                initial_best_score=row.get('initial_best_score'),
                global_nonlinear_AC_optimum_certified=raw.get('global_nonlinear_AC_optimum_certified',False)))
    return dict(forecast_errors=forecast,methods=methods),solvers


def analyze(saved,raw=None,*,tolerance=1e-4):
    """Independent descriptive arithmetic; tolerance classifies paired changes."""
    if not finite(tolerance) or tolerance<0:raise ValueError('nonnegative finite tolerance required')
    raw={} if raw is None else raw;records=saved['records'];methods=saved['methods'];n=len(records)
    if len({str(r['seed']) for r in records})!=n:raise ValueError('duplicate independent history seed')
    history_rows=[];diagnostics=[];solver_rows=[];margins={};diagnostic_by_seed={}
    for record in records:
        seed=str(record['seed']);margins[seed]={}
        extra,solvers=_raw_diagnostics(record,raw[seed],tolerance) if seed in raw else ({},[])
        diagnostic_by_seed[seed]=extra;solver_rows.extend(solvers)
        diag=dict(seed=seed,family=record['family'],R_relative=record.get('training_parameter_error',{}).get('R'),
            X_relative=record.get('training_parameter_error',{}).get('X'),
            full_research_runtime_seconds=record.get('seconds'),raw_diagnostics_available=seed in raw,
            candidate_miss=record.get('candidate_miss',False),training_fit_failure=record.get('training_fit_failure',False))
        for policy,values in extra.get('forecast_errors',{}).items():
            diag.update({policy+'_'+key:value for key,value in values.items()})
        diag.update({'actual_calls_'+key:value for key,value in record.get('actual_solver_calls',{}).items() if finite(value)})
        diagnostics.append(diag)
        for method in methods:
            row=record['methods'][method];M=_margin(row,record['family']);margins[seed][method]=M
            extra_method=dict(recorded_new=row.get('origin',{}).get('selected_is_new_MILP_proposal') is True,
                raw_diagnostics_available=seed in raw)
            extra_method.update(extra.get('methods',{}).get(method,{}))
            history_rows.append(dict(seed=seed,family=record['family'],method=method,status=row.get('status'),
                raw_status=row.get('raw_status'),M=M,failure_reasons=row.get('failure_reasons',[]),
                source_origin_category=row.get('origin',{}).get('category'),
                source_origin_raw=row.get('origin',{}).get('raw'),
                **{key:row.get('energy',{}).get(key) for key in ENERGY},**extra_method))
    paired=[];pair_summary={}
    for name,(left,right) in CONTRASTS.items():
        if left not in methods or right not in methods:continue
        rows=[]
        for record in records:
            seed=str(record['seed']);a,b=margins[seed][left],margins[seed][right]
            delta=a-b if finite(a) and finite(b) else None
            row=dict(seed=seed,family=record['family'],contrast=name,left=left,right=right,
                left_M=a,right_M=b,delta_M=delta,classification=classify(delta,tolerance))
            rows.append(row);paired.append(row)
        pair_summary[name]=dict(left=left,right=right,delta_M=stats([r['delta_M'] for r in rows],n,tolerance=tolerance),
            by_true_family={family:stats([r['delta_M'] for r in rows if r['family']==family],
                sum(r['family']==family for r in rows),tolerance=tolerance) for family in sorted({r['family'] for r in records})})
    method_summary={}
    for method in methods:
        rows=[r for r in history_rows if r['method']==method]
        method_summary[method]=dict(M=stats([r['M'] for r in rows],n),
            status_counts=dict(Counter(r['status'] for r in rows)),
            failure_reasons=dict(Counter(reason for r in rows for reason in set(r['failure_reasons']))),
            energy={key:stats([r[key] for r in rows],n) for key in ENERGY},
            origin_categories=dict(Counter(r['source_origin_category'] for r in rows)),
            lineage_terminal=dict(Counter(r.get('lineage_terminal','raw_unavailable') for r in rows)),
            recorded_new=sum(r.get('recorded_new') is True for r in rows),
            material_gain_checked_histories=sum(finite(r.get('predictive_gain')) for r in rows),
            recorded_new_with_material_predictive_gain=sum(r.get('recorded_new_with_material_gain') is True for r in rows),
            predictive_gain=stats([r.get('predictive_gain') for r in rows],n,tolerance=tolerance),
            by_true_family={family:stats([r['M'] for r in rows if r['family']==family],
                sum(r['family']==family for r in rows)) for family in sorted({r['family'] for r in records})})
    numeric_diagnostics=sorted({k for r in diagnostics for k,v in r.items() if finite(v) and k not in ('seed',)})
    return dict(schema=SCHEMA,policy_id=saved['policy_id'],source_manifest_sha256=saved.get('source_manifest_sha256'),
        independent_histories=n,family_counts=dict(Counter(r['family'] for r in records)),
        methods=methods,method_summary=method_summary,paired_contrasts=pair_summary,
        history_methods=history_rows,history_contrasts=paired,history_diagnostics=diagnostics,
        diagnostic_summary={k:stats([r.get(k) for r in diagnostics],n) for k in numeric_diagnostics},
        solver_conditions=solver_rows,proxy_summary=dict(conditions=len(solver_rows),
            completed=sum(r['proxy_completed'] is True for r in solver_rows),
            unfinished=sum(r['proxy_unfinished'] is True for r in solver_rows),
            gate_passed=sum(r['proxy_gate'] is True for r in solver_rows),
            gap=stats([r['gap'] for r in solver_rows],len(solver_rows)),
            by_policy={policy:dict(conditions=sum(r['policy']==policy for r in solver_rows),
                completed=sum(r['policy']==policy and r['proxy_completed'] is True for r in solver_rows),
                gate_passed=sum(r['policy']==policy and r['proxy_gate'] is True for r in solver_rows))
                for policy in sorted({r['policy'] for r in solver_rows})}),
        candidate_omissions=sum(bool(r.get('candidate_miss')) for r in records),
        raw_diagnostic_histories=len(raw),paired_change_tolerance=tolerance,
        means_use_raw_differences=True,numerical_tolerance_is_significance_threshold=False,
        evaluation=saved.get('evaluation'),analysis_manifest=dict(
            source_sha256={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            imports_science_modules=False,new_fit_calls=0,new_selector_calls=0,new_simulation_calls=0),
        metric='M=min wrong-candidate(validation+future SSE)-true-candidate SSE; fixed training R/X',
        cost_scope='Saved full research runtime includes training, strategies, diagnostics, evaluation and replay; not single-policy deployment time.',
        interpretation='Paired descriptive results per independent base history; model distances and M are not topology probabilities. Failed/missing pairs remain in total denominators.')


def _write_csv(path,rows,fields=None):
    if fields is None:fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(row.get(key),ensure_ascii=False,allow_nan=False)
                if isinstance(row.get(key),(list,dict)) else row.get(key) for key in fields})


def make_figures(result,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    output=Path(directory);rows=result['history_methods'];seeds=list(dict.fromkeys(r['seed'] for r in rows))
    families={r['seed']:r['family'] for r in rows};lookup={(r['seed'],r['method']):r['M'] for r in rows}
    labels=[seed+'\n'+families[seed].replace('clade_','c') for seed in seeds]
    style={'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200}
    with plt.rc_context(style):
        fig,axes=plt.subplots(2,2,figsize=(max(10.,min(18.,len(seeds)*.85)),8.),sharex=True)
        for ax,domain in zip(axes.ravel(),DOMAINS):
            lines=[('none','No EV','#222222','--'),('constant','Constant','#999999',':'),
                ('quality_'+domain,'Quality MILP','#2ca02c','-'),('design_'+domain,'Discrimination MILP','#1f77b4','-')]
            if domain=='charge_fixed':lines.insert(2,('highlow','Preset high/low','#ff7f0e','-.'))
            for method,label,color,linestyle in lines:
                if method not in result['methods']:continue
                values=[lookup.get((seed,method)) for seed in seeds]
                valid=sum(finite(x) for x in values)
                ax.plot(range(len(seeds)),[x if finite(x) else math.nan for x in values],
                    marker='o',markersize=3.5,linewidth=1.,linestyle=linestyle,color=color,
                    label=label+f' ({valid}/{len(seeds)})')
            ax.axhline(0,color='black',linewidth=.7);ax.grid(axis='y',alpha=.2)
            ax.set_title(domain.replace('_',' '));ax.set_ylabel('Actual margin M (SSE)')
            ax.set_xticks(range(len(seeds)),labels,rotation=45 if len(seeds)>6 else 0,ha='right' if len(seeds)>6 else 'center')
            ax.legend(fontsize=7.2,loc='best')
        fig.suptitle('Actual AC margins by independent history',fontsize=13)
        fig.text(.5,.015,'Legend denominators retain failed histories; missing scores are gaps. Lines connect paired labels, not time steps.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.96))
        for suffix in ('png','pdf'):fig.savefig(output/('history_margins.'+suffix))
        plt.close(fig)
        selected=[name for name in FREE_PLOTS if name in result['paired_contrasts']]
        if not selected:selected=list(result['paired_contrasts'])
        if not selected:raise ValueError('no paired contrasts available for second figure')
        fig,ax=plt.subplots(figsize=(max(10.,len(selected)*1.15),6.2))
        family_names=sorted(set(families.values()));colors={f:plt.get_cmap('tab10')(i) for i,f in enumerate(family_names)}
        paired={(r['seed'],r['contrast']):r['delta_M'] for r in result['history_contrasts']}
        for i,seed in enumerate(seeds):
            jitter=(i-(len(seeds)-1)/2)*min(.035,.45/max(1,len(seeds)))
            for j,name in enumerate(selected):
                value=paired.get((seed,name))
                if finite(value):ax.scatter(j+jitter,value,s=24,alpha=.85,color=colors[families[seed]],zorder=3)
        means=[result['paired_contrasts'][name]['delta_M']['mean'] for name in selected]
        ax.scatter(range(len(selected)),[x if finite(x) else math.nan for x in means],marker='D',s=35,
            color='black',label='Paired mean',zorder=4)
        for j,name in enumerate(selected):
            s=result['paired_contrasts'][name]['delta_M']
            ax.text(j,1.015,f"{s['valid']}/{s['total']}; +{s['positive']} ={s['zero']} -{s['negative']}",
                transform=ax.get_xaxis_transform(),ha='center',fontsize=7)
        names={'discharge':'V2G - charge\nfixed','mobile_discharge':'V2G - charge\nmobile',
            'movement':'Mobile - fixed\nV2G','charge_movement':'Mobile - fixed\ncharge',
            'timing':'Design - reverse\nfixed V2G','amplitude':'Full - half\namplitude',
            'forecast_policy':'AR - mean\nmobile V2G'}
        ticklabels=[names.get(name,name.removesuffix('_vs_quality').replace('_',' ')+'\ndesign - quality') for name in selected]
        ax.set_xticks(range(len(selected)),ticklabels,rotation=30,ha='right')
        ax.axhline(0,color='black',linewidth=.8);ax.grid(axis='y',alpha=.2)
        ax.set_ylabel('Paired actual ΔM (left minus right, SSE)')
        ax.set_title('Freedom and objective comparisons: every point is one history',pad=28)
        handles=[Line2D([],[],marker='o',linestyle='',color=colors[f],label=f) for f in family_names]
        handles.append(Line2D([],[],marker='D',linestyle='',color='black',label='Paired mean'))
        ax.legend(handles=handles,ncol=min(5,len(handles)),fontsize=8,loc='best')
        fig.text(.5,.015,f"Counts use |ΔM| <= {result['paired_change_tolerance']:g} as a numerical tie; raw differences are plotted. No significance test is applied.",ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.98))
        for suffix in ('png','pdf'):fig.savefig(output/('paired_freedom_gains.'+suffix))
        plt.close(fig)
    return ['history_margins.png','history_margins.pdf','paired_freedom_gains.png','paired_freedom_gains.pdf']


def run_analysis(*,directory=None,report=None,raw_directory=None,output,tolerance=1e-4,plots=True):
    saved,raw,inputs=load_saved(directory=directory,report=report,raw_directory=raw_directory)
    result=analyze(saved,raw,tolerance=tolerance);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    result['inputs']=inputs
    method_rows=[dict(method=name,**value['M'],recorded_new=value['recorded_new'],
        material_gain_checked_histories=value['material_gain_checked_histories'],
        recorded_new_with_material_predictive_gain=value['recorded_new_with_material_predictive_gain'],
        **{key+'_mean':value['energy'][key]['mean'] for key in ENERGY},
        moves_total=value['energy']['moves']['sum'])
        for name,value in result['method_summary'].items()]
    pair_rows=[dict(contrast=name,left=value['left'],right=value['right'],**value['delta_M'])
        for name,value in result['paired_contrasts'].items()]
    for filename,rows in (('methods.csv',method_rows),('paired_contrasts.csv',pair_rows),
            ('history_methods.csv',result['history_methods']),('history_contrasts.csv',result['history_contrasts']),
            ('history_diagnostics.csv',result['history_diagnostics']),('solver_conditions.csv',result['solver_conditions'])):
        _write_csv(output/filename,rows)
    result['figures']=make_figures(result,output) if plots else []
    collect.write_json(output/'analysis.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--directory',type=Path,help='raw frozen trial directory')
    inputs.add_argument('--report',type=Path,help='collector summarize/evaluate JSON[.gz]')
    parser.add_argument('--raw-directory',type=Path,help='optional raw supplement for report diagnostics')
    parser.add_argument('--output',type=Path,required=True,help='new analysis artifact directory')
    parser.add_argument('--tolerance',type=float,default=1e-4)
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    if args.directory is not None and args.raw_directory is not None:
        parser.error('--raw-directory applies only to --report')
    result=run_analysis(directory=args.directory,report=args.report,raw_directory=args.raw_directory,
        output=args.output,tolerance=args.tolerance,plots=not args.no_plots)
    print(json.dumps(dict(histories=result['independent_histories'],raw_diagnostics=result['raw_diagnostic_histories'],
        proxy_conditions=result['proxy_summary']['conditions'],output=str(args.output.resolve()),figures=result['figures'])))


if __name__=='__main__':main()
