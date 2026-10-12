"""Evaluator-only line-component errors; independent histories are units.

This module never selects actions, fits parameters, or estimates thresholds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


CLASSES = ("root_stem", "terminal_leaf", "cherry_internal", "other_internal")


def _support(value):
    value = list(value)
    if not value or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in value):
        raise ValueError("supports require nonempty nonnegative integer labels")
    if len(set(value)) != len(value):
        raise ValueError("a support repeats a terminal")
    return tuple(sorted(value))


def _values(value):
    array = np.array([np.nan if v is None else float(v) for v in value])
    if array.ndim != 1:
        raise ValueError("line weights must be vectors")
    return array


def truth_edge_map(truth):
    if truth.get("evaluation_only") is not True:
        raise ValueError("truth must carry the existing evaluation_only=True contract")
    r, x = _values(truth["r_pu"]), _values(truth["x_pu"])
    if len(r) != len(x) or not len(r) or not np.isfinite(r).all() or not np.isfinite(x).all():
        raise ValueError("truth R/X vectors must be finite and aligned")
    supplied = [_support(s) for s in truth.get("edge_supports", truth["supports"])]
    if len(set(supplied)) != len(supplied):
        raise ValueError("duplicate truth supports")
    if "edge_supports" in truth or len(supplied) == len(r):
        supports = supplied
        n = max(i for s in supports for i in s) + 1
        contract = "explicit complete edge-support order"
    else:
        # simulation.truth_metadata exports nontrivial clades only. Its R/X
        # vectors retain RadialCandidate.from_supports canonical edge order.
        n = len(r) - len(supplied) - 1
        if n < 2 or any(not 1 < len(s) < n or max(s) >= n for s in supplied):
            raise ValueError("invalid nontrivial truth-support/canonical-weight contract")
        family = {tuple(range(n)), *((i,) for i in range(n)), *supplied}
        supports = sorted(family, key=lambda s: (-len(s), s))
        contract = "simulation.truth_metadata nontrivial clades, canonical candidate weight order"
    universe = tuple(range(n))
    if n < 2 or len(supports) != len(r) or universe not in supports or any((i,) not in supports for i in range(n)):
        raise ValueError("truth edge set must contain full stem and every terminal leaf")
    sets = list(map(set, supports))
    if any(a & b and not (a <= b or b <= a) for a in sets for b in sets):
        raise ValueError("truth edge supports cross")
    return {support: (float(rr), float(xx)) for support,rr,xx in zip(supports,r,x)}, n, contract


def fit_edge_map(fit):
    supports = [_support(s) for s in fit["supports"]]
    r, x = _values(fit["r_pu"]), _values(fit["x_pu"])
    if len(set(supports)) != len(supports) or len(r) != len(x) or len(r) != len(supports):
        raise ValueError("fit supports and line vectors must be unique and aligned")
    if not np.isfinite(r).all() or not np.isfinite(x).all():
        raise ValueError("true fit saved nonfinite line values")
    return {support: (float(rr),float(xx)) for support,rr,xx in zip(supports,r,x)}


def edge_class(support, n):
    if len(support) == n:
        return "root_stem"
    if len(support) == 1:
        return "terminal_leaf"
    return "cherry_internal" if len(support) == 2 else "other_internal"


def _declared_feasible(fit):
    diagnostics=fit.get('diagnostics',{})
    values=[diagnostics.get('ac_equation_residual'),diagnostics.get('power_balance_residual')]
    return bool(diagnostics.get('physical_voltage_feasible') is True and all(
        value is not None and np.isfinite(float(value)) and 0<=float(value)<=2e-10 for value in values))


def analyze_record(row, record):
    truth, n, contract = truth_edge_map(row['truth'])
    true_index=record.get('true_index',row.get('true_index')); names=record.get('candidate_names',row['candidate_names'])
    if true_index is not None and (isinstance(true_index,bool) or not isinstance(true_index,int) or not 0<=true_index<len(names)):
        raise ValueError('invalid true candidate index')
    true_name=names[true_index] if true_index is not None else None
    fits=[fit for fit in record.get('fits',[]) if fit.get('name')==true_name] if true_name else []
    estimated=None;status='missing';reason=None
    if true_index is None:
        reason='candidate_truth_omission'
    elif not fits:
        reason='true_fit_not_saved'
    elif len(fits)!=1:
        reason='duplicate_true_fit_export'
    else:
        fit=fits[0]
        try:
            estimated=fit_edge_map(fit)
            if set(estimated)!=set(truth):
                raise ValueError('true topology support set differs from truth')
            success=record.get('success',[])
            converged=bool(fit.get('success') is True and true_index<len(success) and success[true_index])
            status='converged' if converged else ('saved_feasible_unconverged' if _declared_feasible(fit) else 'saved_numeric_unverified')
            reason=None if converged else 'true_fit_not_converged'
        except (ValueError,KeyError,TypeError) as error:
            estimated=None;reason=f'invalid_true_fit_export: {error}'
    internal=any(1<len(s)<n for s in truth)
    group='star' if not internal else f"{row.get('case','unknown')}_cherry" if n==3 else 'other_internal_network'
    common=dict(history_id=str(row['history_id']),method=record['method'],round=record.get('round',0),
        stratum=group,case=row.get('case'),truth_topology=row.get('truth_topology'),
        truth_evaluator_only=True,true_name=true_name,truth_support_contract=contract,
        estimate_status=status,missing_reason=reason,true_fit_converged=status=='converged',
        candidate_truth_omitted=true_index is None,
        whole_candidate_bank_fit_complete=bool(record.get('success') and all(record['success'])),
        selection_failure=record.get('selection_status')=='selection_failure_fallback',
        operational_voltage_ok=record.get('operational_voltage_ok'),
        physical_feasibility_from_saved_diagnostics_only=status=='saved_feasible_unconverged',
        values_are_pu=True,topology_correct_probability_reported=False)
    edges=[]
    for support,(r0,x0) in truth.items():
        rr,xx=estimated[support] if estimated is not None else (None,None)
        dr,dx=(rr-r0,xx-x0) if rr is not None else (None,None)
        edges.append(dict(common,support=list(support),edge_class=edge_class(support,n),true_R_pu=r0,true_X_pu=x0,
            estimated_R_pu=rr,estimated_X_pu=xx,signed_R_error_pu=dr,signed_X_error_pu=dx,
            absolute_R_error_pu=abs(dr) if dr is not None else None,
            absolute_X_error_pu=abs(dx) if dx is not None else None,
            signed_relative_R_error=dr/r0 if dr is not None and r0!=0 else None,
            signed_relative_X_error=dx/x0 if dx is not None and x0!=0 else None,
            absolute_relative_R_error=abs(dr/r0) if dr is not None and r0!=0 else None,
            absolute_relative_X_error=abs(dx/x0) if dx is not None and x0!=0 else None))
    classes=[]
    for kind in CLASSES:
        values=[edge for edge in edges if edge['edge_class']==kind]
        item=dict(common,edge_class=kind,edge_count=len(values),applicable=bool(values))
        for coordinate in ('R','X'):
            for relative in (False,True):
                key=f"signed_{'relative_' if relative else ''}{coordinate}_error{'' if relative else '_pu'}"
                metric=f"{coordinate}_{'relative' if relative else 'absolute'}_rmse{'' if relative else '_pu'}"
                array=[edge[key] for edge in values]
                item[metric]=float(np.sqrt(np.mean(np.asarray(array)**2))) if array and all(v is not None for v in array) else None
        for relative in (False,True):
            left=item[f"R_{'relative' if relative else 'absolute'}_rmse{'' if relative else '_pu'}"]
            right=item[f"X_{'relative' if relative else 'absolute'}_rmse{'' if relative else '_pu'}"]
            item[f"RX_{'relative' if relative else 'absolute'}_rmse{'' if relative else '_pu'}"]=float(np.hypot(left,right)/np.sqrt(2)) if left is not None and right is not None else None
        classes.append(item)
    return edges,classes


METRICS=tuple(f'{coordinate}_{scale}_rmse'+('_pu' if scale=='absolute' else '')
              for coordinate in ('R','X','RX') for scale in ('absolute','relative'))


def _measurement(rows,key,status):
    values=[row[key] for row in rows if row['estimate_status']==status and row['applicable'] and row[key] is not None]
    array=np.asarray(values,float)
    return dict(history_count=len(rows),applicable_history_count=sum(row['applicable'] for row in rows),
        available_history_count=len(values),missing_applicable_history_count=sum(row['applicable'] for row in rows)-len(values),
        not_applicable_history_count=sum(not row['applicable'] for row in rows),
        mean=float(array.mean()) if len(array) else None,median=float(np.median(array)) if len(array) else None,
        quantiles=dict(zip(('q05','q25','q50','q75','q95'),map(float,np.quantile(array,[.05,.25,.5,.75,.95])))) if len(array) else None,
        maximum=float(array.max()) if len(array) else None,statistical_unit='independent history',
        edge_values_averaged_within_history_before_aggregation=True)


def summarize(classes):
    keys=list(dict.fromkeys((row['method'],row['round'],row['edge_class']) for row in classes))
    output={}
    for method,round_id,kind in keys:
        rows=[row for row in classes if (row['method'],row['round'],row['edge_class'])==(method,round_id,kind)]
        if len(set(row['history_id'] for row in rows))!=len(rows):
            raise ValueError('duplicate history/method/round/component rows')
        output[f'{method}__round_{round_id}__{kind}']=dict(method=method,round=round_id,edge_class=kind,
            history_count=len(rows),applicable_history_count=sum(row['applicable'] for row in rows),
            status_counts={status:sum(row['estimate_status']==status for row in rows) for status in
                ('converged','saved_feasible_unconverged','saved_numeric_unverified','missing')},
            converged_summary={key:_measurement(rows,key,'converged') for key in METRICS},
            saved_feasible_unconverged_summary={key:_measurement(rows,key,'saved_feasible_unconverged') for key in METRICS})
    return output


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value,ensure_ascii=False) if isinstance(value,(list,dict)) else value for key,value in row.items()})


def run_audit(input,out):
    input=Path(input);config=json.loads((input/'config.json').read_text(encoding='utf-8'))
    files=sorted(input.glob('trial_[0-9]*.json'))
    if len(files)!=config['count']:
        raise ValueError('incomplete predeclared batch; resume it before this audit')
    methods=config['methods'];rows=[json.loads(file.read_text(encoding='utf-8')) for file in files]
    ids=[str(row['history_id']) for row in rows]
    if len(set(ids))!=len(ids):
        raise ValueError('duplicate independent histories')
    edges,classes=[],[]
    traces=[]
    for row in rows:
        if not row.get('independent_history_unit') or row.get('selection_uses_truth') or row.get('selection_uses_future_noise'):
            raise ValueError('independent history / selector separation contract failed')
        if set(record['method'] for record in row['records'])!=set(methods):
            raise ValueError('frozen method omitted from result record')
        trace=tuple((method,tuple(sorted(record.get('round',0) for record in row['records'] if record['method']==method))) for method in methods)
        traces.append(trace)
        for record in row['records']:
            edge,component=analyze_record(row,record);edges.extend(edge);classes.extend(component)
    if traces and any(trace!=traces[0] for trace in traces):
        raise ValueError('histories did not preserve the same method/round trace')
    result=dict(history_count=len(rows),config=config,source_sha256=_sha(__file__),
        input_manifest=dict(config_sha256=_sha(input/'config.json'),trial_sha256={file.name:_sha(file) for file in files}),
        truth_evaluator_only=True,optimization_or_threshold_tuning_performed=False,
        parameters_are_global_MLE_claim=False,topology_correct_probability_reported=False,
        round_results_not_pooled_as_independent_histories=True,
        component_summary=summarize(classes),
        strata={group:summarize([row for row in classes if row['stratum']==group]) for group in dict.fromkeys(row['stratum'] for row in classes)})
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    _write_csv(out/'impedance_edges.csv',edges);_write_csv(out/'impedance_history_components.csv',classes)
    (out/'impedance_component_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    result=run_audit(args.input,args.out)
    print(json.dumps(dict(histories=result['history_count'],truth_evaluator_only=True)))
