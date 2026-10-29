"""One-shot contraction alternatives for complete RNJ and WZZT families.

All new points fit training observations jointly; one unchanged validation rule
chooses among a pre-enumerated set. No repeated validation-directed search,
truth labels, imposed positive weights or new price score is introduced.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter
import traceback
import numpy as np
from research_experiments.wzzt_dr_selection.dr_pool_v2.engine import PoolLP, canonical, compatible
from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import normalize, input_hash
from research_experiments.wzzt_dr_selection.dr_pool_v2.selection import measure, select, package


def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,Path):return str(v)
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    return v


def digest(v):return hashlib.sha256(json.dumps(plain(v),sort_keys=True,allow_nan=False).encode()).hexdigest()


def write(path,v):
    if path is None:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(plain(v),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)


def enumerate_contractions(original_fit):
    """Enumerate all parents and all single nonleaf deletions before fitting."""
    n=len(original_fit['terminal_labels']);leaves={(i,) for i in range(n)}
    records={}
    def add(family, provenance):
        family=canonical(family)
        if not leaves<=set(family):raise ValueError('each input family must retain every leaf')
        if any(not compatible(family[:i],s) for i,s in enumerate(family)):raise ValueError('nonlaminar input parent')
        rec=records.setdefault(family,dict(family=family,provenance=[]))
        if provenance not in rec['provenance']:rec['provenance'].append(provenance)
    def parent(family,source,identifier):
        family=canonical(family)
        add(family,dict(source=source,parent=identifier,operation='parent',deleted=None))
        for block in family:
            if len(block)>1:
                add([s for s in family if s!=block],dict(source=source,parent=identifier,operation='delete_one',deleted=block))
    if not original_fit.get('bank'):raise ValueError('complete RNJ bank is required for a fair baseline')
    for index,point in enumerate(original_fit['bank']):
        parent(point.get('native_family',point['supports']),'rnj_bank',f'bank_{index}:{point.get("config","")}')
    for name,path in original_fit.get('searches',{}).items():
        for index,point in enumerate(path['path']):parent(point['supports'],'wzzt_path',f'{name}:{index}')
    # Sorting is input-independent and avoids changing LP bases according to a
    # candidate's validation loss or simulation outcome.
    return [records[k] for k in sorted(records,key=lambda f:(len(f),f))]


def fit_with_contractions(training,validation,original_fit,*,bound=2.,max_families=200,
                          point_transform=None,lp_factory=None,output_path=None):
    """Return four matched selectors from the same jointly-refitted points.

    point_transform(point, *, family, training, validation) may update R/X
    parameters only. This linear phase always recomputes original MAE and uses
    the unchanged 3% selector/contraction. An AC-ranking experiment must define
    its own held-out AC score separately; this callback is not that experiment.
    """
    started=perf_counter();train,days,_=normalize(training,'training');valid,vdays,_=normalize(validation,'validation')
    labels=train[0].columns.tolist()
    if labels!=original_fit['terminal_labels'] or not valid[0].columns.equals(train[0].columns):raise ValueError('terminal order mismatch')
    hashes=dict(training=input_hash(train,days),validation=input_hash(valid,vdays))
    for name,value in original_fit.get('input_sha256',{}).items():
        if name in hashes and value!=hashes[name]:raise ValueError('original fit observations differ: '+name)
    if max_families<1 or bound<=0 or np.isnan(bound):raise ValueError('positive family cap and positive coefficient bound required')
    plans=enumerate_contractions(original_fit)
    result=dict(status='prepared',input_sha256=hashes,terminal_labels=labels,
        original_fit_sha256=digest(original_fit),implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        settings=dict(bound='unbounded' if np.isinf(bound) else float(bound),max_families=int(max_families),
            continuous_lower_bound=0.,validation_rule='original 3% + support-count tie + original .16 contraction',
            adaptive_validation_search=False,point_transform=point_transform is not None),
        planned_unique_families=len(plans),candidate_plans=plans,candidates=[],failures=[],models={},LP_solves=[],
        original_models=deepcopy(original_fit.get('models',{})),
        scope='One-shot structural candidate expansion, not global optimization or a WZZT-specific price method.')
    if len(plans)>max_families:
        result.update(status='not_run_family_cap',remaining_families=len(plans),elapsed_seconds=perf_counter()-started)
        write(output_path,result);return plain(result)
    all_supports=canonical(s for item in plans for s in item['family'])
    factory=PoolLP if lp_factory is None else lp_factory
    lp=None
    try:
        lp=factory(train,all_supports,'raw',bound=bound)
        for index,item in enumerate(plans):
            try:
                fit=deepcopy(lp.fit(item['family']))
                if point_transform is not None:
                    fit=point_transform(fit,family=item['family'],training=training,validation=validation)
                    if not isinstance(fit,dict):raise ValueError('point_transform must return a parameter point')
                    if not set(canonical(fit['supports']))<=set(item['family']):raise ValueError('point_transform added a support')
                    if not {(i,) for i in range(len(labels))}<=set(canonical(fit['supports'])):raise ValueError('point_transform removed a leaf')
                if not all(np.isfinite(a).all() and np.min(a)>=-1e-9 for a in (np.asarray(fit['r']),np.asarray(fit['x']))):raise ValueError('invalid transformed coefficients')
                point=measure(dict(fit,origin='one_shot_contraction',candidate_index=index,
                                   provenance=item['provenance']),train,valid)
                result['candidates'].append(dict(index=index,family=item['family'],provenance=item['provenance'],point=point))
            except Exception as exc:
                result['failures'].append(dict(index=index,family=item['family'],exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc()))
            result['LP_solves']=deepcopy(lp.solves)
            result['elapsed_seconds']=perf_counter()-started
            write(output_path,result)
        groups={k:[] for k in ('rnj_refit','wzzt_refit','rnj_contractions','wzzt_contractions')}
        group_indices={k:[] for k in groups}
        for candidate in result['candidates']:
            prov=candidate['provenance'];point=candidate['point']
            membership=dict(rnj_refit=any(p['source']=='rnj_bank' and p['operation']=='parent' for p in prov),
                wzzt_refit=any(p['operation']=='parent' for p in prov),
                rnj_contractions=any(p['source']=='rnj_bank' for p in prov),wzzt_contractions=True)
            for name,yes in membership.items():
                if yes:groups[name].append(point);group_indices[name].append(candidate['index'])
        calibration=original_fit['calibrations']['raw']
        for name,points in groups.items():
            if not points:continue
            chosen=select(points,calibration)
            selected_global=group_indices[name][chosen['selection_index']]
            selected=package(chosen,labels,train,valid)
            selected.update(global_candidate_index=selected_global,group_candidate_count=len(points),
                            selected_candidate_provenance=plans[selected_global]['provenance'])
            result['models'][name]=selected
        result.update(status='complete' if not result['failures'] else 'completed_with_failures',
                      fitted_families=len(result['candidates']),remaining_families=0,
                      elapsed_seconds=perf_counter()-started)
        write(output_path,result);return plain(result)
    except Exception as exc:
        result.update(status='failed',exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc(),elapsed_seconds=perf_counter()-started)
        if lp is not None:result['LP_solves']=deepcopy(lp.solves)
        write(output_path,result);raise
    finally:
        if lp is not None:lp.close()
