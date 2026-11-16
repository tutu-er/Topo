"""Postprocess the sealed floor study: residual structure and parameter errors.

No estimator, optimizer, AC engine, or flexibility planner is imported. Truth
is used only to explain already sealed selections. Units of response matrices
and physical branch impedances are reported separately.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
THRESHOLDS = (0., .04, .08, .12)


def file_ref(path):
    p=Path(path).resolve()
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def point_matrix(point,n):
    h=np.zeros((n,2*n))
    for c,r,x in zip(point['supports'],point['r'],point['x'],strict=True):
        h[np.ix_(c,c)]+=r
        h[np.ix_(c,[j+n for j in c])]+=x
    return h


def active(point,tol):
    return {tuple(sorted(c)) for c,r,x in zip(point['supports'],point['r'],point['x'],strict=True)
            if max(r,x)>tol}


def error_classification(candidates, selected, truth, n, tol, *, internal=False):
    """Mutually exclusive hierarchy, separately for internal and full support.

    Correct / active union misses truth / union covers but no exact candidate /
    exact candidate available but not selected. Stored-but-zero support is
    reported separately from structural absence within the second class.
    """
    restrict=lambda s:{c for c in s if not internal or 1<len(c)<n}
    target=restrict({tuple(sorted(c)) for c in truth})
    banks={r['source_candidate_id']:restrict(active(r['point'],tol)) for r in candidates}
    stored=set().union(*(restrict({tuple(sorted(c)) for c in r['point']['supports']}) for r in candidates))
    union=set().union(*banks.values());chosen=banks[selected]
    correct_ids=[i for i,s in banks.items() if s==target]
    missing=target-chosen;extra=chosen-target
    if chosen==target:kind='correct'
    elif target-union:kind='truth_support_missing_from_active_union'
    elif not correct_ids:kind='union_covers_truth_but_no_exact_candidate'
    else:kind='exact_candidate_available_but_selected_wrong'
    encode=lambda s:[list(c) for c in sorted(s,key=lambda c:(len(c),c))]
    return dict(category=kind,correct=chosen==target,missing=encode(missing),extra=encode(extra),
                fn=len(missing),fp=len(extra),truth_count=len(target),
                correct_candidate_ids=correct_ids,bank_has_exact=bool(correct_ids),
                union_missing=encode(target-union),stored_union_missing=encode(target-stored),
                union_missing_but_stored_zero=encode((target-union)&stored))


def physical_coefficients(branches,labels,root,zbase):
    """Aggregate physical series branches sharing a measured descendant set."""
    parent={};children=defaultdict(list);edges={}
    for e in branches:
        if str(e.get('status','true')).lower() not in ('true','1'):continue
        a,b=int(e['from_bus']),int(e['to_bus'])
        if b in parent or b==root:raise ValueError('Physical table is not a rooted tree')
        parent[b]=a;children[a].append(b);edges[b]=e
    positions={int(v):i for i,v in enumerate(labels)};cache={};visiting=set()
    def descendants(node):
        if node in cache:return cache[node]
        if node in visiting:raise ValueError('Physical cycle')
        visiting.add(node);s={positions[node]} if node in positions else set()
        for c in children[node]:s.update(descendants(c))
        visiting.remove(node);cache[node]=tuple(sorted(s));return cache[node]
    if descendants(root)!=tuple(range(len(labels))) or set(parent)-set(cache):raise ValueError('Unreachable physical terminals or branches')
    weights={};line_counts=Counter()
    for child,e in edges.items():
        c=descendants(child)
        if not c:continue
        weights.setdefault(c,np.zeros(2))[:] += [float(e['r_ohm']),float(e['x_ohm'])]
        line_counts[c]+=1
    point=dict(supports=[list(c) for c in weights],r=[2*w[0]/zbase for w in weights.values()],
               x=[2*w[1]/zbase for w in weights.values()])
    return weights,line_counts,point_matrix(point,len(labels))


def parameter_errors(point,Hstar,truth_weights,labels,zbase,tol,resource_nodes):
    n=len(labels);h=point_matrix(point,n);err=h-Hstar
    def rel(a,b):
        den=float(np.linalg.norm(b));return float(np.linalg.norm(a-b)/den) if den else None
    matrices=dict(R_relative_frobenius=rel(h[:,:n],Hstar[:,:n]),X_relative_frobenius=rel(h[:,n:],Hstar[:,n:]))
    resources={}
    for axis,offset in (('P',0),('Q',n)):
        nodes=resource_nodes.get(axis,[]);ix=[labels.index(v)+offset for v in nodes]
        resources[axis]=dict(labels=nodes,node_count=len(ix),relative_frobenius=rel(h[:,ix],Hstar[:,ix]) if ix else None,
                             absolute_response_error_norm=float(np.linalg.norm(err[:,ix])) if ix else None,
                             scope='unit injection response columns; no scheduled action or AC prediction')
    fitted={tuple(c):np.asarray([r,x])*zbase/2 for c,r,x in zip(point['supports'],point['r'],point['x'],strict=True)
            if max(r,x)>tol}
    matched=set(fitted)&set(truth_weights);missing=set(truth_weights)-set(fitted);extra=set(fitted)-set(truth_weights)
    lines=[]
    for c in sorted(set(fitted)|set(truth_weights),key=lambda x:(len(x),x)):
        t=truth_weights.get(c);f=fitted.get(c);status='matched' if c in matched else 'missing' if c in missing else 'extra'
        row=dict(support=list(c),labels=[labels[i] for i in c],status=status,
                 support_kind='singleton' if len(c)==1 else 'common' if len(c)==n else 'internal')
        for j,axis in enumerate(('R','X')):
            row[axis+'_true_ohm']=None if t is None else float(t[j])
            row[axis+'_fitted_ohm']=None if f is None else float(f[j])
            row[axis+'_absolute_error_ohm']=None if status!='matched' else float(abs(f[j]-t[j]))
            row[axis+'_absolute_percentage_error']=None if status!='matched' or t[j]==0 else float(abs(f[j]-t[j])/abs(t[j]))
        lines.append(row)
    line_summary=dict(matched_support_count=len(matched),missing_support_count=len(missing),extra_support_count=len(extra),
                      scope='Same active physical support only; missing and extra supports are separate, not discarded silently')
    for axis in ('R','X'):
        values=[r[axis+'_absolute_percentage_error'] for r in lines if r[axis+'_absolute_percentage_error'] is not None]
        line_summary[axis+'_MAPE']=float(np.mean(values)) if values else None
        line_summary[axis+'_MAPE_count']=len(values)
        line_summary[axis+'_max_APE']=float(max(values)) if values else None
    return dict(matrix=matrices,resource_columns=resources,matched_line=line_summary,lines=lines)


class Reader:
    def __init__(self,audit):
        self.refs={};self.expected={str(Path(r['path']).resolve()):r['sha256'] for r in audit['source_inputs']}
    def path(self,path,expected=None):
        r=file_ref(path);wanted=expected or self.expected.get(r['path'])
        if wanted is None:raise ValueError('File not covered by owning audit: '+r['path'])
        if r['sha256']!=wanted:raise ValueError('Audited source changed: '+r['path'])
        self.refs[r['path']]=r;return Path(r['path'])
    def read(self,path,expected=None):return json.loads(self.path(path,expected).read_text(encoding='utf-8'))
    def referenced(self,r):return self.path(r['path'],r['sha256'])


def aggregate(records):
    grouped=defaultdict(list)
    for r in records:
        common=(r['source_group'],r['phase'],r['threshold'])
        grouped[(*common,'ALL',None,None)].append(r)
        grouped[(*common,r['case'],r['training_points'],None)].append(r)
        if r['source_group']=='large96_pressure':
            grouped[(*common,'ALL',r['source_condition'],r['condition_value'])].append(r)
    result=[]
    for key,rows in grouped.items():
        good=[r for r in rows if r['status']=='complete'];g=dict(source_group=key[0],phase=key[1],threshold=key[2],case=key[3],
            condition=key[4],condition_value=key[5],planned=len(rows),completed=len(good),failed=len(rows)-len(good),
            physical_history_groups=len({r['physical_history_group'] for r in rows}))
        for scope in ('internal','full'):
            classes=Counter(r[scope]['category'] for r in good)
            g[scope]=dict(categories=dict(classes),correct=sum(r[scope]['correct'] for r in good),
                bank_has_exact=sum(r[scope]['bank_has_exact'] for r in good),
                fn=sum(r[scope]['fn'] for r in good),fp=sum(r[scope]['fp'] for r in good),
                union_structural_missing_inputs=sum(bool(r[scope]['stored_union_missing']) for r in good),
                union_zero_only_missing_inputs=sum(bool(r[scope]['union_missing_but_stored_zero']) and not r[scope]['stored_union_missing'] for r in good))
        parameters=[r['parameters'] for r in good if r.get('parameters')]
        g['parameter_count']=len(parameters)
        if parameters:
            g['matrix_error_mean']={axis:float(np.mean([r['matrix'][axis+'_relative_frobenius'] for r in parameters])) for axis in ('R','X')}
            g['matched_line_MAPE_mean']={axis:float(np.mean([r['matched_line'][axis+'_MAPE'] for r in parameters])) for axis in ('R','X')}
            g['resource_column_error_mean']={axis:float(np.mean([r['resource_columns'][axis]['relative_frobenius'] for r in parameters if r['resource_columns'][axis]['relative_frobenius'] is not None])) for axis in ('P','Q')}
        result.append(g)
    return result


def analyze(run,audit_dir,out):
    run,audit_dir,out=map(Path,(run,audit_dir,out))
    if out.exists():raise ValueError('Preserve existing postprocessing output')
    audit=json.loads((audit_dir/'audit.json').read_text(encoding='utf-8'))
    if audit['status']!='passed' or audit['errors']:raise ValueError('Owning audit did not pass')
    reader=Reader(audit);protocol=reader.read(run/'protocol.json');entries=reader.read(protocol['inputs']['path'],protocol['inputs']['sha256'])
    if file_ref(run/'protocol.json')!=audit['source_protocol']:raise ValueError('Owning audit is for a different run')
    original_cells={}
    with (audit_dir/'cells.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line)
            if r['setting_id']=='original_frozen' or (r.get('metric')=='rx75' and r.get('scope')=='non_singleton' and r.get('threshold') in THRESHOLDS[1:]):
                original_cells[(r['input_id'],r['setting_id'],r['phase'])]=r
    records=[];errors=[];line_rows=[]
    for entry in entries:
        folder=run/'inputs'/entry['id'];base=dict(input_id=entry['id'],source_group=entry['source_group'],case=entry['case'],
            source_condition=entry.get('source_condition'),condition_value=entry.get('condition_value'),
            training_points=entry.get('training_points'),physical_history_group=entry.get('physical_history_group'),
            history_repeat=entry.get('history_repeat'),train_days=entry.get('train_days'))
        phases=('historical','cumulative') if entry.get('new_day_ref') else ('historical',)
        try:
            result=reader.read(folder/'result.json')
            seal=reader.read(result['seal']['path'],result['seal']['sha256']);banks=reader.read(seal['banks']['path'],seal['banks']['sha256'])
            meta=json.loads(reader.referenced(entry['evaluation_case_refs']['metadata.json']).read_text(encoding='utf-8'))
            with reader.referenced(entry['evaluation_case_refs']['branches.csv']).open(encoding='utf-8',newline='') as stream:branches=list(csv.DictReader(stream))
            labels=entry['labels'];weights,line_counts,physical=physical_coefficients(branches,labels,int(meta['root_bus']),entry['zbase_ohm'])
            truth=list(weights);Hstar=None
            if entry.get('Hstar_ref'):
                with np.load(reader.referenced(entry['Hstar_ref']),allow_pickle=False) as z:Hstar=np.asarray(z['Hstar'],float)
                if not np.allclose(physical,Hstar,atol=1e-12,rtol=1e-12):raise ValueError('Physical coefficients do not reproduce sealed Hstar')
            for threshold in THRESHOLDS:
                setting_id='original_frozen' if threshold==0 else f'rx75__non_singleton__f{str(threshold).replace(".","p")}'
                plan=banks['baseline'] if threshold==0 else next(g for g in banks['grid'] if g['setting']['id']==setting_id)
                ev=result['baseline'] if threshold==0 else next(g for g in result['grid'] if g['setting']['id']==setting_id)
                for phase in phases:
                    row=dict(base,threshold=threshold,setting_id=setting_id,phase=phase,status='complete')
                    if plan['status']!='complete' or ev['status']!='complete':
                        records.append(dict(row,status='source_failed',error=plan.get('error')));continue
                    selected=ev[phase]['selection']['selected'];old=original_cells[(entry['id'],setting_id,phase)]
                    if selected!=old['selected']:raise ValueError('Saved selection differs from audited row')
                    candidates=plan['candidates'];point=next(c['point'] for c in candidates if c['source_candidate_id']==selected)
                    row.update(selected=selected,candidate_count=len(candidates),parameters=None)
                    for scope in ('internal','full'):
                        v=error_classification(candidates,selected,truth,len(labels),seal['zero_tol'],internal=scope=='internal')
                        if (v['correct'],v['fn'],v['fp'],v['bank_has_exact'])!=(old[scope+'_exact'],old[scope+'_missing'],old[scope+'_extra'],old['bank_has_'+scope+'_truth']):
                            raise ValueError('Independent residual decomposition differs from owning audit')
                        row[scope]=v
                    if Hstar is not None:
                        nodes={axis:old['response']['resource_columns'][axis]['labels'] for axis in ('P','Q')}
                        params=parameter_errors(point,Hstar,weights,labels,entry['zbase_ohm'],seal['zero_tol'],nodes)
                        for axis in ('R','X'):
                            if not np.isclose(params['matrix'][axis+'_relative_frobenius'],old['response'][axis+'_relative_error'],atol=1e-13,rtol=1e-12):raise ValueError('Response error differs from owning audit')
                        for line in params.pop('lines'):
                            line_rows.append(dict(input_id=entry['id'],source_group=entry['source_group'],case=entry['case'],
                                training_points=entry['training_points'],threshold=threshold,phase=phase,selected=selected,
                                physical_branches_in_support=line_counts.get(tuple(line['support']),0),**line))
                        row['parameters']=params
                    records.append(row)
        except Exception as exc:
            errors.append(dict(input_id=entry['id'],error=type(exc).__name__+': '+str(exc)))
            existing={(r['threshold'],r['phase']) for r in records if r['input_id']==entry['id']}
            for threshold in THRESHOLDS:
                for phase in phases:
                    if (threshold,phase) not in existing:records.append(dict(base,threshold=threshold,phase=phase,status='postprocessing_failed',error=str(exc)))
    summary=dict(status='passed' if not errors else 'failed',errors=errors,inputs=len(entries),
        expected_historical_rows=len(entries)*len(THRESHOLDS),
        expected_cumulative_rows=sum(bool(e.get('new_day_ref')) for e in entries)*len(THRESHOLDS),
        records=len(records),failed_rows=sum(r['status']!='complete' for r in records),groups=aggregate(records),
        no_new_fitting=True,no_new_optimization=True,no_new_AC=True,
        scope='Already studied fixed histories; separate historical winner and validation+saved-new-day winner. Full support includes singleton and common supports.',
        uncertainty_scope='Matrix Frobenius error, matched physical-support line MAPE and resource-column response error are different metrics; no confidence or AC guarantee.',
        source_run=str(run.resolve()),source_audit=file_ref(audit_dir/'audit.json'),source_cells=file_ref(audit_dir/'cells.jsonl'),
        sources=list(reader.refs.values()),postprocessing_source=file_ref(__file__))
    out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    with (out/'records.jsonl').open('w',encoding='utf-8') as f:
        for row in records:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    with (out/'parameter_lines.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(line_rows[0]) if line_rows else ['input_id'])
        writer.writeheader();writer.writerows(line_rows)
    md=['# 内部收缩后的残余误差','',f"状态 {summary['status']}；{len(entries)}输入，{len(records)}条分档/计分记录，失败{summary['failed_rows']}。",'',
        '历史winner只用原验证段；累计winner增加原保存新日。旧400无新日。所有完整支持指标仍包括末端线。','',
        '|来源|计分|门槛|n|内部正确|完整正确|内部FN/FP|缺真支持的库|支持齐但无正确候选|有正确候选却选错|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in summary['groups']:
        if g['case']!='ALL' or g['condition'] is not None:continue
        c=g['internal']['categories'];md.append(f"|{g['source_group']}|{g['phase']}|{g['threshold']}|{g['completed']}/{g['planned']}|{g['internal']['correct']}|{g['full']['correct']}|{g['internal']['fn']}/{g['internal']['fp']}|{c.get('truth_support_missing_from_active_union',0)}|{c.get('union_covers_truth_but_no_exact_candidate',0)}|{c.get('exact_candidate_available_but_selected_wrong',0)}|")
    md+=['','矩阵、同支持逐线和资源列误差见summary.json；逐线数值在parameter_lines.csv。资源列表示单位注入方向，没有新增调度或AC。',
         '错误类别互斥；缺支持以活动支持判断，并另列结构从未出现与只有零权重两种情况。多余公共项、遗漏线与共同真实线分别保存。']
    (out/'SUMMARY.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('status','inputs','records','failed_rows')},ensure_ascii=False))
    if errors:print(json.dumps(dict(error_count=len(errors),first_error=errors[0]),ensure_ascii=False))
    return summary


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--audit',type=Path,default=ROOT/'outputs/impedance_floor_audit_20261116_v1')
    args=p.parse_args(argv);analyze(args.run,args.audit,args.out)


if __name__=='__main__':main()
