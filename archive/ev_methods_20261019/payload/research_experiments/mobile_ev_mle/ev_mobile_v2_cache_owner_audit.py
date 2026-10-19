"""Audit source-profile cache ownership without fitting or changing a policy.

The four source-only planners run in DECLARED_ORDER with one cache.  A cache
hit retains the original profile's ``seconds`` although the calling engine
logs actual_seconds=0.  This tool checks whether the first producer belongs
to the caller's declared ancestor closure.  Repeated hits to one unpaid key
are charged once in the missing-fee ledger and separately counted as queries.

Only saved source-profile production time is reconstructed.  It is not a
counterfactual independent run: actions, deadline effects, and results remain
unchanged.  Missing profiles/counters produce explicit incomplete denominators
and lower bounds.  Compact/lite evidence cannot establish zero missing fees.
The tool imports no AC fitter, planner, solver, or physical reviewer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import struct
from time import perf_counter


DECLARED_ORDER=('scalar_fixed','scalar_mobile','cap_fixed','cap_mobile')
DECLARED_DEPENDENCIES={
    'scalar_fixed':{'square_fixed'},
    'scalar_mobile':{'square_fixed','square_mobile','scalar_fixed'},
    'cap_fixed':{'square_fixed','scalar_fixed'},
    'cap_mobile':{'square_fixed','square_mobile','scalar_mobile','cap_fixed'}}
EVENT_FIELDS=('method','cache_hit','actual_seconds','paid_pair_fits','source_profile_count')
SCHEMA='mobile_EV_source_cache_owner_audit_v1'


def _finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def _nonnegative(value):
    return _finite(value) and value>=0


def _integer(value):
    return isinstance(value,int) and not isinstance(value,bool) and value>=0


def _close(a,b):
    return _finite(a) and _finite(b) and abs(a-b)<=1e-9*(1+abs(b))


def _profile_calls(raw):
    """Call sites only; selected_profile is a repeated result, not a new call."""
    calls=[]
    for i,row in enumerate(raw.get('initial_candidates',[])):
        if 'profile' in row:calls.append((f'raw.initial_candidates[{i}].profile',row['profile']))
    for i,row in enumerate(raw.get('grids',[])):
        if 'raw_AC_profile' in row:calls.append((f'raw.grids[{i}].raw_AC_profile',row['raw_AC_profile']))
    return calls


def _key(profile):
    power=profile['power_kw']
    if not isinstance(power,(list,tuple)) or not power:raise ValueError('nonempty time-by-node power matrix required')
    n=len(power[0]);flat=[]
    if n==0:raise ValueError('nonempty node dimension required')
    for row in power:
        if not isinstance(row,(list,tuple)) or len(row)!=n or not all(_finite(v) for v in row):
            raise ValueError('finite rectangular time-by-node power required')
        flat.extend(float(v) for v in row)
    method=profile.get('method','ac')
    if method!='ac':raise ValueError('declared scalar/cap source cache uses AC profiles only')
    digest=sha256(struct.pack('<'+'d'*len(flat),*flat)).hexdigest()
    return f'{method}:{len(power)}x{n}:{digest}'


def _closure(metadata,name):
    paid=set();visiting=set()
    def visit(key):
        if key in visiting:raise ValueError('dependency cycle at '+key)
        if key in paid:return
        if key not in metadata:raise ValueError('missing dependency '+key)
        visiting.add(key)
        for parent in metadata[key].get('dependencies',[]):visit(parent)
        visiting.remove(key);paid.add(key)
    visit(name)
    return paid


def _event_errors(event):
    errors=[]
    if not isinstance(event,dict):return ['event is not an object']
    absent=[k for k in EVENT_FIELDS if k not in event]
    if absent:return ['missing event fields: '+','.join(absent)]
    if event['method']!='ac':errors.append('event method differs from declared AC source cache')
    if not isinstance(event['cache_hit'],bool):errors.append('cache_hit must be boolean')
    if not _nonnegative(event['actual_seconds']):errors.append('invalid actual_seconds')
    for key in ('paid_pair_fits','source_profile_count'):
        if not _integer(event[key]):errors.append('invalid '+key)
    if event.get('cache_hit') is True:
        if event.get('actual_seconds')!=0. or event.get('paid_pair_fits')!=0:
            errors.append('cache hit must have zero incremental time and pair fits')
    elif event.get('cache_hit') is False and not isinstance(event.get('deadline_reached'),bool):
        errors.append('new profile event must declare deadline_reached')
    return errors


def _matches(event,profile):
    """A conservative event/profile pairing; never pair a discarded failed call."""
    if _event_errors(event):return False
    if not _nonnegative(profile.get('seconds')):return False
    if profile.get('method','ac')!=event['method']:return False
    if event['source_profile_count']!=len(profile.get('pairs',[])):return False
    if event['cache_hit']:
        return not profile.get('deadline_reached',False)
    return (_close(event['actual_seconds'],profile['seconds'])
        and event['paid_pair_fits']==profile.get('paid_pair_fits')
        and event['deadline_reached']==profile.get('deadline_reached'))


def _alignment(events,calls):
    """Pair only positions common to every order-preserving valid alignment.

    Cap records omit failed returned profiles.  Usually their nonzero event
    time identifies the surviving calls uniquely.  Ambiguous positions remain
    unknown.  This is bookkeeping, with no optimizer or numerical AC calls.
    """
    E,P=len(events),len(calls)
    if E<P:return {},False
    compatible=[[_matches(event,p) for _,p in calls] for event in events]
    left=[[False]*(P+1) for _ in range(E+1)];left[0][0]=True
    for i in range(E):
        for j in range(P+1):
            if left[i][j]:
                left[i+1][j]=True
                if j<P and compatible[i][j]:left[i+1][j+1]=True
    if not left[E][P]:return {},False
    right=[[False]*(P+1) for _ in range(E+1)];right[E][P]=True
    for i in range(E-1,-1,-1):
        for j in range(P,-1,-1):
            right[i][j]=right[i+1][j] or (j<P and compatible[i][j] and right[i+1][j+1])
    pairs={}
    for j in range(P):
        choices=[i for i in range(E) if left[i][j] and compatible[i][j] and right[i+1][j+1]]
        if len(choices)==1:pairs[choices[0]]=calls[j]
    return pairs,len(pairs)==E==P


def audit_record(record,*,path=None):
    """Return a small ownership ledger; findings do not alter scientific results."""
    errors=[];metadata=record.get('selection',{}).get('methods',{})
    raw_available='delivery_provenance' not in record
    result=dict(schema=SCHEMA,path=None if path is None else str(path),
        identity={k:record.get(k) for k in ('schema','physical_bank','phase','index','seed','status','policy_id','source_unchanged','case_id','family_index')},
        declared_order=list(DECLARED_ORDER),methods={},owners=[],events=[],errors=errors,
        full_raw_evidence_available=raw_available,
        key_scope='within one history with common context, fit settings and seed; AC plus exact little-endian float64 control bytes',
        signature_scope='the frozen caller supplies one common context, source options and seed; engine signature is not separately stored in raw events',
        accounting_bound_scope='lower bounds concern observed original fees of unique keys outside the paid closure; no bound is claimed for counterfactual independent-run wall time',
        fee_scope='saved source-profile production time only; no independent rerun, changed deadline, plan or inference')
    hashes=record.get('source_sha256')
    result['identity']['source_sha256_digest']=None if hashes is None else sha256(
        json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if record.get('schema')!='mobile_reliability_EV_v2':errors.append('unexpected raw record schema')
    cache={};unknown_producers=set();any_missing=False
    for name in DECLARED_ORDER:
        method=metadata.get(name);raw={} if method is None or not raw_available else (method.get('raw') or {})
        row=dict(history_denominator=1,metadata_available=method is not None,
            planning_success=None if method is None else method.get('success'),
            planning_failure=None if method is None else method.get('failure'),
            saved_profile_calls=0,event_calls=0,aligned_event_calls=0,unmapped_event_calls=0,
            cache_hits=0,new_profile_events=0,unowned_hit_queries=0,unowned_unique_keys=0,
            unowned_original_seconds_per_query=0.,unowned_unique_original_seconds=0.,
            unowned_unique_original_pair_fits=0,unresolved_owner_queries=0,
            ownership_provable_paid_without_identified_owner=0,unowned_queries=[],
            logged_actual_profile_seconds=None,logged_attempted_pair_task_calls=None,
            own_actual_seconds=None if method is None else method.get('planning_actual_seconds'),
            reported_planning_seconds=None if method is None else method.get('planning_seconds'))
        result['methods'][name]=row
        if method is not None and set(method.get('dependencies',[]))!=DECLARED_DEPENDENCIES[name]:
            errors.append(name+': dependencies differ from the frozen four-planner declaration')
        try:
            paid=_closure(metadata,name) if method is not None else set()
            row['paid_method_closure']=sorted(paid) if method is not None else None
        except ValueError as exc:
            errors.append(name+': '+str(exc));paid={name};row['paid_method_closure']=None
        calls=_profile_calls(raw);row['saved_profile_calls']=len(calls)
        counters=raw.get('counters');events=None if not isinstance(counters,dict) else counters.get('cache_payment_log')
        if not isinstance(events,list):
            row.update(counter_log_available=False,call_coverage_complete=False,ownership_accounting='lower_bound',
                incomplete_reason='source engine counters/log absent; prior computation cannot be assumed zero')
            unknown_producers.add(name);any_missing=True
            continue
        row['counter_log_available']=True;row['event_calls']=len(events)
        for i,event in enumerate(events):
            for error in _event_errors(event):errors.append(f'{name}.counters.cache_payment_log[{i}]: {error}')
        valid_events=all(not _event_errors(e) for e in events)
        if valid_events:
            row['cache_hits']=sum(e['cache_hit'] for e in events);row['new_profile_events']=len(events)-row['cache_hits']
            for key,expected in (('profile_evaluations',row['new_profile_events']),('profile_cache_hits',row['cache_hits']),
                    ('pair_fit_calls',sum(e['paid_pair_fits'] for e in events))):
                if counters.get(key)!=expected:errors.append(name+': counter mismatch '+key)
            elapsed=sum(e['actual_seconds'] for e in events)
            row['logged_actual_profile_seconds']=elapsed
            row['logged_attempted_pair_task_calls']=sum(e['paid_pair_fits'] for e in events)
            if not _close(counters.get('actual_profile_seconds'),elapsed):errors.append(name+': counter mismatch actual_profile_seconds')
            if not _nonnegative(row['own_actual_seconds']) or elapsed>row['own_actual_seconds']+1e-6:
                errors.append(name+': own planning time omits newly computed profiles')
        pairs,complete=_alignment(events,calls)
        row['aligned_event_calls']=len(pairs);row['unmapped_event_calls']=len(events)-len(pairs)
        row['call_coverage_complete']=complete
        if len(calls)>len(events):errors.append(name+': more saved calls than payment events')
        if len(events)==len(calls) and not complete:errors.append(name+': payment events do not match saved call sequence')
        unpaid={};has_unknown=False
        for i,event in enumerate(events):
            event_path=f'selection.methods.{name}.raw.counters.cache_payment_log[{i}]'
            detail=dict(caller=name,event_index=i,event_path=event_path,event=event,
                profile_path=None,key=None,owner_method=None,owner_path=None,original_seconds=None,ownership='unresolved')
            result['events'].append(detail)
            if i not in pairs:
                has_unknown=True
                if isinstance(event,dict) and event.get('cache_hit'):
                    row['unresolved_owner_queries']+=1
                if not isinstance(event,dict) or (event.get('cache_hit') is False and not event.get('deadline_reached')):
                    unknown_producers.add(name)
                continue
            profile_path,profile=pairs[i];detail['profile_path']=f'selection.methods.{name}.'+profile_path
            try:key=_key(profile)
            except (ValueError,KeyError,TypeError) as exc:
                errors.append(event_path+': '+str(exc));has_unknown=True;unknown_producers.add(name);continue
            detail['key']=key;detail['original_seconds']=profile['seconds']
            if not event['cache_hit']:
                if key in cache:errors.append(event_path+': new profile event repeats an existing cache key')
                owner=dict(key=key,method=name,event_path=event_path,profile_path=detail['profile_path'],
                    seconds=profile['seconds'],paid_pair_fits=profile.get('paid_pair_fits'),
                    cached=not profile.get('deadline_reached',False))
                detail.update(owner_method=name,owner_path=event_path,ownership='own_production')
                result['owners'].append(owner)
                if owner['cached']:cache.setdefault(key,owner)
                continue
            owner=cache.get(key)
            if owner is None:
                possible=set(unknown_producers)
                detail['possible_unmapped_producers']=sorted(possible)
                if not possible:
                    errors.append(event_path+': cache hit lacks any possible earlier producer')
                    has_unknown=True;row['unresolved_owner_queries']+=1
                elif possible<=paid:
                    detail['ownership']='paid_unmapped_producer'
                    row['ownership_provable_paid_without_identified_owner']+=1
                else:
                    has_unknown=True;row['unresolved_owner_queries']+=1
                continue
            detail.update(owner_method=owner['method'],owner_path=owner['event_path'])
            if not _close(profile['seconds'],owner['seconds']):errors.append(event_path+': cached original seconds changed')
            if profile.get('paid_pair_fits')!=owner['paid_pair_fits']:errors.append(event_path+': cached original pair fits changed')
            if owner['method'] in paid:detail['ownership']='paid_ancestor_or_self'
            else:
                detail['ownership']='unowned_hit';row['unowned_hit_queries']+=1
                row['unowned_original_seconds_per_query']+=owner['seconds'];unpaid[key]=owner
                row['unowned_queries'].append({k:detail[k] for k in ('event_path','profile_path','key','owner_method','owner_path','original_seconds')})
        row['unowned_unique_keys']=len(unpaid)
        row['unowned_unique_original_seconds']=sum(p['seconds'] for p in unpaid.values())
        row['unowned_unique_original_pair_fits']=sum(p['paid_pair_fits'] for p in unpaid.values())
        row['ownership_accounting']='exact_for_saved_calls' if complete and not has_unknown and valid_events else 'lower_bound'
        if _nonnegative(row['reported_planning_seconds']):
            row['saved_path_planning_seconds_plus_known_missing_profiles']=row['reported_planning_seconds']+row['unowned_unique_original_seconds']
        any_missing|=row['ownership_accounting']!='exact_for_saved_calls'
    for name,row in result['methods'].items():
        row['errors']=[error for error in errors if error.startswith((name+':',name+'.','selection.methods.'+name+'.'))]
        if row['errors'] or record.get('schema')!='mobile_reliability_EV_v2':row['ownership_accounting']='invalid'
    result['success']=not errors
    result['complete']=not any_missing and not errors
    result['unowned_hit_queries']=sum(r['unowned_hit_queries'] for r in result['methods'].values())
    result['unowned_unique_original_seconds_by_method_sum']=sum(r['unowned_unique_original_seconds'] for r in result['methods'].values())
    return result


def summarize(records):
    result=dict(history_denominator=len(records),valid_history_count=sum(r['success'] for r in records),
        complete_history_count=sum(r['complete'] for r in records),methods={})
    fields=('event_calls','aligned_event_calls','unmapped_event_calls','cache_hits','new_profile_events',
        'unowned_hit_queries','unowned_unique_keys','unowned_original_seconds_per_query',
        'unowned_unique_original_seconds','unowned_unique_original_pair_fits','unresolved_owner_queries',
        'ownership_provable_paid_without_identified_owner')
    for name in DECLARED_ORDER:
        rows=[r['methods'][name] for r in records]
        row=dict(history_denominator=len(rows),metadata_available_count=sum(r['metadata_available'] for r in rows),
            counter_log_available_count=sum(r.get('counter_log_available',False) for r in rows),
            exact_accounting_history_count=sum(r['ownership_accounting']=='exact_for_saved_calls' for r in rows),
            lower_bound_history_count=sum(r['ownership_accounting']=='lower_bound' for r in rows),
            invalid_history_count=sum(r['ownership_accounting']=='invalid' for r in rows),
            histories_with_known_unowned_hit=sum(r['unowned_hit_queries']>0 for r in rows),
            planning_status_counts=dict(Counter(str(r['planning_success']) for r in rows)))
        row.update({k:sum(r[k] for r in rows) for k in fields});result['methods'][name]=row
    return result


def audit_files(files,*,expected_count=None,progress=False):
    paths=sorted({Path(p).resolve() for p in files})
    if expected_count is not None and len(paths)!=expected_count:raise ValueError('input file count differs from expected count')
    began=perf_counter();records=[]
    for i,path in enumerate(paths):
        data=path.read_bytes();payload=gzip.decompress(data) if path.suffix=='.gz' else data
        record=json.loads(payload)
        audited=audit_record(record,path=path)
        audited['input_bytes']=len(data);audited['input_sha256']=sha256(data).hexdigest();records.append(audited)
        if progress:print(f'{i+1}/{len(paths)} {path.name}: hits={audited["unowned_hit_queries"]} complete={audited["complete"]}',flush=True)
    return dict(schema=SCHEMA,success=all(r['success'] for r in records),complete=all(r['complete'] for r in records),
        summary=summarize(records),records=records,seconds=perf_counter()-began,
        auditor_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='source-cache fee ownership only; no optimizer or AC execution; known missing fees are same-saved-path production costs',
        event_schema=dict(required=list(EVENT_FIELDS),new_profile_extra='deadline_reached',
            hit='actual_seconds=0 and paid_pair_fits=0; returned profile retains original seconds and fits',
            paid_pair_fits='attempted directed pair tasks, including failed tasks; not least-squares restart count',
            source_profile_count='all pairs including deadline placeholders; not the number of attempted fits'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',nargs='*',default=[])
    parser.add_argument('--file',nargs='*',default=[])
    parser.add_argument('--expected-count',type=int)
    parser.add_argument('--output',required=True)
    parser.add_argument('--progress',action='store_true')
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError('audit output already exists: '+str(output))
    files=list(map(Path,args.file))
    for directory in args.directory:
        directory=Path(directory)
        if not directory.is_dir():raise FileNotFoundError(directory)
        files.extend(directory.glob('case_*.json.gz'));files.extend(directory.glob('case_*.json'))
    if not files:raise ValueError('explicit input files or nonempty directories required')
    result=audit_files(files,expected_count=args.expected_count,progress=args.progress)
    output.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(result,ensure_ascii=False,sort_keys=True,allow_nan=False,indent=2).encode('utf-8')
    output.write_bytes(gzip.compress(data) if output.suffix=='.gz' else data)
    print(json.dumps(result['summary'],ensure_ascii=False,sort_keys=True),flush=True)
    if not result['success']:raise SystemExit(1)


if __name__=='__main__':main()
