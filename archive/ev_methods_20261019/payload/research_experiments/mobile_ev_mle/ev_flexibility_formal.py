"""Frozen 39-history calibration and 48-history evaluation of v1 EV design."""
from __future__ import annotations
import argparse, base64, gzip, hashlib, json, math, os, platform, sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
import numpy as np
import scipy
from . import ev_flexibility_study as development
from .adaptive_statistics import binomial_exact_interval, paired_continuous_effects
HERE=Path(__file__).resolve().parent
METHODS=tuple(development.METHODS)
STATES=('unique_correct','false_rejection','safe_undecided','compute_failure')
CAL_SEEDS=tuple(range(982001,982040)); EVAL_SEEDS=tuple(range(983001,983049))
QUALITY=('uniform','temporal','spatial')
CASE_NAMES=('star','weak','moderate','strong'); CASE_MASS=(.15,.15,.4,.3)
FAMILIES=(((0,1),),((0,2),),((1,2),)); LABEL_NAMESPACE=72148293
ALPHA=Fraction(1,10); DELTA=Fraction(1,20)
COMPARISONS=(('charge_fixed','constant_fixed','primary_curve'),('charge_mobile','charge_fixed','primary_space'),
             ('v2g_fixed','charge_fixed','secondary_bidirectional'),('charge_staff','charge_mobile','secondary_staff'))
WRAPPER_FILES=('ev_flexibility_formal.py','ev_flexibility_formal_protocol.md')
def require(ok,message):
    if not ok:raise ValueError(message)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def identity(value):return digest(json.dumps(development.io.serial(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def read(path):return development.io.read_json(path)
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(development.io.serial(value),ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
    tmp=path.with_name(path.name+'.tmp');tmp.write_bytes(gzip.compress(raw,mtime=0) if path.suffix=='.gz' else raw)
    os.replace(tmp,path)
def labels_for_seed(seed):
    require(isinstance(seed,(int,np.integer)) and int(seed)>0,'positive integer history seed required')
    streams=np.random.SeedSequence([int(seed),LABEL_NAMESPACE]).spawn(3)
    seeds={name:int(s.generate_state(1)[0]) for name,s in zip(('case','family','quality'),streams)}
    case=str(np.random.default_rng(seeds['case']).choice(CASE_NAMES,p=CASE_MASS))
    family=FAMILIES[int(np.random.default_rng(seeds['family']).integers(3))]
    quality=str(np.random.default_rng(seeds['quality']).choice(QUALITY))
    return dict(seed=int(seed),case=case,family=[] if case=='star' else [list(s) for s in family],
                quality=quality,label_streams=seeds,label_namespace=LABEL_NAMESPACE)
def source_hashes():
    hashes=development.source_hashes()
    hashes.update({n:digest((HERE/n).read_bytes()) for n in WRAPPER_FILES});return hashes
def policy_hashes(freeze):return {k:v for k,v in freeze['source_sha256'].items() if k not in WRAPPER_FILES}
def freeze_payload():
    require(len(development.source_hashes())==21,'v1 source family changed')
    config=development.frozen.configuration();pop=config['population']
    require(tuple(pop['case_names'])==CASE_NAMES and tuple(pop['case_mass'])==CASE_MASS,'v3 case mixture differs')
    require(tuple(pop['quality_mass'])==(1/3,1/3,1/3),'quality mixture differs')
    value=dict(schema='ev_service_formal_v1',policy_config=development.CONFIG,generator_config=config,
        methods=METHODS,comparisons=COMPARISONS,source_sha256=source_hashes(),
        schedule={phase:[labels_for_seed(s) for s in seeds] for phase,seeds in [('calibration',CAL_SEEDS),('evaluation',EVAL_SEEDS)]},
        tolerance=dict(alpha=str(ALPHA),delta=str(DELTA),rank=39,n=39,binomial_tail_exact=str((1-ALPHA)**39),
            binomial_tail=float((1-ALPHA)**39),score='max9 full-joint-SSE truth gap; required fit/physical/coverage failure infinity',
            failed_calibration_stops_gain_evaluation=True,ties='gap<=threshold retained'),
        assumptions=['iid declared simulation mixture across independent H','frozen H-only scientific policy and budgets',
            'candidate coverage audited within three-terminal universe','one public quality per H; max across nine methods',
            'no subgroup, fixed-parameter, global-MLE, field-data or multi-round guarantee'],
        timing=dict(controller_gate_seconds=15.,history_gate_seconds=90.,
            action='diagnostic costs only: keep slow H and continue; no seed replacement',
            hard_solver_budgets='unchanged WZZT .3s, each affine LP/MILP5s, AC multistart2/max_nfev200'),
        environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,
            threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}))
    value=development.io.serial(value);value['policy_id']=identity(value);return value
def initialize_freeze(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);freeze=freeze_payload();path=output/'freeze.json'
    if path.exists():require(read(path)==freeze,'resume identity/config/source/environment differs; preserve prior output')
    else:save(path,freeze)
    snapshot=dict(schema='source_bytes_base64',policy_id=freeze['policy_id'],sources={n:dict(sha256=h,
        bytes_base64=base64.b64encode((HERE/n).read_bytes()).decode()) for n,h in freeze['source_sha256'].items()})
    path=output/'science_source_snapshot.json.gz'
    if path.exists():require(read(path)==snapshot,'saved exact source bytes differ')
    else:save(path,snapshot)
    return freeze
def validate_snapshot(snapshot,freeze):
    require(snapshot.get('policy_id')==freeze['policy_id'],'snapshot identity mismatch')
    require(set(snapshot.get('sources',{}))==set(freeze['source_sha256']),'snapshot coverage mismatch')
    for n,v in snapshot['sources'].items():
        require(digest(base64.b64decode(v['bytes_base64'],validate=True))==v['sha256']==freeze['source_sha256'][n],'source byte hash mismatch')
    return len(snapshot['sources'])
def rows(record):
    result={}
    for item in record.get('records',[]):
        require(item.get('method') not in result,'duplicate method row');result[item['method']]=item
    require(set(result)==set(METHODS),'missing required method row');return result
def candidates(record):
    names=[c['name'] for c in record.get('candidates',[])];require(len(names)==len(set(names)),'duplicate candidates');return tuple(names)
def complete_losses(row,names):
    if row.get('status')!='complete' or not names or set(row.get('losses',{}))!=set(names):return None
    try:values={k:float(v) for k,v in row['losses'].items()}
    except (TypeError,ValueError):return None
    return values if all(math.isfinite(v) and v>=0 for v in values.values()) else None
def valid_loss(value):
    try:number=float(value)
    except (TypeError,ValueError):return False
    return math.isfinite(number) and number>=0
def history_fit_failed(record):
    names=candidates(record);fits=record.get('historical_fits',[])
    historical={f.get('name'):f for f in fits}
    return bool(record.get('runner_exception') or record.get('historical_failures') or not names
        or len(fits)!=len(historical) or set(historical)!=set(names)
        or any(not f.get('success',False) or not valid_loss(f.get('loss')) for f in historical.values()))
def calibration_score(record):
    names=candidates(record);items=rows(record);reasons=[];gaps={}
    if record.get('runner_exception'):reasons.append('runner_exception')
    if history_fit_failed(record):reasons.append('historical_fit_failure')
    for method,row in items.items():
        truth=row.get('true_name');values=complete_losses(row,names)
        if not row.get('true_covered',False) or truth not in names:reasons.append(method+':coverage_failure')
        if values is None:reasons.append(method+':'+row.get('status','missing_status'))
        elif truth in values:gaps[method]=max(0.,values[truth]-min(values.values()))
    infinite=bool(reasons or len(gaps)!=len(METHODS))
    return dict(score=None if infinite else max(gaps.values()),score_is_infinite=infinite,reasons=reasons,method_truth_gaps=gaps)
def tolerance_threshold(scores):
    require(len(scores)==39,'exactly39 calibration H required')
    values=[math.inf if x['score_is_infinite'] else float(x['score']) for x in scores]
    require(all(not math.isnan(v) and v>=0 for v in values),'invalid calibration score')
    value=max(values);tail=(1-ALPHA)**39;require(tail<=DELTA,'insufficient tolerance rank')
    return dict(rank=39,n=39,alpha=str(ALPHA),delta=str(DELTA),threshold=None if math.isinf(value) else value,
        threshold_is_infinite=math.isinf(value),failed_history_count=sum(x['score_is_infinite'] for x in scores),
        binomial_tail_exact=str(tail),binomial_tail=float(tail),scope='random-calibration-set tolerance for next iid mixture H, all9 methods')
def retain_candidates(losses,names,threshold,*,complete=True):
    require(not math.isnan(threshold) and threshold>=0,'invalid threshold')
    require(len(names)==len(set(names)),'duplicate gate candidates')
    if not complete or not names or set(losses)!=set(names):return list(names)
    values={c:float(losses[c]) for c in names}
    if not all(math.isfinite(v) and v>=0 for v in values.values()):return list(names)
    best=min(values.values());return [c for c in names if values[c]-best<=threshold]
def evaluate_gate(record,threshold):
    names=candidates(record);out={};historical_failure=history_fit_failed(record)
    information_methods={'charge_fixed','charge_random','charge_mobile','v2g_fixed','v2g_mobile','charge_staff'}
    for m,row in rows(record).items():
        values=complete_losses(row,names);truth=row.get('true_name');coverage=bool(row.get('true_covered',False) and truth in names)
        complete=values is not None and coverage and not historical_failure
        kept=retain_candidates(values or {},names,threshold,complete=complete)
        state='compute_failure' if not complete else 'false_rejection' if truth not in kept else 'unique_correct' if kept==[truth] else 'safe_undecided'
        status=record.get('selection',{}).get('methods',{}).get(m,{}).get('status')
        activated=bool(record.get('selection',{}).get('activated',False))
        design_failure=bool(m in information_methods and activated and status not in
            ('information_optimized','expanded_domain_incumbent_guard'))
        reason=('historical_fit_failure' if historical_failure else 'candidate_coverage_failure' if not coverage
                else row.get('status','invalid_joint_losses') if values is None else None)
        out[m]=dict(state=state,retained=kept,truth=truth,coverage=coverage,status=row.get('status'),selection_status=status,
            selection_failure=status=='selection_failure_fallback',design_failure=design_failure,
            not_activated=bool(m in information_methods and not activated),
            controller_fallback='fallback' in str(status),activated=activated,
            failure_reason=reason,margin_unavailable_reason=reason if not complete else None,
            signed_margin_full_SSE=row.get('signed_margin_full_SSE') if complete else None,
            wrong_candidates_removed=sum(c!=truth and c not in kept for c in names) if coverage else None)
    return out
def validate_trial(record,phase,label,freeze,runner_kind='real'):
    formal=record.get('formal',{})
    require(formal.get('policy_id')==freeze['policy_id'] and formal.get('phase')==phase and formal.get('labels')==label,
            'trial resume identity/labels differs')
    require(formal.get('runner_kind')==runner_kind,'test-double data cannot resume as real scientific run')
    require(str(record.get('history_id'))==str(label['seed']) and record.get('case')==label['case'] and record.get('quality')==label['quality'],
            'trial H/case/quality mismatch')
    require(record.get('source_sha256')==policy_hashes(freeze),'trial source mismatch')
    require(not record.get('selector_uses_truth',True) and not record.get('selector_uses_future_noise',True),'input isolation failed');rows(record)
def exception_record(label,error,freeze):
    return dict(history_id=str(label['seed']),run_id=str(label['seed'])+'_'+label['quality'],case=label['case'],quality=label['quality'],
        source_sha256=policy_hashes(freeze),candidates=[],historical_fits=[],historical_failures=dict(runner=str(error)),
        runner_exception=str(error),selector_uses_truth=False,selector_uses_future_noise=False,
        seconds=None,total_ac_calls=None,selection=dict(activated=False,seconds=None,methods={m:dict(status='runner_exception') for m in METHODS}),
        records=[dict(method=m,status='runner_exception',true_name=None,true_covered=False) for m in METHODS])
def run_phase(output,phase,freeze,*,limit=None,runner=None):
    require(phase in ('calibration','evaluation'),'invalid phase');schedule=freeze['schedule'][phase]
    require(limit is None or isinstance(limit,int) and 1<=limit<=len(schedule),'limit outside declared phase')
    directory=Path(output)/phase;directory.mkdir(parents=True,exist_ok=True);records=[]
    kind='real' if runner is None else 'test_double';callback=development.run_condition if runner is None else runner
    for label in schedule[:len(schedule) if limit is None else limit]:
        require(source_hashes()==freeze['source_sha256'],'source changed during frozen run; preserve and stop')
        seed=label['seed'];target=directory/('trial_'+str(seed)+'.json.gz')
        if target.exists():record=read(target)
        else:
            try:record,_=callback(seed,label['case'],tuple(tuple(s) for s in label['family']),label['quality'])
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:record=exception_record(label,error,freeze)
            record['formal']=dict(policy_id=freeze['policy_id'],phase=phase,labels=label,runner_kind=kind,
                independent_history_unit=True,one_quality_per_H=True,old_threshold_applied=False,scientific_policy_unchanged=kind=='real')
            validate_trial(record,phase,label,freeze,kind);save(target,record)
        validate_trial(record,phase,label,freeze,kind);score=calibration_score(record);records.append(record)
        last=dict(history_id=str(seed),case=label['case'],quality=label['quality'],score=score,seconds=record.get('seconds'),
            selection_seconds=record.get('selection',{}).get('seconds'),original_cost_gate_passed=bool(record.get('gate_passed',False)))
        save(directory/'checkpoint.json',dict(policy_id=freeze['policy_id'],phase=phase,planned_histories=len(schedule),
            completed_histories=len(records),completed_ids=[r['history_id'] for r in records],last_history=last,
            phase_complete=len(records)==len(schedule),whole_research_complete=False))
        print(json.dumps(last,sort_keys=True),flush=True)
    return records
def calibration_summary(records,freeze):
    scores=[calibration_score(r) for r in records]
    result=dict(policy_id=freeze['policy_id'],planned_n=39,n=len(records),complete=len(records)==39,scores=scores,
        independent_history_ids=[r['history_id'] for r in records],failures=sum(s['score_is_infinite'] for s in scores),
        cost_gate_exceedances=sum(not bool(r.get('gate_passed',False)) for r in records),
        case_counts=dict(Counter(r['case'] for r in records)),quality_counts=dict(Counter(r['quality'] for r in records)),
        calibration_uses_evaluation_records=False,old_threshold_applied=False)
    if result['complete']:result['tolerance']=tolerance_threshold(scores)
    return result
def evaluation_summary(records,threshold,freeze):
    gates=[evaluate_gate(r,threshold) for r in records];n=len(records);methods={}
    for m in METHODS:
        count=Counter(g[m]['state'] for g in gates);require(sum(count.values())==n,'partition denominator mismatch')
        methods[m]=dict(n=n,counts={s:count[s] for s in STATES},rates={s:count[s]/n if n else None for s in STATES},
            activated_histories=sum(g[m]['activated'] for g in gates),selection_failures=sum(g[m]['selection_failure'] for g in gates),
            controller_fallbacks=sum(g[m]['controller_fallback'] for g in gates),
            design_failures=sum(g[m]['design_failure'] for g in gates),not_activated=sum(g[m]['not_activated'] for g in gates))
    comparisons=[]
    for a,b,kind in COMPARISONS:
        ua=[g[a]['state']=='unique_correct' for g in gates];ub=[g[b]['state']=='unique_correct' for g in gates];pairs=[]
        for r,g in zip(records,gates):
            va,vb=g[a]['signed_margin_full_SSE'],g[b]['signed_margin_full_SSE']
            if va is not None and vb is not None and math.isfinite(float(va)) and math.isfinite(float(vb)):pairs.append((r['history_id'],float(va),float(vb)))
        comparisons.append(dict(method=a,baseline=b,kind=kind,primary_full_n=n,
            unique_correct_difference=sum(int(x)-int(y) for x,y in zip(ua,ub))/n if n else None,
            unique_discordant_wins=sum(x and not y for x,y in zip(ua,ub)),unique_discordant_losses=sum(y and not x for x,y in zip(ua,ub)),
            margin_complete_pairs=len(pairs),margin_effect=paired_continuous_effects([p[0] for p in pairs],[p[1] for p in pairs],[p[2] for p in pairs]) if pairs else None))
    any_false=sum(any(g[m]['state']=='false_rejection' for m in METHODS) for g in gates)
    by_case={c:dict(n=sum(r['case']==c for r in records),methods={m:dict(Counter(g[m]['state'] for r,g in zip(records,gates) if r['case']==c)) for m in METHODS},
        margins={a+'_vs_'+b:[dict(history_id=r['history_id'],difference=g[a]['signed_margin_full_SSE']-g[b]['signed_margin_full_SSE'])
            for r,g in zip(records,gates) if r['case']==c and g[a]['signed_margin_full_SSE'] is not None and g[b]['signed_margin_full_SSE'] is not None]
            for a,b,_ in COMPARISONS}) for c in CASE_NAMES}
    return dict(policy_id=freeze['policy_id'],planned_n=48,n=n,complete=n==48,methods=methods,comparisons=comparisons,by_case=by_case,
        any_method_false_rejections=any_false,any_method_false_rejection_interval=binomial_exact_interval(any_false,n) if n else None,
        cost_gate_exceedances=sum(not bool(r.get('gate_passed',False)) for r in records),activation_count=sum(bool(r.get('selection',{}).get('activated',False)) for r in records),
        coverage_failure_histories=sum(not all(g[m]['coverage'] for m in METHODS) for g in gates),quality_counts=dict(Counter(r['quality'] for r in records)),
        total_ac_calls=sum(r.get('total_ac_calls') or 0 for r in records),actual_total_seconds=sum(r.get('seconds') or 0 for r in records),
        threshold=threshold,old_threshold_applied=False,whole_research_complete=False,
        claim='this frozen iid simulation mixture; case-specific results descriptive')
def run_batch(output,*,phase='all',limit=None,runner=None):
    require(phase in ('all','calibration','evaluation'),'invalid requested phase');output=Path(output);freeze=initialize_freeze(output)
    if phase in ('all','calibration'):cal=run_phase(output,'calibration',freeze,limit=limit,runner=runner)
    else:
        cal=[]
        for label in freeze['schedule']['calibration']:
            path=output/'calibration'/('trial_'+str(label['seed'])+'.json.gz');require(path.exists(),'saved full39-H calibration required')
            record=read(path);validate_trial(record,'calibration',label,freeze,'real' if runner is None else 'test_double');cal.append(record)
    summary=calibration_summary(cal,freeze);save(output/'calibration_summary.json',summary)
    if not summary['complete']:
        save(output/'checkpoint.json',dict(policy_id=freeze['policy_id'],stage='calibration_incomplete',completed_calibration=len(cal),whole_research_complete=False));return summary
    tolerance=summary['tolerance']
    if tolerance['threshold_is_infinite']:
        blocked=dict(policy_id=freeze['policy_id'],stage='uninformative_calibration',reason='required calibration fit/physical/coverage failure: max threshold infinity',
            retain_all_candidates=True,evaluation_started=False,whole_research_complete=False,tolerance=tolerance)
        save(output/'checkpoint.json',blocked);return blocked
    if phase=='calibration':return summary
    evaluation=run_phase(output,'evaluation',freeze,limit=limit,runner=runner)
    result=evaluation_summary(evaluation,float(tolerance['threshold']),freeze);result['tolerance']=tolerance;save(output/'evaluation_summary.json',result)
    save(output/'checkpoint.json',dict(policy_id=freeze['policy_id'],stage='formal_evaluation_complete' if result['complete'] else 'evaluation_incomplete',
        completed_calibration=39,completed_evaluation=len(evaluation),whole_research_complete=False,finite_threshold=tolerance['threshold']));return result
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('all','calibration','evaluation'),default='all')
    parser.add_argument('--limit',type=int,default=None,help='Execute/resume planned prefix; final39/48 denominators unchanged')
    parser.add_argument('--output',type=Path,default=HERE/'evidence/ev_flexibility/v1_formal');args=parser.parse_args()
    run_batch(args.output,phase=args.phase,limit=args.limit)
if __name__=='__main__':main()
