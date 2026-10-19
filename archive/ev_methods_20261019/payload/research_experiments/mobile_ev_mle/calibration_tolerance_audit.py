"""Read-only replay from committed compact CSVs; no raw trials or optimizer.

All expected history/method/candidate cells are required. H fits are audited,
but H is not a twelfth max-score method. A failed required fit gives score
+infinity; the affected evaluation method retains all candidates. Strict
score>threshold and retained gap<=threshold preserve ties and infinity rules.
"""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import argparse
import csv
import hashlib
import json
import math
import scipy
from scipy.stats import binom

HERE=Path(__file__).resolve().parent
METHODS=('none','constant_fixed','quality_global','quality_local','information_seed','information_fixed','information_random','information_designed','nominal_information','uniform_information','profile_designed')
CANDIDATES=('clade_01','clade_02','clade_12','star')
STATES=('unique_correct','safe_undecided','false_rejection','compute_failure')
META=('case','quality','true_name')


def require(ok,message):
    if not ok:raise ValueError(message)


def finite(value,label):
    try:v=float(value)
    except (TypeError,ValueError) as error:raise ValueError(label+' is missing or nonnumeric') from error
    require(math.isfinite(v),label+' is nonfinite')
    return v


def boolean(value,label):
    require(value in ('True','False'),label+' is not an explicit boolean')
    return value=='True'


def read_csv(path,required):
    with Path(path).open(encoding='utf-8',newline='') as handle:
        reader=csv.DictReader(handle);header=reader.fieldnames or []
        require(len(header)==len(set(header)) and set(required)<=set(header),'missing or duplicate CSV header: '+str(path))
        rows=list(reader)
    require(bool(rows) and all(None not in r for r in rows),'empty/malformed CSV: '+str(path))
    return rows


def validate_phase(method_rows,fit_rows,expected_histories):
    """Validate the full product, metadata, finite losses, status and true gap."""
    ids=set(expected_histories);require(bool(ids),'empty expected history set')
    method_map={};fit_map={};meta={}
    for r in method_rows:
        require(set(('history_id','method','status','true_gap')+META)<=set(r),'missing method fields')
        key=(r['history_id'],r['method']);require(key not in method_map,'duplicate history/method')
        require(key[0] in ids and key[1] in METHODS,'unexpected history/method')
        require(r['status'] in ('complete','joint_fit_failure'),'unsupported method status')
        require(r['case'] in ('star','weak','moderate','strong') and r['quality'] in ('uniform','temporal','spatial') and r['true_name'] in CANDIDATES,'invalid history metadata')
        values=tuple(r[k] for k in META);require(key[0] not in meta or meta[key[0]]==values,'method metadata conflict')
        meta[key[0]]=values;method_map[key]=dict(r)
    require(set(method_map)=={(h,m) for h in ids for m in METHODS},'missing history/method coverage')
    for r in fit_rows:
        require(set(('history_id','method','name','success','loss')+META)<=set(r),'missing fit fields')
        key=(r['history_id'],r['method'],r['name']);require(key not in fit_map,'duplicate candidate fit')
        require(key[0] in ids and key[1] in METHODS+('H',) and key[2] in CANDIDATES,'unexpected fit cell')
        require(tuple(r[k] for k in META)==meta[key[0]],'fit metadata conflict')
        ok=boolean(r['success'],'fit success');loss=finite(r['loss'],'fit loss');require(loss>=0,'negative SSE')
        fit_map[key]=(ok,loss)
    require(set(fit_map)=={(h,m,c) for h in ids for m in METHODS+('H',) for c in CANDIDATES},'missing candidate/H fit coverage')
    for (h,m),r in method_map.items():
        fits=[fit_map[h,m,c] for c in CANDIDATES];all_success=all(ok for ok,_ in fits)
        if r['status']=='complete':
            require(all_success,'complete method has unsuccessful fit')
            losses={c:fit_map[h,m,c][1] for c in CANDIDATES}
            gap=losses[r['true_name']]-min(losses.values())
            require(finite(r['true_gap'],'complete true_gap')==gap,'true_gap does not match full-SSE loss difference')
        else:
            require(not all_success,'failed method has no failed fit')
            if r['true_gap'] not in ('',None):finite(r['true_gap'],'failed true_gap')
    return dict(ids=sorted(ids),methods=method_map,fits=fit_map,meta=meta)


def history_scores(phase):
    out={}
    for h in phase['ids']:
        failed=any(not phase['fits'][h,'H',c][0] for c in CANDIDATES) or any(phase['methods'][h,m]['status']!='complete' for m in METHODS)
        out[h]=math.inf if failed else max(float(phase['methods'][h,m]['true_gap']) for m in METHODS)
    return out


def tolerance_rank(scores,alpha=Fraction(1,10),delta=Fraction(1,20)):
    """Select by exact decimal-rational tails, independently compare SciPy."""
    alpha=Fraction(str(alpha));delta=Fraction(str(delta));n=len(scores)
    require(n>0 and 0<alpha<1 and 0<delta<1,'invalid exact tolerance parameters')
    require(all(not math.isnan(float(x)) and float(x)>=0 for x in scores),'invalid score')
    p=1-alpha;tails={k:sum((Fraction(math.comb(n,j))*p**j*alpha**(n-j) for j in range(k,n+1)),Fraction(0)) for k in range(1,n+1)}
    require(all(abs(float(v)-float(binom.sf(k-1,n,float(p))))<2e-14 for k,v in tails.items()),'SciPy/exact binomial tail mismatch')
    rank=next((k for k,v in tails.items() if v<=delta),n+1)
    threshold=sorted(scores)[rank-1] if rank<=n else math.inf
    return dict(n=n,alpha_exact=str(alpha),delta_exact=str(delta),rank=rank,threshold=threshold,tail_exact=str(tails.get(rank,Fraction(0))),tail=float(tails.get(rank,0)),tail_by_rank={k:float(v) for k,v in tails.items()})


def retained(phase,h,m,threshold):
    r=phase['methods'][h,m]
    if r['status']!='complete':return set(CANDIDATES)
    losses={c:phase['fits'][h,m,c][1] for c in CANDIDATES};best=min(losses.values())
    return {c for c,v in losses.items() if v-best<=threshold}


def state(status,kept,truth):
    if status!='complete':return 'compute_failure'
    if truth not in kept:return 'false_rejection'
    return 'unique_correct' if kept=={truth} else 'safe_undecided'


def verify_original(phase,threshold):
    for h in phase['ids']:
        for m in METHODS:
            r=phase['methods'][h,m]
            require(set(('retained_candidates','only_truth_retained','false_rejection','unresolved','wrong_candidates_removed'))<=set(r),'missing original gate fields')
            try:saved=json.loads(r['retained_candidates'])
            except (TypeError,json.JSONDecodeError) as error:raise ValueError('missing/malformed original kept') from error
            require(isinstance(saved,list) and all(isinstance(x,str) for x in saved) and len(saved)==len(set(saved)) and set(saved)<=set(CANDIDATES),'invalid original kept')
            kept=retained(phase,h,m,threshold);require(set(saved)==kept,'original kept mismatch')
            truth=r['true_name'];require(boolean(r['only_truth_retained'],'original only truth')==(kept=={truth}),'original unique flag mismatch')
            require(boolean(r['false_rejection'],'original false rejection')==(truth not in kept),'original false flag mismatch')
            require(boolean(r['unresolved'],'original unresolved')==(len(kept)>1),'original unresolved flag mismatch')
            require(str(sum(c!=truth and c not in kept for c in CANDIDATES))==r['wrong_candidates_removed'],'original removed count mismatch')


def re_gate(phase,old_threshold,new_threshold):
    require(new_threshold>=old_threshold,'threshold decreased')
    verify_original(phase,old_threshold)  # Validate every old set/flag before any new gate.
    scores=history_scores(phase);rows=[];changes=[]
    for h in phase['ids']:
        case,quality,truth=phase['meta'][h];item=dict(history_id=h,case=case,old={},new={},old_any_false=False,new_any_false=False,old_score_exceeds=scores[h]>old_threshold,new_score_exceeds=scores[h]>new_threshold)
        for m in METHODS:
            r=phase['methods'][h,m];old=retained(phase,h,m,old_threshold);new=retained(phase,h,m,new_threshold)
            require(old<=new,'retained set not monotone');a=state(r['status'],old,truth);b=state(r['status'],new,truth)
            item['old'][m]=a;item['new'][m]=b;item['old_any_false']|=truth not in old;item['new_any_false']|=truth not in new
            if old!=new:changes.append(dict(history_id=h,case=case,method=m,old_retained=sorted(old),new_retained=sorted(new),old_partition=a,new_partition=b))
        rows.append(item)
    def aggregate(ts):
        out=dict(n=len(ts),methods={},old_any_false_count=sum(r['old_any_false'] for r in ts),new_any_false_count=sum(r['new_any_false'] for r in ts),old_score_exceeds_count=sum(r['old_score_exceeds'] for r in ts),new_score_exceeds_count=sum(r['new_score_exceeds'] for r in ts))
        for m in METHODS:
            old=Counter(r['old'][m] for r in ts);new=Counter(r['new'][m] for r in ts);trans=Counter((r['old'][m],r['new'][m]) for r in ts)
            out['methods'][m]=dict(old={s:old[s] for s in STATES},new={s:new[s] for s in STATES},transitions={a+' -> '+b:v for (a,b),v in trans.items()})
        return out
    return dict(all_eighty=aggregate(rows),by_case={c:aggregate([r for r in rows if r['case']==c]) for c in sorted({r['case'] for r in rows})},retained_set_changes=changes)


def audit(directory=None,diagnostic_path=None):
    directory=Path(directory or HERE/'evidence/structured_information/v3_frozen')
    diagnostic_path=Path(diagnostic_path or directory.parent/'calibration_tolerance_diagnostic.json')
    inputs={};phases={}
    for label,first,count in (('calibration',940001,39),('evaluation',950001,80)):
        paths={key:directory/label/'tables'/f'{key}.csv' for key in ('methods','fits')}
        mr=read_csv(paths['methods'],('history_id','method','status','true_gap','retained_candidates','only_truth_retained','false_rejection','unresolved','wrong_candidates_removed')+META)
        fr=read_csv(paths['fits'],('history_id','method','name','success','loss')+META)
        phases[label]=validate_phase(mr,fr,(str(h) for h in range(first,first+count)))
        inputs.update({label+'/'+key+'.csv':hashlib.sha256(p.read_bytes()).hexdigest() for key,p in paths.items()})
    cal_scores=list(history_scores(phases['calibration']).values());old_threshold=sorted(cal_scores)[35]
    threshold_path=directory/'calibration/thresholds.json';thresholds=json.loads(threshold_path.read_text(encoding='utf-8'))
    require(thresholds['relative']['rank']==36 and thresholds['relative']['sample_count']==39 and thresholds['alpha']==.1 and thresholds['relative']['threshold']==old_threshold,'old threshold/parameters mismatch')
    verify_original(phases['evaluation'],old_threshold)
    choice=tolerance_rank(cal_scores)
    require(choice['rank']==39 and choice['alpha_exact']=='1/10' and choice['delta_exact']=='1/20','unexpected fixed rank parameters')
    expected=json.loads(diagnostic_path.read_text(encoding='utf-8'));cal=expected['calibration_only_checkpoint']
    require(Fraction(str(cal['protocol']['alpha']))==Fraction(1,10) and Fraction(str(cal['protocol']['delta']))==Fraction(1,20) and cal['calibration_count']==39 and tuple(cal['methods'])==METHODS,'diagnostic exact parameters/method family mismatch')
    require(cal['old_rank']==36 and cal['old_threshold']==old_threshold and cal['new_rank']==choice['rank'] and cal['new_threshold']==choice['threshold'],'diagnostic calibration mismatch')
    replay=re_gate(phases['evaluation'],old_threshold,choice['threshold'])
    require(all(replay[key]==expected[key] for key in ('all_eighty','by_case','retained_set_changes')),'diagnostic results mismatch')
    inputs['thresholds.json']=hashlib.sha256(threshold_path.read_bytes()).hexdigest();inputs['diagnostic.json']=hashlib.sha256(diagnostic_path.read_bytes()).hexdigest()
    finite_choice={k:v for k,v in choice.items() if k!='threshold'};finite_choice['threshold']=None if math.isinf(choice['threshold']) else choice['threshold'];finite_choice['threshold_is_infinite']=math.isinf(choice['threshold'])
    return dict(validation_success=True,stage='independent_compact_CSV_replay',coverage=dict(cal_histories=39,cal_methods=39*11,cal_candidate_fits=39*11*4,cal_H_fits=39*4,eval_histories=80,eval_methods=80*11,eval_candidate_fits=80*11*4,eval_H_fits=80*4),old_rank=36,old_threshold=old_threshold,tolerance=finite_choice,original_kept_and_flags_verified=880,retained_sets_monotone=True,all_eighty=replay['all_eighty'],by_case=replay['by_case'],changed_method_histories=len(replay['retained_set_changes']),diagnostic_numeric_comparison_passed=True,input_sha256=inputs,source_sha256={name:hashlib.sha256(HERE.joinpath(name).read_bytes()).hexdigest() for name in ('calibration_tolerance_audit.py','test_calibration_tolerance_audit.py')},scipy_version=scipy.__version__,new_optimization_or_fit_calls=0,raw_trials_read=0,scope='post-formal saved-loss replay; random-calibration-set iid mixture tolerance only; no subgroup/fixed-theta guarantee or new forward evaluation')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path);parser.add_argument('--diagnostic',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=audit(args.directory,args.diagnostic)
    with args.output.open('x',encoding='utf-8',newline='\n') as handle:json.dump(result,handle,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps(dict(validation_success=True,original_kept_verified=880,output=str(args.output)),ensure_ascii=False))
