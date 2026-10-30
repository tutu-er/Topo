"""Read-only attribution of completed controls; no solver or simulator."""
from pathlib import Path
from collections import Counter
from hashlib import sha256
import json

S=Path(r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a')
OUT=S/'wzzt_b_regret_review_v1'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def write(name,v):(OUT/name).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
    if (OUT/'balance_aggregate.json').exists():raise FileExistsError('completed audit retained')
    merged=S/'wzzt_b_completion_v1/merged_evaluations.json'
    proto=S/'wzzt_b_design_v1/protocol.json'
    rows=read(merged);cfg=read(proto);kind={m['id']:m['kind'] for m in cfg['menus']}
    inputs={str(p):sha(p) for p in (merged,proto)};seen={};counts=Counter()
    for row in rows:
        key=(row['flow'],row['menu_id']);rp=Path(row['execution_receipt']);r=read(rp);z=r['controls']
        if key in seen:
            if z!=seen[key]:raise RuntimeError('control trajectory changed across counterfactual models')
        else:seen[key]=z;inputs[str(rp)]=sha(rp)
        counts[key]+=1
    assert len(rows)==308 and len(seen)==44 and set(counts.values())=={7}
    result=[]
    for typ in ('pair_rotation','pair1'):
        controls=[z for (flow,mid),z in seen.items() if kind[mid]==typ]
        for role,events in [('training',(0,1)),('validation',(2,3))]:
            at_least=both=none=0;ec=Counter()
            for z in controls:
                flags=[]
                for e in events:
                    a,b=map(int,z['pairs'][e]);n=int(z['accepted'][e][a])+int(z['accepted'][e][b])
                    ec[{0:'no_response',1:'single',2:'nonzero_balanced'}[n]]+=1;flags.append(n==2)
                    if n==2:
                        for p,q in zip(z['delta_p_kw'][e],z['delta_q_kvar'][e]):
                            assert abs(sum(p))<1e-12 and abs(sum(q))<1e-12
                at_least+=any(flags);both+=all(flags);none+=not any(flags)
            result.append(dict(kind=typ,role=role,deployments=len(controls),at_least_one_balanced=at_least,
                               two_balanced=both,no_balanced=none,event_counts=dict(ec)))
    confirm=S/'wzzt_b_confirm_v1/evaluations.json';cr=read(confirm);inputs[str(confirm)]=sha(confirm)
    trace=[]
    for policy in ('pair1','pair_rotation'):
        row=next(r for r in cr if r['flow']=='r0' and r['case']=='strong_02' and r['policy']==policy)
        rp=Path(row['execution_receipt']);z=read(rp)['controls'];inputs[str(rp)]=sha(rp)
        n=[int(z['accepted'][e][int(z['pairs'][e][0])])+int(z['accepted'][e][int(z['pairs'][e][1])]) for e in (0,1)]
        assert n==[1,0]
        trace.append(dict(policy=policy,fit_id=row['fit_id'],training_participant_counts=n,
                          strict_W_errors=row['evaluated']['current_pool_wzzt']['strict_raw_or_contract']['final_errors']))
    write('input_bindings.json',inputs)
    write('balance_aggregate.json',dict(status='passed_read_only',source_sha256=sha(__file__),
        original_maps=308,distinct_control_trajectories=44,models_per_control=7,counts=result,
        positive_example_trace=trace,new_LP_calls=0,new_AC_calls=0,new_future_generated=False,
        scope='two shared behavior streams across menus; realized counts are not independent trials or risk probabilities'))
    return dict(status='passed_read_only',trajectories=44,new_LP_calls=0)

if __name__=='__main__':print(json.dumps(main()))
