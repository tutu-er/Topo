"""Post-hoc resource/selection accounting; never changes deployment choices."""
from pathlib import Path
import pandas as pd
from .run import ROOT,read,write
from .analyze import solver_records,markdown
from research_experiments.local_rebuild.model import canonical


def diagnose(output):
    output=Path(output);assert read(output/'analysis/verification.json')['passed']
    frame=pd.read_csv(output/'analysis/per_model.csv');pairs=pd.read_csv(output/'analysis/paired_inputs.csv')
    p=read(output/'protocol.json');records=[]
    for job in p['jobs']:
        old=read(output/'jobs'/job/'before/result.json');new=read(output/'jobs'/job/'after/result.json')
        prefix=new['prefix'];last=prefix['attempts'][-1] if prefix['attempts'] else {}
        release=new['branches'].get('release',{});events=release.get('trajectory',{}).get('events',[])
        first=events[0] if events else {};n=len(read(output/'common'/job/'common.json')['labels'])
        parent=prefix['point'];penalty=.003*prefix['reference_MAE']
        parent_J=parent['objective']+penalty*sum(1<len(C)<n for C in parent['supports'])
        same=(last.get('kind')=='joint_tail' and first.get('kind')=='tail' and
            canonical(last.get('keep',()))==canonical(first.get('keep',())) and
            canonical(last.get('candidates',()))==canonical(first.get('candidates',())) and
            not release.get('forbidden'))
        closed=True
        for result in (old,new):
            prefix_calls=solver_records(dict(trajectory=result['prefix']))
            closed=closed and all(e['status']==2 for e in prefix_calls)
            closed=closed and result['branches']['baseline']['exit_event'].get('status')==2
        zeros=[e for e in new['branches']['baseline']['trajectory'].get('zero_events',[]) if e['accepted'] and e['stage']!='final_exit']
        score=first.get('score')
        def topology(choice):
            point=choice['point']
            return {tuple(C) for C,r,x in zip(point['supports'],point['r'],point['x']) if 1<len(C)<n and max(r,x)>1e-9}
        release_same=(topology(new['choices_matched']['baseline'])==topology(new['choices_matched']['release'])) if release else None
        records.append(dict(job=job,both_base_subproblems_gap_met=closed,baseline_zero_events=len(zeros),
            baseline_zero_sets=sum(len(e['removed']) for e in zeros),deletion_triggered=new['recovery_attempts'],
            same_unmasked_tail_domain=same,same_domain_both_gap_met=same and last.get('status')==2 and first.get('status')==2,
            same_domain_resolve_seconds=float(first.get('seconds',0.)) if same else 0.,
            first_release_parent_J_gain=parent_J-score if score is not None else None,
            exceeds_parent_gain_gate=bool(score is not None and parent_J-score>1e-10+.001*parent['objective']),
            prefix_stop=prefix.get('stop_reason'),parent_last_kind=last.get('kind'),release_first_kind=first.get('kind'),
            final_source=new['views']['final']['source'],release_matches_baseline_topology=release_same))
    detail=pd.DataFrame(records);rows=[]
    zero=pairs[pairs.comparison=='zero_path_comparison'].merge(detail,on='job')
    for label,g in [('all',zero),('both_base_subproblems_gap_met',zero[zero.both_base_subproblems_gap_met]),
        ('zero_affected_and_base_gap_met',zero[(zero.baseline_zero_events>0)&zero.both_base_subproblems_gap_met]),
        ('no_baseline_zero_and_base_gap_met',zero[(zero.baseline_zero_events==0)&zero.both_base_subproblems_gap_met])]:
        if not len(g):continue
        rows.append(dict(subset=label,inputs=len(g),delta_F1=float(g.delta_F1.mean()),wins=int((g.delta_F1>1e-10).sum()),
            ties=int((abs(g.delta_F1)<=1e-10).sum()),losses=int((g.delta_F1< -1e-10).sum()),
            new_exact=int(g.new_exact.sum()),lost_exact=int(g.lost_exact.sum())))
    summary=pd.DataFrame(rows);delivery=ROOT/'docs'/output.name
    detail.to_csv(delivery/'domain_diagnostics.csv',index=False);summary.to_csv(delivery/'zero_closed_comparison.csv',index=False)
    after=frame[frame.method=='after'];final=after[after.view=='final'].set_index('job');proposals=[]
    for branch in ('release','masked'):
        candidate=after[after.view=='branch_'+branch].set_index('job')
        for view in ('zero_only','branch_baseline'):
            baseline=after[after.view==view].set_index('job')
            g=candidate.join(baseline[['clade_f1','exact','validation_mae']],rsuffix='_base');delta=g.clade_f1-g.clade_f1_base
            proposals.append(dict(branch=branch,reference=view,inputs=len(g),mean_delta_F1=float(delta.mean()),
                wins=int((delta>1e-10).sum()),ties=int((abs(delta)<=1e-10).sum()),losses=int((delta< -1e-10).sum()),
                new_exact=int((g.exact&~g.exact_base).sum()),lost_exact=int((~g.exact&g.exact_base).sum()),
                higher_F1_not_chosen=int(((delta>1e-10)&(final.loc[g.index].selected_source!=branch)).sum())))
    pd.DataFrame(proposals).to_csv(delivery/'proposal_diagnostics.csv',index=False)
    same=detail[detail.same_unmasked_tail_domain]
    record=dict(inputs=120,both_base_subproblems_gap_met=int(detail.both_base_subproblems_gap_met.sum()),
        recovery_inputs=int(detail.deletion_triggered.sum()),same_unmasked_tail_domain_inputs=len(same),
        same_domain_both_gap_met=int(same.same_domain_both_gap_met.sum()),
        same_domain_resolve_seconds=float(same.same_domain_resolve_seconds.sum()),
        same_domain_gain_exceeds_parent_gate=int(same.exceeds_parent_gain_gate.sum()),
        release_selected_topology_unchanged=int(detail.release_matches_baseline_topology.eq(True).sum()))
    write(delivery/'domain_diagnostics.json',record)
    text='# 零块与恢复邻域的补充核对\n\n'+markdown(summary)+'\n\n'
    text+='双侧base子问题均达到所设精度时，before→新版baseline的差异不由退出阶段的墙钟截断解释。该分组仍是给定算法路径比较；全树最优、物理可识别性和因果噪声分解均未由这些状态证明。\n\n'
    text+=f"停滞恢复触发{record['recovery_inputs']}个输入，其中{len(same)}个输入的首个非屏蔽恢复尝试直接进入末端，且与原末端的保留集合和候选集合完全相同；{record['same_domain_both_gap_met']}项两次均达到精度。所查相同邻域重算累计{record['same_domain_resolve_seconds']:.3f}秒，其中{record['same_domain_gain_exceeds_parent_gate']}项相对原父状态的J改善超过原接收门槛。未统计先扩展后进入相同末端的其他重算，因此这不是所有重复求解的总数。\n\n"
    text+='这个标签只针对恢复首个末端子问题，不能将其等同于整条分支无效：不同的近似最优结构、后续退出候选和验证择优仍可能改变最终F1。实际收益以paired/mechanism_benefits表为准；本核对不改变任何训练或选模结果。\n'
    (delivery/'DIAGNOSTICS.md').write_text(text,encoding='utf-8')
    print(record)


if __name__=='__main__':
    import sys
    diagnose(Path(sys.argv[1]).resolve())
