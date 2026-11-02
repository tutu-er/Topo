"""Public aggregates and LaTeX results, never publish raw P/Q/V, RX or prices."""
import argparse,sys,shutil
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import refine as r
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def export(root,output,document):
    root=Path(root);output=Path(output);document=Path(document);output.mkdir(parents=True,exist_ok=True)
    p=r.read(root/'protocol.json');s=r.read(root/'summary.json');q=r.read(root/'planning_summary.json');a=r.read(root/'audit.json')
    probs=r.read(root/'policy_probability_summary.json')['rows'];rows=r.read(root/'scored_private.json')['rows']
    if not a['passed']:raise ValueError('audit failed')
    r.write(output/'summary.json',s);r.write(output/'planning_summary.json',q);r.write(output/'audit.json',a)
    r.write(output/'independent_probability.json',dict(rows=probs))
    public={k:p[k] for k in ['id','created_utc','planning_samples','planning_seed','response_seed','qualification_samples','qualification_seed','qualification_response_seed','solver_limits','paths','primary_policies','expected_records','laws','scientific_changes','stop_rule','environment']}
    public['cases']=[{k:c[k] for k in ['name','case','labels','base_kw','history_rows','voltage_sigma','pq_sigma']} for c in p['cases']]
    public['source_sha256']=p['source_sha256'];public['private_protocol_sha256']=r.sha(root/'protocol.json')
    public['price_freeze_sha256']=r.sha(root/'price_freeze.json');r.write(output/'protocol_aggregate.json',public)
    order=['zero','rotation','old_diagnostic','candidate'];names=['Natural','Rotation','Previous','Refined'];cn=['无奖励','同费轮换','旧诊断价','新诊断价']
    fig,axes=plt.subplots(2,3,figsize=(11.8,6.8),sharex=True)
    for col,c in enumerate(p['cases']):
        h=next(x['threshold_micro'] for x in q['rows'] if x['condition']==c['name'])
        for rid,law in enumerate(p['laws']):
            ax=axes[rid,col];data=[[v['D_micro'] for v in rows if v['condition']==c['name'] and v['law']==law and v['policy']==key and v['status']=='complete'] for key in order]
            ax.boxplot(data,tick_labels=names,showmeans=True,meanprops=dict(marker='o',markerfacecolor='white',markeredgecolor='black',markersize=4))
            ax.axhline(h,color='tab:red',ls='--',lw=1.2,label='3% exit threshold')
            ax.set_title(c['case']+' / '+law);ax.tick_params(axis='x',rotation=20);ax.grid(axis='y',alpha=.2)
            if col==0:ax.set_ylabel(r'$D$ ($10^{-6}$ p.u.$^2$)')
    axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(output/'loss_thresholds.pdf');fig.savefig(output/'loss_thresholds.png',dpi=150);plt.close(fig)
    figures=document/'figures';figures.mkdir(exist_ok=True)
    for name in ['loss_thresholds.pdf','loss_thresholds.png']:shutil.copy2(output/name,figures/name)
    lines=[r'\subsection{完成范围与计算核验}',
        f"本轮{s['records']}条预锁记录中完成{s['complete']}条，失败/未决{s['failed_unresolved']}条；独立背景{s['independent_backgrounds']}个，每例24个。两种响应律配对，四策略共享路径。不同观察{a['distinct_observations']}份，重复零计划不增加独立分母。",
        f"AC收敛{s['AC_converged_points']}个时点；设备约束违反{s['resource_violations']}条、电压越限{s['voltage_violations']}条。新拟合与新WZZT搜索均为零。5项新增测试（含24组独立积分/梯度比较与已知可行的硬概率降费分支）和原12项测试通过；独立重放{a['checks']}项通过。",
        r'\subsection{规划概率、独立核验与费用}',
        r'\begin{table}[ht]\centering\small\caption{同一4096点核验旧、新报价；费用为场景元/日}',
        r'\begin{tabular}{lrrrrr}\toprule 算例&规划概率&旧价核验&新价核验&旧费&新费\\\midrule']
    for c in p['cases']:
        v=next(v for v in q['rows'] if v['condition']==c['name']);old=next(v for v in probs if v['condition']==c['name'] and v['policy']=='old_diagnostic')
        lines.append(f"{c['case']}&{100*v['planning_probability']:.2f}\\%&{100*old['probability']:.2f}\\%&{100*v['independent_probability']['mean']:.2f}\\%&{v['old_cost']:.2f}&{v['candidate_cost']:.2f}\\\\")
    lines.extend([r'\bottomrule\end{tabular}\end{table}'])
    target=sum(v['planning_target_met'] for v in q['rows']);disputes=sum(v['has_dispute'] for v in q['rows'])
    lines.append(f"{disputes}个有历史争议条件中，本轮找到95\\%规划可行报价{target}/{disputes}。pengwah是旧批事后已正确的零收益对照，但历史仍有争议，不能按真值跳过规划。未达标报价仅供诊断，不能据其称最小费用问题已解。独立核验不回流选价。")
    for c in p['cases']:
        v=next(v for v in q['rows'] if v['condition']==c['name']);prob=v['independent_probability']
        lines.append(f"{c['case']}独立核验标准误为{100*prob['standard_error']:.3f}个百分点；工作分布下的Hoeffding单侧95\\%下界为{100*prob['hoeffding_one_sided_95_lower']:.2f}\\%。求解状态："+r'\texttt{'+', '.join(v['statuses']).replace('_',r'\_').replace(':',': ')+r'}。')
    lines.extend([r'\subsection{实际AC后的完整恢复与错漏}',r'表中恢复的分母均为24，错漏为24日完整错漏总和。旧价与新价费用不同；同费比较只在新价与轮换之间成立。',
        r'\begin{table}[ht]\centering\small\caption{主响应及失配响应；所有失败保留}',
        r'\begin{tabular}{llrrrr}\toprule 算例&策略&主恢复&主错漏&失配恢复&失配错漏\\\midrule'])
    for c in p['cases']:
        for key,label in zip(order,cn):
            main=next(t for t in s['tables'] if t['condition']==c['name'] and t['law']=='main' and t['policy']==key)
            stress=next(t for t in s['tables'] if t['condition']==c['name'] and t['law']=='stress' and t['policy']==key)
            lines.append(f"{c['case']}&{label}&{main['recoveries']}/24&{main['errors']}&{stress['recoveries']}/24&{stress['errors']}\\\\")
        lines.append(r'\midrule')
    lines.extend([r'\bottomrule\end{tabular}\end{table}',r'\begin{figure}[p]\centering\includegraphics[width=\textwidth]{figures/loss_thresholds.pdf}\caption{冻结报价在独立AC路径上的实际损失差与原3\%排除阈值。箱线为24日结果；圆点表示均值，红线为历史阈值。显示指标是否足以触发选模，不能单独代表真实恢复。}\end{figure}'])
    for c in p['cases']:
        vv={t['policy']:t for t in s['tables'] if t['condition']==c['name'] and t['law']=='main'};v=next(v for v in q['rows'] if v['condition']==c['name'])
        lines.append(f"{c['case']}主响应新价/旧价/轮换的实际绝对调节分别为{vv['candidate']['mean_abs_kwh']:.2f}/{vv['old_diagnostic']['mean_abs_kwh']:.2f}/{vv['rotation']['mean_abs_kwh']:.2f} kWh/日；新价平均净调节{vv['candidate']['mean_net_kwh']:.2f} kWh/日。新价$D$均值{vv['candidate']['mean_D']:.3f}、最小{vv['candidate']['min_D']:.3f}，阈值{v['threshold_micro']:.3f}（微标幺平方）。")
    lines.append(r'结果需按代价与可行性共同判断：若新价未达95\%、费用增加而实际恢复相同，就不将其作为已验证优于旧价的方法。保留旧价作为对照，下一步应针对参考预测与局部搜索瓶颈收口分析，不能用提高资源或降低选择阈值补出成功。')
    (document/'results.tex').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    md=['# 连续异质DR概率可行化：独立实测报告','',f"预锁{s['records']}记录，完成{s['complete']}，失败/未决{s['failed_unresolved']}；{s['independent_backgrounds']}独立背景，2配对响应律。{a['checks']}独立检查通过。",'',
        '|算例|规划p|独立p（4096）|费用旧→新|95%规划可行|','|---|---:|---:|---:|---|']
    for v in q['rows']:md.append(f"|{v['condition']}|{v['planning_probability']:.6f}|{v['independent_probability']['mean']:.6f}|{v['old_cost']:.4f}→{v['candidate_cost']:.4f}|{v['planning_target_met']}|")
    md+=['','|算例/响应|策略|恢复/24|总错漏|费用|实际绝对kWh/日|','|---|---|---:|---:|---:|---:|']
    for t in s['tables']:md.append(f"|{t['condition']}/{t['law']}|{t['policy']}|{t['recoveries']}/24|{t['errors']}|{t['fee']:.4f}|{t['mean_abs_kwh']:.4f}|")
    md+=['','完整历史288/288、两完整历史族/RX、w2/common3%/.16保持；0参数重拟合、0新W搜索。效果属于冻结联合候选的验证回选。未达95方案只作诊断，不称可行最优或证明无解。',
        '','费用没有货币上限，物理库存保持；实际包含原AC线损、P/Q/V表计噪声、设备功率/服务能量/SOC因果控制。主/失配共享背景，不能将策略与响应律重复计为独立日。',
        '','工作预测仍3训练日秩至多2；行为和费用为假设。独立工作概率不等于AC成功率、拓扑真实性概率或统计覆盖。',
        '',f"AC收敛{s['AC_converged_points']}时点；设备/电压违反{s['resource_violations']}/{s['voltage_violations']}，热限未核。",'','17项不同单元测试通过；新增条件积分与数值积分、系数/阈值及设备链梯度核验，旧12项保留。']
    (output/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('PUBLIC',output,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',required=True,type=Path);a.add_argument('--run',required=True,type=Path);a.add_argument('--document',required=True,type=Path);args=a.parse_args();export(args.run,args.output,args.document)
