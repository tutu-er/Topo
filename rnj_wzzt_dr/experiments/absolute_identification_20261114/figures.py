"""Compact scientific figures from completed aggregate reports only."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from ... import io


CHINESE=dict(no_dfs='无服务',old_gamma2='旧增量2',nominal='单目标名义恢复',calibrated='单目标均值修正',
    symmetric='多目标绝对恢复',symmetric_guard='对称恢复\n＋方向保护',symmetric_full='全候选绝对恢复',
    symmetric_full_guard='全候选\n＋方向保护',multiplicative_absolute='乘性校正绝对恢复',
    capped_absolute='限幅校正绝对恢复',weighted_nominal='历史加权\n绝对恢复',weighted_capped='历史加权\n＋偏差限幅')
ENGLISH=dict(no_dfs='No service',old_gamma2='Old increment2',nominal='Single-target nominal',
    calibrated='Single-target mean adjusted',symmetric='Symmetric recovery',symmetric_guard='Symmetric recovery\n+ directional guards',
    weighted_nominal='History-weighted\nabsolute recovery',weighted_capped='History-weighted\n+ capped mean correction')
ROLES=('correct_clear','correct_ambiguous','wrong_present','missing')
ROLE_ZH=dict(correct_clear='历史选对、优势明确',correct_ambiguous='历史选对、优势不明确',
    wrong_present='历史选错、库中含真拓扑',missing='库中缺完整真拓扑')
ROLE_EN=dict(correct_clear='Correct historical winner, clear',correct_ambiguous='Correct historical winner, ambiguous',
    wrong_present='Wrong historical winner, truth present',missing='Complete truth absent from bank')
COLORS=('#747b84','#3d7ca6','#d28c45','#4d9d88','#956fa7','#d46f75')


def font_setup():
    candidates=[Path('C:/Windows/Fonts/msyh.ttc'),Path('C:/Windows/Fonts/simhei.ttf')]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            name=font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams.update({'font.family':name,'axes.unicode_minus':False,'font.size':10,
                'svg.fonttype':'path','figure.dpi':120})
            return True,name
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'path'})
    return False,'DejaVu Sans'


def figure_groups(report,physics,partition):
    if partition=='overall':
        return report['overall'][physics],report['by_role'][physics]
    overall=report['by_partition'][physics][partition]
    role_groups={role:report['by_role_partition'][physics][role+'/'+partition]
        for role in ROLES if role+'/'+partition in report['by_role_partition'][physics]}
    return overall,role_groups


def create(report_path,out,*,physics='AC',partition='overall',strategies=None):
    report=io.read(report_path)
    if not report.get('verification',{}).get('passed',False):
        raise ValueError('A source/plan-seal-verified completed aggregate report is required')
    group,roles=figure_groups(report,physics,partition)
    methods=list(report['solver']) if strategies is None else list(strategies)
    if not methods or any(m not in group['strategies'] or m.endswith('_rotation') or m=='no_dfs' for m in methods):
        raise ValueError('Choose existing optimized methods without rotation suffixes')
    cn,font=font_setup()
    names=CHINESE if cn else ENGLISH
    label=lambda s:names.get(s,s.replace('_',' '))
    tr=lambda zh,en:zh if cn else en
    out.mkdir(parents=True,exist_ok=True)
    prefix=physics.lower()+'_'+partition
    files=[]
    n=group['planned_days']
    scope=tr(f"{group['histories']}个历史输入，{n}条新日路径；AC与线性消融共享路径，样本数不翻倍。",
        f"{group['histories']} histories, {n} future-day paths; AC and linear ablation share paths, not extra samples.")
    if group['strategies']['no_dfs']['failed_or_missing_rows']:
        scope+=' '+tr('失败或缺失计作未恢复。','Failed or missing paths count as nonrecovery.')
    physics_title=tr('原AC','Original AC') if physics=='AC' else tr('真实支路线性消融','Physical-branch linear ablation')

    def save(fig,name):
        for ext in ('png','svg'):
            path=out/(prefix+'_'+name+'.'+ext)
            fig.savefig(path,dpi=180,bbox_inches='tight')
            files.append(io.ref(path))
        plt.close(fig)

    # Counts, not conditional-on-output accuracy. Every role retains its full denominator.
    chosen_roles=[r for r in ROLES if r in roles]
    cols=2;rows=int(np.ceil(len(chosen_roles)/cols))
    fig,axes=plt.subplots(rows,cols,figsize=(11,3.25*rows),squeeze=False)
    all_methods=['no_dfs',*methods]
    for ax,role in zip(axes.flat,chosen_roles):
        own=roles[role];denom=own['planned_days'];counts=[own['strategies'][m]['native_correct'] for m in all_methods]
        bars=ax.bar(np.arange(len(all_methods)),counts,color=[COLORS[i%len(COLORS)] for i in range(len(all_methods))])
        ax.set_xticks(np.arange(len(all_methods)),[label(m) for m in all_methods],fontsize=9)
        ax.set_ylim(0,max(1,denom)*1.16)
        ax.set_ylabel(tr('完整恢复次数','Complete recoveries'))
        ax.set_title((ROLE_ZH if cn else ROLE_EN)[role],fontsize=11)
        for bar,count in zip(bars,counts):
            ax.text(bar.get_x()+bar.get_width()/2,count+max(1,denom)*.025,f'{count}/{denom}',ha='center',va='bottom',fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True,nbins=5))
    for ax in list(axes.flat)[len(chosen_roles):]:ax.set_axis_off()
    fig.suptitle(tr('完整原参数库的拓扑恢复：','Native-bank topology recovery: ')+physics_title,fontsize=14,y=.99)
    fig.text(.5,.015,scope,ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.955))
    save(fig,'recovery_by_role')

    # Paired rescue and loss counts; improvements never hide new failures.
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for ax,reference in zip(axes,('no_dfs','rotation')):
        comparisons=[group['comparisons'].get(m+'_vs_'+reference) for m in methods]
        if any(c is None for c in comparisons):
            ax.text(.5,.5,tr('该对照未记录','Comparison unavailable'),ha='center',transform=ax.transAxes);ax.set_axis_off();continue
        gain=np.array([c['new_recoveries'] for c in comparisons]);loss=np.array([c['lost_recoveries'] for c in comparisons]);x=np.arange(len(methods))
        ax.bar(x,gain,color='#4d9d88',label=tr('新增恢复','Recovered only with strategy'))
        ax.bar(x,-loss,color='#d8796d',label=tr('丢失恢复','Recovery lost with strategy'))
        ax.axhline(0,color='#555555',lw=.8)
        limit=max(1,int(gain.max(initial=0)),int(loss.max(initial=0)))
        ax.set_ylim(-limit*1.35,limit*1.35)
        for i,(g,l) in enumerate(zip(gain,loss)):
            ax.text(i,g+limit*.055,str(g),ha='center',va='bottom',fontsize=9)
            ax.text(i,-l-limit*.055,str(l),ha='center',va='top',fontsize=9)
        ax.set_xticks(x,[label(m) for m in methods],fontsize=9)
        ax.set_ylabel(tr('配对路径数（上：新增；下：丢失）','Paired paths (+ recovered; - lost)'))
        ax.set_title(tr('相对无服务','Versus no service') if reference=='no_dfs' else tr('相对同计划费用轮换','Versus same-planned-fee rotation'))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True,nbins=5))
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(loc='upper right',fontsize=8)
    fig.suptitle(tr('新增恢复与丢失恢复同时统计：','Paired recovery gains and losses: ')+physics_title,fontsize=14)
    fig.text(.5,.015,scope,ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.94))
    save(fig,'paired_changes')

    # Fee means keep zero-action/fallback days; solver status is displayed separately.
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    ax=axes[0];fees=[group['strategies'][m]['mean_settled_fee'] for m in methods]
    x=np.arange(len(methods));bars=ax.bar(x,[np.nan if f is None else f for f in fees],color=[COLORS[(i+1)%len(COLORS)] for i in range(len(methods))])
    maxfee=max([f for f in fees if f is not None]+[0.]);ax.set_ylim(0,max(.01,maxfee)*1.4)
    for i,(bar,fee) in enumerate(zip(bars,fees)):
        own=group['strategies'][methods[i]]
        text=tr('无记录','Missing') if fee is None else f'{fee:.3f}\n'+tr(f"非零{own['nonzero_cost_days']}/{own['planned_days']}",f"nonzero {own['nonzero_cost_days']}/{own['planned_days']}")
        ax.text(i,(fee or 0)+max(.01,maxfee)*.035,text,ha='center',va='bottom',fontsize=9)
    ax.set_xticks(x,[label(m) for m in methods],fontsize=9)
    ax.set_ylabel(tr('平均实际激活补偿（元／记录日）','Mean settled activation fee (CNY/recorded day)'))
    ax.set_title(tr('包含零动作与报价失败回退','Includes zero actions and failed-quote fallbacks'))
    ax.spines[['top','right']].set_visible(False)
    ax=axes[1]
    if partition=='overall':
        solvers=[report['solver'][m] for m in methods]
        parts=[]
        for s in solvers:
            remainder=s['planned_quotes']-s['zero_sufficient']-s['nonzero_feasible']-s['infeasibility_proved']-s['unresolved_quotes']
            if remainder<0:raise ValueError('Overlapping solver outcome categories')
            parts.append([s['zero_sufficient'],s['nonzero_feasible'],s['infeasibility_proved'],s['unresolved_quotes'],remainder])
        bottom=np.zeros(len(methods));parts=np.asarray(parts)
        labels=[tr('零动作充分','Zero sufficient'),tr('非零可行','Nonzero feasible'),tr('QCQP不可行','QCQP infeasible'),tr('预算内未决','Unresolved'),tr('其他／异常／缺失','Other / error / missing')]
        for k,(color,name) in enumerate(zip(('#a4cbbf','#4d9d88','#c7cbd1','#e1ae62','#d8796d'),labels)):
            ax.bar(x,parts[:,k],bottom=bottom,color=color,label=name);bottom+=parts[:,k]
        ax.set_ylim(0,max(1,float(bottom.max()))*1.24)
        for i,s in enumerate(solvers):ax.text(i,bottom[i]+max(1,float(bottom.max()))*.02,f"{s['feasible_quotes']}/{s['planned_quotes']}",ha='center',fontsize=9)
        ax.set_xticks(x,[label(m) for m in methods],fontsize=9)
        ax.set_ylabel(tr('报价次数','Quotes'))
        ax.set_title(tr('求解状态（标注：可行／计划）','Solver status (labels: feasible/planned)'))
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,fontsize=8)
        ax.spines[['top','right']].set_visible(False)
    else:
        ax.set_axis_off();ax.text(.5,.5,tr('分组求解状态见完整报告','Grouped solver outcomes are in the full report'),ha='center',transform=ax.transAxes)
    fig.suptitle(tr('费用与求解状态','Activation costs and solver outcomes'),fontsize=14)
    fig.text(.5,.015,tr('费用为示例激活补偿；物理线性计算复用相同采购，费用与报价次数不另计。','Illustrative activation compensation; paired linear evaluation adds no procurement cost or extra quotes.'),ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.94))
    save(fig,'cost_and_solver')
    receipt=dict(source=io.ref(__file__),report=io.ref(report_path),physics=physics,partition=partition,
        strategies=methods,font=font,histories=group['histories'],future_day_paths=n,
        physics_paths_doubled=False,files=files)
    io.write(out/(prefix+'_FIGURES.json'),receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--physics',choices=['AC','linear'],default='AC')
    p.add_argument('--partition',default='overall')
    p.add_argument('--strategies',help='Optional comma-separated optimized method names')
    a=p.parse_args()
    r=create(a.report.resolve(),a.out.resolve(),physics=a.physics,partition=a.partition,
        strategies=None if not a.strategies else a.strategies.split(','))
    print('figures',len(r['files']),'paths',r['future_day_paths'],'font',r['font'])
