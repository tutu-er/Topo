"""Present the recorded zero-maintenance baseline as the four-stage main case.

Read-only with respect to experiments: no fitting, selection changes or timing
estimation. Every displayed metric comes from the verified 120-input records.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'docs/iterative_exit_matched120_20261101'
DEST=ROOT/'docs/mainline_case_results_20261101'
CASES=('paper15','soumalas11','flynn16','pengwah18')
FLOW='''flowchart LR
    A["① RNJ小块预选<br/>至多2个不相交的2～3终端块"]
    B["② 原始WZZT逐步扩展<br/>联合拟合＋及时释放零块"]
    C["③ ≤5直属块联合末端<br/>枚举块并集，耦合MILP重构"]
    D["④ 一次最终退出与选模<br/>收缩、重拟合、验证择优"]
    A --> B
    B -->|"最大直属块数≤5"| C
    C --> D
'''


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def table(frame):
    def cell(x):return f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','|'+'---|'*len(frame.columns),
        *['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]])


def aggregate(main,before):
    rows=[]
    for case in (*CASES,'all'):
        a=main if case=='all' else main[main['case']==case]
        b=before.loc[a.job]
        rows.append(dict(case=case,inputs=len(a),baseline_F1=float(b.clade_f1.mean()),F1=float(a.clade_f1.mean()),
            delta_F1=float(a.clade_f1.mean()-b.clade_f1.mean()),precision=float(a.precision.mean()),recall=float(a.recall.mean()),
            exact=int(a.exact.sum()),baseline_exact=int(b.exact.sum()),recovery_rate=float(a.exact.mean()),
            R_relative=float(a.R_relative.mean()),X_relative=float(a.X_relative.mean())))
    return pd.DataFrame(rows)


def figures(cases,conditions):
    font=font_manager.findfont(font_manager.FontProperties(family='Microsoft YaHei'),fallback_to_default=False)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,
        'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
    grey='#aeb8c4';blue='#2475ad';x=np.arange(5);w=.35
    fig,axes=plt.subplots(1,2,figsize=(12.8,4.9),layout='constrained')
    for ax,old,new,title in [(axes[0],cases.baseline_F1,cases.F1,'平均拓扑F1'),
        (axes[1],cases.baseline_exact/cases.inputs,cases.recovery_rate,'完整恢复率')]:
        ax.bar(x-w/2,old,w,color=grey,label='修改前同预算主线')
        ax.bar(x+w/2,new,w,color=blue,label='当前主算例：及时清零＋一次退出')
        ax.set_xticks(x,['paper15','soumalas11','flynn16','pengwah18','全部120'])
        ax.set_ylim(0,1.105);ax.set_yticks(np.arange(0,1.01,.2));ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.set_title(title,pad=12)
    for i,row in enumerate(cases.itertuples(index=False)):
        axes[0].text(i,float(max(row.baseline_F1,row.F1))+.025,f'{row.F1:.4f}',ha='center',color=blue,fontsize=10)
        axes[0].text(i,.035,f'Δ {row.delta_F1:+.4f}',ha='center',fontsize=9,color='#444444')
        axes[1].text(i,row.recovery_rate+.025,f'{row.exact}/{row.inputs}',ha='center',fontsize=10,color=blue)
    axes[1].set_yticklabels([f'{v:.0%}' for v in np.arange(0,1.01,.2)])
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False,fontsize=10)
    fig.suptitle('四核心主算例：F1小幅改善，完整恢复数保持不变',fontsize=15)
    fig.savefig(DEST/'mainline_results.png',dpi=180);plt.close(fig)

    ns=(24,48,96);sigmas=(.0002,.0005,.001)
    f1=np.empty((3,3));exact=np.empty((3,3),int);delta=np.empty((3,3))
    for i,sigma in enumerate(sigmas):
        for j,N in enumerate(ns):
            r=conditions[(conditions.voltage_sigma==sigma)&(conditions.N==N)].iloc[0]
            assert r.inputs==12
            f1[i,j]=r.F1;exact[i,j]=r.exact;delta[i,j]=r.F1-r.baseline_F1
    fig,axes=plt.subplots(1,3,figsize=(13.1,4.65),layout='constrained')
    data=[(f1,'平均拓扑F1','Blues',0,1),(exact/12,'完整恢复率','Blues',0,1),
          (delta,'相对修改前的F1变化','RdBu',-max(abs(delta).max(),.001),max(abs(delta).max(),.001))]
    for k,(ax,(values,title,cmap,vmin,vmax)) in enumerate(zip(axes,data)):
        im=ax.imshow(values,cmap=cmap,vmin=vmin,vmax=vmax,aspect='auto')
        ax.set_xticks(range(3),ns);ax.set_yticks(range(3),['0.02%','0.05%','0.10%'])
        ax.set_xlabel('训练总样本数 N（三天合计）');ax.set_title(title,pad=12)
        if k==0:ax.set_ylabel('终端与根表电压相对噪声')
        for i in range(3):
            for j in range(3):
                label=f'{values[i,j]:.3f}' if k==0 else f'{exact[i,j]}/12' if k==1 else f'{values[i,j]:+.4f}'
                color='white' if (k<2 and values[i,j]>.58) or (k==2 and abs(values[i,j])>.55*vmax) else '#202020'
                ax.text(j,i,label,ha='center',va='center',fontsize=12,color=color)
        fig.colorbar(im,ax=ax,shrink=.82,pad=.025)
    fig.suptitle('主测试108输入：每格12个；P/Q测量噪声固定0.5%',fontsize=14)
    fig.savefig(DEST/'noise_sample_results.png',dpi=180);plt.close(fig)
    return font


def main():
    DEST.mkdir(parents=True,exist_ok=True);receipt=read(SOURCE/'verification.json');manifest=read(SOURCE/'manifest.json')['files']
    assert receipt['passed'] and receipt['returned']==240
    for name in ('per_model.csv','verification.json','domain_diagnostics.json','protocol.json'):
        assert sha(SOURCE/name)==manifest[name]['sha256']
    frame=pd.read_csv(SOURCE/'per_model.csv');chosen=frame[(frame.method=='after')&(frame.view=='zero_only')].copy()
    before=frame[(frame.method=='before')&(frame.view=='final')].copy().set_index('job')
    assert len(chosen)==120 and chosen.job.is_unique and set(chosen.job)==set(before.index)
    assert chosen.exact.sum()==66 and abs(chosen.clade_f1.mean()-.8287061087061088)<1e-12
    assert len(chosen[chosen.panel=='main'])==108 and len(chosen[chosen.panel=='stress'])==12
    columns=['job','case','panel','bundle','N','voltage_sigma','condition','selected_source','clade_f1','precision','recall',
        'exact','train_mae','validation_mae','test_mae','R_relative','X_relative','internal']
    chosen[columns].to_csv(DEST/'per_input_quality.csv',index=False)
    cases=aggregate(chosen,before);assert cases[cases['case']!='all'].inputs.eq(30).all()
    cases.to_csv(DEST/'case_summary.csv',index=False)
    rows=[]
    for (sigma,N),g in chosen[chosen.panel=='main'].groupby(['voltage_sigma','N']):
        rows.append(dict(voltage_sigma=sigma,N=int(N),inputs=len(g),F1=float(g.clade_f1.mean()),exact=int(g.exact.sum()),
            recovery_rate=float(g.exact.mean()),baseline_F1=float(before.loc[g.job].clade_f1.mean())))
    conditions=pd.DataFrame(rows);conditions.to_csv(DEST/'noise_sample_summary.csv',index=False)
    rows=[]
    for condition,g in chosen[chosen.panel=='stress'].groupby('condition'):
        rows.append(dict(condition=condition,inputs=len(g),F1=float(g.clade_f1.mean()),exact=int(g.exact.sum()),recovery_rate=float(g.exact.mean())))
    stress=pd.DataFrame(rows);stress.to_csv(DEST/'stress_summary.csv',index=False)
    font=figures(cases,conditions)
    (DEST/'four_core_flow.mmd').write_text(FLOW,encoding='utf-8',newline='\n')
    profile=dict(main_case='少量RNJ小块＋原始WZZT及时清零＋五块联合末端＋一次退出及验证选模',
        core_modules=['RNJ小块预选','原始WZZT扩展与及时清零','五块联合末端重构','一次最终退出与输出选择'],
        source_method='after',source_view='zero_only',includes_public_RNJ_fallback=True,includes_singleton_fallback=True,
        extra_stall_recovery=False,mask_branch=False,quality_from_executed_baseline_branch=True,
        standalone_end_to_end_timing_measured=False,runtime_defaults_modified=False,
        parameters=dict(max_RNJ_seeds=2,seed_size=[2,3],strategy='original',zero_tolerance=1e-9,zero_loss_tolerance=1e-10,
            terminal_block_limit=5,exit_solves=1,lambda_coefficient=.003,resolution=.24,response_bounds=[0,2],positive_impedance_floor=None),
        source_files={name:dict(path=str(SOURCE/name),sha256=sha(SOURCE/name)) for name in ('per_model.csv','verification.json','domain_diagnostics.json','protocol.json')},
        original_experiment_limits=read(SOURCE/'protocol.json')['limits'],inputs=120,F1=float(chosen.clade_f1.mean()),exact=66,
        own_path_without_RNJ=dict(F1=.8217551430051431,exact=60),producer_sha256=sha(Path(__file__)),font=font)
    write(DEST/'main_case_profile.json',profile)
    main_part=chosen[chosen.panel=='main'];check=dict(passed=True,inputs=120,source_manifest_verified=True,
        representation='already executed zero_only baseline quality; no new optimization or timing measurement',
        F1=float(chosen.clade_f1.mean()),exact=int(chosen.exact.sum()),main_F1=float(main_part.clade_f1.mean()),main_exact=int(main_part.exact.sum()),
        main_cells=9,inputs_per_cell=12,stress_inputs=12,
        no_standalone_timing_exported=not any('seconds' in c for c in chosen[columns].columns))
    assert check['no_standalone_timing_exported'] and read(SOURCE/'domain_diagnostics.json')['both_base_subproblems_gap_met']==120
    write(DEST/'verification.json',check)
    display=cases[['case','inputs','baseline_F1','F1','precision','recall','exact','recovery_rate']].copy()
    text='# 主算例：四核心RNJ–WZZT简化主线\n\n'
    text+='本次将用户指定简化主线作为主算例展示：RNJ少量小块预选→原始WZZT逐步扩展且及时清零→≤5直属块联合末端→一次最终退出、收缩、LP重拟合和验证选择。额外停滞删块恢复及屏蔽分支不纳入本主算例。原方案和完整恢复版继续保留。\n\n'
    text+='## 四核心流程\n\n```mermaid\n'+FLOW+'```\n\n'
    text+='≤5指直属子树块数。前向停滞可直接进入④；末端清零改变结构后重新判断②/③。清零贯穿初始化、扩展、末端和退出。④包括0.24响应收缩、联合LP和严格验证MAE选择，并保留完整RNJ备用库及单点回退；一次最终退出继续允许正权重临时块撤销。\n\n'
    text+='## 主结果\n\n'+table(display)+'\n\n'
    text+='每例30输入=27主测试+3压力测试。F1/完整恢复按正权重内部集合计算，排除单点和全集。完整恢复66/120=55%；精确率0.834352，召回率0.826984。与修改前同预算基线相比，F1增加0.003472，1胜117平2负，没有新增或丢失完整恢复。单看自身路径、去掉RNJ库保护时为F1=0.821755、完整恢复60/120。\n\n'
    text+='![四例结果](mainline_results.png)\n\n## 噪声与训练样本数\n\n'+table(conditions)+'\n\n'
    text+='每格12输入（4网络×3组随机实现）；N为三天合计训练点数，验证/测试各288点。低噪声0.02%下N48和N96均12/12完整恢复；中噪声0.05%下N96也为12/12。高噪声0.10%下仍有明显误差。\n\n'
    text+='![噪声和样本数](noise_sample_results.png)\n\n## 压力条件\n\n'+table(stress)+'\n\n'
    text+='弱激励F1为0.142857、0/4完整恢复，是本轮明确未解决的条件；根表噪声和异常值各为F1=0.958333、3/4。条件比较是该固定生成器下的样本结果，没有对这些评估输入重新调参。\n\n'
    text+='## 已验证范围与耗时\n\n质量来自前次完整对照中真实执行的after/zero_only baseline分支，共120输入；双侧baseline子问题全部达到原设精度。它不是本次新运行的独立关闭恢复版本。独立端到端耗时尚未测量；原summary中16.15小时等时间列属于整套含恢复分支版本，不用于本主算例。当前只建立展示配置，没有修改fit_mainline的运行默认。\n\n'
    text+='本次未运行仿真、LP或MILP，未更改训练/选模结果。输入CSV、协议、检查回执和展示脚本SHA256记录在main_case_profile.json。运行`python -B -m research_experiments.iterative_benchmark.present_mainline`可从原始结果重新生成此展示。\n'
    (DEST/'MAIN_RESULTS.md').write_text(text,encoding='utf-8',newline='\n')
    write(DEST/'manifest.json',dict(files={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(DEST.iterdir()) if f.is_file() and f.name!='manifest.json'}))
    print(cases[['case','inputs','F1','exact','recovery_rate']].to_string(index=False));print('Presentation verified; no optimization or timing run')


if __name__=='__main__':main()
