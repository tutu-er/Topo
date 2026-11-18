"""Plot sealed P/Q profiles; no AC, topology fitting, or resource dispatch.

The figures use the current eleven-day worlds before the project's additional
meter error. Source mapping and all scaled values are independently checked.
Detailed curves and figures remain in ignored private outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

from ... import io


ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = (0, 3, 8)  # The same source feeders 1/4/9 in both worlds.
EXAMPLE_COLORS = ('#2563eb', '#db6b1f', '#008674')
PHASES = ((0, 768, '#e4eff9', '拟合：8天 / 768点'),
          (768, 960, '#fff0ca', '验证：2天 / 192点'),
          (960, 1056, '#dff3e6', '评价：第11天 / 96点'))
NAMES = {'soumalas11': 'Soumalas（11个末端节点）', 'pengwah18': 'Pengwah（18个末端节点）'}


def checked(reference):
    path = Path(reference['path'])
    if io.ref(path)['sha256'] != reference['sha256']:
        raise ValueError('Changed sealed source: '+str(path))
    return path


def font():
    path = Path('C:/Windows/Fonts/msyh.ttc')
    if path.exists():
        font_manager.fontManager.addfont(str(path))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(path)).get_name()
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titleweight': 'normal', 'savefig.facecolor': 'white',
        'svg.fonttype': 'path', 'pdf.fonttype': 42})


def stats(power):
    return dict(minimum=float(power.min()), maximum=float(power.max()),
        mean=float(power.mean()), rms=float(np.sqrt(np.mean(power**2))))


def load_worlds(run, archive_receipt):
    receipt = io.read(archive_receipt)
    archive_path = checked(receipt['archive'])
    worlds_path = Path(run)/'worlds.json'
    worlds = io.read(worlds_path)
    refs = [io.ref(worlds_path), io.ref(archive_receipt), receipt['archive']]
    with np.load(archive_path, allow_pickle=False) as z:
        raw = {key:z[key].copy() for key in ('p_kw','q_kvar','timestamps_utc','feeder_ids','valid_minutes')}
    positions = {str(t):i for i,t in enumerate(raw['timestamps_utc'])}
    loaded = []
    for world in worlds:
        meta_ref = world['files']['profile_metadata.json']
        meta = io.read(checked(meta_ref)); ref = world['files']['clean.npz']
        with np.load(checked(ref), allow_pickle=False) as z:
            arrays = {key:z[key].copy() for key in ('P','Q','labels','timestamps')}
        refs += [meta_ref, ref]
        mapping = meta['node_mapping']; labels = list(map(int, arrays['labels']))
        if [r['label'] for r in mapping] != labels or not all(r['scale']>0 for r in mapping):
            raise ValueError('Terminal mapping changed')
        n = len(labels); base = float(world['base_kw'])
        if arrays['P'].shape != (1056,n) or arrays['Q'].shape != (1056,n):
            raise ValueError('Expected complete 11-day observations')
        ix = np.asarray([positions[str(t)] for t in arrays['timestamps']])
        fi = [list(map(int,raw['feeder_ids'])).index(int(r['feeder_id'])) for r in mapping]
        scale = np.array([r['scale'] for r in mapping])
        expected_p = raw['p_kw'][np.ix_(ix,fi)]*scale/base
        expected_q = raw['q_kvar'][np.ix_(ix,fi)]*scale/base
        if not np.array_equal(arrays['P'],expected_p) or not np.array_equal(arrays['Q'],expected_q):
            raise ValueError('Physical world differs from paired source times / frozen scaling')
        count = raw['valid_minutes'][np.ix_(ix,fi)]
        if np.any(count<12) or not np.isfinite(expected_p).all() or not np.isfinite(expected_q).all():
            raise ValueError('Incomplete source bins')
        times = pd.DatetimeIndex(pd.to_datetime(arrays['timestamps'],utc=True)).tz_convert('Europe/Berlin')
        if not np.all(times[1:]-times[:-1] == pd.Timedelta(minutes=15)):
            raise ValueError('Time axis is not contiguous 15 minutes')
        if world['train_indices']['8']!=list(range(768)) or world['validation_indices']!=list(range(768,960)) or world['new_day_indices']!=list(range(960,1056)):
            raise ValueError('Source train/validation/evaluation split changed')
        p, q = arrays['P']*base, arrays['Q']*base
        node_rows = []
        for j,label in enumerate(labels):
            pp, qq = p[:768,j],q[:768,j]
            k = float(pp@qq/(pp@pp))
            fraction = float(np.sum((qq-k*pp)**2)/np.sum(qq**2))
            node_rows.append(dict(label=label,source_feeder=int(mapping[j]['feeder_id']),
                scale=float(scale[j]),kappa=k,ratio_residual_energy_fraction=fraction,
                pearson_correlation=float(np.corrcoef(pp,qq)[0,1]),
                negative_P_points=int(np.sum(p[:,j]<0)),negative_Q_points=int(np.sum(q[:,j]<0))))
        loaded.append(dict(case=world['case'],labels=labels,p=p,q=q,times=times,
            meta=meta,mapping=mapping,node_stats=node_rows,base_kw=base,
            total_p=stats(p.sum(axis=1)),total_q=stats(q.sum(axis=1)),
            negative_P_cells=int(np.sum(p<0)),negative_Q_cells=int(np.sum(q<0)),
            source_to_world_bitwise_equal=True,
            node_Q_norm_over_P_norm=float(np.linalg.norm(q[:768])/np.linalg.norm(p[:768]))))
    return loaded, refs


def save(fig, out, stem):
    paths = []
    for extension in ('png','svg','pdf'):
        p = Path(out)/(stem+'.'+extension)
        fig.savefig(p,dpi=180)
        paths.append(io.ref(p))
    plt.close(fig)
    return paths


def overview(worlds, out):
    fig, axes = plt.subplots(2,2,figsize=(14.5,7.5),sharex=True)
    fig.subplots_adjust(left=.065,right=.985,bottom=.12,top=.80,hspace=.38,wspace=.20)
    fig.suptitle('当前P/Q曲线：两网末端净功率合计',fontsize=17,y=.975)
    fig.text(.5,.925,'同步实测馈线 → 同节点P/Q共用正缩放系数 → 15分钟基线；保留原始正负号',ha='center',fontsize=11)
    fig.legend(handles=[Patch(facecolor=c,label=label) for _,_,c,label in PHASES],
               loc='upper center',bbox_to_anchor=(.5,.895),ncol=3,frameon=False)
    x=np.arange(1056)/96.
    for row,w in enumerate(worlds):
        for col,(key,unit,color) in enumerate((('p','kW','#2563eb'),('q','kvar','#d56a24'))):
            ax=axes[row,col]
            for first,stop,c,_ in PHASES:ax.axvspan(first/96,stop/96,color=c,zorder=0)
            ax.plot(x,w[key].sum(axis=1),lw=1.05,color=color)
            ax.axhline(0,color='#606875',lw=.65,alpha=.65)
            for b in (8,10):ax.axvline(b,color='#6b7280',ls='--',lw=.8)
            ax.set_title(NAMES[w['case']]+' · '+('总有功P' if key=='p' else '总无功Q'),loc='left',fontsize=12)
            ax.set_ylabel(unit);ax.grid(axis='y',alpha=.22);ax.set_xlim(0,11)
            ax.set_xticks([0,2,4,6,8,10,11],['1/21\n08:00','1/23\n08:00','1/25\n08:00','1/27\n08:00','1/29\n08:00','1/31\n08:00','2/1\n08:00'])
    fig.text(.5,.025,'横轴为源数据德国当地时间（CET）；总量为末端P/Q相加，未含线损。图为未加本项目表计噪声、未调用额外灵活性的净负荷。',ha='center',fontsize=9.5,color='#49515d')
    return save(fig,out,'01_system_PQ_11days')


def daily(worlds, out):
    fig, axes=plt.subplots(2,2,figsize=(14.5,7.4),sharex=True)
    fig.subplots_adjust(left=.065,right=.985,bottom=.13,top=.86,hspace=.37,wspace=.20)
    fig.suptitle('第11天：节点有功、无功曲线',fontsize=17,y=.98)
    fig.text(.5,.93,'2024/1/31 08:00—2/1 07:45（CET）；彩色显示源馈线1、4、9，其余节点全部用灰线保留',ha='center',fontsize=11)
    x=np.arange(96)/4.
    for row,w in enumerate(worlds):
        for col,(key,unit) in enumerate((('p','kW'),('q','kvar'))):
            ax=axes[row,col]; data=w[key][960:]
            for j in range(len(w['labels'])):ax.plot(x,data[:,j],color='#b8bec8',alpha=.65,lw=.75,zorder=1)
            for j,color in zip(EXAMPLES,EXAMPLE_COLORS):
                ax.plot(x,data[:,j],color=color,lw=1.6,label='节点'+str(w['labels'][j]),zorder=3)
            ax.axhline(0,color='#343d49',lw=.8,ls='--',zorder=2)
            ax.set_title(NAMES[w['case']]+' · '+('有功P' if key=='p' else '无功Q'),loc='left',fontsize=12)
            ax.set_ylabel(unit);ax.grid(alpha=.18);ax.set_xlim(0,24)
            ax.set_xticks([0,4,8,12,16,20,24],['08:00','12:00','16:00','20:00','00:00','04:00','08:00'])
            ax.legend(loc='upper right',ncol=3,fontsize=8.3,frameon=True,framealpha=.85)
    fig.text(.5,.025,'P>0：消费；P<0：反送。Q>0：感性净无功；Q<0：容性净无功。两网前11个末端复用相同源馈线，节点缩放不同。',ha='center',fontsize=9.5,color='#49515d')
    return save(fig,out,'02_node_PQ_evaluation_day')


def relationship(worlds, out):
    w=next(w for w in worlds if w['case']=='pengwah18')
    fig, axes=plt.subplots(1,3,figsize=(14.5,5.1))
    fig.subplots_adjust(left=.06,right=.99,bottom=.20,top=.76,wspace=.28)
    fig.suptitle('P与Q的对应关系：Pengwah三个示例节点',fontsize=17,y=.98)
    fig.text(.5,.88,'灰点：8天训练；蓝点：第11天；虚线：只用训练段计算的固定比例Q=κP，作为诊断对照',ha='center',fontsize=10.5)
    for ax,j in zip(axes,EXAMPLES):
        pp,qq=w['p'][:,j],w['q'][:,j];s=w['node_stats'][j];k=s['kappa']
        ax.scatter(pp[:768],qq[:768],s=8,c='#a0a8b4',alpha=.35,edgecolors='none',label='训练')
        ax.scatter(pp[960:],qq[960:],s=13,c='#2563eb',alpha=.75,edgecolors='none',label='评价日')
        ends=np.array([pp.min(),pp.max()]);ax.plot(ends,k*ends,color='#d56a24',ls='--',lw=1.5,label='Q=κP')
        ax.axhline(0,c='#77808b',lw=.6);ax.axvline(0,c='#77808b',lw=.6)
        ax.set_title('节点'+str(w['labels'][j])+' / 源馈线'+str(s['source_feeder']),loc='left',fontsize=12)
        ax.set_xlabel('P / kW');ax.set_ylabel('Q / kvar');ax.grid(alpha=.16)
        ax.text(.04,.96,f"κ={k:.3f}\n比例残差能量占比 {100*s['ratio_residual_energy_fraction']:.1f}%",transform=ax.transAxes,va='top',fontsize=9.5,bbox=dict(facecolor='white',alpha=.87,edgecolor='none'))
    axes[-1].legend(loc='lower right',fontsize=8.5,framealpha=.9)
    fig.text(.5,.035,'比例残差能量占比 = Σ(Q−κP)² / ΣQ²（仅训练段）；用于展示固定比例近似的偏差，不参与拓扑生成或选模。',ha='center',fontsize=9.5,color='#49515d')
    return save(fig,out,'03_PQ_relationship')


def execute(run, archive_receipt, out):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve earlier plots: '+str(out))
    worlds,refs=load_worlds(run,archive_receipt)
    font();out.mkdir(parents=True)
    figures=[*overview(worlds,out),*daily(worlds,out),*relationship(worlds,out)]
    summaries=[]
    for w in worlds:
        summaries.append({k:v for k,v in w.items() if k not in ('p','q','times','meta')})
        frame=pd.DataFrame({'timestamp_CET':w['times'].astype(str)})
        for j,label in enumerate(w['labels']):
            frame['P_'+str(label)+'_kW']=w['p'][:,j]
            frame['Q_'+str(label)+'_kvar']=w['q'][:,j]
        frame.to_csv(out/(w['case']+'_curves_private.csv'),index=False,encoding='utf-8-sig')
    source=io.ref(Path(__file__))
    result=dict(status='passed',created_utc=datetime.now(timezone.utc).isoformat(),
        no_new_topology_or_impedance_fitting=True,no_new_AC=True,no_new_dispatch=True,source_to_world_bitwise_equal=True,
        proportional_diagnostic='Closed-form training-only kappa; display diagnostic, not a new topology estimator',
        curve_law='Before project-added meter noise; measured source error remains',
        system_total='Sum of terminal net powers, excluding line losses',
        timezone='Europe/Berlin (CET in this window)',minutes_per_point=15,
        split=dict(train=[0,768],validation=[768,960],evaluation=[960,1056]),
        example_source_feeders=[1,4,9],source=source,sources=refs,figures=figures,
        worlds=summaries,environment=io.environment(),source_unchanged=all(io.ref(r['path'])['sha256']==r['sha256'] for r in refs))
    io.write(out/'summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'data/history10_rx_20261116_v1')
    p.add_argument('--archive-receipt',type=Path,default=ROOT/'data/measured_pq_sources_20261114/archive.json')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=execute(a.run,a.archive_receipt,a.out)
    print({k:r[k] for k in ('status','source_to_world_bitwise_equal','no_new_AC','source_unchanged')})


if __name__=='__main__':main()
