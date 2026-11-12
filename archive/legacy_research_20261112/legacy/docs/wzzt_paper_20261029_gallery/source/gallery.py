"""Comprehensive recorded-data figures. Never executes an estimator or solver."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import gzip
import html
import json
import platform
import shutil
import sqlite3
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages
from research_experiments.wzzt_systematic.common import read,write,sha,linear_loss
from research_experiments.wzzt_fine.evaluate import signature
from .statistics import interval
from .gallery_records import Sources,create_snapshot,query,load_record,COHORTS
from .gallery_settings import EXTRA_LABELS,settings_table

ROOT=Path(__file__).resolve().parents[2]
METHODS={
 'NJ':('NJ','#808b96','o'), 'RNJ_native':('RNJ','#56a6c9','s'),
 'RNJ_L1_bank':('RNJ bank (L1)','#6c9fbe','D'),
 'WZZT_leaf_path':('WZZT leaf / finite pool','#d55e00','^'),
 'WZZT_leaf_bank':('WZZT leaf + RNJ bank','#e69f00','v'),
 'baseline25':('Single RNJ start','#6473aa','d'),
 'multistart':('RNJ+WZZT / multistart','#0072b2','o'),
 'restart_control':('Multistart + pool restart','#536e92','s'),
 'ridge_union':('Multistart + ridge pool','#009e73','D'),
 'multistart_guard0':('Multistart + delete','#b783b2','P'),
 'ridge_guard0':('Ridge pool + delete','#7b2cbf','X'),
 'WZZT_free_path':('WZZT leaf / free domain','#b74720','^'),
 'WZZT_free_bank':('Free domain + RNJ bank','#dc913c','v'),
 'ridge_postfilter_guard':('Validation filter guard','#bd5294','P'),
 'joint_control25':('Additional single-step / 25 s','#808b96','s'),
 'joint_width5':('Two-step / width 5','#009e73','D'),
 'joint_width5_delete':('Two-step + delete','#7b2cbf','X')}
CORE=['NJ','RNJ_native','WZZT_leaf_path','multistart','ridge_union','ridge_guard0']
CASES=['paper15','soumalas11','flynn16','pengwah18']
SIZES=[24,48,96,288];SIGMAS=[0.,.0002,.0005,.001]


def label(method):return METHODS.get(method,(method,'#555555','o'))[0]
def style(method):return dict(color=METHODS.get(method,('', '#555555','o'))[1],marker=METHODS.get(method,('', '', 'o'))[2])
def frame(*shape,figsize=None,**kwargs):return plt.subplots(*shape,figsize=figsize,layout='constrained',**kwargs)
def legend(fig,methods,ncol=3):
    fig.legend([Line2D([0],[0],**style(m),lw=1.5,ms=5) for m in methods],[label(m) for m in methods],
        loc='outside lower center',ncol=ncol,frameon=False,fontsize=9)
def grid(ax):ax.grid(alpha=.18);ax.set_axisbelow(True)
def sample_axis(ax):ax.set_xscale('log',base=2);ax.set_xticks(SIZES,labels=[str(n) for n in SIZES]);ax.set_xlabel('Training snapshots over 3 days');grid(ax)
def effect_errors(d):return np.maximum(0,np.vstack([d['mean']-d.lower,d.upper-d['mean']]))


def summarize(d,groups,metric,*,conditional=False,block=None):
    rows=[]
    for key,g in d.groupby(groups,dropna=False,sort=True):
        key=key if isinstance(key,tuple) else (key,)
        use=g[g.model_available] if conditional and 'model_available' in g else g
        unit=block or ('case' if 'panel' in g and set(g.panel)=={'topology'} else 'bundle')
        values=use.groupby(unit)[metric].mean().dropna()
        rows.append({**dict(zip(groups,key)),**interval(values),'n_planned':len(g),
            'n_available':int(g.model_available.sum()) if 'model_available' in g else len(g),
            'n_metric':int(use[metric].notna().sum()),'block_unit':unit,'metric':metric,
            'conditional_on_available':conditional})
    return pd.DataFrame(rows)


def paired(d,method,reference,*,cohort='original'):
    a=d[(d.cohort==cohort)&d.variant.eq(method)].copy()
    b=d[(d.cohort==cohort)&d.variant.eq(reference)].set_index('job_id')
    if not a.job_id.isin(b.index).all():raise ValueError('unmatched comparison')
    a['delta']=a.clade_f1-a.job_id.map(b.clade_f1)
    a['reference']=reference;return a


class Gallery:
    def __init__(self,sources,destination,tables,book_path=None):
        self.sources=sources;self.path=Path(destination);self.t=tables;self.figures=[]
        self.pdf=PdfPages(book_path or self.path/'all_figures.pdf',metadata={'Title':'WZZT recorded-data figure collection','CreationDate':None,'ModDate':None})
        (self.path/'figures').mkdir(exist_ok=True);(self.path/'plot_data').mkdir(exist_ok=True)
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
            'axes.labelsize':10,'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':240,
            'axes.spines.top':False,'axes.spines.right':False})

    def save(self,slug,title,category,fig,data,caption,*,cohorts='original'):
        self.pdf.savefig(fig,bbox_inches='tight')
        for ext in ('png','pdf'):
            kw=dict(metadata={'CreationDate':None,'ModDate':None}) if ext=='pdf' else {}
            fig.savefig(self.path/'figures'/(slug+'.'+ext),bbox_inches='tight',**kw)
        plt.close(fig);data.to_csv(self.path/'plot_data'/(slug+'.csv'),index=False)
        self.figures.append(dict(figure_id=slug,title=title,category=category,cohorts=cohorts,caption=caption,
            png='figures/'+slug+'.png',pdf='figures/'+slug+'.pdf',data='plot_data/'+slug+'.csv',plot_rows=len(data)))
        print('FIGURE',slug,flush=True)

    def curve(self,ax,d,methods,*,ylabel='',title='',scale=1,log=False):
        for m in methods:
            q=d[d.variant==m].sort_values('N')
            if len(q):ax.errorbar(q.N,q['mean']*scale,yerr=effect_errors(q)*scale,**style(m),lw=1.5,ms=4,capsize=2)
        sample_axis(ax);ax.set_ylabel(ylabel);ax.set_title(title)
        if log:ax.set_yscale('log')

    def main(self):
        f=self.t['model_results'];main=f[(f.cohort=='original')&(f.panel=='main')];methods=list(METHODS)[:11]
        stats=[]
        for metric in ('clade_f1','exact_clades'):stats.append(summarize(main,['variant'],metric))
        q=pd.concat(stats);q=q[q.variant.isin(methods)];fig,axes=frame(1,2,figsize=(13,6.8),sharey=True)
        for ax,metric,title in zip(axes,('clade_f1','exact_clades'),('Mean clade F1','Exact nontrivial clade family')):
            d=q[q.metric==metric].set_index('variant').loc[methods].reset_index();y=np.arange(len(d))
            ax.barh(y,d['mean'],color=[style(m)['color'] for m in methods],alpha=.8)
            ax.errorbar(d['mean'],y,xerr=effect_errors(d),fmt='none',ecolor='#222222',capsize=3,lw=1)
            ax.set_yticks(y,labels=[label(m) for m in methods]);ax.set_ylim(len(d)-.4,-.6);ax.set_xlim(0,1.12);ax.set_title(title);grid(ax)
            for i,r in d.iterrows():ax.text(r['upper']+.015,i,f"{r['mean']:.3f}",va='center',fontsize=8)
        fig.suptitle('Main panel: 640 inputs per method; 10 independent data bundles')
        self.save('01_main_methods','主面板：完整方法比较','主结果',fig,q,'F1与exact均针对非平凡末端簇集合。误差棒按十个bundle均值重采样，显示逐项95%区间；640条件共享四个固定网络。')
        q=summarize(main[main.variant.isin(CORE)],['voltage_sigma','N','variant'],'clade_f1')
        fig,axes=frame(2,2,figsize=(12,8),sharey=True)
        for ax,sigma in zip(axes.ravel(),SIGMAS):
            self.curve(ax,q[q.voltage_sigma==sigma],CORE,ylabel='Mean clade F1',title=f'V/root noise SD = {100*sigma:g}%');ax.set_ylim(-.02,1.04)
        legend(fig,CORE);self.save('02_noise_sample_size','噪声 × 数据量：六种主要方法','主结果',fig,q,'每点四网络×十bundle；区间按bundle分块。N为三个物理日的嵌套采样总量，V=0仍保留P/Q噪声0.5%。')
        methods=['NJ','multistart','ridge_guard0'];q=summarize(main[main.variant.isin(methods)],['case','voltage_sigma','N','variant'],'clade_f1')
        fig,axes=frame(4,4,figsize=(16,12),sharex=True,sharey=True)
        for i,c in enumerate(CASES):
            for j,sigma in enumerate(SIGMAS):
                ax=axes[i,j];self.curve(ax,q[(q['case']==c)&(q.voltage_sigma==sigma)],methods,title=f'{c} | SD {100*sigma:g}%');ax.set_ylim(-.02,1.04)
                if i<3:ax.set_xlabel('')
                if j==0:ax.set_ylabel('Clade F1')
        legend(fig,methods);self.save('03_each_network','逐网络的噪声与数据量曲线','主结果',fig,q,'每个小图固定网络和噪声，每点十个独立数据bundle；区间逐点计算。')
        self.heatmaps(main,CORE,'clade_f1','04_F1_heatmaps','各方法F1全条件矩阵',0,1,'viridis')
        self.heatmaps(main,CORE,'exact_clades','05_exact_heatmaps','各方法exact全条件矩阵',0,1,'viridis')
        changes=[]
        for method,ref in [('multistart','NJ'),('ridge_union','multistart'),('ridge_guard0','ridge_union')]:
            d=paired(main,method,ref);changes.append(summarize(d,['voltage_sigma','N','variant','reference'],'delta'))
        q=pd.concat(changes);fig,axes=frame(1,3,figsize=(14,4.8))
        for ax,(m,ref) in zip(axes,[('multistart','NJ'),('ridge_union','multistart'),('ridge_guard0','ridge_union')]):
            d=q[(q.variant==m)&(q.reference==ref)];p=d.pivot(index='voltage_sigma',columns='N',values='mean').loc[SIGMAS,SIZES]
            limit=max(abs(p.to_numpy()).max(),.005);im=ax.imshow(p,vmin=-limit,vmax=limit,cmap='RdBu_r',aspect='auto')
            self.annotate_heat(ax,p,fmt='+.3f',diverging=True);ax.set_title(label(m)+'\nminus '+label(ref));fig.colorbar(im,ax=ax,shrink=.8,label='Mean paired F1 gain')
        self.save('06_component_gain_maps','配对增量：多初值、扩池、删除','组件分析',fig,q,'同一输入上先相减，再按十bundle平均。各小图使用独立对称色标，数字给出实际效应量。')
        q=main[main.variant.isin(CORE)].groupby(['variant','voltage_sigma','N']).agg(precision=('precision','mean'),recall=('recall','mean'),n_planned=('record_id','size')).reset_index()
        fig,axes=frame(2,2,figsize=(11,8),sharex=True,sharey=True)
        for ax,sigma in zip(axes.ravel(),SIGMAS):
            for m in CORE:
                d=q[(q.voltage_sigma==sigma)&(q.variant==m)].sort_values('N');ax.plot(d.recall,d.precision,color=style(m)['color'],lw=1.2)
                ax.scatter(d.recall,d.precision,s=[18,34,58,92],**style(m),zorder=3)
            ax.set_xlim(-.02,1.05);ax.set_ylim(-.02,1.06);ax.set_xlabel('Mean recall');ax.set_ylabel('Mean precision');ax.set_title(f'SD {100*sigma:g}% | marker area increases with N');grid(ax)
        legend(fig,CORE);self.save('07_precision_recall','精确率、召回率与采样量','主结果',fig,q,'每个点为固定噪声/N下四网络×十bundle的均值。线连接同一方法的N24→48→96→288；标记由小到大对应这四个N，用于观察漏簇与过细分。')
        self.effects(main)
        q=summarize(main[main.variant.isin(CORE)],['voltage_sigma','N','variant'],'test_mae',conditional=True)
        fig,axes=frame(2,2,figsize=(12,8))
        for ax,sigma in zip(axes.ravel(),SIGMAS):self.curve(ax,q[q.voltage_sigma==sigma],CORE,ylabel='Test squared-drop MAE (pu$^2$)',title=f'SD {100*sigma:g}%',log=True)
        legend(fig,CORE);self.save('09_linear_test_error','独立测试日的平方压降误差','预测与AC',fig,q,'固定训练参数后评价独立测试日；纵轴为平方压降MAE，单位p.u.^2。区间按十bundle；误差统计以可用模型为条件。')
        out=[];methods=['NJ','multistart','ridge_union','ridge_guard0']
        fig,axes=frame(2,2,figsize=(11,8))
        for i,metric in enumerate(('R_error','X_error')):
            for j,sigma in enumerate((.0002,.001)):
                q=summarize(main[(main.voltage_sigma==sigma)&main.variant.isin(methods)],['N','variant'],metric,conditional=True);q['voltage_sigma']=sigma;out.append(q)
                self.curve(axes[i,j],q,methods,ylabel=f'Relative {metric[0]} response error',title=f'{metric[0]} | SD {100*sigma:g}%',log=True)
        legend(fig,methods);self.save('10_response_errors','R/X响应矩阵误差','预测与AC',fig,pd.concat(out),'相对Frobenius误差；R/X是平方压降响应矩阵，不是逐条线路阻抗误差。每点四网络×十bundle。')

    def annotate_heat(self,ax,p,fmt='.2f',diverging=False):
        a=p.to_numpy(dtype=float);ax.set_xticks(range(len(p.columns)),labels=[str(x) for x in p.columns]);ax.set_yticks(range(len(p.index)),labels=[f'{100*x:g}%' if isinstance(x,(float,np.floating)) else str(x) for x in p.index])
        ax.set_xlabel('Training snapshots');ax.set_ylabel('V/root noise SD')
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):ax.text(j,i,format(a[i,j],fmt),ha='center',va='center',fontsize=9,
                color='white' if (abs(a[i,j])>.65*np.nanmax(abs(a)) if diverging else a[i,j]<.55) else '#111111')

    def heatmaps(self,d,methods,metric,slug,title,vmin,vmax,cmap):
        q=summarize(d[d.variant.isin(methods)],['variant','voltage_sigma','N'],metric);fig,axes=frame(2,3,figsize=(12,7.7))
        for ax,m in zip(axes.ravel(),methods):
            p=q[q.variant==m].pivot(index='voltage_sigma',columns='N',values='mean').loc[SIGMAS,SIZES]
            im=ax.imshow(p,vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto');self.annotate_heat(ax,p);ax.set_title(label(m))
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label='Mean '+metric)
        self.save(slug,title,'主结果',fig,q,'每格40输入，按四固定网络×十数据bundle平均；原分母保留。图中数字为均值，逐点区间见绘图CSV。')

    def effects(self,main):
        pairs=[('multistart','NJ'),('ridge_union','multistart'),('ridge_union','restart_control'),('ridge_guard0','ridge_union')]
        rows=[];fig,axes=frame(1,4,figsize=(14,5),sharey=True)
        for ax,(m,ref) in zip(axes,pairs):
            d=paired(main,m,ref);b=d.groupby('bundle').delta.mean();summary=interval(b)
            rows.extend(dict(variant=m,reference=ref,row_type='bundle',bundle=int(k),value=v) for k,v in b.items())
            rows.append(dict(variant=m,reference=ref,row_type='summary',value=summary['mean'],**{k:v for k,v in summary.items() if k!='mean'}))
            ax.scatter(b,np.arange(len(b)),color=style(m)['color'],s=35);ax.axvline(0,color='#777777',lw=1)
            ax.errorbar([summary['mean']],[-1.2],xerr=[[max(0,summary['mean']-summary['lower'])],[max(0,summary['upper']-summary['mean'])]],fmt='D',capsize=4,color='#111111')
            ax.set_yticks([-1.2,*range(10)],labels=['Mean + 95% CI',*[f'b{i}' for i in range(10)]]);ax.set_ylim(9.6,-2);grid(ax)
            ax.set_title(label(m)+'\nminus '+label(ref));ax.set_xlabel('Paired F1 gain')
        self.save('08_paired_bundle_effects','每个独立数据重复上的配对收益','组件分析',fig,pd.DataFrame(rows),'每点一个bundle，先在该bundle内平均全部64个网络/噪声/N条件。黑色点为均值与95%区间。原三主检验的Holm结果保留于原报告。')

    def scenarios(self):
        f=self.t['model_results'];original=f[f.cohort=='original'];stress=original[original.panel=='stress'];methods=['NJ','multistart','ridge_union','ridge_guard0']
        q=summarize(stress[stress.variant.isin(methods)],['condition','N','variant'],'clade_f1');names=sorted(stress.condition.unique())
        fig,axes=frame(1,2,figsize=(13,7),sharey=True)
        for ax,N in zip(axes,(48,288)):
            for offset,m in zip(np.linspace(-.27,.27,len(methods)),methods):
                d=q[(q.N==N)&(q.variant==m)].set_index('condition').loc[names].reset_index()
                ax.errorbar(d['mean'],np.arange(len(names))+offset,xerr=effect_errors(d),**style(m),ls='none',capsize=2,ms=4)
            ax.set_yticks(range(len(names)),labels=[n.replace('_',' ') for n in names]);ax.set_xlim(-.03,1.04);ax.set_title(f'N = {N}');ax.set_xlabel('Mean clade F1');grid(ax)
        legend(fig,methods,ncol=2);self.save('11_stress_conditions','十类场景 × 两档数据量','场景与泛化',fig,q,'每个场景/N含四网络×五bundle；区间按五bundle。所有方法使用同一条件数据。')
        out=[]
        for m in ['NJ','restart_control','ridge_union','ridge_guard0']:
            out.append(summarize(paired(stress,m,'multistart'),['condition','N','variant'],'delta'))
        q=pd.concat(out);fig,axes=frame(1,2,figsize=(12,7),sharey=True)
        for ax,N in zip(axes,(48,288)):
            p=q[q.N==N].pivot(index='condition',columns='variant',values='mean').loc[names,['NJ','restart_control','ridge_union','ridge_guard0']]
            limit=max(.005,abs(q['mean']).max());im=ax.imshow(p,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
            ax.set_xticks(range(4),labels=['NJ','Restart','Ridge pool','Ridge + delete'],rotation=25,ha='right');ax.set_yticks(range(10),labels=[n.replace('_',' ') for n in names]);ax.set_title(f'N = {N}')
            for i in range(10):
                for j in range(4):ax.text(j,i,f'{p.iloc[i,j]:+.3f}',ha='center',va='center',fontsize=9,color='white' if abs(p.iloc[i,j])>.6*limit else 'black')
        fig.colorbar(im,ax=axes,label='Paired F1 gain vs multistart',shrink=.8)
        self.save('12_stress_gains','各场景相对多初值的收益','场景与泛化',fig,q,'先对相同输入配对相减；原池追加修正控制额外搜索预算。颜色统一，完整五bundle区间在CSV中。')
        topo=original[(original.panel=='topology')&original.variant.isin(CORE)].copy();topo['terminals']=topo['case'].str.extract(r'random(\d+)').astype(int)
        q=summarize(topo,['terminals','voltage_sigma','variant'],'clade_f1',block='case');fig,axes=frame(1,2,figsize=(12,5.5),sharey=True)
        for ax,sigma in zip(axes,(.0002,.001)):
            for m in CORE:
                d=q[(q.voltage_sigma==sigma)&(q.variant==m)].sort_values('terminals');ax.errorbar(d.terminals,d['mean'],yerr=effect_errors(d),**style(m),lw=1.4,capsize=3,ms=4)
            ax.set_xticks([8,16,24,32]);ax.set_ylim(-.02,1.04);ax.set_xlabel('Observed terminals');ax.set_ylabel('Mean clade F1');ax.set_title(f'SD {100*sigma:g}% | 3 trees / size');grid(ax)
        legend(fig,CORE);self.save('13_unseen_size','新拓扑的规模、噪声与恢复性能','场景与泛化',fig,q,'每档三棵新树，按树均值重采样；共享三个数据bundle和N48/288。原LP15缺失保留为零分，同时CSV记录可用数。')
        q=summarize(topo,['case','variant'],'clade_f1',block='bundle');order=[f'random{n}_t{i}' for n in (8,16,24,32) for i in range(3)]
        p=q.pivot(index='case',columns='variant',values='mean').loc[order,CORE];fig,ax=frame(figsize=(11,7.5));im=ax.imshow(p,vmin=0,vmax=1,cmap='viridis',aspect='auto')
        ax.set_xticks(range(len(CORE)),labels=[label(m) for m in CORE],rotation=22,ha='right');ax.set_yticks(range(12),labels=order)
        for i in range(12):
            for j in range(len(CORE)):ax.text(j,i,f'{p.iloc[i,j]:.2f}',ha='center',va='center',color='white' if p.iloc[i,j]<.55 else 'black',fontsize=9)
        fig.colorbar(im,ax=ax,label='Mean clade F1');self.save('14_each_unseen_tree','十二棵新树逐一比较','场景与泛化',fig,q,'各格平均该树的三个数据bundle、两噪声与两个N；逐树展示，避免总体均值掩盖网络差异。')
        recovered=self.sources.frame('case33_recovered_coverage.csv');methods=['NJ','multistart','ridge_union','ridge_postfilter_guard']
        d=recovered[recovered.variant.isin(methods)].copy();d['model_available']=True;d['panel']='case33';q=summarize(d,['N','voltage_sigma','variant','fixed_LP_seconds'],'clade_f1')
        fig,axes=frame(1,2,figsize=(12,5.5),sharey=True)
        for ax,sigma in zip(axes,(.0002,.001)):
            for m in methods:
                g=q[(q.voltage_sigma==sigma)&(q.variant==m)].sort_values('N');ax.errorbar(g.N,g['mean'],yerr=effect_errors(g),**style(m),lw=1.5,capsize=3)
            ax.set_xscale('log',base=2);ax.set_xticks([48,288],labels=['48\nLP15 original','288\nLP60 retry']);ax.set_ylim(-.02,1.04);ax.set_ylabel('Mean clade F1');ax.set_xlabel('Training N and fixed-LP budget');ax.set_title(f'Case33 derivative | SD {100*sigma:g}%');grid(ax)
        legend(fig,methods,ncol=2);self.save('15_case33_recovered','case33恢复覆盖：逐行注明预算','场景与泛化',fig,q,'同一棵端点化case33，五个数据bundle。N48为原LP15，N288为独立LP60重试；预算不同，数据量效应与预算不作单因素因果分解。过滤保护为探索性输出。',cohorts='original,lp60_retry,postfilter_guard')
        self.physical()

    def physical(self):
        d=self.t['physical_regimes'];d=d[d.condition=='reference'].copy();order=CASES+[f'random{n}_t{i}' for n in (8,16,24,32) for i in range(3)]+['case33_terminal']
        q=d.groupby('case').agg(bundles=('bundle','size'),drop_median=('drop_median_pct','mean'),drop_p05=('drop_p05_pct','mean'),drop_p95=('drop_p95_pct','mean'),Vmin=('minimum_V_pu','min'),Vmax=('maximum_V_pu','max')).loc[order].reset_index()
        fig,axes=frame(1,2,figsize=(12,9),sharey=True);y=np.arange(len(q))
        axes[0].errorbar(q.drop_median,y,xerr=np.maximum(0,np.vstack([q.drop_median-q.drop_p05,q.drop_p95-q.drop_median])),fmt='o',color='#0072b2',capsize=3)
        axes[0].set_xlabel('Root-minus-terminal V (% of nominal pu)');axes[0].set_title('Mean bundle median and 5th / 95th percentiles')
        axes[1].hlines(y,q.Vmin,q.Vmax,color='#009e73',lw=3);axes[1].scatter(q.Vmin,y,color='#0072b2',s=22);axes[1].scatter(q.Vmax,y,color='#d55e00',s=22)
        axes[1].set_xlabel('Voltage magnitude (pu)');axes[1].set_title('Observed terminal voltage range over all bundles')
        for ax in axes:ax.set_yticks(y,labels=order);ax.set_ylim(len(q)-.4,-.6);grid(ax)
        self.save('16_physical_voltage','实际AC电压与压降尺度','物理数据',fig,q,'无噪声物理数据，合并训练/验证/测试九日。左侧为每bundle分位数再平均的描述性范围，右侧为全体极值；均非置信区间。')
        d=self.t['daily_diagnostics'];d=d[(d['split']=='train')&(d['case'].isin(CASES))&(d.bundle<5)].copy()
        d=d[((d.condition=='reference')&(d.voltage_sigma==.0005)) | (d.condition!='reference')]
        # Each physical/noise condition has a single voltage-noise setting in the stress panel.
        q=d.groupby(['case','condition','bundle']).agg(condition_number=('condition_number','median'),signal=('dynamic_signal_rms','mean'),noise=('dynamic_noise_rms','mean')).reset_index()
        names=sorted(q.condition.unique());fig,axes=frame(1,2,figsize=(13,7),sharey=True)
        med=[]
        for i,c in enumerate(names):
            a=q[q.condition==c]
            axes[0].scatter(a.condition_number,np.full(len(a),i)+np.linspace(-.16,.16,len(a)),s=13,color='#0072b2',alpha=.5)
            axes[0].scatter([a.condition_number.median()],[i],marker='D',s=45,color='black')
            axes[1].scatter(a.signal.mean(),i-.13,marker='o',color='#009e73',s=40);axes[1].scatter(a.noise.mean(),i+.13,marker='x',color='#d55e00',s=40)
            med.append(dict(condition=c,network_bundles=len(a),median_condition_number=a.condition_number.median(),mean_signal=a.signal.mean(),mean_noise=a.noise.mean()))
        axes[0].set_xscale('log');axes[0].set_xlabel('Centered PQ design condition number\nDots: network/bundle; diamonds: median')
        axes[1].set_xlabel('Dynamic squared-drop RMS (pu$^2$)');axes[1].legend([Line2D([],[],marker='o',ls='',color='#009e73'),Line2D([],[],marker='x',ls='',color='#d55e00')],['Signal','Meter error'],loc='lower right')
        for ax in axes:ax.set_yticks(range(len(names)),labels=[n.replace('_',' ') for n in names]);grid(ax)
        self.save('17_excitation_meter_noise','激励条件、信号与计量误差','物理数据',fig,q,'每个点先汇总同一网络/bundle的三个训练日；原诊断使用每日96点，不是训练N子集。此图为描述性分布，未把日内快照当独立重复。')

    def components(self):
        a=self.sources.frame('ablations.csv');a=a[(a.panel=='main')&~a.variant.isin(['ridge_two_step','ridge_two_step_guard0'])].copy()
        a['comparison']=a.variant+' − '+a.reference;fig,ax=frame(figsize=(11,6));y=np.arange(len(a))
        ax.errorbar(a['mean'],y,xerr=effect_errors(a),fmt='o',capsize=4,color='#0072b2');ax.axvline(0,color='#777777');ax.set_yticks(y,labels=[f'{r.comparison}\n(n={r.inputs}, bundles={r.independent_blocks})' for r in a.itertuples()]);ax.set_xlabel('Paired clade F1 gain');grid(ax)
        self.save('18_predeclared_ablations','预定消融：正则化、预算、删除与候选库','组件分析',fig,a,'每行仅与自身匹配输入上的参考比较，子面板分母不同。区间按各自bundle；原单bundle两步结果保留在数据索引，十bundle扩展另图展示。')
        a=self.sources.frame('joint_extension_paper_comparisons.csv');d=self.t['model_results'];j=d[d.cohort=='joint_extension'];base=d[d.cohort=='original'].set_index(['job_id','variant'])
        pairs=[('joint_width5','joint_control25'),('joint_width5_delete','ridge_guard0'),('joint_width5_delete','joint_width5')]
        rows=[];fig,axes=frame(1,2,figsize=(13,6))
        for k,(m,ref) in enumerate(pairs):
            p=j[j.variant==m].copy();reference=j[j.variant==ref].set_index('job_id').clade_f1 if ref.startswith('joint_') else base.xs(ref,level='variant').clade_f1
            p['delta']=p.clade_f1-p.job_id.map(reference);b=p.groupby('bundle').delta.mean();summary=interval(b)
            axes[0].scatter(b,np.full(len(b),k)+np.linspace(-.16,.16,len(b)),s=25,color=style(m)['color'])
            axes[0].errorbar([summary['mean']],[k],xerr=[[max(0,summary['mean']-summary['lower'])],[max(0,summary['upper']-summary['mean'])]],fmt='D',color='black',capsize=4)
            counts=[int((p.delta>1e-10).sum()),int((abs(p.delta)<=1e-10).sum()),int((p.delta<-1e-10).sum())];left=0
            for count,color in zip(counts,['#009e73','#c5cbd0','#d55e00']):
                axes[1].barh(k,count,left=left,color=color)
                if count:axes[1].text(left+count/2,k,str(count),ha='center',va='center',fontsize=9)
                left+=count
            rows.extend(dict(row_type='bundle',comparison=m+' minus '+ref,bundle=int(bi),delta=v) for bi,v in b.items())
            rows.extend(dict(row_type='input',comparison=m+' minus '+ref,job_id=row.job_id,bundle=int(row.bundle),delta=row.delta) for row in p.itertuples())
            rows.append(dict(row_type='summary',comparison=m+' minus '+ref,**summary,inputs=len(p),wins=counts[0],ties=counts[1],losses=counts[2]))
        labels=['Two-step − single-step','Two-step + delete − delete only','Delete after two-step − two-step']
        for ax in axes:ax.set_yticks(range(3),labels=labels);grid(ax)
        axes[0].axvline(0,color='#777777');axes[0].set_xlabel('10 bundle means and 95% mean-gain interval')
        axes[1].set_xlim(0,160);axes[1].set_xlabel('Inputs: improve / tie / degrade (160 per comparison)')
        self.save('19_joint_extension','十bundle两步扩展的收益与分布','组件分析',fig,pd.DataFrame(rows),'次要扩展复用原主面板数据，固定父模型库与算法。每点一个bundle；并列展示稀疏收益与退化计数。完整区间、符号检验及最终来源保留在数据库/原表。',cohorts='joint_extension,original')
        s=self.t['filter_sensitivity'];fig,axes=frame(2,4,figsize=(16,7.5));rows=[]
        for j,panel in enumerate(['main','stress','topology','case33']):
            a=s[s.panel==panel]
            for i,metric in enumerate(['clade_f1','test_mae']):
                q=a.groupby('floor').agg(mean=(metric,'mean'),n_planned=('job_id','size'),n_available=('model_available','sum')).reset_index();q['panel']=panel;q['metric']=metric;rows.append(q)
                axes[i,j].plot(q.floor,q['mean'],'o-',color='#0072b2');axes[i,j].axvline(.16,ls='--',color='#d55e00');axes[i,j].set_xlabel('Final strength floor');axes[i,j].set_title(f'{panel} | {len(a)//5} inputs');grid(axes[i,j])
                if metric=='clade_f1':axes[i,j].set_ylim(-.02,1.04)
                else:axes[i,j].set_yscale('log')
        axes[0,0].set_ylabel('Mean clade F1');axes[1,0].set_ylabel('Available-model test MAE (pu$^2$)')
        self.save('20_filter_sensitivity','最终过滤阈值的结构与预测影响','组件分析',fig,pd.concat(rows),'固定同一收缩前预选模型，仅改变最终阈值；未重新拟合或搜索。F1保留全部输入分母，MAE按可用模型统计；五阈值属于探索性诊断。',cohorts='exploratory_filter_sensitivity')
        g=d[d.cohort=='postfilter_guard'].copy();r=d[(d.cohort=='original')&d.variant.eq('ridge_union')].set_index('job_id')
        g['delta_F1']=g.clade_f1-g.job_id.map(r.clade_f1);g['delta_validation_mae']=g.validation_mae-g.job_id.map(r.validation_mae)
        g=g[g.model_available];fig,axes=frame(2,2,figsize=(11,8))
        for ax,panel in zip(axes.ravel(),['main','stress','topology','case33']):
            q=g[g.panel==panel];ax.scatter(q.delta_validation_mae,q.delta_F1,s=17,alpha=.5,color='#7b2cbf');ax.axhline(0,color='#777777');ax.axvline(0,color='#777777');ax.set_xscale('symlog',linthresh=1e-6)
            ax.set_title(f'{panel}: {len(q)} available inputs');ax.set_xlabel('Guard minus fixed-floor validation MAE (pu$^2$)');ax.set_ylabel('Guard minus fixed-floor F1');grid(ax)
        self.save('21_filter_guard_tradeoff','验证过滤保护：预测与拓扑的不同变化','组件分析',fig,g,'同一预选模型上验证误差不增，拓扑F1仍可能下降。图只展示可用的1191个模型；原13条不可用记录仍在模型总表。新树/case33的预先冻结补充检查标记在prospective_filter_holdout字段。',cohorts='postfilter_guard,original')

    def diagnostics(self):
        f=self.t['model_results'];original=f[f.cohort=='original'];methods=['multistart','ridge_union','ridge_guard0'];rows=[]
        fig,axes=frame(1,4,figsize=(16,5),sharey=True)
        for ax,panel in zip(axes,['main','stress','topology','case33']):
            for k,m in enumerate(methods):
                d=original[(original.panel==panel)&original.variant.eq(m)];a=d[d.model_available]
                crossing=a.diagnostic_json.fillna('{}').map(lambda s:bool(json.loads(s).get('crossing_pairs',[])))
                vals=[float((a.candidate_pool_true_recall<1-1e-12).mean()),float(crossing.mean()),float((a.predicted_clades>a.true_clades).mean()),1-len(a)/len(d)]
                for i,v in enumerate(vals):ax.bar(i+(k-1)*.23,v,width=.22,color=style(m)['color'])
                rows.extend(dict(panel=panel,variant=m,diagnostic=n,fraction=v,n_planned=len(d),n_available=len(a),denominator='planned' if i==3 else 'available') for i,(n,v) in enumerate(zip(['pool misses true','final crossing','too many clades','unavailable'],vals)))
            ax.set_xticks(range(4),labels=['Pool miss','Crossing','Overfine','Unavailable'],rotation=35,ha='right');ax.set_title(panel);ax.set_ylim(0,1.04);grid(ax)
        axes[0].set_ylabel('Fraction of inputs');legend(fig,methods)
        self.save('22_failure_diagnostics','候选覆盖、错误交叉与过细分','失败与求解',fig,pd.DataFrame(rows),'前三项以可用模型为分母，unavailable以全部计划输入为分母。过细分指非平凡预测簇多于真簇；最终交叉是结构诊断，时间先后证据另见逐步事件。')
        e=self.t['greedy_events'];e=e[e.full_sweep & (e.true_candidates_tested>0)].copy();fig,axes=frame(1,2,figsize=(12,5.5));colors={'main':'#0072b2','stress':'#009e73','topology':'#7b2cbf','case33':'#d55e00'}
        counts=[]
        for i,(panel,g) in enumerate(e.groupby('panel')):
            x=np.sort(g.loss_margin_vs_best_true);axes[0].plot(x,np.arange(1,len(x)+1)/len(x),color=colors[panel],label=f'{panel}: {len(g)} events / {g.job_id.nunique()} inputs')
            strict=int(g.strictly_better_than_tested_true.sum());axes[1].barh(i,strict,color=colors[panel]);axes[1].barh(i,len(g)-strict,left=strict,color='#c5cbd0');axes[1].text(len(g)+20,i,f'{strict}/{len(g)}',va='center',fontsize=9);counts.append(panel)
        axes[0].set_xscale('symlog',linthresh=1e-8);axes[0].axvline(0,color='#777777');axes[0].set_xlabel('Best tested true loss − accepted false loss (pu$^2$)');axes[0].set_ylabel('Descriptive event ECDF');axes[0].legend(fontsize=8);grid(axes[0])
        axes[1].set_yticks(range(len(counts)),labels=counts);axes[1].set_xlabel('Strict false preference / eligible false-accept events');axes[1].set_xlim(0,e.groupby('panel').size().max()*1.2);grid(axes[1])
        self.save('23_greedy_objective','完整扫描中的一步训练目标偏好','失败与求解',fig,e,'仅保留完整有限池扫描且测过真簇提案的错误正权簇接受事件。严格偏好阈值1e-8；多个事件可属于同一输入，不作为独立重复或长期阻挡的充分证据。')
        fig,axes=frame(1,4,figsize=(16,5),sharey=True);rows=[];methods=['WZZT_leaf_path','restart_control','ridge_union'];categories=['pool_local_optimum','pool_wall_limit','unavailable','other'];colors2=['#009e73','#e69f00','#b64747','#aeb6bf']
        for ax,panel in zip(axes,['main','stress','topology','case33']):
            for i,m in enumerate(methods):
                d=original[(original.panel==panel)&original.variant.eq(m)];status=d.wzzt_search_status.fillna('unavailable');bottom=0
                for cat,col in zip(categories,colors2):
                    count=int(status.eq(cat).sum()) if cat!='other' else int((~status.isin(categories[:-1])).sum())
                    ax.bar(i,count/len(d),bottom=bottom,color=col);bottom+=count/len(d);rows.append(dict(panel=panel,variant=m,status=cat,count=count,n_planned=len(d)))
            ax.set_xticks(range(3),labels=['Leaf','Restart','Ridge'],rotation=25);ax.set_title(panel);ax.set_ylim(0,1.03);grid(ax)
        axes[0].set_ylabel('Fraction of planned inputs');fig.legend([plt.Rectangle((0,0),1,1,color=c) for c in colors2],['Pool local stop','Path budget','Unavailable','Other'],loc='outside lower center',ncol=4,frameon=False)
        self.save('24_finite_pool_stops','有限池局部停止、预算与不可用记录','失败与求解',fig,pd.DataFrame(rows),'局部停止针对当前有限候选邻域；不可用是共享工作进程阶段未生成记录，不直接归因于各算法不可行。')
        free=f[f.cohort=='free_domain'];ids=set(free.job_id);methods=['WZZT_leaf_path','WZZT_leaf_bank','multistart','ridge_union','WZZT_free_path','WZZT_free_bank']
        d=pd.concat([original[original.job_id.isin(ids)&original.variant.isin(methods)],free]);q=summarize(d,['cohort','variant'],'clade_f1');q=q.set_index('variant').loc[methods].reset_index()
        fig,axes=frame(1,2,figsize=(13,6.5));y=np.arange(len(q));axes[0].barh(y,q['mean'],color=[style(m)['color'] for m in methods],alpha=.8);axes[0].errorbar(q['mean'],y,xerr=effect_errors(q),fmt='none',ecolor='black',capsize=3);axes[0].set_yticks(y,labels=[label(m) for m in methods]);axes[0].set_xlim(0,1.05);axes[0].set_xlabel('Clade F1 on the same 80 inputs');grid(axes[0])
        events=self.t['mip_events'];status_labels={2:'Gap target reached (2)',9:'Time limit (9)',11:'Callback interrupt (11)'}
        for status,g in events.groupby('raw_status'):
            x=np.sort(g.relative_gap);axes[1].plot(x,np.arange(1,len(x)+1)/len(x),label=f'{status_labels.get(status,str(status))}: {len(g)} calls')
        axes[1].set_xlabel('Native single-MILP relative gap');axes[1].set_ylabel('Descriptive ECDF');axes[1].legend(fontsize=8);grid(axes[1])
        data=pd.concat([q.assign(row_type='F1_summary'),events.assign(row_type='MIP_event')],ignore_index=True,sort=False)
        self.save('25_free_domain_and_gap','自由域策略与原生MILP gap','失败与求解',fig,data,'左侧80匹配输入、五bundle区间。自由域100秒路径采用兼容簇追加，有限池25秒修正允许交叉替换；比较完整策略。右侧626次调用，gap属于单步模型，状态2按本轮1%容差终止。',cohorts='original,free_domain')
        retry=f[f.cohort=='lp60_retry'];ids=set(retry.job_id);base=original[original.job_id.isin(ids)];methods=[m for m in list(METHODS)[:11] if m in set(retry.variant)]
        fig,axes=frame(1,2,figsize=(14,6));rows=[]
        for offset,d,name,col in [(-.18,base,'Original LP15','#808b96'),(.18,retry,'LP60 retry','#009e73')]:
            q=d[d.variant.isin(methods)].groupby('variant').model_available.sum().reindex(methods);axes[0].barh(np.arange(len(methods))+offset,q,height=.34,color=col,label=name)
            rows.extend(dict(row_type='availability',variant=m,budget=name,available=int(q[m]),planned=13) for m in methods)
        axes[0].set_yticks(range(len(methods)),labels=[label(m) for m in methods]);axes[0].set_xlim(0,14);axes[0].set_xlabel('Available model records out of 13 failed-worker inputs');axes[0].legend(fontsize=8);grid(axes[0])
        raw=self.sources.frame('lp60_retry.csv').drop_duplicates('job_id').sort_values('job_id');axes[1].plot(range(len(raw)),raw.longest_LP_seconds,'o',color='#0072b2');axes[1].axhline(15,color='#d55e00',ls='--',label='Original LP cap: 15 s');axes[1].axhline(60,color='#777777',ls=':',label='Retry LP cap: 60 s');axes[1].set_xticks(range(len(raw)),labels=[s.split('__')[1].replace('case33_terminal','C33').replace('random32_t2','R32')+' '+s.split('__')[2]+' '+s.split('__')[-2] for s in raw.job_id],rotation=65,ha='right',fontsize=7)
        axes[1].set_ylabel('Longest completed fixed-family LP (seconds)');axes[1].legend(fontsize=8);grid(axes[1]);rows.extend(dict(row_type='LP_time',**r) for r in raw.to_dict('records'))
        self.save('26_LP60_recovery','原共享进程缺失与LP60恢复','失败与求解',fig,pd.DataFrame(rows),'十三组原LP15触时输入均以独立LP60重试恢复；左侧可用性包含原checkpoint已有结果。不同预算分别保留，原触时不被覆盖。',cohorts='original,lp60_retry')

    def runtime_accuracy(self):
        f=self.t['model_results'];original=f[f.cohort=='original'];timing=self.t['timing'].copy();methods=['NJ','RNJ_native','multistart','ridge_union'];lookup=original.set_index(['job_id','variant']).clade_f1
        timing['clade_f1']=[lookup.loc[(j,m)] for j,m in zip(timing.job_id,timing.variant)];fig,axes=frame(1,2,figsize=(12,6))
        for i,m in enumerate(methods):
            d=timing[timing.variant==m];axes[0].boxplot(d.estimator_seconds,positions=[i],widths=.5,showfliers=False,patch_artist=True,boxprops=dict(facecolor=style(m)['color'],alpha=.45),medianprops=dict(color='black'))
            axes[0].scatter(np.full(len(d),i)+np.linspace(-.14,.14,len(d)),d.estimator_seconds,s=18,color=style(m)['color'],alpha=.7)
            axes[1].scatter(d.estimator_seconds,d.clade_f1,s=17,alpha=.25,color=style(m)['color'])
            x=d.estimator_seconds.median();y=d.clade_f1.mean();axes[1].scatter(x,y,s=100,**style(m))
            offsets={'NJ':(-35,-24),'RNJ_native':(6,20),'multistart':(-105,-24),'ridge_union':(8,20)}
            axes[1].annotate(label(m),(x,y),xytext=offsets[m],textcoords='offset points',fontsize=8,
                bbox=dict(boxstyle='round,pad=.2',facecolor='white',edgecolor='none',alpha=.85))
        axes[0].set_xticks(range(4),labels=['NJ','RNJ','Multistart','Ridge pool']);axes[0].set_yscale('log');axes[0].set_ylabel('Estimator seconds; fresh process, no shared estimates');grid(axes[0])
        axes[1].set_xscale('log');axes[1].set_xlabel('Estimator seconds (large marker: median)');axes[1].set_ylabel('Clade F1 (large marker: mean over matched 16 inputs)');axes[1].set_ylim(-.02,1.06);grid(axes[1])
        self.save('27_runtime_accuracy','独立运行成本与配对精度','成本',fig,timing,'每方法16个相同输入，各用新进程顺序重放。CPU/Gurobi环境见原hardware.json；不含AC后处理。右侧为固定bundle0的描述性成本/精度，不将16条件当独立重复。')
    def computation(self):
        self.runtime_accuracy()
        original=self.t['model_results'];original=original[original.cohort=='original']
        methods=['NJ','multistart','ridge_union'];a=original[(original.panel=='main')&original.ac_evaluated&original.variant.isin(methods)].copy()
        q=summarize(a,['voltage_sigma','N','variant'],'test_ac_rmse',conditional=True);fig,axes=frame(1,2,figsize=(12,5.7))
        for ax,sigma in zip(axes,(.0002,.001)):
            self.curve(ax,q[q.voltage_sigma==sigma],methods,ylabel='Independent-test voltage RMSE (%)',title=f'V/root meter SD {100*sigma:g}%',scale=100)
            ax.set_xticks([48,288],labels=['48','288']);ax.set_xlim(40,320)
        legend(fig,methods);self.save('28_AC_test_prediction','AC重估后的独立测试电压误差','预测与AC',fig,q,'只用训练数据重估线参数，冻结后评价独立测试日。纵轴为相对电压幅值RMSE百分数；按十bundle计算逐点区间。与平方压降MAE分开解释。')
        fits=self.t['ac_fits'];a=fits[fits.cohort=='original'].copy();links=original[original.ac_key.notna()].drop_duplicates('ac_key').set_index('ac_key')
        for col in ('panel','case','N','bundle','clade_f1'):a[col]=a.ac_key.map(links[col])
        fig,axes=frame(1,2,figsize=(12,5.5));colors={'main':'#0072b2','stress':'#009e73','topology':'#7b2cbf','case33':'#d55e00'}
        for panel,g in a.groupby('panel'):
            for converged,marker in [(True,'o'),(False,'x')]:
                d=g[g.locally_converged.eq(converged)]
                axes[0].scatter(d.clade_f1,100*d.test_rmse,s=20,c=colors[panel],marker=marker,alpha=.5)
                axes[1].scatter(d.clade_f1,100*d.physical_test_rmse,s=20,c=colors[panel],marker=marker,alpha=.5)
        for ax,title in zip(axes,['Measured PQ/root/V test','Noiseless physical test']):
            ax.set_yscale('log');ax.set_xlabel('Nontrivial clade F1');ax.set_ylabel('Frozen-parameter voltage RMSE (%)');ax.set_title(title);grid(ax)
        handles=[Line2D([],[],marker='o',ls='',color=c,label=p) for p,c in colors.items()]+[Line2D([],[],marker='o',ls='',color='black',label='Local fit converged'),Line2D([],[],marker='x',ls='',color='black',label='Local fit stopped without convergence')]
        fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False,fontsize=8)
        self.save('29_AC_error_topology','AC残差与拓扑F1的对应关系','预测与AC',fig,a,'原批次580个去重的输入/参数组合；同一模型被多个方法选中时只画一次。标记区分局部参数拟合收敛与否，低残差作为预测指标。')
        inputs=self.t['inputs'].set_index('job_id');a['terminals']=a.job_id.map(inputs.labels_json.map(lambda x:len(json.loads(x))))
        fig,axes=frame(1,2,figsize=(12,5.5))
        for conv,marker,col in [(True,'o','#009e73'),(False,'x','#d55e00')]:
            g=a[a.locally_converged.eq(conv)];axes[0].scatter(g.terminals,g.fitting_seconds,s=18,marker=marker,c=col,alpha=.45,label='Converged' if conv else 'Not converged')
        axes[0].set_yscale('log');axes[0].set_xlabel('Observed terminals');axes[0].set_ylabel('AC training refit seconds');axes[0].legend();grid(axes[0])
        counts=a.groupby(['nfev','locally_converged']).size().unstack(fill_value=0);bottom=np.zeros(len(counts))
        for conv,col in [(True,'#009e73'),(False,'#d55e00')]:
            vals=counts.get(conv,pd.Series(0,index=counts.index));axes[1].bar(counts.index,vals,bottom=bottom,color=col,label='Converged' if conv else 'Not converged');bottom+=vals
        axes[1].set_xlabel('least_squares nfev (numerical Jacobian calls excluded)');axes[1].set_ylabel('Unique fitted models');axes[1].legend();grid(axes[1])
        self.save('30_AC_cost_convergence','AC计算成本和局部收敛','成本',fig,a,'max_nfev=12约束求解器的主函数评价计数，数值Jacobian额外潮流调用另计。图用实际seconds记录成本，580个原批去重模型均保留。')

    def topology_examples(self):
        from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
        results=self.t['model_results'];params=self.t['parameters'].set_index('record_id');inputs=self.t['inputs'].set_index('job_id')
        examples=[('main',c,48,'original') for c in CASES]+[('topology','random32_t1',288,'original'),('case33','case33_terminal',288,'lp60_retry')]
        for k,(panel,case,N,cohort) in enumerate(examples,31):
            selected=results[(results.panel==panel)&results['case'].eq(case)&results.N.eq(N)&results.bundle.eq(0)&results.voltage_sigma.eq(.001)&results.cohort.eq(cohort)]
            methods=['NJ','multistart','ridge_union','ridge_postfilter_guard' if cohort=='lp60_retry' else 'ridge_guard0']
            lookup=selected.set_index('variant');job=selected.job_id.iloc[0];network=Path(inputs.loc[job,'truth_network'])
            for name in ('metadata.json','branches.csv'):self.sources.consume(network/name)
            p=params.loc[lookup.loc['NJ','record_id']];labels=json.loads(p.labels_json);positions={v:i for i,v in enumerate(labels)}
            truth={tuple(sorted(positions[v] for v in s)) for s in physical_truth(network,labels)['clades']};n=len(labels)
            fig,axes=frame(2,3,figsize=(16,9));records=[]
            for ax,m in zip(axes.ravel(),['truth',*methods]):
                if m=='truth':family=truth;title='True canonical clade tree'
                else:
                    row=lookup.loc[m];point=json.loads(params.loc[row.record_id,'point_json']);family={tuple(s) for s in point['supports'] if 2<=len(s)<n}
                    TP=len(family&truth);FP=len(family-truth);FN=len(truth-family);title=f'{label(m)}\nF1={row.clade_f1:.3f} | TP {TP}, FP {FP}, FN {FN}'
                full=tuple(range(n));family=set(family)|{(i,) for i in range(n)}|{full};parent={s:min((a for a in family if set(s)<set(a)),key=lambda a:(len(a),a)) for s in family if s!=full}
                children={s:sorted([t for t in family if parent.get(t)==s],key=lambda t:(min(t),len(t))) for s in family};order=[];depth={full:0}
                def visit(s):
                    if len(s)==1:order.append(s[0])
                    for c in children[s]:depth[c]=depth[s]+1;visit(c)
                visit(full);leafx={v:i for i,v in enumerate(order)};bottom=max(depth.values())+1
                coords={s:(float(np.mean([leafx[i] for i in s])),-bottom if len(s)==1 else -depth[s]) for s in family}
                for s in sorted(family,key=lambda s:(len(s),s)):
                    x,y=coords[s];par=parent.get(s)
                    if par is not None:
                        px,py=coords[par];ax.plot([px,x,x],[py,py,y],color='#70777e',lw=.8,zorder=1)
                    color='#0072b2' if len(s)==1 else '#333333' if s==full else '#009e73' if s in truth else '#d55e00'
                    ax.scatter([x],[y],s=12 if len(s)==1 else 24,color=color,zorder=2)
                    if len(s)==1:ax.text(x,y-.2,str(labels[s[0]]),rotation=90 if n>20 else 0,ha='center',va='top',fontsize=6 if n<=20 else 5.5)
                    records.append(dict(variant=m,support_json=json.dumps(s),parent_json=json.dumps(par),x=x,y=y,nontrivial_true_clade=s in truth,terminal_ids_json=json.dumps([labels[i] for i in s])))
                ax.set_title(title,fontsize=10);ax.set_xlim(-1,n);ax.set_ylim(-bottom-1,1);ax.axis('off')
            axes.ravel()[-1].axis('off');axes.ravel()[-1].text(.02,.8,'Canonical clade trees\n\nBlue: observed terminal\nGreen: correct nontrivial clade\nOrange: false nontrivial clade\nBlack: root\n\nEdges show hierarchy; lengths are layout only.\nTerminal order follows each tree traversal.',va='top',fontsize=11,linespacing=1.6)
            fig.suptitle(f'{case} | bundle 0 | N={N} | V/root SD=0.1% | '+('LP60 retry' if cohort=='lp60_retry' else 'original LP15'))
            self.save(f'{k:02d}_topology_{case}',f'拓扑示例：{case}','拓扑示例',fig,pd.DataFrame(records),'示例按网络、bundle0、固定噪声与N选取，不按结果优劣筛选。图为非平凡簇的规范层级，压缩根/叶的平凡结构；边长只作布局。case33为LP60补充，过滤保护为探索性方法。',cohorts=cohort)


GLOSSARY={
 'record_id':('cohort::job_id::variant；每条模型结果的唯一键','identifier'),
 'cohort':('original/free_domain/joint_extension/lp60_retry/postfilter_guard；不同来源分别查询','category'),
 'job_id':('原1204输入条件唯一键；补充处理仍复用同一输入键','identifier'),
 'panel':('main/stress/topology/case33','category'),'case':('网络名称','category'),
 'condition':('物理、激励或计量场景','category'),'bundle':('独立数据组编号；多条件共享同组随机路径','integer'),
 'N':('训练三物理日的总快照数，嵌套采样；验证和测试各288点','snapshots'),
 'voltage_sigma':('端点电压计量相对高斯标准差；根电压设置见configuration_json','relative SD'),
 'pq_sigma':('P/Q计量相对标准差','relative SD'),'variant':('方法代码，见methods.csv','category'),
 'model_available':('是否存在可用的保存模型；原缺失记录仍保留','boolean'),
 'score_policy':('标记可用模型评分或缺失保留为零分的统计口径','category'),
 'clade_f1':('非平凡末端簇F1；排除叶簇与全末端簇','0..1'),
 'exact_clades':('可用模型的非平凡簇集合完全一致','boolean'),
 'precision':('正确预测非平凡簇/预测簇数','0..1'),'recall':('正确预测非平凡簇/真簇数','0..1'),
 'predicted_clades':('预测非平凡簇数','count'),'true_clades':('真实非平凡簇数','count'),'matched_clades':('匹配真簇数','count'),
 'R_error':('R平方压降响应矩阵的相对Frobenius误差','ratio'),'X_error':('X平方压降响应矩阵的相对Frobenius误差','ratio'),
 'train_mae':('训练平方压降MAE','pu^2'),'validation_mae':('验证平方压降MAE','pu^2'),'test_mae':('独立测试平方压降MAE','pu^2'),
 'test_ac_rmse':('训练后冻结参数的独立测试相对电压幅值RMSE','relative voltage'),
 'physical_test_ac_rmse':('冻结AC参数在无噪声测试物理路径上的相对电压RMSE','relative voltage'),
 'fixed_LP_seconds':('固定家族LP时间上限；不等于整条路径或工作进程上限','seconds'),
 'worker_seconds':('整共享进程耗时，不是单方法独立部署耗时','seconds'),
 'shared_worker_seconds':('原主批共享进程耗时，不作单方法计时','seconds'),
 'estimator_seconds':('新进程中单个研究入口的估计耗时，排除导入/读取','seconds'),
 'process_wall_seconds':('包括新进程启动的端到端重放耗时','seconds'),
 'search_seconds':('实际路径耗时；检查预算可能被单次LP越过','seconds'),
 'evidence_role':('预定主/组件或次要扩展、探索性分析','category'),
 'prospective_filter_holdout':('是否属于过滤规则冻结后的148输入补充检查；与主数据复用关系见原报告','boolean'),
 'point_json':('完整保存模型；supports为端点位置集合，r/x为平方压降响应系数','JSON'),
 'labels_json':('端点位置到实际bus ID的映射','JSON'),
 'point_signature':('沿用原signature函数的结构/参数指纹','SHA-256'),
 'model_signature':('与parameters.point_signature关联','SHA-256'),
 'ac_key':('cohort命名空间内去重的AC输入/参数键','identifier'),
 'locally_converged':('least_squares报告的局部重估收敛标记','boolean'),
 'nfev':('least_squares函数评价计数；不含数值Jacobian额外潮流调用','count'),
 'testing_valid':('全部测试点潮流计算成功；本轮未配运行电压/热限约束','boolean'),
 'relative_gap':('单次原生MILP相对gap；不是全路径/全树最优性gap','ratio'),
 'condition_number':('每天96点中心化[P,Q]设计矩阵条件数','ratio'),
 'dynamic_signal_rms':('真动态平方压降信号RMS','pu^2'),'dynamic_noise_rms':('平方压降计量误差RMS','pu^2'),
 'drop_median_pct':('九物理日根电压减端点电压的中位数','nominal-pu percentage points'),
 'source_table':('原指标表文件名','path'),'source_table_sha256':('实际读入指标表哈希','SHA-256'),
 'source_result':('原始结果或checkpoint路径；探索性后处理的完整点另保存在本索引','path'),
 'source_result_sha256':('原始结果/前置结果文件哈希','SHA-256'),
 'source_path':('具体AC结果文件路径','path'),'source_sha256':('来源文件哈希','SHA-256'),
}
GLOSSARY.update({
 'train_initial_rmse':('AC重估前的训练相对电压RMSE','relative voltage'),
 'train_rmse':('AC重估后的训练相对电压RMSE','relative voltage'),
 'test_rmse':('冻结AC参数的独立测试相对电压RMSE','relative voltage'),
 'physical_test_rmse':('冻结AC参数在无噪声物理测试目标上的相对电压RMSE','relative voltage'),
 'r_refit_json':('AC重估后的R响应系数，物理标幺阻抗为其一半','JSON'),
 'x_refit_json':('AC重估后的X响应系数，物理标幺阻抗为其一半','JSON'),
 'fitting_seconds':('AC训练重估实际时间，含数值Jacobian潮流调用','seconds'),
 'testing_seconds':('AC固定参数观测测试耗时','seconds'),'physical_seconds':('AC固定参数无噪声目标测试耗时','seconds'),
 'train_rows':('AC训练时实际潮流行数，应等于N','snapshots'),'test_rows':('AC测试实际行数，当前为288','snapshots'),
 'mean_drop':('逐物理日根电压减端点电压的均值','pu voltage'),'min_drop':('逐物理日最小根到端点压降','pu voltage'),
 'max_drop':('逐物理日最大根到端点压降','pu voltage'),
 'drop_p05_pct':('九日根到端点压降的5%分位数','nominal-pu percentage points'),
 'drop_p95_pct':('九日根到端点压降的95%分位数','nominal-pu percentage points'),
 'minimum_V_pu':('九日端点电压最小值','pu voltage'),'maximum_V_pu':('九日端点电压最大值','pu voltage'),
 'voltage_min':('逐物理日电压最小值','pu voltage'),'voltage_max':('逐物理日电压最大值','pu voltage'),
 'physical_snapshots':('合并九日的物理时刻数；不是独立数据bundle数','snapshots'),
 'bound':('原生MILP下界，对应单步平方压降MAE','pu^2'),'incumbent':('原生MILP可行目标值','pu^2'),
 'selected_upper':('最终采用的已核验可行解目标上界','pu^2'),
 'settings_id':('cohort::variant，算法配置与预算的键','identifier'),
 'nominal_stage_budgets_json':('分阶段名义预算；实际可能因单次LP越过检查点，不能当独立部署总耗时','JSON seconds'),
 'MILP_relative_gap':('自由域单步MILP的原生相对gap目标；其他方法为空','ratio'),
 'worker_failure_kind':('原共享进程异常类别，区分固定LP触时与其他错误','category')})


def documentation(gallery):
    p=gallery.path;tables=gallery.t;registry=pd.DataFrame(gallery.figures);registry.to_csv(p/'figure_index.csv',index=False)
    METHODS.update(EXTRA_LABELS)
    tables['methods']=pd.DataFrame([dict(variant=m,label_en=v[0],color=v[1],marker=v[2]) for m,v in METHODS.items()])
    tables['methods'].to_csv(p/'methods.csv',index=False)
    tables['method_settings']=settings_table(tables['model_results']);tables['method_settings'].to_csv(p/'method_settings.csv',index=False)
    with sqlite3.connect(p/'records.sqlite') as conn:
        for name in ('methods','method_settings'):tables[name].to_sql(name,conn,index=False,if_exists='replace')
        conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS settings_key ON method_settings(cohort,variant)')
    dictionary=[]
    for name,d in tables.items():
        for c in d:
            description,unit=GLOSSARY.get(c,('保留原表同名字段；来源、指标口径见原实验HANDOFF及本目录README','source-defined'))
            dictionary.append(dict(table=name,column=c,dtype=str(d[c].dtype),missing=int(d[c].isna().sum()),description=description,unit=unit))
    pd.DataFrame(dictionary).to_csv(p/'data_dictionary.csv',index=False)
    snapshot=read(p/'snapshot_summary.json')
    lines=['# WZZT数据记录与全面绘图','',
        f"基于已完成的2026-10-29实验生成；本次未重跑估计器或求解器。模型结果共{snapshot['model_records']}条、可用{snapshot['available_models']}条，覆盖原1204个输入条件。五类结果通过cohort区分；补充处理不增加独立输入/重复数量。",'',
        '## 浏览与取图','',
        f"共{len(registry)}张图，其中六张为固定条件的规范拓扑示例。每图提供高分辨率PNG、矢量PDF及实际绘图数据CSV。完整PDF合集为[all_figures.pdf](all_figures.pdf)，离线图册为[index.html](index.html)，机器可读图目为[figure_index.csv](figure_index.csv)。",'',
        '英文图注适合论文排版；中文说明在下列各图条目中。主统计区间沿用固定seed20261029的20000次bundle均值重采样；新树按树分块，图中逐点区间是描述性逐点区间。', '',
        '## 数据与统计口径','',
        '- 原实验13500条主记录保持原LP15预算及原始缺失分母；13组LP60重试独立存储。',
        '- 自由域80输入/160模型、两步扩展160输入/480模型、LP60重试13输入/156模型、过滤保护1204条分别归档。五类记录合计15500条，参数15358条。',
        '- F1和exact在非平凡末端簇上计算；缺失F1保留为零。MAE、R/X误差和AC误差以实际有指标的可用模型为条件，CSV记录分母。',
        '- 平方压降MAE单位p.u.^2；AC相对电压幅值RMSE原表为比例、图中乘100显示百分数。R/X为响应矩阵误差。',
        '- 日内设计矩阵诊断使用每日96点；N24/48/96/288是同一三天的嵌套采样，独立重复按数据bundle计算。',
        '- case33恢复视图保留N48原LP15与N288补充LP60的来源和预算；较大网络原始失败图仍保留。',
        '- 过滤保护和两步扩展为次要/探索性结果。完整协议、原始失败及原54项交付核验位于[原报告](../wzzt_paper_20261029/README.md)。', '',
        '## 记录与查询','',
        'model_records.csv.gz为长表。records.sqlite.zip包含可恢复数据库，模型、参数、输入路径、逐日诊断、AC拟合、搜索事件及计时通过键关联。数据库原文件在本机保留，Git中使用压缩归档。data_dictionary.csv给出字段和单位；methods.csv完整记录23个方法代码；method_settings.csv按cohort/variant记录35组算法配置与预算。', '',
        '```python',
        'from research_experiments.wzzt_paper.gallery_records import query, load_record',
        'folder = r"D:\\0-github_workspace\\Topo\\docs\\wzzt_paper_20261029_gallery"',
        'rows = query(folder, cohort="original", panel="main", case="paper15",',
        '             voltage_sigma=.001, N=48, bundle=0, variant="ridge_guard0")',
        'model = load_record(folder, rows.iloc[0].record_id, with_observations=True)',
        'R, X = model["R"], model["X"]',
        'data_config, method_config = model["configuration"], model["settings"]',
        '# cohort="lp60_retry" 显式读取60秒重试；cohort=None才跨批查询。',
        '```','',
        'SQL主要表：model_results、parameters、inputs、method_settings、methods、ac_fits、daily_diagnostics、physical_regimes、greedy_events、mip_events、timing、joint_diagnostics、filter_sensitivity；case33_recovered为带来源的恢复覆盖视图。主键record_id由cohort、job_id、variant构成；算法配置按cohort/variant关联。AC字段ac_key连接584条去重AC记录，模型表中共有876条AC选择关联。', '',
        '原始观测、网络、checkpoint及求解器记录保留在outputs/wzzt_paper_20261029_full，哈希与路径位于本索引及source_manifest.json。仅查询参数、绘图和指标不依赖原始观测；with_observations或重绘真拓扑示例需要本机原始路径。', '',
        '## 复现','',
        '```powershell',
        '& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_paper.gallery docs/wzzt_paper_20261029 outputs/wzzt_paper_20261029_full docs/wzzt_paper_20261029_gallery',
        '& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_paper.gallery --verify docs/wzzt_paper_20261029_gallery',
        '```','',
        'PDF逐页检查使用装有pypdf/Pillow的Python运行 `python -m research_experiments.wzzt_paper.gallery_pdf_qa <directory>`；脚本调用Poppler渲染。检查或补充目视记录后，使用 `gallery --verify --seal <directory>` 更新交付哈希并复核。仅修改文字字典可用 `gallery --refresh-metadata <directory>`，不会重新生成图形数据。','',
        '源码副本与环境记录在source/和environment.json。manifest.json绑定每张图、绘图数据和数据库归档；verification.json记录数值、关联、分母和归档核验。visual_review.json在实际目视检查后写入。','',
        '## 图目','']
    for category,g in registry.groupby('category',sort=False):
        lines+=['### '+category,'','|图号|内容|文件|','|---|---|---|']
        for r in g.to_dict('records'):lines.append(f"|{r['figure_id'].split('_')[0]}|{r['title']}|[PNG]({r['png']}) · [PDF]({r['pdf']}) · [CSV]({r['data']})|")
        lines+=['']
    for r in registry.to_dict('records'):
        lines+=['### '+r['figure_id']+' '+r['title'],'',r['caption'],'',f"![{r['title']}]({r['png']})",'']
    (p/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    cards=[]
    for r in registry.to_dict('records'):
        cards.append(f"<article data-category='{html.escape(r['category'])}'><h2>{html.escape(r['figure_id'])} {html.escape(r['title'])}</h2><a href='{r['png']}'><img loading='lazy' src='{r['png']}' alt='{html.escape(r['title'])}'></a><p>{html.escape(r['caption'])}</p><p><a href='{r['png']}'>PNG</a> · <a href='{r['pdf']}'>PDF</a> · <a href='{r['data']}'>绘图CSV</a></p></article>")
    page="""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>WZZT 结果图册</title><style>body{font:16px/1.65 system-ui,sans-serif;background:#f5f7fa;color:#1d2939;margin:0}header{padding:28px 4%;background:#18374c;color:white}main{max-width:1450px;margin:auto;padding:22px}nav{position:sticky;top:0;background:white;padding:12px;z-index:2;border-bottom:1px solid #ccd4df}a{color:#1767a0}header a{color:#d2edff}button{margin:4px;padding:8px 13px;border:1px solid #bac7d5;border-radius:6px;background:white;cursor:pointer}article{background:white;margin:24px 0;padding:20px;border-radius:10px;box-shadow:0 2px 9px #dde3eb}h2{font-size:20px}img{width:100%;height:auto}p{max-width:1100px}</style><header><h1>WZZT 数据记录与全面绘图</h1><p>已完成实验的36张主题图 · 15500条分批模型记录 · 每图附PNG / 矢量PDF / 绘图数据</p><p><a href='all_figures.pdf'>全部图PDF</a> · <a href='README.md'>说明与查询</a> · <a href='figure_index.csv'>图目</a> · <a href='data_dictionary.csv'>字段字典</a></p></header><nav>FILTER_BUTTONS</nav><main>CARDS</main><script>function filter(c){for(const a of document.querySelectorAll('article'))a.hidden=c!=='全部'&&a.dataset.category!==c}</script></html>"""
    buttons=''.join(f'<button onclick="filter(\'{c}\')">{c}</button>' for c in ['全部',*registry.category.drop_duplicates()])
    (p/'index.html').write_text(page.replace('FILTER_BUTTONS',buttons).replace('CARDS',''.join(cards)),encoding='utf-8')


def seal(destination):
    p=Path(destination)
    files={f.relative_to(p).as_posix():sha(f) for f in sorted(p.rglob('*')) if f.is_file() and f.name!='manifest.json' and f.name!='records.sqlite' and 'review' not in f.relative_to(p).parts}
    write(p/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),database_sha256=sha(p/'records.sqlite'),files=files))


def verify(destination):
    from .gallery_records import restore_database
    p=Path(destination).resolve();manifest=read(p/'manifest.json');checks=[]
    def check(name,passed):checks.append(dict(check=name,passed=bool(passed)))
    check('all recorded artifact bytes',all((p/n).is_file() and sha(p/n)==h for n,h in manifest['files'].items()))
    check('saved generator sources match current modules',all(sha(f)==sha(Path(__file__).parent/f.name) for f in (p/'source').glob('*.py')))
    sources=read(p/'source_manifest.json');check('all consumed source files unchanged',all(Path(n).is_file() and sha(Path(n))==d['sha256'] for n,d in sources.items()))
    db=restore_database(p)
    with sqlite3.connect(db) as conn:
        check('SQLite integrity',conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok')
        r=pd.read_sql_query('SELECT * FROM model_results',conn);params=pd.read_sql_query('SELECT * FROM parameters',conn)
        inputs=pd.read_sql_query('SELECT * FROM inputs',conn);ac=pd.read_sql_query('SELECT * FROM ac_fits',conn);covered=pd.read_sql_query('SELECT * FROM case33_recovered',conn)
        methods=pd.read_sql_query('SELECT * FROM methods',conn);settings=pd.read_sql_query('SELECT * FROM method_settings',conn)
    expected={'original':13500,'free_domain':160,'joint_extension':480,'lp60_retry':156,'postfilter_guard':1204}
    check('all 15500 records with separate cohort denominators',r.groupby('cohort').size().to_dict()==expected and len(r)==15500)
    check('all 1204 original input conditions retained',len(inputs)==1204 and inputs.job_id.nunique()==1204)
    check('all 23 method codes and 35 cohort-specific settings documented',set(r.variant)==set(methods.variant) and len(methods)==23 and len(settings)==35 and set(zip(r.cohort,r.variant))==set(zip(settings.cohort,settings.variant)))
    check('unique record keys and parameter bijection',not r.record_id.duplicated().any() and not params.record_id.duplicated().any() and set(params.record_id)==set(r[r.model_available.eq(1)].record_id))
    check('missing scores retain explicit zero-score policy',r.loc[r.model_available.eq(0),'clade_f1'].eq(0).all() and r.loc[r.model_available.eq(0),'score_policy'].eq('unavailable_retained_as_zero').all())
    check('876 AC links to 584 actual unique fits',r.ac_key.notna().sum()==876 and len(ac)==584 and set(r.ac_key.dropna())==set(ac.ac_key))
    check('case33 mixed-budget view retains 240 records',len(covered)==240 and covered.groupby('variant').job_id.nunique().eq(20).all())
    point_ok=True;legal=True;allfinite=True
    for row in params.itertuples():
        point=json.loads(row.point_json);labels=json.loads(row.labels_json);point_ok&=signature(point)==row.point_signature
        family=[frozenset(s) for s in point['supports']];legal&=len(set(family))==len(family) and all(frozenset([i]) in family for i in range(len(labels)))
        legal&=all(not(a&b) or a<=b or b<=a for i,a in enumerate(family) for b in family[i+1:])
        values=np.asarray(point['r']+point['x']);allfinite&=np.isfinite(values).all() and (values>=-1e-9).all() and (values<=2+1e-9).all()
    check('every parameter signature recomputed',point_ok);check('all stored clade families laminar with complete observed leaves',legal);check('all response coefficients finite and within recorded bounds',allfinite)
    records=pd.read_csv(p/'model_records.csv.gz',low_memory=False);check('CSV and SQLite keys and F1 identical',records.record_id.tolist()==r.record_id.tolist() and np.allclose(records.clade_f1,r.clade_f1,atol=1e-12,rtol=0))
    old_source=Path(next(n for n in sources if Path(n).name=='per_job.csv')).parent
    original=pd.read_csv(old_source/'per_job.csv');new=r[r.cohort=='original'].set_index(['job_id','variant']).loc[pd.MultiIndex.from_frame(original[['job_id','variant']])]
    check('all original structural scores unchanged',np.allclose(new.clade_f1,original.clade_f1,atol=1e-12,rtol=0) and np.array_equal(new.model_available,original.model_available.astype(int)))
    q=pd.read_csv(p/'plot_data/04_F1_heatmaps.csv');raw=original[(original.panel=='main')&original.variant.eq('ridge_union')].groupby(['voltage_sigma','N']).clade_f1.mean()
    plotted=q[q.variant=='ridge_union'].set_index(['voltage_sigma','N'])['mean'];check('all main ridge plot cells match direct raw-record means',np.allclose(plotted.loc[raw.index],raw,atol=1e-12,rtol=0))
    figures=pd.read_csv(p/'figure_index.csv');check('36 figures with PDF PNG and CSV',len(figures)==36 and all((p/row.png).is_file() and (p/row.pdf).is_file() and len(pd.read_csv(p/row.data))==row.plot_rows for row in figures.itertuples()))
    overview=pd.read_csv(p/'plot_data/01_main_methods.csv')
    check('overview plot exports only the eleven displayed 640-input methods',len(overview)==22 and overview.variant.nunique()==11 and overview.n_planned.eq(640).all())
    joint_plot=pd.read_csv(p/'plot_data/19_joint_extension.csv')
    check('joint plot exports all input deltas, block means and win/tie/loss counts',len(joint_plot[joint_plot.row_type=='input'])==480 and len(joint_plot[joint_plot.row_type=='bundle'])==30 and
        len(joint_plot[joint_plot.row_type=='summary'])==3 and joint_plot[joint_plot.row_type=='summary'][['wins','ties','losses']].sum(axis=1).eq(160).all())
    pdf_check=p/'pdf_render_verification.json'
    if pdf_check.exists():
        receipt=read(pdf_check)
        check('independent PDF page counts and rendering',receipt['all_passed'] and receipt['pages']==36 and receipt['individual_PDFs']==36 and receipt['all_figures_sha256']==sha(p/'all_figures.pdf'))
    visual=p/'visual_review.json'
    if visual.exists():
        receipt=read(visual);check('visual review binds the current 36-page book',receipt['all_passed'] and receipt['PDF_pages_reviewed']==36 and receipt['book_sha256']==sha(p/'all_figures.pdf'))
    from PIL import Image
    dimensions=[]
    for row in figures.itertuples():
        with Image.open(p/row.png) as im:im.verify()
        with Image.open(p/row.png) as im:dimensions.append(im.width>=1600 and im.height>=900)
    check('all figure raster outputs decode at publication size',all(dimensions))
    for cohort in COHORTS:
        row=r[(r.cohort==cohort)&r.model_available.eq(1)].iloc[0];loaded=load_record(p,row.record_id,with_observations=True)
        check('cohort-aware reader '+cohort,loaded['record']['cohort']==cohort and loaded['R'].shape==loaded['X'].shape==(len(loaded['labels']),)*2)
        check('method and data settings linked '+cohort,loaded['settings']['cohort']==cohort and loaded['settings']['variant']==row.variant and 'root_meter_sigma' in loaded['configuration'])
        train_mae=linear_loss(loaded['train'],loaded['R'],loaded['X'])['mae']
        val_mae=linear_loss(loaded['validation'],loaded['R'],loaded['X'])['mae'];test_mae=linear_loss(loaded['test'],loaded['R'],loaded['X'])['mae']
        check('observation/parameter/score alignment '+cohort,abs(train_mae-loaded['point']['objective'])<1e-8 and
            abs(val_mae-loaded['record']['validation_mae'])<1e-8 and abs(test_mae-loaded['record']['test_mae'])<1e-8)
    missing=r[r.model_available.eq(0)].iloc[0]
    rejected=False
    try:load_record(p,missing.record_id)
    except ValueError:rejected=True
    check('unavailable model remains explicitly unavailable',rejected)
    example=query(p,cohort='original',panel='main',case='paper15',N=48,voltage_sigma=.001,bundle=0,variant='ridge_guard0')
    loaded=load_record(p,example.iloc[0].record_id,with_observations=True)
    check('exact 48-row training and 288-row independent observations',len(loaded['train'][0])==48 and len(loaded['validation'][0])==len(loaded['test'][0])==288)
    with zipfile.ZipFile(p/'records.sqlite.zip') as z:check('database archive member exact',z.namelist()==['records.sqlite'] and __import__('hashlib').sha256(z.read('records.sqlite')).hexdigest()==manifest['database_sha256'])
    unchanged=all(c['passed'] for c in checks if c['check'] in ('all consumed source files unchanged','all original structural scores unchanged'))
    summary=dict(all_passed=all(c['passed'] for c in checks),checks=checks,model_records=len(r),available_models=int(r.model_available.sum()),stored_parameter_records=len(params),figures=len(figures),AC_unique=len(ac),AC_links=int(r.ac_key.notna().sum()),original_experiment_unchanged=unchanged)
    write(p/'verification.json',summary);seal(p)
    print('GALLERY VERIFICATION',dict(all_passed=summary['all_passed'],checks=len(checks),failed=[c['check'] for c in checks if not c['passed']]),flush=True)
    if not summary['all_passed']:raise AssertionError('gallery verification failed')


def review_sheets(destination):
    from PIL import Image,ImageOps,ImageDraw
    p=Path(destination);folder=p/'review';folder.mkdir(exist_ok=True);figures=pd.read_csv(p/'figure_index.csv')
    for offset in range(0,len(figures),6):
        canvas=Image.new('RGB',(2400,2400),'#e9edf2');draw=ImageDraw.Draw(canvas)
        for i,row in enumerate(figures.iloc[offset:offset+6].itertuples()):
            with Image.open(p/row.png) as im:
                thumb=ImageOps.contain(im.convert('RGB'),(1180,735));x=(i%2)*1200+10+(1180-thumb.width)//2;y=(i//2)*800+40+(735-thumb.height)//2;canvas.paste(thumb,(x,y))
            draw.text(((i%2)*1200+15,(i//2)*800+10),row.figure_id,fill='black')
        canvas.save(folder/f'sheet_{offset//6+1:02d}.png')


def build(source,output,destination):
    p=Path(destination).resolve();sources=Sources(source,output);tables=create_snapshot(sources,p);g=Gallery(sources,p,tables)
    try:g.main();g.scenarios();g.components();g.diagnostics();g.computation();g.topology_examples()
    finally:g.pdf.close()
    documentation(g)
    with zipfile.ZipFile(p/'records.sqlite.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(p/'records.sqlite','records.sqlite')
    (p/'.gitignore').write_text('/records.sqlite\n/review/\n',encoding='utf-8');(p/'.gitattributes').write_text('* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n',encoding='utf-8')
    source_dir=p/'source';source_dir.mkdir(exist_ok=True)
    for name in ('gallery.py','gallery_records.py','gallery_pdf_qa.py','gallery_settings.py','statistics.py'):shutil.copy2(Path(__file__).parent/name,source_dir/name)
    write(p/'source_manifest.json',sources.used)
    write(p/'environment.json',dict(utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,bootstrap_seed=20261029,bootstrap_repeats=20000,source_directory=str(sources.source),raw_output_directory=str(sources.output),no_new_estimator_runs=True))
    review_sheets(p);seal(p);verify(p)


def refresh_metadata(destination):
    """Refresh registries without recomputing points or re-rendering unchanged figures."""
    from types import SimpleNamespace
    p=Path(destination).resolve()
    with sqlite3.connect(p/'records.sqlite') as conn:
        names=[r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        tables={name:pd.read_sql_query('SELECT * FROM "'+name+'"',conn) for name in names}
    documentation(SimpleNamespace(path=p,t=tables,figures=pd.read_csv(p/'figure_index.csv').to_dict('records')))
    with zipfile.ZipFile(p/'records.sqlite.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(p/'records.sqlite','records.sqlite')
    for name in ('gallery.py','gallery_records.py','gallery_pdf_qa.py','gallery_settings.py','statistics.py'):shutil.copy2(Path(__file__).parent/name,p/'source'/name)
    seal(p);verify(p)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');p.add_argument('--verify',action='store_true');p.add_argument('--seal',action='store_true');p.add_argument('--refresh-metadata',action='store_true');a=p.parse_args()
    if a.refresh_metadata:refresh_metadata(a.paths[0])
    elif a.verify:
        if a.seal:seal(a.paths[0])
        verify(a.paths[0])
    elif len(a.paths)==3:build(*a.paths)
    else:p.error('supply source output destination, or --verify destination')
