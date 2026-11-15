"""Plot paired development outcomes from the independently audited report."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    data=json.loads(a.report.read_text(encoding='utf8'))
    cases=data['cases']
    modes=('normal_fixed','mgf_fixed','mgf_forecast')
    titles=('正态约束','固定功率指数约束','含预测误差指数约束')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(len(cases),3,figsize=(12.8,3.25*len(cases)),squeeze=False)
    colors=('#8b97a6','#4378b9','#d88e35')
    for i,case in enumerate(cases):
        groups=case['strategies']
        name=case['id'].replace('__2024-01-04__s2__','  ').replace('m0p5','噪声 ×0.5').replace('m1p0','噪声 ×1').replace('m2p0','噪声 ×2')
        for j,(mode,title) in enumerate(zip(modes,titles)):
            ax=axes[i,j]
            keys=(mode,mode+'__pq',mode+'__response')
            ax.set_title(name+'\n'+title,fontsize=10)
            ax.axhline(groups['no_dfs']['recovered'],ls='--',color='#555555',lw=1,label='零采购')
            n=max(groups[key]['n'] for key in keys if key in groups)
            for x,key in enumerate(keys):
                if key not in groups:
                    continue
                g=groups[key];fallback=g.get('zero_fallback_rows',0)>0
                ax.bar(x,g['recovered'],width=.62,color=colors[x],hatch='//' if fallback else None,
                    edgecolor='#555555' if fallback else 'none')
                ax.text(x,g['recovered']+.35,f"{g['recovered']}/{g['n']}",ha='center',fontsize=10)
                fee=g['settled_total_yuan']['mean']
                ax.text(x,-2.1,f'¥{fee:.2f}'+(' 回退' if fallback else ''),ha='center',fontsize=9)
            ax.set_xticks(np.arange(3),['原方案','P/Q 预算','响应预算'])
            ax.set_ylim(-3,n+3);ax.set_yticks(np.arange(0,n+1,max(1,n//4)))
            ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
            if j==0:ax.set_ylabel('内部拓扑恢复次数')
            if i==0 and j==2:ax.legend(frameon=False,fontsize=9)
    fig.suptitle('冻结 R/X：限制调度方向后的配对比较',fontsize=16,y=.995)
    fig.text(.5,.006,'同一输入的各策略使用相同随机路径；费用为平均实际结算总费用。斜线表示报价失败后的零采购回退。\n内部拓扑指标不含公共项和全部响应支持的完全一致性；这是已知开发输入比较。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.97))
    a.out.mkdir(parents=True,exist_ok=False)
    fig.savefig(a.out/'recovery_cost.png',dpi=180,bbox_inches='tight')
    fig.savefig(a.out/'recovery_cost.pdf',bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
