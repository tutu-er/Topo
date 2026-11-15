"""Show paired AC recovery and settlement from the verified aggregate report."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ... import io


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    data=io.read(a.report)
    if data['status']!='complete':raise ValueError('Verified completed report required')
    cases=data['cases'];modes=('normal_fixed','mgf_fixed','mgf_forecast');titles=('正态约束','固定功率指数约束','含预测误差指数约束')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(len(cases),3,figsize=(12.5,3.2*len(cases)),squeeze=False)
    for i,case in enumerate(cases):
        groups=case['strategies'];label=case['id'].replace('__2024-01-04__s2__','  ').replace('m2p0','噪声×2').replace('m0p5','噪声×0.5').replace('m1p0','噪声×1')
        for j,mode in enumerate(modes):
            ax=axes[i,j];keys=('no_dfs',mode,'single_'+mode)
            for x,(key,color) in enumerate(zip(keys,('#999999','#477bb6','#df963b'))):
                g=groups[key];fallback=g['zero_fallback_rows']>0
                ax.bar(x,g['recovered'],width=.62,color=color,hatch='//' if fallback else None,edgecolor='#555555' if fallback else 'none')
                ax.text(x,g['recovered']+.4,f"{g['recovered']}/{g['n']}",ha='center')
                ax.text(x,-2.0,f"¥{g['settled_total_yuan']['mean']:.2f}"+(' 回退' if fallback else ''),ha='center',fontsize=9)
            n=max(groups[k]['n'] for k in keys);ax.set_ylim(-3,n+3);ax.set_yticks(np.arange(0,n+1,max(1,n//4)));ax.set_xticks(range(3),('零采购','原多目标','历史单目标'))
            ax.set_title(label+'\n'+titles[j],fontsize=10);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
            if j==0:ax.set_ylabel('AC执行后内部拓扑正确次数')
    fig.suptitle('历史单目标与原多目标：冻结参数的配对AC比较',fontsize=15,y=.995)
    fig.text(.5,.012,'同一输入采用相同预测、设备响应及量测噪声；费用为平均预约加调用总费。\n内部拓扑与完整响应支持分别报告；同物理日重复路径属于开发重放。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.97));a.out.mkdir(parents=True,exist_ok=False)
    fig.savefig(a.out/'paired_ac.png',dpi=180,bbox_inches='tight');fig.savefig(a.out/'paired_ac.pdf',bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
