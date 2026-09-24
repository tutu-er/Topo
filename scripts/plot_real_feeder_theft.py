"""Reproducible scientific figure for the 65037 feeder and null-loss finding."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
import networkx as nx

try:
    from scripts.real_feeder_joint_theft import read_feeder,ROOT
except ModuleNotFoundError:
    from real_feeder_joint_theft import read_feeder,ROOT


def main():
    output=ROOT/'theft_wzzt/outputs/theft_real65037'
    font=FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc')
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10})
    feeder=read_feeder();tree=nx.bfs_tree(feeder.graph,0)
    positions={};cursor=[0]
    def place(node,depth):
        children=list(tree.successors(node))
        if not children:x=cursor[0];cursor[0]+=1
        else:x=sum(place(child,depth+1) for child in children)/len(children)
        positions[node]=(x,-depth);return x
    place(0,0)
    fig,(ax,bar)=plt.subplots(1,2,figsize=(15,7),gridspec_kw={'width_ratios':[2.1,1]})
    nx.draw_networkx_edges(tree,positions,ax=ax,arrows=False,edge_color='#b0b8c1',width=1.3)
    nx.draw_networkx_nodes(tree,positions,ax=ax,node_size=125,node_color='#edf0f3',edgecolors='#a0a7b0',linewidths=.7)
    nx.draw_networkx_nodes(tree,positions,ax=ax,nodelist=feeder.meters,node_size=180,node_color='#2763a7',edgecolors='white')
    nx.draw_networkx_nodes(tree,positions,ax=ax,nodelist=[0],node_size=240,node_color='#1c665b',node_shape='s')
    nx.draw_networkx_nodes(tree,positions,ax=ax,nodelist=[5],node_size=260,node_color='#d97820',node_shape='D')
    region=[39,41,42]
    nx.draw_networkx_nodes(tree,positions,ax=ax,nodelist=region,node_size=280,node_color='none',edgecolors='#d97820',linewidths=2)
    for node,(x,y) in positions.items():
        ax.text(x,y,str(node),ha='center',va='center',fontsize=6.8,color='white' if node in feeder.meters or node in [0,5] else '#334155')
    ax.set_title('65037：53 母线、14 个有功率记录的电表节点',fontproperties=font,fontsize=13,pad=18)
    handles=[Line2D([0],[0],marker='s',color='none',markerfacecolor='#1c665b',markersize=9,label='已知电压参考点'),
             Line2D([0],[0],marker='o',color='none',markerfacecolor='#2763a7',markersize=9,label='电表节点'),
             Line2D([0],[0],marker='D',color='none',markerfacecolor='#d97820',markersize=9,label='合成窃电位置：母线 5')]
    ax.legend(handles=handles,loc='lower left',prop=font,frameon=False)
    ax.text(.02,1.01,'橙圈：所选 z 对应电表 {39, 41, 42}；图为拓扑示意，非地理位置。',transform=ax.transAxes,fontproperties=font,fontsize=9,color='#4b5563')
    ax.axis('off')
    old=json.loads((output/'v1_scalar/seed0_unbalanced_clean.json').read_text())
    new=json.loads((output/'v3_tree_selection/seed0_unbalanced_clean.json').read_text())
    values=[old['amplitude_mean_kw'],new['amplitude_mean_kw']]
    bars=bar.bar([0,1],values,width=.6,color=['#8996a8','#2763a7'])
    bar.set_xticks([0,1]);bar.set_xticklabels(['标量线损估计','按相线损＋正常历史校准'],fontproperties=font,fontsize=9)
    bar.set_ylabel('估计额外功率均值 / kW',fontproperties=font)
    bar.set_title('无窃电的不平衡窗口（种子 0）',fontproperties=font,fontsize=13,pad=18)
    bar.set_ylim(0,max(values)*1.25);bar.spines[['top','right']].set_visible(False)
    bar.yaxis.grid(True,alpha=.2);bar.set_axisbelow(True)
    for rect,value in zip(bars,values):bar.text(rect.get_x()+rect.get_width()/2,value+.3,f'{value:.3f}',ha='center',fontsize=12)
    bar.text(.02,.96,'真实额外功率为 0 kW\n幅值残差不等于窃电证据\n本图不是误报率或置信度统计',transform=bar.transAxes,va='top',fontproperties=font,fontsize=10,color='#475569')
    fig.subplots_adjust(left=.035,right=.98,bottom=.12,top=.85,wspace=.3)
    fig.savefig(output/'feeder_and_loss_diagnostic.png',dpi=190)
    fig.savefig(output/'feeder_and_loss_diagnostic.svg')
    print(output/'feeder_and_loss_diagnostic.png')


if __name__=='__main__':main()
