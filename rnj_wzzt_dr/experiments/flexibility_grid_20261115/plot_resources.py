"""Plot sealed resource offers on the four physical trees, for offline display.

Reads JSON/CSV and the already sealed forecast. No optimizer, identifier or AC
engine is imported. The physical tree is never passed back into procurement.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from ... import io


ROOT = Path(__file__).resolve().parents[3]
CASES = ('paper15', 'flynn16', 'pengwah18', 'soumalas11')
TITLES = dict(paper15='Paper15', flynn16='Flynn16', pengwah18='Pengwah18', soumalas11='Soumalas11')
COLORS = dict(pv='#b87908', ev='#16846a', thermal='#bd4650', reactive='#276cba')
MARKERS = dict(pv='^', ev='s', thermal='D')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                    ensure_ascii=False,allow_nan=False).encode()).hexdigest()


class Sources:
    def __init__(self):
        self.refs = {}

    def file(self, path, expected=None):
        path = Path(path).resolve(); actual = io.ref(path)
        if expected is not None and actual['sha256'] != expected['sha256']:
            raise ValueError('Changed sealed input: '+str(path))
        self.refs[str(path)] = actual
        return path

    def read(self, path, expected=None):
        return io.read(self.file(path, expected))

    def reference(self, reference):
        return self.read(reference['path'], reference)

    def csv(self, path, expected):
        with self.file(path,expected).open(encoding='utf-8-sig',newline='') as stream:
            return list(csv.DictReader(stream))

    def verify(self):
        for reference in self.refs.values():
            self.file(reference['path'],reference)


def tree_layout(buses, edges, root, terminals):
    """Deterministic leaf-order tree drawing; horizontal lengths are schematic."""
    ids = {int(b['bus_id']) for b in buses}
    if len(ids) != len(buses) or len(edges) != len(ids)-1:
        raise ValueError('Expected one radial physical tree')
    children = defaultdict(list); parent = {}
    for a,b in edges:
        if a not in ids or b not in ids or b in parent or b == root:
            raise ValueError('Invalid physical parent relationship')
        parent[b]=a; children[a].append(b)
    if set(parent) != ids-{root}:
        raise ValueError('Disconnected physical tree')
    if {node for node in ids if not children[node]} != set(terminals):
        raise ValueError('Physical leaves and measured terminal labels differ')
    visited=set(); descendants={}; depth={root:0}
    def walk(node):
        if node in visited: raise ValueError('Cycle in physical tree')
        visited.add(node)
        leaves=[]
        for child in children[node]:
            depth[child]=depth[node]+1; leaves.extend(walk(child))
        descendants[node]=leaves or [node]
        return descendants[node]
    walk(root)
    if visited != ids: raise ValueError('Unvisited physical bus')
    for node in ids:
        children[node].sort(key=lambda child:min(descendants[child]))
    order=[]
    def collect(node):
        if not children[node]:order.append(node)
        for child in children[node]:collect(child)
    collect(root)
    y={node:float(len(order)-1-i) for i,node in enumerate(order)}
    deepest=max(depth[node] for node in ids-set(terminals))
    positions={}
    for node in ids:
        if node in terminals:
            positions[node]=(.54,y[node])
        else:
            values=[y[leaf] for leaf in descendants[node]]
            positions[node]=(.035+.375*depth[node]/max(1,deepest),.5*(min(values)+max(values)))
    return positions,order


def collect(run):
    run=Path(run).resolve(); source=Sources()
    complete=source.read(run/'complete.json')
    if complete.get('status') != 'complete' or complete.get('source_unchanged') is not True:
        raise ValueError('Completed unchanged grid run required')
    protocol=source.read(run/'protocol.json')
    entries=source.reference(protocol['inputs'])
    if len(entries)!=36 or protocol.get('smoke'):
        raise ValueError('The formal36-cell resource catalogue is required')
    data_index=source.reference(protocol['input_index'])
    data_protocol=source.reference(data_index['protocol'])
    frozen_sources={str(Path(r['path']).resolve()):r for r in data_protocol['sources']}
    result=[]; node_rows=[]; resource_rows=[]
    for case in CASES:
        cells=[e for e in entries if e['case']==case]
        if len(cells)!=9 or len({e['original_id'] for e in cells})!=1:
            raise ValueError('Each world must have all nine fixed grid cells')
        if {(e['stride'],e['meter_multiplier']) for e in cells}!={(s,m) for s in (1,2,4) for m in (.5,1.,2.)}:
            raise ValueError('Incomplete sampling/noise conditions')
        e=cells[0];spec=source.reference(e['resource_spec']);context=source.reference(e['resource_context'])
        resources=e['p_resources']+e['q_resources']
        if resources != spec['p_resources']+spec['q_resources'] or spec['status']!='ready':
            raise ValueError('Resource list differs from its original sealed specification')
        config_digest=canonical_hash(resources);forecast=None;labels=None;cell_refs=[]
        for cell in cells:
            if (cell['p_resources']!=e['p_resources'] or cell['q_resources']!=e['q_resources']
                    or cell['resource_spec']!=e['resource_spec'] or cell['base_kw']!=e['base_kw']):
                raise ValueError('Offers or base changed across grid conditions')
            directory=run/'fraction0.10'/cell['id']/'forecast00'
            seal=source.read(directory/'seal.json')
            config=source.reference(seal['resource_case'])
            p=source.file(seal['planner']['path'],seal['planner'])
            with np.load(p,allow_pickle=False) as data:
                current=data['forecast'].copy()
            actual_labels=list(map(int,config['expected_terminal_labels']))
            if labels is None:labels=actual_labels;forecast=current
            elif actual_labels!=labels or not np.array_equal(current,forecast):
                raise ValueError('Forecast or terminal ordering changed across paired grid')
            actual=config['resources']
            if len(actual)!=len(resources):raise ValueError('Resource count changed in executed case')
            for requested,used in zip(resources,actual):
                if any(used.get(k)!=v for k,v in requested.items()):
                    raise ValueError('Resource definition changed in executed case')
            if config['action_units']!=['kW']*len(e['p_resources'])+['kvar']*len(e['q_resources']):
                raise ValueError('Mixed P/Q action units changed')
            cell_refs.append(dict(id=cell['id'],seal=io.ref(directory/'seal.json'),
                                  resource_case=seal['resource_case'],planner=seal['planner']))
        case_root=ROOT/'topology_identification_standalone'/'cases'/case
        paths={name:case_root/name for name in ('metadata.json','buses.csv','branches.csv')}
        for path in paths.values():
            if str(path.resolve()) not in frozen_sources:raise ValueError('Physical table absent from data source seal')
        metadata=source.read(paths['metadata.json'],frozen_sources[str(paths['metadata.json'].resolve())])
        buses=source.csv(paths['buses.csv'],frozen_sources[str(paths['buses.csv'].resolve())])
        branches=source.csv(paths['branches.csv'],frozen_sources[str(paths['branches.csv'].resolve())])
        edges=[(int(b['from_bus']),int(b['to_bus'])) for b in branches if b['status'].lower()=='true']
        root=int(metadata['root_bus']);positions,order=tree_layout(buses,edges,root,set(labels))
        if set(labels)!={int(b['bus_id']) for b in buses if b['is_terminal'].lower()=='true'}:
            raise ValueError('Resource terminal and physical bus identities differ')
        if any(r['node_label'] not in labels for r in resources):
            raise ValueError('Root or hidden resource placement is forbidden')
        clock=(8+np.arange(24)+.5)%24
        export=np.maximum(0.,-forecast[:,:len(labels)]*e['base_kw']).reshape(24,4,len(labels)).min(1)
        detail=[]
        for r in resources:
            kind=r['kind'];reactive=kind=='reactive';cap=float(r['capacity_kvar'] if reactive else r['contract_kw'])
            hours=[0,24] if kind in ('thermal','reactive') else [10,14] if kind=='pv' else [18,7]
            available=np.ones(24,bool) if hours==[0,24] else ((clock>=hours[0])&(clock<hours[1]) if hours[0]<hours[1] else ((clock>=hours[0])|(clock<hours[1])))
            lower=np.where(available,-cap,0.);upper=np.where(available,cap,0.)
            if kind=='pv':
                lower[:]=0.;upper=np.where(available,np.minimum(cap,export[:,labels.index(r['node_label'])]),0.)
            elif reactive and r.get('available_clock_hours',[0,24])!=[0,24]:
                raise ValueError('This figure expects the sealed all-day Q contract')
            info=dict(case=case,original_id=e['original_id'],resource_id=r['id'],node_label=r['node_label'],kind=kind,
                unit='kvar' if reactive else 'kW',maximum_offered_capacity=cap,
                forecast_envelope_lower_minimum=float(lower.min()),forecast_envelope_upper_maximum=float(upper.max()),
                available_clock_hours=hours,clock_timezone='Europe/Berlin',decision_start_local_hour=8,
                registered_source_kw=r.get('registered_kw'),scaled_registered_kw=r.get('scaled_registered_kw'),
                assumed_equipment_rating_kvar=r.get('assumed_equipment_rating_kvar'),
                source_to_network_scale=r.get('source_to_network_scale'),feeder_id=r.get('feeder_id'),
                registration_date=r.get('registration_date'),
                buffer_kwh=(2.*cap if kind=='ev' else cap if kind=='thermal' else None),
                retention_factor_per_hour=1. if kind=='ev' else .98 if kind=='thermal' else None,
                terminal_energy_restoration=kind in ('ev','thermal'),
                sign='positive reactive absorption; negative reactive injection' if reactive else
                     'positive net-demand increment (PV curtailment)' if kind=='pv' else 'positive extra consumption; negative demand reduction',
                activation_price_yuan_per_energy=r['fee_positive_yuan_per_kvarh'] if reactive else r['fee_positive_yuan_per_kwh'],
                illustrative_capacity_price_yuan_per_native_capacity_day=float(protocol['reservation_prices']['Q' if reactive else 'P']))
            detail.append(info);resource_rows.append(info)
        for bus in buses:
            node=int(bus['bus_id']);node_resources=[r for r in detail if r['node_label']==node]
            def value(kind,field):return sum(r[field] for r in node_resources if r['kind']==kind)
            node_rows.append(dict(case=case,original_id=e['original_id'],node_label=node,
                role='root' if node==root else 'observed_terminal' if node in labels else 'hidden_internal',
                PV_contract_kw=value('pv','maximum_offered_capacity'),
                PV_forecast_available_peak_kw=value('pv','forecast_envelope_upper_maximum'),
                EV_contract_kw=value('ev','maximum_offered_capacity'),thermal_contract_kw=value('thermal','maximum_offered_capacity'),
                Q_bidirectional_contract_kvar=value('reactive','maximum_offered_capacity'),
                Q_assumed_equipment_kvar=sum(r['assumed_equipment_rating_kvar'] for r in node_resources if r['kind']=='reactive'),
                fixed_grid_conditions=9))
        result.append(dict(case=case,original_id=e['original_id'],history_start=e['start_date'],base_kw=e['base_kw'],
            root_bus=root,terminal_labels=labels,buses=[dict(id=int(b['bus_id']),role=b['bus_type']) for b in buses],
            edges=edges,schematic_positions={str(k):list(v) for k,v in positions.items()},leaf_order=order,
            resources=detail,node_rows=[r for r in node_rows if r['case']==case],
            terminal_count=len(labels),hidden_count=len(buses)-len(labels)-1,
            P_contract_total_kw=sum(r['contract_kw'] for r in e['p_resources']),
            non_PV_contract_total_kw=sum(r['contract_kw'] for r in e['p_resources'] if r['kind']!='pv'),
            Q_contract_total_kvar=sum(r['capacity_kvar'] for r in e['q_resources']),
            Q_resource_nodes=len(e['q_resources']),PV_forecast_available_peak_sum_kw=sum(r['forecast_envelope_upper_maximum'] for r in detail if r['kind']=='pv'),
            identical_resources_across_nine_cells=True,identical_forecast_across_nine_cells=True,
            resource_definition_hash=config_digest,original_resource_spec=e['resource_spec'],resource_context=e['resource_context'],
            physical_sources={name:io.ref(path) for name,path in paths.items()},cell_sources=cell_refs,
            physical_topology_scope='Offline drawing only; no resource relocation or new optimization',
            layout_scope='Radial connectivity is exact; positions and line lengths are schematic'))
    for name in ('experiments/measured_pq_flex_20261114/domain.py','experiments/reactive_flex_20261114/domain.py',
                 'experiments/distribution_panel_20261115/resources.py'):
        path=ROOT/'rnj_wzzt_dr'/name
        source.file(path)
    source.verify()
    return result,node_rows,resource_rows,source,protocol


def figure(networks,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.font_manager import FontProperties
    from matplotlib import font_manager
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if not font.exists():raise FileNotFoundError('Chinese figure font required: '+str(font))
    font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(23,16))
    for index,(ax,network) in enumerate(zip(axes.ravel(),networks)):
        positions={int(k):tuple(v) for k,v in network['schematic_positions'].items()}
        nodes={r['node_label']:r for r in network['node_rows']};n=network['terminal_count']
        for i,node in enumerate(network['leaf_order']):
            y=positions[node][1]
            if i%2==0:ax.fill_between([.57,1.335],y-.43,y+.43,color='#f3f6f9',zorder=0)
        for a,b in network['edges']:
            xa,ya=positions[a];xb,yb=positions[b]
            ax.plot([xa,xb],[ya,yb],color='#8a96a3',lw=1.05,zorder=1)
        for bus in network['buses']:
            node=bus['id'];x,y=positions[node]
            if node==network['root_bus']:
                ax.scatter([x],[y],s=65,marker='D',color='#263442',zorder=5)
                ax.annotate('根 '+str(node),(x,y),xytext=(-3,-17),textcoords='offset points',ha='center',fontsize=9,color='#263442')
            elif node not in network['terminal_labels']:
                ax.scatter([x],[y],s=36,facecolor='#adb7c1',edgecolor='white',linewidth=.6,zorder=4)
                ax.annotate(str(node),(x,y),xytext=(5,3),textcoords='offset points',fontsize=8.5,color='#5b6570')
            else:
                q=nodes[node]['Q_bidirectional_contract_kvar']
                if q:ax.scatter([x],[y],s=50+q*19,facecolor='#d7e7fa',edgecolor=COLORS['reactive'],linewidth=.8,zorder=3)
                ax.scatter([x],[y],s=23,facecolor='white',edgecolor='#263442',linewidth=.9,zorder=4)
                ax.text(.595,y,str(node),ha='left',va='center',fontsize=10.5,color='#263442')
                ax.text(.735,y,'—' if not q else f'±{q:g}',ha='center',va='center',fontsize=10,color=COLORS['reactive'])
                for kind,xc,field in [('pv',.835,'PV_contract_kw'),('ev',1.045,'EV_contract_kw'),('thermal',1.205,'thermal_contract_kw')]:
                    cap=nodes[node][field]
                    if not cap:continue
                    ax.scatter([xc-.022],[y],s=34,marker=MARKERS[kind],color=COLORS[kind],zorder=5)
                    label=f"{cap:.2f}/{nodes[node]['PV_forecast_available_peak_kw']:.2f}" if kind=='pv' else f'±{cap:.2f}'
                    ax.text(xc,y,label,ha='left',va='center',fontsize=9.3,color=COLORS[kind])
        for x,label,color in [(.595,'终端', '#263442'),(.735,'Q / kvar',COLORS['reactive']),
                (.87,'PV / kW',COLORS['pv']),(1.07,'EV / kW',COLORS['ev']),(1.23,'热水 / kW',COLORS['thermal'])]:
            ax.text(x,n-.05,label,ha='center',va='center',fontsize=10,fontweight='bold',color=color)
        title=f"({chr(97+index)}) {TITLES[network['case']]} · {n}个终端 / {network['hidden_count']}个隐藏节点"
        ax.set_title(title,loc='left',fontsize=13.5,pad=33,color='#172c40',fontweight='bold')
        ax.text(.0,1.025,f"历史窗口起日 {network['history_start']}   |   Q合同 ±{network['Q_contract_total_kvar']:g} kvar，{network['Q_resource_nodes']}个供给点",
                transform=ax.transAxes,fontsize=10,color='#536575',va='bottom')
        ax.set_xlim(-.02,1.34);ax.set_ylim(-.8,n+.55);ax.axis('off')
    handles=[Line2D([],[],marker='D',color='none',markerfacecolor='#263442',markersize=7,label='根节点'),
        Line2D([],[],marker='o',color='none',markerfacecolor='#adb7c1',markersize=7,label='隐藏零注入节点'),
        Line2D([],[],marker='o',color='none',markerfacecolor='white',markeredgecolor='#263442',markersize=7,label='可测终端'),
        Line2D([],[],marker='o',color='none',markerfacecolor='#d7e7fa',markeredgecolor=COLORS['reactive'],markersize=11,label='Q合同容量（圆大小）'),
        *[Line2D([],[],marker=MARKERS[k],color='none',markerfacecolor=COLORS[k],markersize=7,label=t)
          for k,t in [('pv','光伏弃光合同'),('ev','EV充电调整'),('thermal','电热水调整')]]]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.943),ncol=7,frameon=False,fontsize=11,
               handletextpad=.5,columnspacing=1.6)
    fig.suptitle('四个实验网络的固定灵活性资源配置',fontsize=19,fontweight='bold',y=.985,color='#172c40')
    fig.text(.5,.958,'每网九格共用相同资源目录和预测；容量为可提供上限，图中没有按实际收益调整布点。',ha='center',fontsize=11,color='#536575')
    fig.text(.5,.066,'PV数字 = 合同容量 / 本预测可用峰值；12个PV合同的当前峰值均为0。PV窗口10–14，EV窗口18–次日07，热水与Q全天。',ha='center',fontsize=11,color='#263442')
    fig.text(.5,.042,'净负荷正号：+P增加用电或弃光，−P减少用电；+Q吸收无功，−Q注入补偿。时钟为当地Europe/Berlin，决策日从08:00开始。',ha='center',fontsize=11,color='#263442')
    fig.text(.5,.018,'Q为另行假设的独立补偿合同，并非登记数据证明的逆变器余量；物理树仅用于事后展示，线长与位置为示意。',ha='center',fontsize=10.5,color='#536575')
    fig.subplots_adjust(left=.035,right=.985,bottom=.105,top=.87,wspace=.11,hspace=.24)
    fig.savefig(out/'resource_configuration.png',dpi=180,facecolor='white')
    fig.savefig(out/'resource_configuration.pdf',facecolor='white')
    plt.close(fig)


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(source_run,out):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('Keep previous figure and receipts: '+str(out))
    networks,nodes,resources,sources,protocol=collect(source_run)
    out.mkdir(parents=True)
    figure(networks,out)
    write_csv(out/'configuration_nodes.csv',nodes)
    write_csv(out/'configuration_resources.csv',resources)
    sources.file(__file__);sources.verify()
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=io.ref(__file__),
        source_run=str(Path(source_run).resolve()),phase='offline_resource_configuration_display',
        fixed_grid_conditions_per_world=9,fixed_resources_verified=True,fixed_forecasts_verified=True,
        new_AC=False,new_optimization=False,new_fitting=False,sources_unchanged=True,
        networks=networks,resource_count=len(resources),physical_node_count=len(nodes),
        sign_convention=dict(P='positive net consumption; PV curtailment is a positive delta',
                             Q='positive absorption; negative injection/compensation'),
        distinction='Registered source capacities are scaled metadata; participation and independently available Q equipment/contracts are declared assumptions. Offered capacity, per-forecast envelope and purchased/delivered action are different.',
        energy_constraints='EV zero daily net energy and 2-hour contract buffer. Hot water retention0.98/hour, 1-hour buffer and terminal restoration. Q has no daily energy or ramp constraint.',
        probability_or_recovery_comparison=False,input_refs=list(sources.refs.values()))
    io.write(out/'configuration.json',receipt)
    notes=['# 四网资源配置图','', '图展示原物理拓扑上的封存供给目录，不展示优化动作或恢复效果。每网九格的P/Q资源列表、合同容量和封存预测逐项相同；未比较不同布点。','',
        '|网络|可测终端|隐藏节点|P合同总量kW|其中非PV合同kW|Q合同±kvar|Q节点数|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for n in networks:notes.append(f"|{TITLES[n['case']]}|{n['terminal_count']}|{n['hidden_count']}|{n['P_contract_total_kw']:.4f}|{n['non_PV_contract_total_kw']:.4f}|{n['Q_contract_total_kvar']:g}|{n['Q_resource_nodes']}|")
    notes+=['','全部12个PV合同在各自10–14当地时段的预测净反送代理上限为0。图中PV为合同/预测可用峰值；不把登记容量当成本日出力证明。EV与热水的±功率上限仍受逐时能量缓冲与末端恢复约束。',
        'Q采用独立双向补偿器假设，可吸收或注入无功。正Q增加节点无功负荷，负Q为补偿注入；无功不附带有功变化。全天能力为声明合同，没有由PV登记kW推导夜间逆变器裕度。',
        'P调用单价PV0.30、EV0.10、热水0.15元/kWh，乘既有分时系数；Q0.05元/kvarh。容量结算P/Q均为示例0.5元/(kW或kvar·日)，按请求峰值购买。图中的供给上限不是购买量，也不含设备投资。',
        '根与隐藏节点未布资源；全部合同在已量测终端。物理表、资源来源、36格seal/planner哈希及配置一致性见configuration.json。图的空间位置与支路长度为排版示意，连接关系来自原物理bus/branch表。','']
    (out/'README.md').write_text('\n'.join(notes),encoding='utf8')
    shutil.copyfile(__file__,out/'plot_resources.py')
    io.write(out/'complete.json',dict(status='complete',sources_unchanged=True,
        png=io.ref(out/'resource_configuration.png'),pdf=io.ref(out/'resource_configuration.pdf'),
        configuration=io.ref(out/'configuration.json'),nodes=io.ref(out/'configuration_nodes.csv'),
        resources=io.ref(out/'configuration_resources.csv'),notes=io.ref(out/'README.md')))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('outputs/flexibility_grid_execution_20261115_v1'))
    parser.add_argument('--out',type=Path,default=Path('outputs/flexibility_resource_configuration_20261115_v1'))
    args=parser.parse_args();answer=run(args.run,args.out)
    print(dict(networks=len(answer['networks']),nodes=answer['physical_node_count'],resources=answer['resource_count'],
               fixed_resources_verified=answer['fixed_resources_verified'],sources_unchanged=answer['sources_unchanged']))
