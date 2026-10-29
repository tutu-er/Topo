import numpy as np


def random_network(n,topology):
    from rnj_wzzt.data.paper_style_case_bank import _make_case
    seed=202610290+n*100+topology;rng=np.random.default_rng(seed);edges=[];hidden=[]
    leaves=list(range(1001,1001+n));rng.shuffle(leaves)
    def expand(parent,members):
        if len(members)==1:
            edges.append((parent,members[0],float(rng.uniform(15,45)),'service'));return
        child=2+len(hidden);hidden.append(child)
        edges.append((parent,child,float(rng.uniform(25,90)),'backbone'))
        k=int(rng.integers(2,min(4,len(members))+1));cuts=sorted(rng.choice(np.arange(1,len(members)),k-1,replace=False))
        for group in np.split(np.asarray(members),cuts):expand(child,group.tolist())
    # At least two root laterals; no unary hidden vertices are introduced.
    k=int(rng.integers(2,4));cuts=sorted(rng.choice(np.arange(1,n),k-1,replace=False))
    for group in np.split(np.asarray(leaves),cuts):expand(1,group.tolist())
    terminals=[(v,float(rng.uniform(2,6)*15/n),float(rng.uniform(.6,1.7)*15/n)) for v in sorted(leaves)]
    net=_make_case(case_name=f'random{n}_t{topology}',hidden=hidden,terminals=terminals,edges=edges,
                   impedance_scale=1.,note='new random terminal-only topology; no outcome-based selection')
    net.metadata.update(topology_seed=seed,generator='random arity 2..4 partitions; fixed impedance/terminal-load distributions')
    return net


def case33_network():
    from terminal_case33.data.case33bw_raw import load_raw_case33bw
    from terminal_case33.scenario.terminalize import terminalize_case33
    from rnj_wzzt.models.network import TerminalizedNetwork
    raw=load_raw_case33bw(include_tie_lines=False)
    old=terminalize_case33(raw,mode='hybrid_leaf',service_impedance_mode='scaled_original',seed=0)
    net=TerminalizedNetwork(old.buses.copy(),old.branches.copy(),old.root_bus,old.base_kv,old.base_mva,
                            old.original_to_terminal,old.terminal_to_original,dict(old.metadata))
    net.metadata.update(impedance_scale=1.,case_name='case33_terminal',
        provenance='local Baran-Wu/MATPOWER-layout case33bw; hybrid_leaf terminalization, service impedance 0.2 adjacent median',
        comparison_scope='modified terminal-only feeder, not unmodified IEEE33 nor field measurements')
    return net


def register(case,simulation):
    if case.startswith('random'):
        n,t=case.removeprefix('random').split('_t');builder=lambda:random_network(int(n),int(t))
    elif case=='case33_terminal':builder=case33_network
    else:return
    simulation.CASE_BUILDERS[case]=builder
