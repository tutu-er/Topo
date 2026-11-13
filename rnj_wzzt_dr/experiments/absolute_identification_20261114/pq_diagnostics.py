"""Read-only P/Q identifiability diagnostics for every native recovery change.

The threshold is descriptive and fixed at singular-value ratio 0.05. No
observation, resource, fitted parameter or procurement policy is changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from ... import io
from .data import load_entry
from .failure_explanations import read_observation


def design(power, threshold=.05):
    """Original uncentered design: the fitted voltage model has no intercept."""
    power=np.asarray(power,float)
    _,s,V=np.linalg.svd(power,full_matrices=False)
    ratio=s/s[0]
    weak=V[ratio<threshold].T
    train_energy=float(np.sum(power*power))
    return dict(singular_values=s.tolist(),singular_ratios=ratio.tolist(),
        weak_directions=int(weak.shape[1]),directions=int(power.shape[1]),
        training_weak_energy_fraction=float(np.sum((power@weak)**2)/train_energy),
        minimum_singular_ratio=float(ratio[-1])),weak


def fraction(x,basis):
    total=float(np.sum(x*x))
    return None if total<1e-30 else float(np.sum((x@basis)**2)/total)


def pq_departure(power,slopes):
    """Energy normal to each node's historical q=k*p line, in original units."""
    n=len(slopes);power=np.asarray(power,float);k=np.asarray(slopes,float)
    normal=(power[:,n:]-power[:,:n]*k)/np.sqrt(1+k*k)
    return float(np.sum(normal*normal)/np.sum(power*power))


def run(run_path,attribution,destination):
    if not (run_path/'complete.json').exists():raise ValueError('Batch is incomplete')
    if destination.exists():raise FileExistsError('Preserve completed diagnostics')
    destination.mkdir(parents=True)
    io.write(destination/'protocol.json',dict(source=io.ref(__file__),
        batch_protocol=io.ref(run_path/'protocol.json'),pairs=io.ref(attribution/'private_pairs.json'),
        selected='All successfully attributed AC native recovery gains and losses; no effect filtering.',
        singular_ratio_threshold=.05,centering='Main uncentered [P,Q]; centered sensitivity also reported.',
        uses='Descriptive hypothesis check only; no policy tuning, no causal claim.',
        bias_split='Same AC-selected native pair for before/after; linear physical voltage is read from stored paired evaluation. AC bias = frozen-parameter bias against physical linear network + AC nonlinear correction.',new_AC_days=0))
    entries={e['id']:e for e in io.read(run_path/'inputs.json')}
    events=[e for e in io.read(attribution/'private_pairs.json') if e['physics']=='AC' and e['explanation_status']=='complete']
    cache={};rows=[]
    for event in events:
        ident=event['id'];entry=entries[ident]
        if ident not in cache:
            score,train,validation,_=load_entry(entry)
            X=np.c_[train.p,train.q]
            info,V=design(X);centered,Vc=design(X-X.mean(0))
            nodes=[]
            for i,label in enumerate(train.labels):
                p,q=train.p[:,i],train.q[:,i]
                nodes.append(dict(label=int(label),correlation=float(np.corrcoef(p,q)[0,1]),
                    origin_slope_q_over_p=float(np.dot(p,q)/np.dot(p,p)),
                    centered_slope_q_over_p=float(np.dot(p-p.mean(),q-q.mean())/np.sum((p-p.mean())**2))))
            slopes=[node['origin_slope_q_over_p'] for node in nodes]
            cache[ident]=(score,V,Vc,dict(id=ident,case=entry['case'],role=entry['role'],
                history_partition=entry['history_partition'],training_points=int(len(train.p)),
                nodes=nodes,design=info,centered_design=centered,
                training_PQ_departure_fraction=pq_departure(X,slopes)))
        score,V,Vc,history=cache[ident]
        d=run_path/'days'/ident/f"day{event['day']:03d}"
        with np.load(d/'background.npz') as bg:root=bg['root']
        b=read_observation(d/'no_dfs_AC.npz',root)
        a=read_observation(d/(event['strategy']+'_AC.npz'),root)
        X=b['true_power'];U=a['true_power']-X;n=X.shape[1]//2
        Hi=np.asarray(score['native_bank'][event['true_native_index']]['H'])
        D=Hi-np.asarray(score['native_bank'][event['wrong_native_index']]['H'])
        bias_parts={}
        for label,strategy,observed in (('before','no_dfs',b),('after',event['strategy'],a)):
            linear=read_observation(d/(strategy+'_linear.npz'),root)
            if not np.allclose(linear['true_power'],observed['true_power'],rtol=0,atol=1e-14):
                raise ArithmeticError('AC/linear physical powers differ')
            pair_signal=observed['true_power']@D.T
            linear_bias=float(np.sum((linear['true_y']-observed['true_power']@Hi.T)*pair_signal)/score['sigma2'])
            ac_correction=float(np.sum((observed['true_y']-linear['true_y'])*pair_signal)/score['sigma2'])
            if abs(linear_bias+ac_correction-event[label]['physical_bias'])>1e-8:
                raise ArithmeticError('AC bias split identity failed')
            bias_parts[label]=dict(linear_parameter_bias=linear_bias,AC_nonlinear_correction=ac_correction,
                physical_bias=float(event[label]['physical_bias']))
        bias_parts['delta']={k:bias_parts['after'][k]-bias_parts['before'][k] for k in bias_parts['before']}
        Uw=U@V@V.T;Us=U-Uw
        # Squared prediction energies are nonadditive: retain their cross term.
        w=Uw@D.T;s=Us@D.T;t=U@D.T
        prediction=dict(total=float(np.sum(t*t)),weak=float(np.sum(w*w)),strong=float(np.sum(s*s)),
            cross=float(2*np.sum(w*s)))
        if abs(prediction['total']-prediction['weak']-prediction['strong']-prediction['cross'])>1e-12:
            raise ArithmeticError('Prediction projection identity failed')
        rows.append(dict(id=ident,day=event['day'],strategy=event['strategy'],event=event['event'],
            case=event['case'],role=event['role'],history_partition=event['history_partition'],
            noise_voltage_sigma=event['noise_voltage_sigma'],profile=event['profile'],profile_level=event['profile_level'],
            action_weak_energy_fraction=fraction(U,V),baseline_weak_energy_fraction=fraction(X,V),
            action_weak_energy_fraction_centered_design=fraction(U,Vc),
            baseline_weak_energy_fraction_centered_design=fraction(X,Vc),
            action_PQ_departure_fraction=pq_departure(U,[node['origin_slope_q_over_p'] for node in history['nodes']]),
            baseline_PQ_departure_fraction=pq_departure(X,[node['origin_slope_q_over_p'] for node in history['nodes']]),
            action_Q_squared_fraction=float(np.sum(U[:,n:]**2)/np.sum(U*U)),
            action_pair_prediction_energy=prediction,physical_bias_decomposition=bias_parts,
            delta=event['delta'],action=event['action']))
    histories=[v[3] for v in cache.values()]
    summary=dict(histories=len(histories),change_events=len(rows),
        minimum_node_correlation=min((n['correlation'] for h in histories for n in h['nodes']),default=None),
        maximum_node_correlation=max((n['correlation'] for h in histories for n in h['nodes']),default=None),
        history_weak_directions={h['id']:h['design']['weak_directions'] for h in histories},
        action_weak_fraction_range=[min([r['action_weak_energy_fraction'] for r in rows],default=None),max([r['action_weak_energy_fraction'] for r in rows],default=None)],
        baseline_weak_fraction_range=[min([r['baseline_weak_energy_fraction'] for r in rows],default=None),max([r['baseline_weak_energy_fraction'] for r in rows],default=None)],
        interpretation='Weak-design action exposure can explain sensitivity of frozen parameters, but it is not evidence by itself that those directions caused a particular gain or loss. Gains and losses use the same fixed threshold and are all retained.')
    io.write(destination/'private_histories.json',histories);io.write(destination/'private_events.json',rows)
    io.write(destination/'summary.json',summary);io.write(destination/'complete.json',dict(summary=io.ref(destination/'summary.json')))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','attribution','destination'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(run(a.run.resolve(),a.attribution.resolve(),a.destination.resolve()))
