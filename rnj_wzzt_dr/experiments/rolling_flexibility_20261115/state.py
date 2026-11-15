"""Frozen full-native scoring and sequential quarter-hour score state."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import numpy as np
from scipy.special import logsumexp

from ...dr.selection_mse import choose_mse
from ..flexibility_sweep_20261113.fitting import calibrated_means


def _frozen(value, dtype=float):
    out = np.array(value, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class FrozenBank:
    H: np.ndarray
    means: np.ndarray
    initial_sse: np.ndarray
    topology_ids: tuple
    internal: np.ndarray
    sigma2: float
    labels: tuple
    reference_mse: float
    tie: float
    initial_cells: int
    root_median: float
    voltage_median: float
    fingerprint: str

    @classmethod
    def from_score(cls, score, validation):
        native = score['native_bank']
        H = np.asarray([row['H'] for row in native], float)
        labels = tuple(int(v) for v in validation.labels)
        n = len(labels)
        X = np.column_stack((validation.p, validation.q))
        Y = np.asarray(validation.y, float)
        sigma2, reference, tie = (float(score[k]) for k in ('sigma2', 'reference_mse', 'validation_tie'))
        if (not len(native) or H.shape != (len(native), n, 2*n)
                or X.shape != (len(Y), 2*n) or Y.shape != (len(Y), n)
                or not all(np.isfinite(a).all() for a in (H, X, Y))
                or not np.isfinite([sigma2, reference, tie]).all()
                or min(sigma2, reference) <= 0 or tie < 0
                or int(score['cells']) != Y.size or len(set(labels)) != n):
            raise ValueError('Invalid frozen native bank or historical validation')
        sse = np.asarray([p['history_sse'] for p in native], float)
        direct = np.square(Y[None]-np.einsum('tc,knc->ktn', X, H)).mean(axis=(1,2))*Y.size
        if not np.allclose(sse, direct, rtol=1e-11, atol=64*np.finfo(float).eps*max(float(sse.max()),1e-300)):
            raise ValueError('Native historical SSE does not match validation')
        counts=np.asarray([p['internal'] for p in native],float)
        if not np.isfinite(counts).all() or (counts<0).any() or not np.equal(counts,counts.astype(int)).all():
            raise ValueError('Native stored support counts must be nonnegative integers')
        internal = counts.astype(int)
        topologies = tuple(str(p['topology_id']) for p in native)
        # Active topology can agree while stored zero-weight supports differ.
        # Preserve each native count for the original strict tie rule.
        chosen = choose_mse(sse/Y.size, internal, K=1/reference, epsilon_scaled=tie)['selected']
        if chosen != int(score['native_index']):
            raise ValueError('Initial strict winner differs from the frozen record')
        if validation.root is None or validation.voltage is None:
            raise ValueError('Historical voltage and root measurements required')
        root, voltage = np.asarray(validation.root,float), np.asarray(validation.voltage,float)
        if (root.shape != (len(Y),) or voltage.shape != Y.shape
                or not np.isfinite(root).all() or not np.isfinite(voltage).all()
                or (root<=0).any() or (voltage<=0).any()):
            raise ValueError('Invalid historical voltage measurements')
        # Exact original row-scale formula, now evaluated at EVERY native point.
        means = calibrated_means(dict(parameters=[dict(H=h) for h in H], sigma2=sigma2), validation)
        if not np.isfinite(means).all(): raise ValueError('Invalid frozen planning means')
        digest = hashlib.sha256()
        for a in (H, means, sse, internal): digest.update(np.asarray(a,dtype='<f8').tobytes())
        digest.update(json.dumps(dict(labels=labels,topologies=topologies,sigma2=sigma2,
            reference=reference,tie=tie,cells=Y.size,root_median=float(np.median(root)),
            voltage_median=float(np.median(voltage))),sort_keys=True).encode())
        return cls(_frozen(H),_frozen(means),_frozen(sse),topologies,_frozen(internal,int),sigma2,
            labels,reference,tie,Y.size,float(np.median(root)),float(np.median(voltage)),digest.hexdigest())

    @property
    def N(self): return len(self.labels)

    @property
    def native_count(self): return len(self.H)

    def voltage_covariance(self, meter_relative_sd=.0005):
        sd=float(meter_relative_sd)
        if not np.isfinite(sd) or sd<0: raise ValueError('Invalid meter relative SD')
        root=(2*sd*self.root_median**2)**2
        individual=(2*sd*self.voltage_median**2)**2
        return root*np.ones((self.N,self.N))+individual*np.eye(self.N)


class ScoreState:
    def __init__(self, bank):
        self.bank=bank
        self.sse=bank.initial_sse.copy()
        self.step=0

    def update(self, observation, step=None):
        """Consume exactly one unseen 15-minute row. step is zero based."""
        if step is not None and (isinstance(step,bool) or int(step)!=step or int(step)!=self.step):
            raise ValueError('Observation step is duplicated, skipped or out of order')
        if tuple(observation.labels)!=self.bank.labels:
            raise ValueError('Observation labels differ from the frozen native bank')
        X=np.column_stack((observation.p,observation.q));Y=np.asarray(observation.y,float)
        if X.shape!=(1,2*self.bank.N) or Y.shape!=(1,self.bank.N) or not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise ValueError('ScoreState.update requires one finite 15-minute row')
        self.sse += np.square(Y[None]-np.einsum('tc,knc->ktn',X,self.bank.H)).sum(axis=(1,2))
        self.step += 1
        return self.snapshot()

    def snapshot(self):
        cells=self.bank.initial_cells+self.step*self.bank.N
        strict=choose_mse(self.sse/cells,self.bank.internal,K=1/self.bank.reference_mse,
            epsilon_scaled=self.bank.tie)
        target=int(strict['selected'])
        families=list(dict.fromkeys(self.bank.topology_ids))
        points=[min((i for i,g in enumerate(self.bank.topology_ids) if g==family),
                    key=lambda i:(self.sse[i],i)) for family in families]
        scores=self.sse[points]
        logw=-(scores-scores.min())/(2*self.bank.sigma2)
        logw-=logsumexp(logw)
        weights={g:float(w) for g,w in zip(families,np.exp(logw))}
        foreign=[i for i,g in enumerate(self.bank.topology_ids) if g!=self.bank.topology_ids[target]]
        competitor=min(foreign,key=lambda i:(self.sse[i],self.bank.internal[i],i)) if foreign else None
        return dict(bank_fingerprint=self.bank.fingerprint,step=self.step,observed_steps=self.step,
            cells=cells,sse=self.sse.tolist(),native_target=target,current=target,
            target_family=self.bank.topology_ids[target],weights=weights,
            profile_family_scores={g:float(s) for g,s in zip(families,scores)},
            profile_family_points={g:int(i) for g,i in zip(families,points)},
            profile_max_family=families[int(np.argmax(logw))],
            target_weight=weights[self.bank.topology_ids[target]],competitor_native=competitor,
            strict_selection=strict,
            weight_interpretation='Normalized family-minimum frozen SSE; profile evidence, not integrated Bayesian posterior')

    def restore(self, snapshot):
        if snapshot.get('bank_fingerprint')!=self.bank.fingerprint:
            raise ValueError('Score checkpoint belongs to another frozen bank')
        step=snapshot['step'];sse=np.asarray(snapshot['sse'],float)
        if (isinstance(step,bool) or int(step)!=step or step<0 or sse.shape!=self.sse.shape
                or not np.isfinite(sse).all() or (sse<self.bank.initial_sse).any()):
            raise ValueError('Invalid cumulative score checkpoint')
        self.step=int(step);self.sse=sse.copy()
        actual=self.snapshot()
        if actual['native_target']!=snapshot['native_target'] or actual['cells']!=snapshot['cells']:
            raise ValueError('Score checkpoint winner/cells differ from direct recomputation')
        return actual
