"""Remaining-horizon full-native QCQP gates and frozen-epoch progress."""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from scipy.stats import norm

from ..distribution_gate_20261114.evidence import (
    DistributionEvidence, gaussian_power_integral, quadratic_box_upper, symmetric,
)
from .state import ScoreState


class EpochResetRequired(ValueError):
    pass


class Ledger:
    """Positive step contributions frozen at the first verified feasible plan.

    Epoch changes are explicit. A changed target/lambda/physical epoch never
    silently reuses or replaces an earlier schedule. Normal-mode progress is
    expected-score progress using the reference plan's remaining risk reserve;
    its complete normal gate, not this progress row, supplies the nominal bound.
    """
    def __init__(self, record):
        self.record=deepcopy(record)

    @classmethod
    def from_plan(cls, evidence, action, *, epoch_id=0):
        checked=evidence.verify_expansion(action)
        if not checked['feasible']:
            raise ValueError('Progress weights require a verified gate-feasible remaining plan')
        if not evidence.case.audit(action)['passed']:
            raise ValueError('Progress weights require a verified physical remaining plan')
        plan=evidence.step_values(action);base=evidence.step_values(np.zeros(evidence.case.shape))
        positive=np.maximum(plan['contribution']-base['contribution'],0.)
        denominator=positive.sum(axis=1,keepdims=True)
        weights=np.divide(positive,denominator,out=np.zeros_like(positive),where=denominator>0)
        return cls(dict(epoch_id=epoch_id,start_step=evidence.step,end_step=evidence.step+evidence.T,
            signature=evidence.epoch_signature(),weights=weights.tolist(),
            positive_contributions=positive.tolist(),reference_variance=plan['variance'].tolist(),
            first_verified_plan=np.asarray(action,float).tolist(),
            rule='Freeze positive extra contributions until explicit target/lambda/physical epoch change; normalize only over remaining frozen weights',
            normal_progress='Mean score deficit with remaining variance of the first feasible plan; full normal gate remains mandatory'))

    def snapshot(self): return deepcopy(self.record)

    def requirements(self,evidence):
        r=self.record
        if r['signature']!=evidence.epoch_signature():
            raise EpochResetRequired('Target, lambda, response map or physical epoch changed; explicitly rebuild progress ledger')
        offset=evidence.step-int(r['start_step'])
        if offset<0 or evidence.step+evidence.T!=int(r['end_step']):
            raise EpochResetRequired('Progress ledger time range differs from remaining horizon')
        weights=np.asarray(r['weights'],float)[:,offset:]
        variances=np.asarray(r['reference_variance'],float)[:,offset:]
        if weights.shape!=(len(evidence.rows),evidence.T) or variances.shape!=weights.shape:
            raise ValueError('Invalid progress checkpoint dimensions')
        if not np.isfinite(weights).all() or (weights<0).any() or not np.isfinite(variances).all() or (variances<0).any():
            raise ValueError('Invalid progress checkpoint weights or variance')
        base=evidence.step_values(np.zeros(evidence.case.shape))['contribution']
        result=[]
        for k,row in enumerate(evidence.rows):
            if evidence.mode=='normal_fixed':
                required=row['threshold']-row['h']+row['z']*np.sqrt(variances[k].sum())
            else:
                required=row['lambda_selected']*(row['threshold']-row['h'])+np.log(1/row['alpha'])
            deficit=max(0.,float(required-base[k].sum()))
            remaining=float(weights[k].sum())
            if deficit>1e-10 and remaining==0:
                raise EpochResetRequired('Remaining frozen positive weights exhausted while a new evidence deficit remains')
            fraction=float(weights[k,0]/remaining) if remaining>0 else 0.
            result.append(dict(target_native=row['target_native'],competitor_native=row['competitor_native'],
                deficit=deficit,remaining_weight=remaining,first_weight=fraction,
                gamma=fraction*deficit,active=bool(deficit>0),epoch_id=r['epoch_id'],
                meaning='First-step extra expected score' if evidence.mode=='normal_fixed' else 'First-step extra fixed-lambda exponent'))
        return result


class RollingEvidence(DistributionEvidence):
    def __init__(self,case,bank,snapshot,forecast,*,mode='mgf_forecast',beta=.9,eta=.9,
                 lambda_start=.5,progress=None,meter_relative_sd=.0005,forecast_relative_sd=.05,
                 forecast_sd=None):
        if mode not in ('normal_fixed','mgf_fixed','mgf_forecast'):
            raise ValueError(mode)
        if not 0<beta<1 or not .5<eta<1:
            raise ValueError('Require beta in (0,1) and eta in (.5,1)')
        if not np.isfinite([lambda_start,meter_relative_sd,forecast_relative_sd]).all() or lambda_start<=0 or min(meter_relative_sd,forecast_relative_sd)<0:
            raise ValueError('Invalid lambda or error scales')
        validated=ScoreState(bank).restore(snapshot)
        if case.repeat!=1: raise ValueError('Rolling controls must be one 15-minute row per step')
        self.case,self.bank,self.mode=case,bank,mode
        self.step=int(validated['step']);self.snapshot=deepcopy(snapshot)
        self.T,self.N,self.R=case.H,bank.N,case.R
        self.base_kw=float(case.base_kw if hasattr(case,'base_kw') else case.base)
        self.forecast=np.asarray(forecast,float)
        if self.T<=0 or self.forecast.shape!=(self.T,2*self.N) or not np.isfinite(self.forecast).all():
            raise ValueError('Expected finite remaining-step P/Q forecast')
        if hasattr(case,'labels') and tuple(case.labels)!=bank.labels:
            raise ValueError('Resource and frozen-bank terminal order differ')
        self.H,self.means=bank.H,bank.means
        self.native_H=self.H;self.native_indices=list(range(bank.native_count))
        self.native_topology=list(bank.topology_ids);self.sigma2=bank.sigma2
        self.history=np.asarray(validated['sse'],float);self.native_history=self.history
        self.K=bank.native_count;self.target=int(validated['native_target']);self.targets=[self.target]
        self.beta,self.eta=float(beta),float(eta)
        self.family_count=len(set(bank.topology_ids))
        self.profile_threshold=(float(np.log((self.family_count-1)*beta/(1-beta))) if self.family_count>1 else None)
        final_cells=int(validated['cells'])+self.T*self.N
        self.strict_tie=final_cells*bank.reference_mse*bank.tie/(2*bank.sigma2)
        self.tie=max(1e-7,self.strict_tie,self.profile_threshold or 0.)
        self.sigma_e=bank.voltage_covariance(meter_relative_sd)
        self.meter_relative_sd=float(meter_relative_sd);self.forecast_relative_sd=float(forecast_relative_sd)
        declared_sd=forecast_relative_sd*np.abs(self.forecast) if forecast_sd is None else np.asarray(forecast_sd,float)
        if declared_sd.shape!=self.forecast.shape or not np.isfinite(declared_sd).all() or (declared_sd<0).any():
            raise ValueError('Declared remaining forecast SD must match all power channels')
        self.power_sd=declared_sd.copy() if mode=='mgf_forecast' else np.zeros_like(self.forecast)
        self.sigma_p_diag=self.power_sd**2
        self.control_map=np.asarray(case.response_effect(np.eye(2*self.N),self.base_kw),float)
        if self.control_map.shape!=(2*self.N,self.R) or not np.isfinite(self.control_map).all():
            raise ValueError('Invalid fixed per-epoch control response map')
        probe=np.arange(self.T*self.R,dtype=float).reshape(case.shape)/(self.T*self.R+1)
        if not np.allclose(probe@self.control_map.T,case.shift_pq_pu(probe,self.base_kw),atol=1e-13,rtol=1e-12):
            raise ValueError('Power shift and gain-adjusted response map differ')
        competitors=[j for j,g in enumerate(bank.topology_ids) if g!=bank.topology_ids[self.target]]
        self.rows=[];self.mean_matrices=[];self.variance_matrices=[]
        self.effective_matrices=[];self.logdet_terms=[];self.step_logdet=[]
        self.guards=[];self.pairs=[]
        for kind in ('mu','variance','gate'):
            for suffix in ('q','a','c'): setattr(self,kind+'_'+suffix,[])
        for j in competitors:
            D=self.H[self.target]-self.H[j];E=self.means[self.target]-self.H[self.target]
            B=symmetric((D.T@D+E.T@D+D.T@E)/(2*self.sigma2))
            V=symmetric(D.T@self.sigma_e@D/self.sigma2**2)
            alpha=(1-eta)/len(competitors);h=float((self.history[j]-self.history[self.target])/(2*self.sigma2))
            row=dict(target_rep=self.target,target_native=self.target,competitor_native=j,
                target_topology_id=bank.topology_ids[self.target],competitor_topology_id=bank.topology_ids[j],
                alpha=float(alpha),h=h,tie=self.tie,threshold=self.tie,profile_threshold=self.profile_threshold,
                z=float(norm.isf(alpha)),lambda_start=float(lambda_start),lambda_selected=None,lambda_halvings=0)
            self.mean_matrices.append(B);self.variance_matrices.append(V)
            mq,ma,mc=self.expand(np.broadcast_to(B,(self.T,)+B.shape))
            vq,va,vc=self.expand(np.broadcast_to(V,(self.T,)+V.shape))
            self.mu_q.append(mq);self.mu_a.append(ma);self.mu_c.append(mc+h-self.tie)
            self.variance_q.append(vq);self.variance_a.append(va);self.variance_c.append(vc)
            # Same forward expected-score increment guard, now for all native competitors.
            self.guards.append((self.target,j,self.means[self.target],mq[0],ma))
            self.pairs.append((self.target,j))
            if mode!='normal_fixed':
                lam=float(lambda_start)
                for halvings in range(80):
                    try:
                        Q,logdet,mineig=gaussian_power_integral(symmetric(lam*B-lam*lam*V/2),self.power_sd)
                        break
                    except np.linalg.LinAlgError: lam*=.5
                else: raise ArithmeticError('No finite Gaussian exponential integral')
                q,a,c=self.expand(Q)
                self.gate_q.append(q);self.gate_a.append(a)
                self.gate_c.append(c+float(logdet.sum())+lam*(h-self.tie)-np.log(1/alpha))
                self.effective_matrices.append(Q);self.logdet_terms.append(float(logdet.sum()));self.step_logdet.append(logdet)
                row.update(lambda_selected=lam,lambda_=lam,lambda_halvings=halvings,
                    gaussian_domain_positive_definite=True,minimum_G_eigenvalue=mineig)
                row['lambda']=lam
            self.rows.append(row)
        count=len(self.rows)
        for kind in ('mu','variance','gate'):
            k=0 if kind=='gate' and mode=='normal_fixed' else count
            for suffix,shape in (('q',(k,self.T,self.R,self.R)),('a',(k,self.T,self.R)),('c',(k,))):
                name=kind+'_'+suffix;setattr(self,name,np.asarray(getattr(self,name),float).reshape(shape))
        self.mean_matrices=np.asarray(self.mean_matrices,float).reshape(count,2*self.N,2*self.N)
        self.variance_matrices=np.asarray(self.variance_matrices,float).reshape(count,2*self.N,2*self.N)
        self.alphas=np.asarray([r['alpha'] for r in self.rows]);self.z=np.asarray([r['z'] for r in self.rows])
        self.variance_upper=np.maximum(0.,quadratic_box_upper(self.variance_q,self.variance_a,self.variance_c,case.lower,case.upper))
        self.gate_upper=None if mode=='normal_fixed' else quadratic_box_upper(self.gate_q,self.gate_a,self.gate_c,case.lower,case.upper)
        self.progress=Ledger(progress) if isinstance(progress,dict) else progress
        self.progress_rows=[] if self.progress is None else self.progress.requirements(self)
        self.progress_q=np.zeros((len(self.progress_rows),self.T,self.R,self.R))
        self.progress_a=np.zeros((len(self.progress_rows),self.T,self.R))
        self.progress_c=np.zeros(len(self.progress_rows))
        for k,row in enumerate(self.progress_rows):
            if row['active']:
                kind='mu' if mode=='normal_fixed' else 'gate'
                self.progress_q[k,0]=getattr(self,kind+'_q')[k,0]
                self.progress_a[k,0]=getattr(self,kind+'_a')[k,0]
                self.progress_c[k]=-row['gamma']
        self.metadata=dict(mode=mode,dt_hours=.25,step=self.step,remaining_steps=self.T,
            bank_fingerprint=bank.fingerprint,target_native=self.target,target_family=bank.topology_ids[self.target],
            family_count=self.family_count,native_count=bank.native_count,beta=beta,eta=eta,
            profile_threshold=self.profile_threshold,strict_score_tie=self.strict_tie,threshold=self.tie,
            target_rule='Current full-native strict cumulative MSE winner; may change next quarter hour',
            score_scale='Frozen initial validation sigma2; raw SSE is accumulated without re-fitting',
            mean_source='Original validation row-scaling formula frozen separately at every native point',
            guard_scope='Forward mean-increment guards for every foreign native point; no representative pruning',
            confidence_scope='Per-replan conditional nominal profile-weight bound; not a closed-loop daily or actual-AC guarantee',
            progress_rule='Hard first-step extra contribution from an explicitly frozen feasible-plan epoch; none before such a plan exists',
            progress_rows=self.progress_rows,progress_ledger=None if self.progress is None else self.progress.snapshot(),
            covariance_scope='Frozen historical root-common and independent-terminal Gaussian variance; independent time blocks in declared model',
            power_sd_source='Explicit frozen day-ahead SD slice' if forecast_sd is not None else 'Relative SD times this supplied mean forecast; explicit day-ahead slice recommended',
            gate_box_upper=self.gate_upper,variance_box_upper=self.variance_upper)

    def epoch_signature(self):
        return dict(bank_fingerprint=self.bank.fingerprint,mode=self.mode,target=self.target,
            competitors=[r['competitor_native'] for r in self.rows],
            lambdas=[r['lambda_selected'] for r in self.rows],threshold=self.tie,
            physical_epoch=self.snapshot.get('physical_epoch',0),control_map=self.control_map.tolist())

    def step_values(self,u):
        power=self.forecast+self.case.shift_pq_pu(self.case.array(u),self.base_kw)
        mean=np.einsum('tc,kcd,td->kt',power,self.mean_matrices,power,optimize=True)
        variance=np.einsum('tc,kcd,td->kt',power,self.variance_matrices,power,optimize=True)
        if np.min(variance,initial=0.) < -1e-8: raise ArithmeticError('Negative per-step variance')
        variance=np.maximum(variance,0.)
        if self.mode=='normal_fixed': contribution=mean
        else:
            contribution=np.asarray([np.einsum('tc,tcd,td->t',power,Q,power,optimize=True)+ld
                for Q,ld in zip(self.effective_matrices,self.step_logdet)]).reshape(len(self.rows),self.T)
        return dict(mean=mean,variance=variance,contribution=contribution)

    def direct(self,u):
        info=super().direct(u)
        progress=[]
        if self.progress_rows:
            values=self.step_values(u)['contribution']
            base=self.step_values(np.zeros(self.case.shape))['contribution']
            progress=[float(values[k,0]-base[k,0]-r['gamma']) if r['active'] else 0.
                      for k,r in enumerate(self.progress_rows)]
        info.update(progress_margins=progress,minimum_progress_margin=min(progress) if progress else None,
            nominal_profile_weight_lower=self.beta,nominal_success_requirement=self.eta,
            progress_active=bool(self.progress_rows))
        info['feasible']=bool(info['feasible'] and all(x>=-1e-7 for x in progress))
        return info

    def verify_expansion(self,u):
        info=super().verify_expansion(u)
        u=self.case.array(u)
        expanded=np.einsum('hi,khij,hj->k',u,self.progress_q,u)+2*np.einsum('khi,hi->k',self.progress_a,u)+self.progress_c
        if not np.allclose(expanded,info['progress_margins'],atol=1e-8,rtol=1e-8):
            raise ArithmeticError('First-step progress expansion differs from direct calculation')
        info['expansion_max_absolute_errors']['progress']=float(np.max(np.abs(expanded-info['progress_margins']),initial=0.))
        return info
