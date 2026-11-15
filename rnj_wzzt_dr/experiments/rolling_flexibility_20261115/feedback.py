"""Pre-dispatch pair monitoring; nominal model diagnostics, not causal gain."""
from __future__ import annotations

from copy import deepcopy
import numpy as np


def seal_monitor(bank,snapshot,*,step=None,meter_relative_sd=.0005):
    """Freeze the ordered pair BEFORE acquiring its next measured P/Q/Y."""
    if snapshot.get('bank_fingerprint')!=bank.fingerprint:
        raise ValueError('Monitor snapshot belongs to another bank')
    k=int(snapshot['step']) if step is None else int(step)
    if k!=snapshot['step']: raise ValueError('Monitor must precede exactly the next score row')
    target=int(snapshot['native_target']);competitor=snapshot['competitor_native']
    return dict(bank_fingerprint=bank.fingerprint,step=k,target_native=target,
        competitor_native=None if competitor is None else int(competitor),
        pair=None if competitor is None else [target,int(competitor)],
        available=competitor is not None,meter_relative_sd=float(meter_relative_sd),
        means_rule='Every frozen native planning mean explains the same pre-dispatch ordered score',
        interpretation='Conditional mean-model diagnostic at measured power, not action versus no-action causal benefit',
        omitted_laws='Measured-power error and latent whole-device delivery uncertainty are not calibrated by this diagnostic')


def evaluate_monitor(bank,monitor,x_meas,y):
    if monitor.get('bank_fingerprint')!=bank.fingerprint:
        raise ValueError('Monitor bank changed')
    x=np.asarray(x_meas,float).reshape(-1);y=np.asarray(y,float).reshape(-1)
    if x.shape!=(2*bank.N,) or y.shape!=(bank.N,) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('One finite measured P/Q and Y row required')
    if not monitor['available']:
        return dict(available=False,reason='No different-topology competitor in frozen bank',
                    step=monitor['step'],pair=None,passed=None,outside_union=None)
    t,j=monitor['pair']
    if t==j or bank.topology_ids[t]==bank.topology_ids[j]:
        raise ValueError('Monitoring pair must have different topologies')
    d=(bank.H[t]-bank.H[j])@x
    actual=float((np.square(y-bank.H[j]@x).sum()-np.square(y-bank.H[t]@x).sum())/(2*bank.sigma2))
    predictions=bank.means@x
    target_prediction=bank.H[t]@x
    means=(.5*np.sum(d*d)+np.sum((predictions-target_prediction[None])*d[None],axis=1))/bank.sigma2
    covariance=bank.voltage_covariance(monitor['meter_relative_sd'])
    variance=float(d@covariance@d/bank.sigma2**2)
    if variance < -1e-10: raise ArithmeticError('Negative monitoring variance')
    variance=max(0.,variance);half_width=2*np.sqrt(variance)
    # A union of separate intervals, NOT the convex hull of their endpoints.
    tolerance=64*np.finfo(float).eps*max(1.,abs(actual),float(np.max(np.abs(means),initial=0.)))
    fits=np.abs(actual-means)<=half_width+tolerance
    explained=np.flatnonzero(fits).tolist()
    return dict(available=True,step=monitor['step'],pair=list(monitor['pair']),
        actual_new_logodds=actual,mean_under_each_native=means.tolist(),variance=variance,
        interval_lower=(means-half_width).tolist(),interval_upper=(means+half_width).tolist(),
        explained_by_native=explained,outside_union=not bool(explained),passed=bool(explained),
        target_mean_explains=bool(fits[t]),competitor_mean_explains=bool(fits[j]),
        actual_minus_target_mean=float(actual-means[t]),
        interpretation='A negative score may overturn the old target; outside all means diagnoses model mismatch, not causal loss')


class FeedbackController:
    def __init__(self,dt=.25,confirmation_hours=2.,recovery_hours=2.,caution_fraction=.25):
        vals=np.asarray([dt,confirmation_hours,recovery_hours,caution_fraction],float)
        if not np.isfinite(vals).all() or min(vals[:3])<=0 or not 0<caution_fraction<=1:
            raise ValueError('Invalid feedback durations or action fraction')
        durations=np.asarray([confirmation_hours,recovery_hours])/dt
        if not np.allclose(durations,np.rint(durations),atol=1e-12,rtol=0):
            raise ValueError('Physical confirmation/recovery durations must contain whole control steps')
        self.dt=float(dt);self.confirmation_steps=int(round(durations[0]));self.recovery_steps=int(round(durations[1]))
        self.caution_fraction=float(caution_fraction)
        self.state='normal';self.outside_streak=0;self.recovery_streak=0;self.last_step=-1
        self.last_record=None

    def observe(self,bank,monitor,x_meas,y,*,previous_target,current_target):
        if int(monitor['step'])!=self.last_step+1:
            raise ValueError('Feedback observations must arrive once in dispatch order')
        if int(previous_target)!=monitor['target_native']:
            raise ValueError('Previous target differs from the pre-dispatch sealed target')
        report=evaluate_monitor(bank,monitor,x_meas,y)
        changed=int(previous_target)!=int(current_target)
        reasons=[];old_state=self.state
        if not report['available']:
            # Missing information breaks both confirmation streaks; it cannot
            # clear a pause or count as a successful monitoring observation.
            self.outside_streak=0;self.recovery_streak=0
            if self.state!='pause': self.state='caution' if changed else self.state
            reasons.append('monitor_unavailable')
        elif report['outside_union']:
            self.outside_streak+=1;self.recovery_streak=0
            self.state='pause' if self.state=='pause' or self.outside_streak>=self.confirmation_steps else 'caution'
            reasons.append('outside_all_frozen_mean_intervals')
        elif self.state=='pause':
            self.outside_streak=0;self.recovery_streak+=1
            if self.recovery_streak>=self.recovery_steps:
                self.state='caution';self.recovery_streak=0
                reasons.append('confirmed_model_explanation_recovered')
            else: reasons.append('pause_recovery_pending')
        else:
            self.outside_streak=0;self.recovery_streak=0
            self.state='normal'
        if changed:
            if self.state!='pause': self.state='caution'
            reasons.append('target_changed_one_step_caution')
        elif report.get('available') and not report['target_mean_explains']:
            if self.state!='pause': self.state='caution'
            reasons.append('alternative_frozen_mean_explains_observation')
        self.last_step=int(monitor['step'])
        fraction=0. if self.state=='pause' else self.caution_fraction if self.state=='caution' else 1.
        self.last_record=dict(report,previous_state=old_state,state=self.state,reasons=reasons,
            target_changed=changed,previous_target=int(previous_target),current_target=int(current_target),
            extra_action_fraction=fraction,outside_streak=self.outside_streak,recovery_streak=self.recovery_streak,
            confirmation_steps=self.confirmation_steps,recovery_steps=self.recovery_steps,
            elapsed_confirmed_outside_hours=self.outside_streak*self.dt,
            low_profile_weight_pauses=False,causal_procurement_gain_observed=False,
            safeguard_scope='Only extra identification action is limited; firm service is not cancelled')
        return deepcopy(self.last_record)

    def snapshot(self):
        return dict(dt=self.dt,confirmation_steps=self.confirmation_steps,recovery_steps=self.recovery_steps,
            caution_fraction=self.caution_fraction,state=self.state,outside_streak=self.outside_streak,
            recovery_streak=self.recovery_streak,last_step=self.last_step,last_record=deepcopy(self.last_record))

    def restore(self,record):
        for key in ('dt','confirmation_steps','recovery_steps','caution_fraction'):
            if record[key]!=getattr(self,key): raise ValueError('Feedback checkpoint configuration differs')
        if record['state'] not in ('normal','caution','pause'):
            raise ValueError('Unknown feedback checkpoint state')
        for key in ('outside_streak','recovery_streak'):
            if int(record[key])!=record[key] or record[key]<0: raise ValueError('Invalid feedback streak')
        for key in ('state','outside_streak','recovery_streak','last_step','last_record'):
            setattr(self,key,deepcopy(record[key]))
        return self.snapshot()
