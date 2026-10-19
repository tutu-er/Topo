"""Deterministic failure probes, distinct from independent statistical cohorts."""
from pathlib import Path
import numpy as np
from .simulation import generate,observe,model
from .candidates import propose
from .likelihood import Observations,fit_candidate
from .design import EVConstraints,enumerate_actions,select_action
from .run_study import dump,fit_bank,fit_record


def main():
    path=generate(81001)
    obs,_=observe(path,slice(0,12))
    candidates,proposal=propose(obs,run_wzzt=False)
    meter=model()
    hist,fail=fit_bank(candidates,obs,meter)
    constraints=EVConstraints(availability=np.zeros((8,3),bool),initial_site=0)
    actions,rejected=enumerate_actions(8,3,constraints,max_actions=11)
    selection=select_action(candidates,obs,hist,meter,actions,constraints=constraints)
    assert not actions and selection["status"]=="no_feasible_action"
    limited=[fit_record(c,fit_candidate(c,obs,meter,max_nfev=1,multistart=1))
             for c in candidates]
    assert all(not f["success"] for f in limited)
    no_anchor=Observations(obs.p,obs.q,obs.v,obs.root_v)
    no_anchor_meter=type(meter)(meter.p_sigma,meter.q_sigma,meter.v_sigma,
        bias_p=meter.bias_p,bias_q=meter.bias_q,bias_v=meter.bias_v)
    try:
        fit_candidate(candidates[0],no_anchor,no_anchor_meter)
    except ValueError as error:
        anchor_failure=str(error)
    else:
        raise AssertionError("Unanchored free P/Q source accepted")
    evidence=dict(seed=81001,scope="deterministic diagnostic probes; not statistical evaluation",
        access_failure=dict(selection=selection,rejected_actions=rejected),
        optimization_budget_failure=limited,unanchored_model_restriction=anchor_failure,
        historical_fit_failures=fail)
    dump(Path(__file__).resolve().parent/"evidence"/"failure_diagnostics.json",evidence)
    print("access failure, iteration-budget failure and source-anchor restriction verified")


if __name__=="__main__":
    main()
