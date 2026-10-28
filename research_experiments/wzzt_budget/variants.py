"""Observation-only selections derived from already frozen search trajectories."""
from research_experiments.wzzt_systematic.common import matrices,linear_loss
from research_experiments.wzzt_retune.selection import select
from .worker import measured


def multistart_original_rule(result,parent,train,validation):
    points=list(parent["methods"]["raw"]["RNJ_WZZT"])
    for name,trajectory in result["searches"].items():
        if not name.startswith("multistart_"):continue
        for point in trajectory["path"]:
            r,x=matrices(len(result["labels"]),point["supports"],point["r"],point["x"])
            points.append(dict(point,validation_mae=linear_loss(validation,r,x)["mae"],origin=name))
    temp=dict(parent,methods={"raw":{"RNJ_WZZT":points}})
    return measured(select(temp,"RNJ_WZZT",result["config"]),train,validation)


def ac_gate(baseline,alternative,*,fraction=.03):
    """Adopt an extra-start model only for a material held-out AC RMSE gain."""
    old=baseline.get("validation",{});new=alternative.get("validation",{})
    if not new.get("valid",False):return False
    if not old.get("valid",False):return True
    return new["relative_voltage_rmse"] < (1-fraction)*old["relative_voltage_rmse"]-1e-12
