"""Audit a frozen full-price schedule; no actual future or graph is read."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import full_price as full
import run as bounded


def main(args):
    data=bounded.build_problem(args.archive.resolve(),args.public.resolve())
    problem=full.FullPriceProblem(data)
    plan=bounded.unpickle(args.quote.resolve())
    x=np.asarray(plan["x"])
    u=problem.device.nominal(x)
    if np.max(np.abs(u-np.asarray(plan["mean_u"])))>1e-10:
        raise ValueError("Frozen nominal schedule differs from original actuator")
    device=problem.device
    fee,unused=device.fees(u,gradient=True)
    if abs(fee-plan["fee"])>1e-8:
        raise ValueError("Frozen fee differs from original cost")
    reward=np.asarray(plan["unit_reward"])
    reward_fee=float(device.dt*np.sum(reward*np.abs(u)))
    if abs(reward_fee-fee)>1e-8:
        raise ValueError("Frozen unit reward differs from nominal settlement")
    names=("shift","curtail","ev","battery")
    old_x=np.asarray(data["frozen_x"])
    old_u=device.nominal(old_x)
    old_fee=device.fees(old_u)
    def totals(values,selection=None):
        if selection is not None: values=values[selection]
        positive=float(device.dt*np.maximum(values,0).sum())
        negative=float(device.dt*np.maximum(-values,0).sum())
        return dict(increase_kwh=positive,decrease_kwh=negative,absolute_kwh=positive+negative,
                    net_kwh=positive-negative,max_sum_absolute_kw=float(np.abs(values).sum(axis=0).max()),
                    max_absolute_single_resource_kw=float(np.abs(values).max(initial=0.)))
    kinds=[]
    for k,name in enumerate(names):
        select=device.kind==k
        kinds.append(dict(kind=k,name=name,**totals(u,select),
            fee=float(device.dt*np.sum(reward[select]*np.abs(u[select]))),
            nonzero_request_hours=int(np.count_nonzero(np.abs(x[select])>1e-8)),
            nonzero_delivery_slots=int(np.count_nonzero(np.abs(u[select])>1e-8))))
    nodes=[]
    for n,label in enumerate(device.labels):
        select=device.node==n
        nodes.append(dict(node=int(label),**totals(u,select),
            fee=float(device.dt*np.sum(reward[select]*np.abs(u[select]))),
            peak_slot=int(np.abs(u[select]).sum(axis=0).argmax()),
            nonzero_resource_hours=int(np.count_nonzero(np.abs(x[select])>1e-8))))
    hours=[]
    for h in range(24):
        values=u[:,4*h:4*h+4]
        hours.append(dict(hour=h,increase_kwh=float(device.dt*np.maximum(values,0).sum()),
            decrease_kwh=float(device.dt*np.maximum(-values,0).sum()),
            fee=float(device.dt*np.sum(reward[:,4*h:4*h+4]*np.abs(values))),
            max_sum_absolute_kw=float(np.abs(values).sum(axis=0).max()),
            active_resource_requests=int(np.count_nonzero(np.abs(x[:,h])>1e-8))))
    positive_reward=reward[np.abs(u)>1e-8]
    aggregate=dict(**totals(u),fee=float(fee),old_fee=float(old_fee),
        active_resource_hours=int(problem.active.size),raw_resource_hours=int(x.size),
        nonzero_request_hours=int(np.count_nonzero(np.abs(x)>1e-8)),
        changed_request_hours=int(np.count_nonzero(np.abs(x-old_x)>1e-6)),
        newly_activated_request_hours=int(np.count_nonzero((np.abs(old_x)<=1e-8)&(np.abs(x)>1e-6))),
        changed_delivery_slots=int(np.count_nonzero(np.abs(u-old_u)>1e-6)),
        hourly_request_min=float(x.min()),hourly_request_max=float(x.max()),
        unit_reward_min=float(positive_reward.min(initial=np.inf)),
        unit_reward_max=float(positive_reward.max(initial=0.)),
        peak_delivery_slot=int(np.abs(u).sum(axis=0).argmax()),
        active_nodes=int(sum(np.any(np.abs(u[device.node==n])>1e-8) for n in range(len(device.labels)))),
        audit=device.audit(u))
    result=dict(quote=bounded.reference(args.quote.resolve()),
        source=bounded.reference(Path(__file__).resolve()),summary=aggregate,by_type=kinds,by_node=nodes,by_hour=hours,
        settlement="dt * sum(frozen unit_reward * absolute actually delivered kW); nominal delivery used in this audit",
        time_index="slots0..95; 15minutes each; peak hour=slot//4",true_future_used=False)
    bounded.write(args.out.resolve(),result)
    print(bounded.plain(aggregate))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("archive","public","quote","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    main(parser.parse_args())
