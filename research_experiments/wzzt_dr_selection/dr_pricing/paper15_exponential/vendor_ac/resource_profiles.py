"""Verbatim historical resource-profile functions; no data-loading CLI."""
import numpy as np

def period_array(values, cfg):
    result = np.zeros(cfg["time"]["slots"])
    for value, (start, end) in zip(values, cfg["time"]["period_hours"], strict=True):
        result[round(start/cfg["time"]["dt_hours"]):round(end/cfg["time"]["dt_hours"])] = value
    return result

def profiles(node, cfg):
    """Full inventory envelopes. Enroll a share -> scale power AND energy.

    Arrays have [resource,slot] order: shiftable, curtailable, EV, battery.
    Net increments enter the old AC bridge only once; baselines are embedded.
    """
    T=cfg["time"]["slots"]; dt=cfg["time"]["dt_hours"]; hour=np.arange(T)*dt
    avail=period_array(cfg["user"]["availability_fraction_by_period"],cfg)
    base=np.zeros((4,T)); lo=np.zeros_like(base); hi=np.zeros_like(base)
    for i,key in enumerate(("shiftable_kw","curtailable_kw")):
        base[i]=node[key]*avail
        lo[i]=-base[i]
    hi[0]=base[0]
    ev=cfg["ev"]; connected=np.zeros(T,dtype=bool)
    windows=[]
    for start,end in ev["connected_hours"]:
        connected |= (hour>=start)&(hour<end)
        windows.append([round(start/dt),round(end/dt)])
    total_hours=dt*connected.sum()
    base[2,connected]=node["ev_grid_task_kwh"]/total_hours
    lo[2]=-base[2];hi[2,connected]=node["ev_charger_kw"]-base[2,connected]
    lo[3]=-node["battery_power_kw"];hi[3]=node["battery_power_kw"]
    costs=np.array([period_array(cfg["price"]["full_activation_price_by_period"][k],cfg)*node["cost_factor"][k]
        for k in ("user","user","ev","battery")])
    sd=np.array([np.full(T,cfg["response"]["relative_sd_by_type"][k]*node["sd_factor"][k])
        for k in ("user","user","ev","battery")])
    peak=((hour>=6)&(hour<9))|((hour>=17)&(hour<22))
    sd[:2,peak]*=cfg["response"]["user_peak_sd_multiplier"]
    sd[2,(hour>=6)&(hour<8)]*=cfg["response"]["ev_last_two_hours_before_08_departure_sd_multiplier"]
    sd[2,~connected]=0.
    return dict(baseline_kw=base,delta_lower_kw=lo,delta_upper_kw=hi,cost_full_activation=costs,
        relative_sd=sd,ev_windows=windows,
        q_over_p=np.array([node["user_q_over_p"],node["user_q_over_p"],0.,0.]))
