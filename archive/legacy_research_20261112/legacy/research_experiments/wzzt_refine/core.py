from time import perf_counter
import numpy as np
import pandas as pd
from research_experiments.wzzt_systematic.common import matrices,linear_loss
from research_experiments.wzzt_retune.engine import arrays,canonical,compatible


def scored(point,validation,origin=None):
    p=dict(point)
    r,x=matrices(validation[0].shape[1],p["supports"],p["r"],p["x"])
    p["validation_mae"]=linear_loss(validation,r,x)["mae"]
    if origin is not None:p["origin"]=origin
    return p


def select_pre(points,fraction=.03):
    best=min(p["validation_mae"] for p in points)
    eligible=[i for i,p in enumerate(points) if p["validation_mae"]<=best*(1+fraction)+1e-12]
    index=min(eligible,key=lambda i:(len(points[i]["supports"]),points[i]["validation_mae"],i))
    return dict(points[index],selection_index=index)


def finalize(points,train,validation,calibration):
    p=select_pre(points);p["ranking_validation_mae"]=p["validation_mae"]
    keep=[i for i,s in enumerate(p["supports"]) if len(s)==1 or
        calibration["a"]*p["r"][i]+calibration["b"]*p["x"][i]+1e-10>=.16*calibration["median_root_depth"]]
    p.update(supports=[p["supports"][i] for i in keep],r=[p["r"][i] for i in keep],x=[p["x"][i] for i in keep])
    r,x=matrices(train[0].shape[1],p["supports"],p["r"],p["x"])
    p["pre_contraction_objective"]=p["objective"]
    p["objective"]=linear_loss(train,r,x)["mae"]
    return scored(p,validation)


def backward(lp,initial,validation,*,seconds=15.,max_deletions=2):
    """Propose subset families even when training loss rises; use validation."""
    if seconds<=0 or not np.isfinite(seconds) or isinstance(max_deletions,bool) or not isinstance(max_deletions,int) or max_deletions<0:
        raise ValueError("positive finite seconds and nonnegative integer max_deletions required")
    t0=perf_counter();current=scored(lp.fit(initial),validation,"backward_initial")
    proposals=[];decisions=[];reason="deletion_limit"
    for depth in range(max_deletions):
        family=canonical(current["supports"]);candidates=[current];records=[];full=True
        for removed in family:
            if len(removed)==1:continue
            if perf_counter()-t0>=seconds:full=False;break
            fit=scored(lp.fit([s for s in family if s!=removed]),validation,"backward_delete")
            fit.update(deletion_depth=depth+1,requested_removed=list(removed))
            if fit["objective"]<current["objective"]-1e-8:
                raise AssertionError("subset LP improved the superset optimum")
            proposals.append(fit);candidates.append(fit)
            records.append(dict(removed=removed,training_objective=fit["objective"],validation_mae=fit["validation_mae"],
                supports=fit["supports"],training_loss_rise=fit["objective"]-current["objective"]))
        chosen=select_pre(candidates)
        accepted=len(chosen["supports"])<len(current["supports"])
        decisions.append(dict(depth=depth,current_supports=current["supports"],current_objective=current["objective"],
            current_validation_mae=current["validation_mae"],full_sweep=full,candidates=records,
            accepted=accepted,selected_index=chosen["selection_index"],selected_supports=chosen["supports"]))
        if not accepted:reason="validation_no_simpler_model" if full else "backward_wall_limit";break
        current=chosen
        if not full:reason="backward_wall_limit";break
    return dict(points=proposals,decisions=decisions,reason=reason,seconds=perf_counter()-t0,
                acceptance="validation 3pct simplicity, not positive training gain")


def ridge_alpha(p,q,strength):
    if not np.isfinite(strength) or strength<=0:raise ValueError("positive finite ridge strength required")
    design=np.column_stack((p,q))
    scale=max(float(np.sum(design*design)/design.shape[1]),1e-12)
    return float(strength*scale),scale


def regularized_pool(obs,base_pool,*,strengths=(.001,.01),root_label=1):
    """Augment only candidates; keep the original fit and calibration unchanged."""
    from rnj_wzzt.estimation.multiscenario import fit_projected_sensitivity
    from rnj_wzzt.graph.rooted_neighbor_joining import rooted_neighbor_joining
    from rnj_wzzt.graph.rooted_hierarchy import rooted_clades
    from research_experiments.rnj.rooted_study_baselines import infer_classical_nj
    from research_experiments.rnj.rx75_edge_floor import calibrate_rx75_floor
    t0=perf_counter();labels=list(obs[0].columns);positions={v:i for i,v in enumerate(labels)}
    pool=set(canonical(base_pool));fronts=[]
    for preprocessing in ("raw","demean"):
        p,q,y=arrays(obs,preprocessing)
        scenes=[dict(P_terminal=pd.DataFrame(p,columns=labels),Q_terminal=pd.DataFrame(q,columns=labels),drop_target=pd.DataFrame(y,columns=labels))]
        for strength in strengths:
            alpha,scale=ridge_alpha(p,q,strength);diag={}
            r,x,_,_=fit_projected_sensitivity(scenes,alpha=alpha,diagnostics=diag)
            cal=calibrate_rx75_floor(r,x)
            front=dict(preprocess=preprocessing,strength=strength,alpha=alpha,gram_scale=scale,
                R=r.tolist(),X=x.tolist(),diagnostics=diag,tree_families=[])
            for geometry,w in (("R",1.),("RX75",.75),("X",0.)):
                kernel=w*r/cal["mu_r"]+(1-w)*x/cal["mu_x"]
                for threshold in (0.,.04,.08,.16,.24):
                    nj=infer_classical_nj(kernel,kernel,labels,root_label,collapse_factor=threshold)
                    rnj=rooted_neighbor_joining(kernel,np.diag(kernel),labels,root_label,
                        group_tolerance=threshold*max(float(np.median(np.diag(kernel))),1e-12))
                    for method,tree in (("NJ",nj),("RNJ",rnj)):
                        family=canonical([(i,) for i in range(len(labels))]+[tuple(positions[v] for v in c) for c in rooted_clades(tree.edges,root_label,labels)])
                        pool.update(family)
                        front["tree_families"].append(dict(geometry=geometry,threshold=threshold,method=method,family=family))
            fronts.append(front)
    return dict(pool=canonical(pool),fronts=fronts,seconds=perf_counter()-t0,
                old_pool_size=len(canonical(base_pool)),new_pool_size=len(pool),truth_used=False)


def pair_family(family,first,second):
    if not compatible([first],second):return None
    retained=[s for s in family if compatible([s],first) and compatible([s],second)]
    return canonical([*retained,first,second])


def two_step(lp,initial,validation,*,seconds=25.,width=3):
    """Keep non-improving first moves; fit two compatible additions jointly."""
    if seconds<=0 or not np.isfinite(seconds) or isinstance(width,bool) or not isinstance(width,int) or width<1:
        raise ValueError("positive finite seconds and positive integer width required")
    t0=perf_counter();initial=canonical(initial);current=lp.fit(initial)
    first=[];records=[];stage1_complete=True
    for c in lp.pool:
        if len(c)<2 or c in initial:continue
        if perf_counter()-t0>=.4*seconds:stage1_complete=False;break
        retained=[s for s in initial if compatible([s],c)]
        fit=lp.fit([*retained,c])
        first.append((fit["objective"],c,fit))
    first.sort(key=lambda item:(item[0],len(item[1]),item[1]))
    beam=first[:width];kept=[];pairs=set();stage2_complete=True
    tolerance=1e-10+.001*current["objective"]
    for _,c,fit in beam:
        if current["objective"]-fit["objective"]>tolerance:
            kept.append(dict(fit,origin="two_step_single",added=[list(c)]))
    for first_loss,c1,_ in beam:
        for c2 in lp.pool:
            if len(c2)<2 or c2 in initial or c2==c1:continue
            if perf_counter()-t0>=seconds:stage2_complete=False;break
            pair=tuple(sorted((c1,c2)))
            if pair in pairs:continue
            family=pair_family(initial,c1,c2)
            if family is None:continue
            pairs.add(pair);fit=lp.fit(family)
            gain=current["objective"]-fit["objective"]
            accepted=gain>tolerance
            records.append(dict(first=c1,second=c2,first_objective=first_loss,endpoint_objective=fit["objective"],
                gain=gain,accepted=accepted,removed=[s for s in initial if s not in family],family=family))
            if accepted:kept.append(dict(fit,origin="two_step_pair",added=[list(c1),list(c2)],first_objective=first_loss))
        if not stage2_complete:break
    kept.sort(key=lambda p:(p["objective"],len(p["supports"]),canonical(p["supports"])))
    unique=[];seen=set()
    for p in kept:
        family=canonical(p["supports"])
        if family not in seen:unique.append(scored(p,validation));seen.add(family)
        if len(unique)>=width:break
    return dict(points=unique,current_objective=current["objective"],initial=initial,
        first_candidates=[dict(candidate=c,objective=loss) for loss,c,_ in first],
        beam=[list(c) for _,c,_ in beam],pairs=records,first_sweep_complete=stage1_complete,
        second_sweep_complete=stage2_complete,reason="beam_exhausted" if stage1_complete and stage2_complete else "beam_wall_limit",
        seconds=perf_counter()-t0,width=width,certificate_scope="heuristic width-limited two-support neighborhood")
