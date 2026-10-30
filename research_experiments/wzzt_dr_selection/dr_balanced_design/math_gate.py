"""Independent three-terminal B-direction LP gate; no project solver imports.

The finite calculations check the stated algebra; the algebraic claims are not
inferred from a finite coefficient grid. There is no household or future data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from scipy.optimize import linprog


KAPPA = .35
ALPHA = .2
LEAVES = [(0,), (1,), (2,)]
PAIRS = [(0, 1), (0, 2), (1, 2)]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def families():
    result = []
    for root in (False, True):
        for pair in (None, *PAIRS):
            supports = [*LEAVES]
            if pair is not None:
                supports.append(pair)
            if root:
                supports.append((0, 1, 2))
            result.append(dict(id=f"{'root' if root else 'noroot'}_{'star' if pair is None else ''.join(map(str, pair))}",
                               root=root, pair=pair, supports=supports))
    return result


def feature(supports, U):
    columns = []
    for support in supports:
        z = np.zeros(3)
        z[list(support)] = 1.
        columns.append(np.outer(z, z @ U).ravel())
    F = np.column_stack(columns)
    # r and x are independent nonnegative optimization variables. Their
    # identical-direction columns deliberately preserve P/Q confounding.
    return np.column_stack((F, KAPPA * F))


def solve_l1(A, y, *, equality=None, rhs=None):
    n, p = A.shape
    objective = np.r_[np.zeros(p), np.ones(n)]
    inequality = np.r_[np.column_stack((A, -np.eye(n))), np.column_stack((-A, -np.eye(n)))]
    bound = np.r_[y, -y]
    Aeq = None if equality is None else np.column_stack((equality, np.zeros((len(rhs), n))))
    started = time.perf_counter()
    result = linprog(objective, A_ub=inequality, b_ub=bound, A_eq=Aeq, b_eq=rhs,
                     bounds=(0., None), method="highs",
                     options={"dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9})
    receipt = dict(status=int(result.status), success=bool(result.success), message=result.message,
                   seconds=time.perf_counter()-started, lower_bounds=0., upper_bounds=None,
                   r_x_independent=True, coefficient_lower_bound_not_positive=True,
                   scalar_response_count=n, coefficient_count=p)
    if result.success:
        coefficients = result.x[:p]
        residual = A @ coefficients-y
        dual = float(bound @ result.ineqlin.marginals)
        if rhs is not None:
            dual += float(rhs @ result.eqlin.marginals)
        receipt.update(objective=float(result.fun), primal_l1=float(np.abs(residual).sum()),
                       mean_absolute_error=float(np.abs(residual).mean()),
                       numerical_dual_objective=dual, numerical_primal_dual_gap=float(result.fun-dual),
                       coefficients=coefficients.tolist(), residual=residual.tolist(),
                       max_equality_residual=0. if equality is None else float(np.max(np.abs(equality@coefficients-rhs))),
                       max_inequality_violation=float(max(0., np.max(inequality@result.x-bound))))
    return receipt


def prepare(root):
    root = Path(root)
    if (root/"protocol.json").exists():
        raise FileExistsError("frozen protocol already exists")
    root.mkdir(parents=True, exist_ok=True)
    protocol = dict(created_utc=utc(), stage="B three-terminal independent mathematical gate",
                    source=str(Path(__file__).resolve()), source_sha256=digest(__file__),
                    python=sys.version, platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
                    solver="scipy.optimize.linprog(method='highs')", kappa=KAPPA, amplitude=ALPHA,
                    leaf_profiles=[[.3,.4,.5],[.3,0.,.5]], root_effect=.4,
                    proper_effects=[0.,.02,.6], proper_locations=[list(p) for p in PAIRS],
                    candidate_families="all eight: all singleton supports + optional root + zero or one proper pair",
                    coefficient_domain="every r and x independently nonnegative with no upper bound",
                    actions=["u12", "u13", "two", "common", "history_plus_two"],
                    intended_profile_LPs=720, intended_uniform_separation_LPs=84,
                    no_project_solver=True, no_measurement_noise=True, no_actual_future=True,
                    evaluation_tolerance=1e-8, no_rerun=True,
                    claim_scope="finite coefficient checks validate algebra only; continuous sign and shared-witness proofs stated separately")
    write(root/"protocol.json", protocol)
    write(root/"protocol.sha256.json", dict(sha256=digest(root/"protocol.json")))
    return protocol


def run(root):
    root = Path(root)
    protocol = read(root/"protocol.json")
    if digest(root/"protocol.json") != read(root/"protocol.sha256.json")["sha256"]:
        raise RuntimeError("protocol drift")
    if digest(__file__) != protocol["source_sha256"]:
        raise RuntimeError("source drift")
    if (root/"run_started.json").exists():
        raise FileExistsError("prior run retained")
    write(root/"run_started.json", dict(utc=utc(), protocol_sha256=digest(root/"protocol.json")))
    started = time.perf_counter()
    u12 = np.array([1.,-1.,0.])*ALPHA
    u13 = np.array([1.,0.,-1.])*ALPHA
    # Each experiment includes its return, so energy is balanced over time too.
    U12 = np.column_stack((u12,-u12))
    U13 = np.column_stack((u13,-u13))
    U2 = np.column_stack((U12,U13))
    H = np.outer(np.ones(3), np.array([1.,2.]))
    actions = dict(u12=U12,u13=U13,two=U2,common=H,history_plus_two=np.column_stack((H,U2)))
    models = families()
    lp = []
    checks = []
    def check(name, okay, **detail):
        checks.append(dict(name=name, passed=bool(okay), **detail))
    index = {}
    for leaf_index, leaves in enumerate(protocol["leaf_profiles"]):
        for c in protocol["proper_effects"]:
            for pair in PAIRS:
                z = np.zeros(3); z[list(pair)] = 1.
                K = .4*np.ones((3,3))+np.diag(leaves)+c*np.outer(z,z)
                key = f"leaf{leaf_index}_c{c}_{pair[0]}{pair[1]}"
                for action, U in actions.items():
                    target = (K@U).ravel()
                    for candidate in models:
                        receipt = solve_l1(feature(candidate["supports"], U), target)
                        receipt.update(case=key, leaf_profile=leaf_index, true_pair=list(pair), true_effect=c,
                                       action=action, family=candidate["id"], fitted_pair=candidate["pair"], fitted_root=candidate["root"])
                        lp.append(receipt)
                        index[key,action,candidate["id"]] = receipt
                        write(root/"lp_receipts"/f"{len(lp):04d}.json", receipt)
                        check("profile_lp_solved", receipt["success"], case=key,action=action,family=candidate["id"])
                        if not receipt["success"]:
                            continue
                        check("profile_primal_dual_replay", abs(receipt["objective"]-receipt["primal_l1"])<1e-8 and
                              abs(receipt["numerical_primal_dual_gap"])<1e-8 and receipt["max_inequality_violation"]<1e-8,
                              lp_id=len(lp))
                        if action == "common":
                            check("synchronous_history_absorbed_by_any_family", receipt["objective"]<1e-8, lp_id=len(lp))
                        if c == 0. and action in ("u12","u13","two"):
                            check("zero_internal_effect_not_distinguishable", receipt["objective"]<1e-8, lp_id=len(lp))
                        if c > 0. and candidate["pair"] != pair and action == "two":
                            check("two_directions_wrong_internal_family_sign_bound", receipt["objective"]+1e-8 >= 2*ALPHA*c,
                                  lp_id=len(lp),lower_bound=2*ALPHA*c)
                        selected = "u13" if pair == (0,1) else "u12"
                        if c > 0. and candidate["pair"] != pair and action == selected:
                            check("selected_single_direction_sign_bound", receipt["objective"]+1e-8 >= 2*ALPHA*c,
                                  lp_id=len(lp),lower_bound=2*ALPHA*c)
                        if candidate["pair"] == pair and action in ("u12","u13","two"):
                            check("correct_internal_family_fits_without_root", receipt["objective"]<1e-8, lp_id=len(lp))
                for action, invisible in (("u12",(0,1)),("u13",(0,2))):
                    if pair == invisible:
                        for candidate in models:
                            r = index[key,action,candidate["id"]]
                            if candidate["pair"] is None:
                                check("one_direction_can_completely_miss_true_pair", r["objective"]<1e-8,
                                      case=key,action=action,family=candidate["id"])
                for b in (0.,1.7):
                    Kb = b*np.ones((3,3))+np.diag(leaves)+c*np.outer(z,z)
                    check("balanced_actions_cancel_arbitrary_common_root", np.max(abs((Kb-K)@U2))<1e-12,
                          case=key,other_root=b)
    # Uniform set-distance check: a shared diagonal explanation exists under
    # synchronized history even when its generating model has c=.6>0.
    z = np.array([0.,1.,1.])
    K = .4*np.ones((3,3))+np.diag([.3,.4,.5])+.6*np.outer(z,z)
    yh = (K@H).ravel()
    shared_leaves = K@np.ones(3)
    uniform = []
    for i, left in enumerate(models):
        for right in models[i+1:]:
            Hl=feature(left["supports"],H); Hr=feature(right["supports"],H)
            pl,pr=Hl.shape[1],Hr.shape[1]
            E=np.block([[Hl,np.zeros((len(yh),pr))],[np.zeros((len(yh),pl)),Hr]])
            eh=np.r_[yh,yh]
            for action in ("u12","u13","two"):
                U=actions[action]
                D=np.column_stack((feature(left["supports"],U),-feature(right["supports"],U)))
                receipt=solve_l1(D,np.zeros(D.shape[0]),equality=E,rhs=eh)
                receipt.update(left=left["id"],right=right["id"],action=action,
                               history_tolerance=0.,historical_generator_effect=.6,
                               shared_leaf_witness=shared_leaves.tolist())
                uniform.append(receipt)
                write(root/"uniform_receipts"/f"{len(uniform):04d}.json",receipt)
                check("uniform_set_distance_zero_despite_positive_generator",receipt["success"] and receipt["objective"]<1e-8
                      and receipt["max_equality_residual"]<1e-8,uniform_lp=len(uniform))
    # Explicit continuous witness proof, not a discretization inference.
    witness=[]
    for f in models:
        beta=np.zeros(2*len(f["supports"]));beta[:3]=shared_leaves
        witness.append(dict(family=f["id"],history_max_abs=float(np.max(abs(feature(f["supports"],H)@beta-yh))),
                            extra_block_coefficients_exactly_zero=True))
    check("one_shared_witness_in_every_family",all(x["history_max_abs"]<1e-12 for x in witness))
    summary=dict(status="passed" if all(c["passed"] for c in checks) else "failed",
                 profile_LPs=len(lp),uniform_LPs=len(uniform),total_LPs=len(lp)+len(uniform),
                 checks=len(checks),failed_checks=[c for c in checks if not c["passed"]],
                 elapsed_seconds=time.perf_counter()-started,
                 finite_named_truths=18,unique_truths=14,unique_truths_note="two leaf profiles x (one c0 + three c.02 + three c.6)",
                 coefficient_domain="r>=0,x>=0 with no upper bound; all jointly reestimated",
                 positive_result="two balanced directions distinguish every positive proper pair from other proper families; root itself remains unexcited",
                 negative_results=["one direction can erase the true pair","c0 is indistinguishable","synchronized history admits a shared leaf-only explanation", "all 84 two-sided historical set-distance LPs equal zero"],
                 all_profile_LPs_successful=all(x["success"] for x in lp),all_uniform_LPs_successful=all(x["success"] for x in uniform),
                 maximum_uniform_distance=max(x["objective"] for x in uniform if x["success"]),
                 source_sha256=digest(__file__),protocol_sha256=digest(root/"protocol.json"),
                 claim_scope="independent finite numerical checks plus explicit sign/shared-witness algebra; no WZZT recovery test, no price performance claim")
    write(root/"shared_witness.json",witness)
    write(root/"checks.json",checks)
    write(root/"summary.json",summary)
    write(root/"run_ended.json",dict(utc=utc(),status=summary["status"],total_LPs=summary["total_LPs"]))
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["prepare","run"])
    parser.add_argument("--root",required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.root) if args.mode=="prepare" else run(args.root),ensure_ascii=False))
