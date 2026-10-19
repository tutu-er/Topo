"""Predeclared finite AC family for independently auditable small examples.

The finite support is a verification model, not a quadrature claim about the
existing continuous free-line/free-state AC likelihood. No observations or
truth enter construction of its atoms, their prior or its action library.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import hashlib
import numpy as np

from .likelihood import Observations, ac_predict
from .simulation import model as meter_model
from .candidates import RadialCandidate
from .design import EVAction, EVConstraints, enumerate_actions, check_action


@dataclass
class FiniteACBank:
    candidates: list
    atoms: list
    labels: np.ndarray
    prior: np.ndarray
    history_means: np.ndarray
    future_means: dict
    actions: dict
    constraints: EVConstraints
    meter: object
    history_count: int
    future_count: int
    diagnostics: dict


def vectorize(obs, meter):
    """Whiten all measured channels, without subtracting the fixed offsets."""
    chunks = []
    for channel in ("p", "q", "v", "master_p", "master_q"):
        value = getattr(obs, channel)
        sigma = getattr(meter, channel + "_sigma")
        if value is None or sigma is None:
            raise ValueError("audit family requires all P/Q/V and master P/Q channels")
        chunks.append((np.asarray(value) / np.asarray(sigma)).ravel())
    return np.concatenate(chunks)


def unvectorize(value, meter, count, n=3):
    value = np.asarray(value, float)
    if value.shape != (count * (3*n+2),):
        raise ValueError("whitened observation dimension differs from all-channel model")
    values, offset = {}, 0
    for channel in ("p", "q", "v", "master_p", "master_q"):
        shape = (count, n) if channel in ("p", "q", "v") else (count,)
        size = int(np.prod(shape))
        values[channel] = value[offset:offset+size].reshape(shape) * getattr(meter, channel+"_sigma")
        offset += size
    return Observations(**values, root_v=np.ones(count))


def latent_path(path_id, total=20, n=3):
    """Two proper, predeclared time-varying load/source trajectories."""
    t = np.arange(total, dtype=float)
    i = np.arange(n, dtype=float)
    phase = .65 * path_id
    common = .010 * np.sin(.43*t+phase)
    p = .12 + common[:,None] + .003*np.cos(.91*t[:,None]+i[None,:]+phase)
    q = .04 + .25*common[:,None] + .0025*np.sin(.73*t[:,None]+1.2*i[None,:]+phase)
    sp = .10 + .03*np.sin(.89*t+phase)
    sq = .035 + .012*np.cos(.69*t+phase)
    return p, q, sp, sq


def atom_mean(candidate, atom, meter, rows, action=None):
    p, q = atom["p"][rows].copy(), atom["q"][rows].copy()
    if action is not None:
        p += action.p
        q += action.q
    pred = ac_predict(candidate, atom["r"], atom["x"], p, q,
                      atom["source_p"][rows], atom["source_q"][rows], root_v=1.)
    obs = Observations(p=p+meter.bias_p, q=q+meter.bias_q,
                       v=pred["v"]+meter.bias_v, root_v=1.,
                       master_p=pred["master_p"], master_q=pred["master_q"])
    return vectorize(obs, meter), pred


def build_bank(history_count=12, future_count=8):
    if (history_count, future_count) != (12,8):
        raise ValueError("frozen first-stage family has H=12, N=8")
    candidates = [RadialCandidate.from_supports("clade01",3,((0,1),)),
                  RadialCandidate.from_supports("clade02",3,((0,2),))]
    meter = meter_model(3)
    access = np.ones((future_count,3), bool)
    access[:2,2] = False
    move = np.ones((3,3))-np.eye(3)
    constraints = EVConstraints(p_max=.3,energy=.3,base_kw=20.,
        availability=access,travel_slots=move.astype(int),travel_energy=.01*move,
        soc_min=.1,initial_site=0)
    feasible, rejected = enumerate_actions(future_count,3,constraints,max_actions=11)
    zero = EVAction("none",np.zeros((future_count,3)),np.zeros((future_count,3)),
                    np.zeros(future_count,int),"none")
    actions = {a.name:a for a in [zero]+feasible}
    atoms, labels = [], []
    for label, candidate in enumerate(candidates):
        for rscale,xscale,strength,path_id in product((.85,1.15),(.85,1.15),range(3),range(2)):
            internal = ((.00003,.00002),(.006,.004),(.012,.007))[strength]
            r = np.array([.005 if len(s)==3 else .028+.003*s[0]
                          if len(s)==1 else internal[0] for s in candidate.supports])*rscale
            x = np.array([.0035 if len(s)==3 else .011+.002*s[0]
                          if len(s)==1 else internal[1] for s in candidate.supports])*xscale
            p,q,sp,sq = latent_path(path_id)
            atoms.append(dict(topology=label,r=r,x=x,p=p,q=q,source_p=sp,source_q=sq,
                              rscale=rscale,xscale=xscale,strength=strength,path_id=path_id))
            labels.append(label)
    labels = np.asarray(labels,int)
    historical = []
    future = {name:[] for name in actions}
    max_eq, max_balance, vmin, vmax = 0.,0.,np.inf,-np.inf
    # A failed atom invalidates the bank. Never condition on successful atoms.
    for atom,label in zip(atoms,labels):
        mean, physical = atom_mean(candidates[label],atom,meter,slice(0,history_count))
        historical.append(mean)
        for name,action in actions.items():
            mean, physical = atom_mean(candidates[label],atom,meter,slice(history_count,None),action)
            future[name].append(mean)
            max_eq = max(max_eq,physical["ac_equation_residual"])
            max_balance = max(max_balance,physical["power_balance_residual"])
            vmin = min(vmin,float(np.min(abs(physical["voltage"]))))
            vmax = max(vmax,float(np.max(abs(physical["voltage"]))))
            if not (.9 <= np.min(abs(physical["voltage"])) and np.max(abs(physical["voltage"])) <= 1.1):
                raise ValueError(f"unsafe finite atom/action: {label}, {name}")
    prior = np.full(len(atoms),1/len(atoms))
    checks = {name:check_action(a,constraints) for name,a in actions.items() if name!="none"}
    if not all(row["feasible"] for row in checks.values()):
        raise ValueError("frozen finite action library contains infeasible action")
    return FiniteACBank(candidates,atoms,labels,prior,np.asarray(historical),
        {name:np.asarray(means) for name,means in future.items()},actions,constraints,meter,
        history_count,future_count,dict(atom_count=len(atoms),atoms_per_topology=24,
            source_point="node1",ac_equation_residual=max_eq,power_balance_residual=max_balance,
            voltage_min=vmin,voltage_max=vmax,action_checks=checks,rejected_actions=rejected,
            continuous_prior_approximation_claim=False))


def changed_prior(bank, topology_mass=.5, weak_mass=1/3):
    if not 0<topology_mass<1 or not 0<weak_mass<1:
        raise ValueError("sensitivity masses must be interior proper probabilities")
    result = np.empty(len(bank.atoms))
    for i,atom in enumerate(bank.atoms):
        tm = topology_mass if atom["topology"]==0 else 1-topology_mass
        sm = weak_mass if atom["strength"]==0 else (1-weak_mass)/2
        result[i] = tm * sm / 8  # independent R/X scales and two paths
    if not np.isclose(result.sum(),1):
        raise AssertionError("prior masses do not sum to one")
    return result


def evaluation_atom(bank, seed, strength, topology):
    """Evaluation-only selector of generator; never called by design methods."""
    possible = [i for i,a in enumerate(bank.atoms)
                if a["strength"]==strength and a["topology"]==topology]
    return possible[int(np.random.default_rng(seed+111).integers(len(possible)))]


def array_hash(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        value = np.asarray(value,dtype="<f8")
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()
