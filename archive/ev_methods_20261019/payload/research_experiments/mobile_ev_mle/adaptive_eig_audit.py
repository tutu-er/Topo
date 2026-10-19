"""Read-only prospective EIG replay of frozen adaptive-MILP pilot plans.

The prior is a declared 80-atom engineering model, not a quadrature of the
continuous AC fitting family or the pilot's generating population. Only past
meters and each already planned action's P/Q enter this audit. Pilot truth,
future observations and stored measurement-noise arrays are never accessed.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
from scipy.special import logsumexp

try:
    from . import audit_common, bayes, bayes_rb, likelihood, simulation
    from .adaptive_statistics import replay_eig_shortlist
    from .candidates import RadialCandidate
except ImportError:
    from research_experiments.mobile_ev_mle import audit_common, bayes, bayes_rb, likelihood, simulation
    try:
        from .adaptive_statistics import replay_eig_shortlist
    except ImportError:
        from adaptive_statistics import replay_eig_shortlist
    from research_experiments.mobile_ev_mle.candidates import RadialCandidate


CHANNELS = ("p", "q", "v", "master_p", "master_q")
HISTORY_KEYS = tuple("history_" + key for key in (*CHANNELS, "root_v"))
TOPOLOGY_NAMES = ("star", "clade_01", "clade_02", "clade_12")
FAMILIES = ((), ((0, 1),), ((0, 2),), ((1, 2),))
INTERNAL = ((.00003, .00002), (.006, .004), (.012, .007))
SENSITIVITY_GRID = tuple(product((.1, .25, .5), (.25, .75)))


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _readonly(value, dtype=float):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def array_hash(*arrays):
    digest = sha256()
    for value in arrays:
        array = np.ascontiguousarray(value, dtype="<f8")
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _json_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _meter_hash(meter):
    return array_hash(*[getattr(meter, name+"_sigma") for name in CHANNELS],
                      *[getattr(meter, "bias_"+name) for name in ("p", "q", "v")])


@dataclass(frozen=True)
class PlannedAction:
    """Actual frozen terminal EV injections in pu, independent of atom truth."""
    name: str
    p: np.ndarray
    q: np.ndarray

    def __post_init__(self):
        p, q = np.asarray(self.p, float), np.asarray(self.q, float)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("action name must be a nonempty string")
        if p.ndim != 2 or not all(p.shape) or p.shape != q.shape or p.shape[1] != 3:
            raise ValueError("planned P/Q must have common [future_slots,3] shape")
        if not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError("planned P/Q must be finite")
        object.__setattr__(self, "p", _readonly(p))
        object.__setattr__(self, "q", _readonly(q))

    def payload(self):
        return dict(p=self.p.tolist(), q=self.q.tolist(), unit="pu", base_kw=20.)


def extract_planned_actions(trial: Mapping):
    """Whitelist access: records -> method, action -> p/q; no other fields."""
    result = {}
    for row in trial["records"]:
        name = row["method"]
        given = row["action"]
        action = PlannedAction(name, given["p"], given["q"])
        if name in result:
            raise ValueError("duplicate planned method name")
        result[name] = action
    if not result or len({a.p.shape for a in result.values()}) != 1:
        raise ValueError("a nonempty common-horizon planned action set is required")
    return {name: result[name] for name in sorted(result)}


def history_from_arrays(arrays: Mapping):
    """Load only the six allowed historical NPZ entries, never future arrays."""
    values = {key.removeprefix("history_"): np.asarray(arrays[key], float).copy()
              for key in HISTORY_KEYS}
    observation = likelihood.Observations(**values)
    if observation.n != 3:
        raise ValueError("this declared prior is for three terminals only")
    return observation


def read_history(path):
    # Lazy NPZ indexing is essential: unrelated object/truth/noise entries are
    # not unpickled or inspected. Do not replace this whitelist by dict(npz).
    with np.load(path, allow_pickle=False) as arrays:
        return history_from_arrays(arrays)


def source_hashes():
    modules = (audit_common, bayes, bayes_rb, likelihood, simulation)
    files = {Path(module.__file__).name: sha256(Path(module.__file__).read_bytes()).hexdigest()
             for module in modules}
    files[Path(__file__).name] = sha256(Path(__file__).read_bytes()).hexdigest()
    import inspect
    replay_file = Path(inspect.getfile(replay_eig_shortlist))
    files[replay_file.name] = sha256(replay_file.read_bytes()).hexdigest()
    # The imported candidate's actual definition can live in the old AC module.
    candidate_file = Path(inspect.getfile(RadialCandidate))
    files[candidate_file.name] = sha256(candidate_file.read_bytes()).hexdigest()
    return files


@dataclass(frozen=True)
class EngineeringBank:
    candidates: tuple
    atoms: tuple
    labels: np.ndarray
    history_means: np.ndarray
    meter: object
    history_count: int
    future_count: int
    history_root_v: np.ndarray
    atom_hash: str
    diagnostics: dict


def engineering_prior(bank, *, star_mass=.25, weak_mass=1/3):
    """Topology mass first, then each topology's own normalized atom masses."""
    if (not np.isfinite(star_mass) or not np.isfinite(weak_mass)
            or not 0 < star_mass < 1 or not 0 < weak_mass < 1):
        raise ValueError("star_mass and weak_mass must be interior proper probabilities")
    prior = []
    for atom in bank.atoms:
        if atom["topology"] == 0:
            weight = star_mass / 8
        else:
            strength_mass = weak_mass if atom["strength"] == 0 else (1-weak_mass)/2
            weight = ((1-star_mass)/3) * strength_mass / 8
        prior.append(weight)
    result = np.asarray(prior)
    if np.any(result <= 0) or not np.isclose(result.sum(), 1, atol=1e-12, rtol=0):
        raise AssertionError("engineering prior is not normalized positive mass")
    return result


def _atom_response(candidate, atom, meter, rows, root_v, action=None):
    p, q = atom["p"][rows].copy(), atom["q"][rows].copy()
    if action is not None:
        if action.p.shape != p.shape:
            raise ValueError("planned action does not match future atom path")
        p += action.p
        q += action.q
    physical = likelihood.ac_predict(candidate, atom["r"], atom["x"], p, q,
        atom["source_p"][rows], atom["source_q"][rows], root_v=root_v)
    voltage = abs(physical["voltage"])
    if (voltage.min() < .85 or voltage.max() > 1.15
            or np.max(abs(p)) > 1 or np.max(abs(q)) > 1
            or np.max(abs(atom["source_p"][rows])) > .5
            or np.max(abs(atom["source_q"][rows])) > .5):
        raise ValueError("required engineering atom/action violates the declared AC domain")
    observation = likelihood.Observations(p=p+meter.bias_p, q=q+meter.bias_q,
        v=physical["v"]+meter.bias_v, master_p=physical["master_p"],
        master_q=physical["master_q"], root_v=root_v)
    return audit_common.vectorize(observation, meter), physical


def build_finite_bank(history_count=12, future_count=8, *, history_root_v=None):
    """80 atoms: star8 plus three cherries24; no observations define the prior."""
    h = _positive_integer(history_count, "history_count")
    n = _positive_integer(future_count, "future_count")
    root = np.ones(h) if history_root_v is None else np.broadcast_to(
        np.asarray(history_root_v, float), (h,)).copy()
    if not np.isfinite(root).all() or np.any(root <= 0):
        raise ValueError("historical known root voltage must be finite and positive")
    candidates = tuple(RadialCandidate.from_supports(name, 3, family)
                       for name, family in zip(TOPOLOGY_NAMES, FAMILIES))
    meter = simulation.model(3)
    atoms, labels, history_means = [], [], []
    maximum_equation = maximum_balance = 0.
    minimum_voltage, maximum_voltage = np.inf, -np.inf
    for label, candidate in enumerate(candidates):
        strengths = (None,) if label == 0 else range(3)
        for rscale, xscale, strength, path_id in product((.85, 1.15), (.85, 1.15), strengths, range(2)):
            internal = None if strength is None else INTERNAL[strength]
            r = np.array([.005 if len(s) == 3 else .028+.003*s[0]
                          if len(s) == 1 else internal[0] for s in candidate.supports])*rscale
            x = np.array([.0035 if len(s) == 3 else .011+.002*s[0]
                          if len(s) == 1 else internal[1] for s in candidate.supports])*xscale
            p, q, sp, sq = audit_common.latent_path(path_id, total=h+n, n=3)
            atom = dict(topology=label, r=_readonly(r), x=_readonly(x), p=_readonly(p), q=_readonly(q),
                        source_p=_readonly(sp), source_q=_readonly(sq), rscale=rscale, xscale=xscale,
                        strength=strength, path_id=path_id)
            mean, physical = _atom_response(candidate, atom, meter, slice(0, h), root)
            atoms.append(atom); labels.append(label); history_means.append(mean)
            maximum_equation = max(maximum_equation, physical["ac_equation_residual"])
            maximum_balance = max(maximum_balance, physical["power_balance_residual"])
            minimum_voltage = min(minimum_voltage, float(abs(physical["voltage"]).min()))
            maximum_voltage = max(maximum_voltage, float(abs(physical["voltage"]).max()))
    atom_identity = [dict(topology=a["topology"], rscale=a["rscale"], xscale=a["xscale"],
        strength=a["strength"], path_id=a["path_id"], arrays=array_hash(
            a["r"], a["x"], a["p"], a["q"], a["source_p"], a["source_q"])) for a in atoms]
    diagnostics = dict(atom_count=len(atoms), atoms_per_topology={name:int(labels.count(i))
        for i, name in enumerate(TOPOLOGY_NAMES)}, topology_names=list(TOPOLOGY_NAMES),
        atom_manifest=atom_identity, source_point="common physical node1", future_root_v=1.,
        shared_history_future_lines=True, baseline_paths="audit_common.latent_path: two declared sinusoidal paths",
        known_calibrated_fixed_bias=True, history_channels=list(CHANNELS),
        meter_hash=_meter_hash(meter), historical_ev_prior="zero; this audit is for the original pilot history",
        measurement_model="independent known scalar Gaussian standard deviations; root known",
        ac_equation_residual=maximum_equation, power_balance_residual=maximum_balance,
        history_voltage_min=minimum_voltage, history_voltage_max=maximum_voltage,
        scope="declared finite engineering prior; distinct from pilot generating population",
        continuous_quadrature=False, continuous_parameter_error_control=False,
        posterior_concentration_is_topology_correctness_evidence=False,
        prior_chosen_before_reading_history=True)
    if len(atoms) != 80 or tuple(labels.count(i) for i in range(4)) != (8, 24, 24, 24):
        raise AssertionError("declared all-four engineering bank changed unexpectedly")
    return EngineeringBank(candidates, tuple(atoms), _readonly(labels, int), _readonly(history_means),
        meter, h, n, _readonly(root), _json_hash(atom_identity), diagnostics)


def replay_action(bank, action):
    """Replay this exact planned P/Q at every atom; no nearest-library action."""
    if action.p.shape != (bank.future_count, 3):
        raise ValueError("actual planned action horizon differs from bank")
    means = []
    vmin, vmax, equation, balance = np.inf, -np.inf, 0., 0.
    for atom, label in zip(bank.atoms, bank.labels):
        mean, physical = _atom_response(bank.candidates[label], atom, bank.meter,
            slice(bank.history_count, bank.history_count+bank.future_count), 1., action)
        means.append(mean)
        vmin = min(vmin, float(abs(physical["voltage"]).min()))
        vmax = max(vmax, float(abs(physical["voltage"]).max()))
        equation = max(equation, physical["ac_equation_residual"])
        balance = max(balance, physical["power_balance_residual"])
    return np.asarray(means), dict(voltage_min=vmin, voltage_max=vmax,
        ac_equation_residual=equation, power_balance_residual=balance,
        actual_planned_pq=True, ev_added_once=True, source_node=1,
        operational_feasibility_checked_by_this_module=False)


def _history_fit_diagnostics(bank, history_y, posterior):
    best = int(np.argmin(posterior.history_loss))
    residual = bank.history_means[best]-history_y
    offset, channels = 0, {}
    for channel in CHANNELS:
        size = bank.history_count*(3 if channel in ("p", "q", "v") else 1)
        part = residual[offset:offset+size]; offset += size
        channels[channel] = dict(whitened_rmse=float(np.sqrt(np.mean(part**2))),
                                  squared_residual=float(part@part))
    return dict(best_history_loss=float(posterior.history_loss[best]),
        history_dimension=len(history_y), best_loss_per_channel_dimension=float(
            posterior.history_loss[best]/len(history_y)), best_atom_index=best,
        per_channel=channels, loss_is_not_calibrated_goodness_of_fit_pvalue=True,
        support_misspecification_requires_external_predictive_check=True)


def _validation_summary(results, chosen, names):
    comparisons, action_summary = {}, {}
    for name in names:
        rows = [result[name] for result in results]
        action_summary[name] = dict(eig=float(np.mean([row["eig"] for row in rows])),
            eig_se=float(np.sqrt(sum(row["eig_se"]**2 for row in rows))/len(rows)),
            seed_estimates=[row["eig"] for row in rows],
            seed_standard_errors=[row["eig_se"] for row in rows])
        pairs = [result[chosen]["paired_comparisons"][name] for result in results]
        comparisons[name] = dict(eig_gain=float(np.mean([row["eig_gain"] for row in pairs])),
            eig_gain_se=float(np.sqrt(sum(row["eig_gain_se"]**2 for row in pairs))/len(pairs)),
            seed_gains=[row["eig_gain"] for row in pairs],
            seed_standard_errors=[row["eig_gain_se"] for row in pairs])
    return dict(action_summary=action_summary, frozen_winner_comparisons=comparisons,
                independent_integration_repeats=len(results))


def _entropy_diagnostics(posterior, tolerance):
    log_mass = posterior.log_topology_probabilities
    negative = log_mass < 0
    # This log-domain diagnostic retains terms whose probability exponentials
    # underflow. It is computed from rounded double log masses, not arbitrary
    # precision or a certified entropy interval.
    log_entropy = float(logsumexp(log_mass[negative]+np.log(-log_mass[negative]))) if negative.any() else None
    return dict(tolerance=float(tolerance), saturated=bool(posterior.topology_entropy <= tolerance),
        maximum_topology_mass=float(posterior.topology_probabilities.max()),
        zero_displayed_topology_mass_count=int(np.count_nonzero(posterior.topology_probabilities == 0)),
        log_entropy_from_preserved_double_log_masses=log_entropy,
        log_entropy_is_arbitrary_precision_certificate=False,
        interpretation="finite-support conditional entropy upper bound; numerical zero is not exact zero information or correct topology")


def audit_history(history, actions, *, bank=None, star_mass=.25, weak_mass=1/3,
                  selection_samples=2048, validation_samples=(2048, 8192),
                  validation_repeats=3, seed=630001, batch_size=128,
                  entropy_tolerance=1e-10, scorer=None, replay_cache=None):
    """Prospective shortlist audit; selection and validation streams are distinct."""
    if not isinstance(actions, Mapping) or not actions:
        raise ValueError("actual planned actions must be a nonempty mapping")
    names = sorted(actions)
    if any(name != actions[name].name for name in names):
        raise ValueError("action mapping key and method name differ")
    nfuture = len(actions[names[0]].p)
    if bank is None:
        bank = build_finite_bank(history.t, nfuture, history_root_v=history.root_v)
    if bank.history_count != history.t or nfuture != bank.future_count or not np.array_equal(
            bank.history_root_v, history.root_v):
        raise ValueError("bank does not condition on these known historical root inputs")
    if _meter_hash(bank.meter) != bank.diagnostics["meter_hash"]:
        raise ValueError("bank meter changed after historical means were frozen")
    selection_samples = _positive_integer(selection_samples, "selection_samples")
    budgets = tuple(_positive_integer(value, "validation_samples") for value in validation_samples)
    if not budgets or len(set(budgets)) != len(budgets):
        raise ValueError("validation budgets must be nonempty and unique")
    repeats = _positive_integer(validation_repeats, "validation_repeats")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integration seed")
    if not np.isfinite(entropy_tolerance) or entropy_tolerance < 0:
        raise ValueError("entropy tolerance must be finite nonnegative")
    scorer = bayes_rb.rb_eig_scores if scorer is None else scorer
    prior = engineering_prior(bank, star_mass=star_mass, weak_mass=weak_mass)
    historical_y = audit_common.vectorize(history, bank.meter)
    posterior = bayes.posterior_weights(historical_y, bank.history_means, bank.labels, prior)
    cache = {} if replay_cache is None else replay_cache
    means, physical = {}, {}
    # Only exact controls are shared. Different source/atom family or horizons
    # have a different cache key; approximate nearest-action lookup is absent.
    for name in names:
        action = actions[name]
        key = (bank.atom_hash, bank.future_count, bank.diagnostics["meter_hash"], array_hash(action.p, action.q))
        if key not in cache:
            cache[key] = replay_action(bank, action)
        means[name], physical[name] = cache[key]
    context_id = _json_hash(dict(atom_hash=bank.atom_hash, prior_hash=array_hash(prior),
        history_hash=array_hash(historical_y, history.root_v),
        actions={name:array_hash(actions[name].p, actions[name].q) for name in names}))
    selection = replay_eig_shortlist(posterior, actions, lambda action: means[action.name],
        samples_per_topology=selection_samples, selection_seed=int(seed), validation_seeds=(),
        batch_size=batch_size, replay_context_id=context_id, scorer=scorer)
    chosen = selection["selected_action"]
    validation = []
    stride = max(1000, repeats+1)
    for level, samples in enumerate(budgets):
        seeds = [int(seed)+(level+1)*stride+repeat+1 for repeat in range(repeats)]
        results = [scorer(posterior, means, samples_per_topology=samples,
                          seed=integration_seed, batch_size=batch_size) for integration_seed in seeds]
        validation.append(dict(samples_per_topology=samples, integration_seeds=seeds,
            scores=results, **_validation_summary(results, chosen, names)))
    all_scores = [selection["selection_scores"], *[result for level in validation for result in level["scores"]]]
    maximum_abs_identity = max(abs(result[name]["identity_difference"])
                               for result in all_scores for name in names)
    entropy = _entropy_diagnostics(posterior, entropy_tolerance)
    mean_hashes = {name:array_hash(means[name]) for name in names}
    equivalents = {}
    for name, digest in mean_hashes.items():
        equivalents.setdefault(digest, []).append(name)
    return dict(status="complete", selected_action=chosen,
        selection_is_audit_shortlist_only=True, original_pilot_actions_modified=False,
        action_order=names, actions={name:actions[name].payload() for name in names},
        prior=dict(star_mass=float(star_mass), each_cherry_mass=float((1-star_mass)/3),
            weak_mass_given_cherry=float(weak_mass), r_x_scales=[.85,1.15],
            atom_weights=prior.tolist(), prior_hash=array_hash(prior), scope=bank.diagnostics["scope"]),
        posterior=posterior.summary(), posterior_log_atom_weights=posterior.log_weights.tolist(),
        history_fit=_history_fit_diagnostics(bank, historical_y, posterior),
        entropy_saturation=entropy,
        shortlist_rank_is_numerical_only_when_entropy_saturated=entropy["saturated"],
        shortlist_ranking_requires_independent_validation=True,
        identical_predictive_mean_action_groups=[group for group in equivalents.values() if len(group)>1],
        selection=selection, validation=validation, maximum_abs_identity_difference=maximum_abs_identity,
        atom_hash=bank.atom_hash, history_mean_hash=array_hash(bank.history_means),
        meter_hash=bank.diagnostics["meter_hash"],
        history_input_hash=array_hash(*[getattr(history, key) for key in (*CHANNELS, "root_v")]),
        future_mean_hashes=mean_hashes,
        action_hashes={name:array_hash(actions[name].p, actions[name].q) for name in names},
        physical_replay=physical, integration_seed=int(seed),
        selector_uses_evaluation_truth=False, selector_uses_realized_future_noise=False,
        selector_reads_future_observations=False,
        topology_false_rejection_control_claim=False, continuous_prior_approximation_claim=False,
        engineering_prior_matches_pilot_population=False)


def _dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run_audit(input_dir, output_dir, *, limit=8, sensitivity_limit=2,
              selection_samples=2048, validation_samples=(2048,8192), validation_repeats=3,
              seed=630001, batch_size=128):
    """Read current pilot files, access only whitelisted past/action fields."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("audit output must differ from frozen input directory")
    limit = _positive_integer(limit, "limit")
    if not isinstance(sensitivity_limit, int) or isinstance(sensitivity_limit, bool) or sensitivity_limit < 0:
        raise ValueError("sensitivity_limit must be a nonnegative integer")
    files = sorted(input_dir.glob("trial_*.json"))[:limit]
    if not files:
        raise ValueError("no trial_*.json pilot inputs found")
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared, input_manifest = [], []
    # Freeze the actually accessed projection before any integration. Changing
    # truth fields does not affect this manifest; changing past meters/plans
    # does, and cannot silently overwrite a previous audit configuration.
    for path in files:
        name = path.stem
        try:
            history = read_history(path.with_name(name+"_observations.npz"))
            actions = extract_planned_actions(json.loads(path.read_text(encoding="utf-8")))
            prepared.append((path, history, actions, None))
            input_manifest.append(dict(input_identifier=name, status="read_allowed_fields",
                history_hash=array_hash(*[getattr(history,key) for key in (*CHANNELS,"root_v")]),
                actions_hash=_json_hash({method:action.payload() for method,action in actions.items()})))
        except (ValueError,KeyError,OSError) as error:
            prepared.append((path, None, None, str(error)))
            input_manifest.append(dict(input_identifier=name,status="input_failure_preserved",error=str(error)))
    config = dict(version="all4_engineering_prior_eig_audit_v1", source_sha256=source_hashes(),
        limit=limit, sensitivity_limit=sensitivity_limit, selection_samples=selection_samples,
        validation_samples=list(validation_samples), validation_repeats=validation_repeats,
        seed=seed, batch_size=batch_size, prior_scope="80-atom declared finite engineering prior",
        allowed_input_manifest=input_manifest,
        sensitivity_grid=[dict(star_mass=s,weak_mass=w) for s,w in SENSITIVITY_GRID],
        input_contract=dict(history_npz_keys=list(HISTORY_KEYS), trial_json_fields=["records.method","records.action.p","records.action.q"],
            truth_case_seed_noise_and_future_fields_accessed=False),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    if (output_dir/"config.json").exists():
        previous = json.loads((output_dir/"config.json").read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("refusing to mix audit configurations in an existing output directory")
    _dump(output_dir/"config.json", config)
    summaries = []
    for index, (path, history, actions, input_error) in enumerate(prepared):
        started = perf_counter()
        name = path.stem
        try:
            if input_error is not None:
                raise ValueError(input_error)
            bank = build_finite_bank(history.t, len(next(iter(actions.values())).p), history_root_v=history.root_v)
            cache = {}
            base = audit_history(history, actions, bank=bank, selection_samples=selection_samples,
                validation_samples=validation_samples, validation_repeats=validation_repeats,
                seed=seed+index*100000, batch_size=batch_size, replay_cache=cache)
            sensitivities = []
            if index < sensitivity_limit:
                for prior_index, (star,weak) in enumerate(SENSITIVITY_GRID):
                    sensitivities.append(audit_history(history, actions, bank=bank, star_mass=star,
                        weak_mass=weak, selection_samples=selection_samples, validation_samples=validation_samples,
                        validation_repeats=validation_repeats, seed=seed+index*100000+(prior_index+1)*10000,
                        batch_size=batch_size, replay_cache=cache))
            row = dict(input_identifier=name, baseline=base, prior_sensitivities=sensitivities,
                       finite_bank=bank.diagnostics, seconds=perf_counter()-started, status="complete")
            summary = dict(input_identifier=name,status="complete",selected_action=base["selected_action"],
                topology_entropy=base["posterior"]["topology_entropy"], saturated=base["entropy_saturation"]["saturated"],
                best_history_loss=base["history_fit"]["best_history_loss"],
                history_dimension=base["history_fit"]["history_dimension"],
                prior_sensitivity_selections=[dict(star_mass=s["prior"]["star_mass"],weak_mass=s["prior"]["weak_mass_given_cherry"],
                    selected_action=s["selected_action"],topology_entropy=s["posterior"]["topology_entropy"]) for s in sensitivities],
                seconds=row["seconds"])
        except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError,KeyError,OSError) as error:
            row = dict(input_identifier=name,status="audit_failure_preserved",error=str(error),seconds=perf_counter()-started)
            summary = row.copy()
        _dump(output_dir/(name+"_eig.json"),row)
        summaries.append(summary)
        _dump(output_dir/"progress.json",dict(config=config,completed=len(summaries),histories=summaries))
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    result = dict(config=config,history_count=len(summaries),histories=summaries,
        complete_history_count=sum(row["status"]=="complete" for row in summaries),
        topology_correctness_inferred=False, continuous_parameter_guarantee=False)
    _dump(output_dir/"summary.json",result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--limit",type=int,default=8)
    parser.add_argument("--sensitivity-limit",type=int,default=2)
    parser.add_argument("--selection-samples",type=int,default=2048)
    parser.add_argument("--validation-samples",type=int,nargs="+",default=[2048,8192])
    parser.add_argument("--validation-repeats",type=int,default=3)
    parser.add_argument("--seed",type=int,default=630001)
    parser.add_argument("--batch-size",type=int,default=128)
    args = parser.parse_args()
    run_audit(args.input,args.output,limit=args.limit,sensitivity_limit=args.sensitivity_limit,
        selection_samples=args.selection_samples,validation_samples=args.validation_samples,
        validation_repeats=args.validation_repeats,seed=args.seed,batch_size=args.batch_size)


if __name__ == "__main__":
    main()
