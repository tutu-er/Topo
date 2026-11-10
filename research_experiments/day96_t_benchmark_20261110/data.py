"""Generate paired AC observations; estimators use only the flattened loader."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CASES = ("paper15", "soumalas11", "flynn16", "pengwah18")
SIGMAS = (.0002, .0005, .001)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observed(path):
    """Load every stored slot: train 1x96; validation 3x96; test 1x96."""
    with np.load(Path(path), allow_pickle=False) as archive:
        labels = archive["labels"].astype(int).tolist()
        frames = [pd.DataFrame(np.asarray(archive[k], float).reshape(-1, len(labels)), columns=labels)
                  for k in ("P", "Q", "V")]
        root = pd.Series(np.asarray(archive["root"], float).reshape(-1))
    assert all(f.shape == frames[0].shape for f in frames) and len(root) == len(frames[0])
    return *frames, root


def generate(output, *, bundles=4):
    """Fresh source-matched physical seeds; common random numbers across sigma."""
    from rnj_wzzt.scenario import simulation as sim
    from research_experiments.wzzt_paper.prepare import measurements, settings

    output = Path(output).resolve()
    catalog, errors, all_seeds = {}, [], []
    for case in CASES:
        context = sim._prepare_case_context(case, 96, scenario_suite="reference", root_observation="noisy",
                                             root_meter_noise_rel=.0005, root_sigma=.003, impedance_scale=1.)
        network = output / "networks" / case / "reference"
        network.mkdir(parents=True, exist_ok=True)
        context.net.buses.to_csv(network / "buses.csv", index=False)
        context.net.branches.to_csv(network / "branches.csv", index=False)
        write(network / "metadata.json", dict(root_bus=int(context.net.root_bus), base_kv=float(context.net.base_kv),
                                               base_mva=float(context.net.base_mva), source=context.net.metadata))
        for seed_index in range(bundles):
            bundle = 1000 + seed_index
            physical, error = {}, None
            try:
                for split, offset, days in (("train", 0, (0,)), ("validation", 1, (0, 1, 2)), ("test", 2, (0,))):
                    replicate = 10000 + 10 * seed_index + offset
                    physical[split] = []
                    for day in days:
                        base = sim._build_base_trajectory(context, day, replicate)
                        ac, voltage = sim._solve_physical_ac(context, base.p_base, base.q_base, base.root_true)
                        assert int(ac["converged"].sum()) == 96 and np.isfinite(voltage.to_numpy()).all()
                        p, q = base.p_base.loc[:, context.terminals], base.q_base.loc[:, context.terminals]
                        physical[split].append((base, p, q, voltage))
                        all_seeds.append(dict(case=case, bundle=bundle, split=split, day=day,
                                              replicate=replicate, profile_seed=base.shifted_seed, profile=base.profile))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            for sigma in SIGMAS:
                job = f"day96__{case}__b{bundle}__reference__v{round(sigma*10000):04d}__N96"
                folder = output / "data" / case / f"b{bundle}" / f"v{round(sigma*10000):04d}"
                folder.mkdir(parents=True, exist_ok=True)
                config, diagnostics = settings("reference", sigma), []
                ctx = replace(context, settings={**context.settings, "root_meter_noise_rel": sigma})
                try:
                    if error:
                        raise RuntimeError(error)
                    for split, records in physical.items():
                        true, measured = {k: [] for k in ("P", "Q", "V", "root")}, {k: [] for k in ("P", "Q", "V", "root")}
                        for day, (base, p, q, v) in enumerate(records):
                            obs = measurements(sim, ctx, base, p, q, v, config)
                            for key, value in zip(true, (p, q, v, base.root_true)):
                                true[key].append(value.to_numpy()); measured[key].append(obs[key])
                            design = np.c_[p.to_numpy(), q.to_numpy()]
                            design -= design.mean(axis=0)
                            signal = base.root_true.to_numpy()[:, None] ** 2 - v.to_numpy() ** 2
                            noise = obs["root"][:, None] ** 2 - obs["V"] ** 2 - signal
                            diagnostics.append(dict(split=split, day=day, replicate=base.replicate, profile=base.profile,
                                                    seed=base.shifted_seed, slots=96, voltage_min=float(v.to_numpy().min()),
                                                    voltage_max=float(v.to_numpy().max()), design_rank=int(np.linalg.matrix_rank(design)),
                                                    condition_number=(float(np.linalg.cond(design))
                                                                      if np.isfinite(np.linalg.cond(design)) else None),
                                                    dynamic_signal_rms=float(np.sqrt(np.mean((signal-signal.mean(axis=0))**2))),
                                                    dynamic_noise_rms=float(np.sqrt(np.mean((noise-noise.mean(axis=0))**2)))))
                        for kind, arrays in (("truth", true), ("observed", measured)):
                            np.savez_compressed(folder / f"{split}_{kind}.npz", labels=context.terminals,
                                                **{k: np.stack(v) for k, v in arrays.items()})
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    errors.append(dict(job=job, error=error))
                write(folder / "diagnostics.json", dict(config=config, records=diagnostics, data_error=error))
                catalog[job] = dict(job=job, panel="day96", case=case, bundle=bundle, seed_index=seed_index,
                                    N=96, voltage_sigma=sigma, condition="reference", data_error=error,
                                    train=str(folder / "train_observed.npz"), validation=str(folder / "validation_observed.npz"),
                                    test=str(folder / "test_observed.npz"), truth_network=str(network),
                                    split_points=dict(train=96, validation=288, test=96), train_days=[0],
                                    train_replicate=10000+10*seed_index, labels=list(map(int, context.terminals)),
                                    root_label=int(context.net.root_bus), base_kw=1000.*float(context.net.base_mva))
                print("DATA", job, error or "OK", flush=True)
    write(output / "catalog.json", catalog)
    write(output / "data_errors.json", errors)
    write(output / "physical_seeds.json", all_seeds)
    return catalog
