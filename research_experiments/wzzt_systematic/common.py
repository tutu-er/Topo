from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

CASES = ("paper15", "soumalas11", "flynn16", "pengwah18")
SIZES = (24, 48, 96, 288)
METHODS = ("NJ", "WZZT", "RNJ+WZZT")
CONDITIONS = {
    "ideal_ac": dict(pq_noise=0., v_noise=0., load_factor=1., low_xr=False),
    "reference": dict(pq_noise=.005, v_noise=.0002, load_factor=1., low_xr=False),
    "high_voltage_noise": dict(pq_noise=.005, v_noise=.001, load_factor=1., low_xr=False),
    "higher_load": dict(pq_noise=.005, v_noise=.0002, load_factor=1.5, low_xr=False),
    "low_xr": dict(pq_noise=.005, v_noise=.0002, load_factor=1., low_xr=True),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    import math
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple, set, frozenset)):
            return [clean(v) for v in item]
        if hasattr(item, "tolist"):
            return clean(item.tolist())
        if isinstance(item, float) and not math.isfinite(item):
            return str(item)
        return item
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def indices(total):
    import numpy as np
    if total not in SIZES:
        raise ValueError("undeclared total sample size")
    return np.arange(0, 96, 96 // (total // 3), dtype=int)


def observed(path, total=None):
    import numpy as np
    import pandas as pd
    with np.load(path) as data:
        pick = np.arange(96) if total is None else indices(total)
        labels = data["labels"].astype(int).tolist()
        frames = [pd.DataFrame(data[name][:, pick].reshape(-1, len(labels)), columns=labels)
                  for name in ("P", "Q", "V")]
        root = pd.Series(data["root"][:, pick].reshape(-1))
    return (*frames, root)


def scenes_from_observed(obs):
    import numpy as np
    import pandas as pd
    p, q, v, root = obs
    return [{"name": f"day{day}", "P_terminal": pp, "Q_terminal": q.iloc[rows].copy(),
             "drop_target": pd.DataFrame(root.iloc[rows].to_numpy()[:, None]**2-v.iloc[rows].to_numpy()**2,
                                         index=pp.index, columns=pp.columns)}
            for day, rows in enumerate(np.array_split(np.arange(len(p)), 3))
            for pp in (p.iloc[rows].copy(),)]


def raw_model(model):
    import numpy as np
    return dict(objective_coefficients=np.asarray(model.variables.objective),
        lower_bounds=np.asarray(model.variables.lower), upper_bounds=np.asarray(model.variables.upper),
        integrality=np.asarray(model.variables.integrality),
        constraint_row_index=np.asarray(model.rows.row_index, dtype=int),
        constraint_col_index=np.asarray(model.rows.col_index, dtype=int),
        constraint_data=np.asarray(model.rows.data), constraint_lower=np.asarray(model.rows.lower),
        constraint_upper=np.asarray(model.rows.upper))


def matrices(n, supports, r, x):
    import numpy as np
    result = []
    for weights in (r, x):
        m = np.zeros((n, n))
        for support, weight in zip(supports, weights, strict=True):
            m[np.ix_(support, support)] += weight
        result.append(m)
    return result


def linear_loss(obs, r, x):
    import numpy as np
    p, q, v, root = obs
    target = root.to_numpy()[:, None]**2-v.to_numpy()**2
    error = p.to_numpy()@np.asarray(r).T+q.to_numpy()@np.asarray(x).T-target
    return dict(mae=float(np.mean(abs(error))), rmse=float(np.sqrt(np.mean(error**2))))


def ac_diagnostic(labels, supports, r, x, obs, *, refit):
    import numpy as np
    from research_experiments.wzzt_solver_tuning.ac_screen import ac_score, atom_network
    from rnj_wzzt.models.ac_powerflow import solve_ac_power_flow_timeseries
    result = ac_score(labels, supports, r, x, obs, refit=refit, max_nfev=12)
    if result["valid"]:
        net = atom_network(labels, supports, result["r_refit"], result["x_refit"])
        ac = solve_ac_power_flow_timeseries(net, obs[0], obs[1], v_root=obs[3], max_iter=200, tol=1e-11)
        error = (ac["V_bus_mag"].loc[:, labels].to_numpy()-obs[2].to_numpy())/obs[2].to_numpy()
        day_mse = [float(np.mean(block**2)) for block in np.array_split(error, 3)]
        result.update(block_mse=day_mse, block_mse_se=float(np.std(day_mse, ddof=1)/np.sqrt(3)),
            scope="three physical-day blocks; training-only local impedance fit" if refit else "frozen training parameters; independent test days; no refit")
    return result
