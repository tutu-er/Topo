import json
from pathlib import Path

from topoident.innovation_active import run_active_probe_design
from topoident.innovation_bayesian import run_bayesian_hidden_loads
from topoident.innovation_conformal import run_conformal_adaptive_pmu
from topoident.innovation_hmm import run_dynamic_hmm
from topoident.innovation_potential import run_potential_sparse_recovery
from topoident.innovation_topology_free import run_topology_free_latent_tree


if __name__ == "__main__":
    results = {
        "innovation_1": run_bayesian_hidden_loads(),
        "innovation_2": run_active_probe_design(),
        "innovation_3": run_dynamic_hmm(),
        "innovation_4": run_potential_sparse_recovery(),
        "innovation_5": run_conformal_adaptive_pmu(),
        "innovation_6": run_topology_free_latent_tree(),
    }
    path = Path("results/innovation_suite.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
