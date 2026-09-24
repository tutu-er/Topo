import json
from pathlib import Path

from topoident.innovation_topology_free import run_topology_free_latent_tree


if __name__ == "__main__":
    result = run_topology_free_latent_tree()
    path = Path("results/innovation_6_topology_free.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
