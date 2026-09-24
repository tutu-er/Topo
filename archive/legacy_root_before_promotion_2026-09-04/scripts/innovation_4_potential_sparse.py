import json
from pathlib import Path

from topoident.innovation_potential import run_potential_sparse_recovery


if __name__ == "__main__":
    result = run_potential_sparse_recovery()
    path = Path("results/innovation_4_potential.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
