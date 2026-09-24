import json
from pathlib import Path

from topoident.innovation_bayesian import run_bayesian_hidden_loads


if __name__ == "__main__":
    result = run_bayesian_hidden_loads()
    path = Path("results/innovation_1_bayesian.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
