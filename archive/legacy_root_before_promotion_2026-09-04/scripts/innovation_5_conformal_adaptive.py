import json
from pathlib import Path

from topoident.innovation_conformal import run_conformal_adaptive_pmu


if __name__ == "__main__":
    result = run_conformal_adaptive_pmu()
    path = Path("results/innovation_5_conformal.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
