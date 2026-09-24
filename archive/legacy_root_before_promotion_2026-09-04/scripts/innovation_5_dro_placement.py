import json
from pathlib import Path

from topoident.innovation_dro import run_dro_placement


if __name__ == "__main__":
    result = run_dro_placement()
    path = Path("results/innovation_5_dro.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
