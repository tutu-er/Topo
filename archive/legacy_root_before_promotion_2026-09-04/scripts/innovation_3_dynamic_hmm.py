import json
from pathlib import Path

from topoident.innovation_hmm import run_dynamic_hmm


if __name__ == "__main__":
    result = run_dynamic_hmm()
    path = Path("results/innovation_3_hmm.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
