import json
from pathlib import Path

from topoident.innovation_active import run_active_probe_design


if __name__ == "__main__":
    result = run_active_probe_design()
    path = Path("results/innovation_2_active.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
