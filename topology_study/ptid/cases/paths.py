"""Where the data are. Case files and results are not in Git (outputs/ at the repository root); set PTID_OUTPUTS to
use another place."""
import os
from pathlib import Path

STUDY = Path(__file__).resolve().parents[2]                       # topology_study/
OUTPUTS = Path(os.environ.get("PTID_OUTPUTS", STUDY.parent/"outputs"))

LIBRARY = OUTPUTS/"case_library_20261125"                         # dev/ and test/ case files (test is sealed)
REAL = OUTPUTS/"hourly_probing_dev_20261120"/"eulv"/"inputs"      # 66 real-feeder development inputs
RESULTS = OUTPUTS/"topology_study"                                # this study's run files
MANIFEST_DEV = Path(__file__).resolve().parent/"manifest_dev.json"
