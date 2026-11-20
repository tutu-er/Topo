# Data and Archives

The November 18, 2026 cleanup moved the AC simulator and four network tables into `physical/`. `identification/` contains the maintained topology estimator. The [provenance manifest](PROVENANCE.json) records the packaged source hashes and the original locations of extracted and archived files.

The package runs on user-supplied observations. The original benchmark and study data are local assets under the repository's ignored `data/` directory.

| Local path | Contents |
| --- | --- |
| `data/benchmarks/` | Retained inputs, frozen models, and related assets from the main benchmark |
| `data/impedance_floor_20261116_v1/inputs.json` | Source catalog for the 296-entry default evaluation scope |
| `data/frozen_studies/wzzt_long_grid_20261117_v2/` | Final candidate banks, per-task outcomes, and report for the paired loss study |
| `data/frozen_studies/w_importance_20261117_v1/` | Final candidate banks, per-task outcomes, and report for the W ablation |
| `data/legacy_sources/` | Small original source files required by legacy hash checks |

`paths.py` resolves references to retained assets when their historical paths have moved. The loader adjusts references in memory. The stored JSON and its SHA-256 digest remain the evidence used to check each original run.

The full solver traces and controller outputs for the two studies live in private archives outside the development tree. To replay a sealed study, use the archived run and its source snapshot. A new run from this edition records its own source and protocol hashes. See the [cleanup manifest](../cleanup_manifest.json) for the archive inventory and the local backup locator.
