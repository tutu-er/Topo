# Real feeders: converted from the Non-Synthetic European LV Test System

These 33 network cases are derived from: Koirala, A.; Suárez-Ramón, L.; Mohamed, B.; Arboleya, P. "Non-synthetic European low voltage test system", *International Journal of Electrical Power & Energy Systems* 118 (2020) 105712, and its data set (Mendeley Data, DOI 10.17632/685vgp64sm.1), built from the GIS and smart meters of EDP (Spain). Licence: **CC BY 4.0**; the derived tables here are shared under the same licence with this attribution.

**What was changed** (`../eulv_convert.py`, see `../REAL_NETWORK_CASE.md`):
- each case is one LV feeder (the subtree behind one line leaving a transformer's LV bus), with the normally-open fuses open, taking feeders with 8–25 building service connections;
- each service connection is one metered terminal (its customers summed);
- branches without customers downstream are dropped and chains through hidden degree-2 nodes are merged (series R and X added);
- R and X are the phase values of the line codes (balanced single-phase equivalent);
- base 0.4 kV / 0.2 MVA as in the study's other cases; `transformer_kva` is the feeding transformer's rating.

`index.json` lists every case with its transformer, feeder line, terminals, hidden nodes, metered internal nodes, customers and electrical size. The customers' hourly loads are not stored here: the converter writes them to `data/nonsynthetic_eu_lv_20261123/loads/`, which is outside Git, from the downloaded data set.
