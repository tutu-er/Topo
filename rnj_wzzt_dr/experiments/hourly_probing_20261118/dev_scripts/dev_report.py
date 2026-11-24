import sys
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import report
conds = sys.argv[1].split(",")
ref = sys.argv[2] if len(sys.argv) > 2 else "bgreedy_P"
arms = ("passive_bic", "cert_P", "bgreedy_P", "random_matched_P", "qbc_matched_P", "qbc_P", "greedy_P")
report.compare(conds, ref=ref, arms=tuple(a for a in arms if a != ref))
