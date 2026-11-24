import json, sys, collections
for c in sys.argv[1:]:
    d = json.load(open(rf"C:\0-workspace\Topo\repo\outputs\hourly_probing_dev_20261120\{c}\inputs.json"))
    err = [r for r in d if "error" in r]
    d = [r for r in d if "error" not in r]
    print(c, "n", len(d), "errors", len(err), "truth_in_bank", sum(r["truth_in_bank"] for r in d),
          "idbank_truth", sum(r["identifier_bank_has_truth"] for r in d), "leader", sum(r["leader_correct"] for r in d))
    print("  strata", sorted(collections.Counter((r["case"], r["stratum"]) for r in d).items()))
    print("  bank sizes", sorted(collections.Counter(r["bank"] for r in d).items())[-3:])
