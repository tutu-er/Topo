"""Scientific figures for verified fixed-proposal E-gate calibration."""
from pathlib import Path
import argparse
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def figures(output):
    output=Path(output).resolve();v=read(output/"analysis/verification.json")
    assert v["passed"] and v["complete"] and v["inputs"]==48 and v["variant_rows"]==288
    d=ROOT/"docs"/output.name;d.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(output/"analysis/summary.csv").set_index("variant")
    names=["old_J","tau_0","tau_0.0001","tau_0.001","tau_0.003","tau_0.01"]
    labels=["Old J","0","0.0001","0.001","0.003","0.01"]
    selected="tau_"+format(v["full48_selected_tau"],"g")
    colors=["#888888" if n=="old_J" else "#228833" if n==selected else "#4477aa" for n in names]
    fig,axes=plt.subplots(1,3,figsize=(12,4.4),layout="constrained")
    for ax,field,title in zip(axes,("E_accepted","F1","exact"),
                            ("Accepted E proposals / 48","Mean final clade F1","Exact final topology / 48")):
        values=[float(frame.loc[n,field]) for n in names]
        bars=ax.bar(range(6),values,color=colors,width=.68)
        ax.bar_label(bars,labels=[format(x,".6f" if field=="F1" else ".0f") for x in values],
                     rotation=90 if field=="F1" else 0,padding=3,fontsize=7.5)
        ax.set(xticks=range(6),xticklabels=labels,xlabel="Pure-MSE threshold tau",ylabel=title,
               ylim=(0,1.14 if field=="F1" else 52 if field=="exact" else 32))
        ax.tick_params(axis="x",rotation=35,labelsize=8)
        ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
    fig.suptitle("Fixed E proposals: gate changes versus final topology\nGreen: threshold chosen by validation MSE, before topology evaluation",fontsize=11)
    for suffix in ("png","svg"):fig.savefig(d/f"gate_comparison.{suffix}",dpi=180)
    plt.close(fig)
    receipt=dict(script_sha256=sha(__file__),verification_sha256=sha(output/"analysis/verification.json"),
                 summary_sha256=sha(output/"analysis/summary.csv"),selection_sha256=sha(output/"selection.json"),
                 files={n:sha(d/n) for n in ("gate_comparison.png","gate_comparison.svg")},
                 visual_review_pending=True,new_MIQP_calls=0)
    (d/"figure_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8",newline="\n")
    print("Saved gate_comparison PNG/SVG; visual review pending")
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("output",type=Path)
    figures(p.parse_args().output)
