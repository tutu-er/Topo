"""Plot verified loss/structure contrasts and selector-only sensitivity views."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LOSSES = ("mae", "scaled_mse")
METHODS = ("noT", "T003")
LOSS_LABELS = {"mae":"L1 training", "scaled_mse":"Squared training"}
COLORS = {"mae":"#4477aa", "scaled_mse":"#228833"}


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream: return list(csv.DictReader(stream))


def save(fig, destination, name):
    for suffix in ("png","svg"): fig.savefig(destination/f"{name}.{suffix}", dpi=180)
    plt.close(fig)


def figures(output):
    output = Path(output).resolve(); analysis = output/"analysis"
    verification = read(analysis/"verification.json")
    assert verification["passed"] and verification["complete"]
    assert verification["new_native_verified"] == verification["reused_parent_verified"] == 96
    destination = ROOT/"docs"/output.name; destination.mkdir(parents=True, exist_ok=True)
    source_files = [analysis/name for name in ("summary.csv","noise.csv","view_inputs.csv")]
    summary_rows = rows(source_files[0])
    summary = {(r["fit_loss"],r["method"],r["view"]):r for r in summary_rows}
    assert len(summary) == 8 and all(int(v["verified"]) == int(v["planned"]) == 48 for v in summary.values())
    fig, axes = plt.subplots(1,3,figsize=(11.6,4.2),layout="constrained")
    specs = (("F1",1.,"Mean clade F1",".3f"),("exact",1.,"Exact topology / 48 inputs",".0f"),
             ("total_seconds",1/3600,"Successful method time (hours)",".2f"))
    x = np.arange(2); width=.34
    for ax,(field,factor,label,fmt) in zip(axes,specs):
        values_all=[]
        for i,loss_name in enumerate(LOSSES):
            values=[float(summary[loss_name,m,"primary_mae"][field])*factor for m in METHODS]
            values_all.extend(values)
            bars=ax.bar(x+(i-.5)*width,values,width=width,color=COLORS[loss_name],label=LOSS_LABELS[loss_name])
            ax.bar_label(bars,labels=[format(v,fmt) for v in values],padding=3,fontsize=8)
        ax.set(xticks=x,xticklabels=["No joint tail","Joint tail .003"],ylabel=label)
        ax.set_ylim(0,1.08 if field=="F1" else (52 if field=="exact" else max(values_all)*1.18))
        ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
    axes[0].legend(frameon=False,fontsize=8,loc="lower left")
    fig.suptitle("Training-loss comparison with the same MAE validation rule",fontsize=12)
    save(fig,destination,"loss_structure_comparison")

    noise = rows(source_files[1]); fig,axes=plt.subplots(1,3,figsize=(11.6,4.25),layout="constrained")
    for loss_name in LOSSES:
        for method in METHODS:
            selected=sorted((r for r in noise if r["fit_loss"]==loss_name and r["method"]==method and r["view"]=="primary_mae"),
                            key=lambda r:float(r["voltage_sigma"]))
            assert len(selected)==3 and all(int(r["verified"])==16 for r in selected)
            sigma=[100*float(r["voltage_sigma"]) for r in selected]
            values=([float(r["F1"]) for r in selected],[100*float(r["exact"])/16 for r in selected],
                    [float(r["total_seconds"])/16 for r in selected])
            label=LOSS_LABELS[loss_name]+(" / no T" if method=="noT" else " / T")
            for ax,ys in zip(axes,values):
                ax.plot(sigma,ys,color=COLORS[loss_name],ls="-" if method=="noT" else "--",
                        marker="o" if method=="noT" else "^",ms=5,lw=1.7,label=label)
    for ax,label in zip(axes,("Mean clade F1","Exact topology recovery (%)","Mean successful method time (s)")):
        ax.set(xlabel="V/root meter noise SD (%)",ylabel=label,xticks=[.02,.05,.1])
        ax.spines[["top","right"]].set_visible(False);ax.grid(alpha=.18)
    axes[0].set_ylim(-.02,1.05);axes[1].set_ylim(-2,104)
    axes[2].legend(frameon=False,fontsize=7.5,loc="best")
    fig.suptitle("Noise-stratified comparison: 16 paired inputs per point",fontsize=12)
    save(fig,destination,"noise_comparison")

    views=rows(source_files[2]); cells=[(loss_name,m) for loss_name in LOSSES for m in METHODS]
    labels=[("L1" if loss_name=="mae" else "Squared")+("\nNo T" if m=="noT" else "\nT .003") for loss_name,m in cells]
    view_map={(r["job"],r["fit_loss"],r["method"],r["view"]):r for r in views if r["status"]=="verified"}
    changes=[]
    for loss_name,method in cells:
        matching=[r for r in views if r["fit_loss"]==loss_name and r["method"]==method and r["view"]=="primary_mae"]
        assert len(matching)==48
        changes.append(sum(int(r["selected_index"])!=int(view_map[r["job"],loss_name,method,"secondary_mse"]["selected_index"]) for r in matching))
    fig,axes=plt.subplots(1,3,figsize=(12.2,4.3),layout="constrained");x=np.arange(4)
    for ax,field,label,fmt in ((axes[0],"F1","Mean clade F1",".3f"),(axes[1],"exact","Exact topology / 48 inputs",".0f")):
        paired_values=[]
        for i,(view,color,view_label) in enumerate((("primary_mae","#4477aa","MAE validation"),("secondary_mse","#cc6677","MSE validation"))):
            values=[float(summary[l,m,view][field]) for l,m in cells]
            paired_values.append(values)
            ax.bar(x+(i-.5)*width,values,width=width,color=color,label=view_label)
        pad=.012 if field=="F1" else .6
        for j,(a,b) in enumerate(zip(*paired_values)):
            if abs(a-b)<1e-12:
                ax.text(x[j],max(a,b)+pad,format(a,fmt),ha="center",va="bottom",fontsize=8)
            else:
                for i,value in enumerate((a,b)):
                    ax.text(x[j]+(i-.5)*width,value+pad,format(value,fmt),ha="center",va="bottom",fontsize=7,rotation=90)
        ax.set(ylabel=label,xticks=x,xticklabels=labels,ylim=(0,1.08 if field=="F1" else 52))
    bars=axes[2].bar(x,changes,width=.6,color="#aa4499")
    axes[2].bar_label(bars,labels=list(map(str,changes)),padding=3,fontsize=8)
    axes[2].set(xticks=x,xticklabels=labels,ylabel="Changed saved-point index / 48",ylim=(0,max(5,max(changes)+3)))
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
    axes[0].legend(frameon=False,fontsize=8,loc="lower left")
    fig.suptitle("Selector-only sensitivity on each unchanged fitted output library",fontsize=12)
    save(fig,destination,"same_library_validation")
    names=[f"{name}.{suffix}" for name in ("loss_structure_comparison","noise_comparison","same_library_validation") for suffix in ("png","svg")]
    receipt=dict(verification_sha256=sha(analysis/"verification.json"),input_csv_hashes={f.name:sha(f) for f in source_files},
                 figure_sha256={name:sha(destination/name) for name in names},script_sha256=sha(__file__),files=names,
                 visual_review_pending=True,new_fit_calls=0,new_optimizer_calls=0,
                 time_scope="successful method wall plus charged common costs; L1 is reused, interruption costs are separate")
    (destination/"figure_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print("Saved three PNG/SVG figure pairs; manual visual review remains pending",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("output",type=Path)
    figures(parser.parse_args().output)
