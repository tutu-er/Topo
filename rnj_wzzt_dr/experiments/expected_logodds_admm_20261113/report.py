"""Independent audits and complete reporting of both successful and failed DR."""
from pathlib import Path
import argparse
from collections import defaultdict,Counter
import numpy as np
from scipy.special import logsumexp

from ... import io
from ...observations import load
from ..unified_posterior_dr_20261112.history_scan import require,sha,utc,ROOT
from ..posterior_mse_20261112.deliver import write_csv
from .run import REFERENCE
from .convex_run import FIRST
from .confirm import means_interval,BETAS


def audit(run,out,protocol,actual):
    seal=io.read(out/"policy_seal.json");require(sha(out/"protocol.json")==seal["protocol_sha256"],"Policy protocol changed")
    for path,digest in seal["policy_files"].items():require(sha(path)==digest,"Executed policy changed")
    for relative,ref in protocol["sources"].items():
        require(sha(ROOT/relative)==ref["sha256"],"Execution source changed since freeze: "+relative)
        require(sha(out/"source_snapshot"/relative)==ref["sha256"],"Frozen execution source differs")
    history_seal=io.read(REFERENCE/"selection_seal.json")
    truth={r["id"]:r for r in io.read(REFERENCE/"evaluation.json")}
    entries={(r["id"],r["target"]):r for r in protocol["policies"]}
    checked=0;maximum=0.;seen=set()
    for row in actual:
        key=row["id"],row["target"],row["strategy"],row["replicate"]
        require(key not in seen,"Duplicate physical trial");seen.add(key)
        if row["status"]!="complete":continue
        ref=row["observation"];require(sha(ref["path"])==ref["sha256"],"Observed data changed")
        with np.load(ref["path"],allow_pickle=False) as a:require(set(a.files)=={"P","Q","V","root","labels"},"Latent variables exposed")
        observed=load(ref["path"]);scorefile=REFERENCE/"scores"/(row["id"]+".json")
        require(sha(scorefile)==history_seal["scores"][row["id"]],"Historical score changed")
        score=io.read(scorefile);loss=[]
        for p in score["parameters"]:
            predicted=np.zeros_like(observed.y,dtype=np.longdouble);point=p["point"]
            for C,r,x in zip(point["supports"],point["r"],point["x"],strict=True):
                contribution=np.longdouble(r)*observed.p[:,C].astype(np.longdouble).sum(axis=1)
                contribution+=np.longdouble(x)*observed.q[:,C].astype(np.longdouble).sum(axis=1)
                predicted[:,C]+=contribution[:,None]
            loss.append(np.sum((observed.y.astype(np.longdouble)-predicted)**2)+np.longdouble(p["history_sse"]))
        loss=np.asarray(loss,dtype=np.longdouble);ll=-(loss-loss.min())/(2*np.longdouble(score["sigma2"]))
        weights=np.exp(ll);weights/=weights.sum()
        require(sha(row["prefix"]["path"])==row["prefix"]["sha256"],"Evidence path changed")
        with np.load(row["prefix"]["path"],allow_pickle=False) as a:
            expected=np.exp(a["final_log_probability"])
            require(np.max(abs(np.exp(a["prefix_log_probability"][-1])-expected))<1e-10,"Prefix endpoint differs")
        difference=float(np.max(abs(weights-expected)));maximum=max(maximum,difference)
        require(difference<1e-8,"Independent final posterior differs")
        best=int(np.argmax(weights));ids=[m["topology_id"] for m in score["models"]]
        require(ids[best]==row["selected_topology_id"],"Independent selected topology differs")
        entry=entries[row["id"],row["target"]]
        target_index=ids.index(entry["target_topology_id"]);true_index=ids.index(truth[row["id"]]["truth_id"])
        require(row["target_index"]==target_index and row["true_index"]==true_index,"Target/truth index differs")
        require(abs(float(weights[target_index])-row["p_target"])<1e-8,"Target posterior field differs")
        require(abs(float(weights[true_index])-row["p_true"])<1e-8,"True posterior field differs")
        require(row["map_correct"]==(ids[best]==truth[row["id"]]["truth_id"]),"Recovery field differs")
        odds=(loss-loss[target_index])/(2*np.longdouble(score["sigma2"]))
        require(np.max(abs(odds[entry["competitors"]]-row["final_major_log_odds"]))<1e-7,"Actual major log odds differ")
        predicted=set(map(tuple,score["models"][best]["active_clades"]));real=set(map(tuple,truth[row["id"]]["truth_clades"]))
        require(row["missing"]==len(real-predicted) and row["extra"]==len(predicted-real),"Structure counts differ")
        for beta in BETAS:
            order=np.argsort(-weights,kind="stable");count=min(int(np.searchsorted(np.cumsum(weights[order]),beta))+1,len(order))
            kept=[int(i) for i in order if weights[i]>=weights[order[count-1]]];contains=row["true_index"] in kept
            d=row["decisions"][str(beta)]
            require(d["size"]==len(kept) and d["truth_retained"]==contains,"Retained set differs")
            require(d["correct_unique"]==(len(kept)==1 and contains),"Correct unique differs")
            require(d["wrong_unique"]==(len(kept)==1 and not contains),"Wrong unique differs")
            require(d["target_pass"]==bool(weights[target_index]>=beta),"Target pass field differs")
        checked+=1
    receipt=dict(status="passed",created_utc=utc(),physical_scored_rows=checked,planned_rows=len(seen),
        maximum_independent_probability_error=maximum,all_source_and_policy_hashes_unchanged=True,
        observed_channels_only=True,structure_and_decisions_checked=True,new_parameter_fits=0,source=io.ref(__file__))
    io.write(out/"independent_audit.json",receipt);return receipt


def tables(run,protocol,actual):
    evaluated={r["id"]:r for r in io.read(REFERENCE/"evaluation.json")}
    entries={(r["id"],r["target"]):r for r in protocol["policies"]}
    groups=defaultdict(list)
    for r in actual:groups[r["id"],r["target"],r["strategy"]].append(r)
    rows=[];paired=[]
    for key,entry in entries.items():
        correct=entry["target_topology_id"]==evaluated[key[0]]["truth_id"]
        for strategy,policy in entry["policies"].items():
            trials=groups.get((*key,strategy),[]);valid=[r for r in trials if r["status"]=="complete"]
            available=policy["available"]
            row=dict(id=key[0],target=key[1],strategy=strategy,target_correct=correct,available=available,
                planned=32,valid=len(valid),normalized_fee=policy["fee"]/entry["fee_scale"] if available else None,
                mean_target_posterior=float(np.mean([r["p_target"] for r in valid])) if valid else None,
                mean_true_posterior=float(np.mean([r["p_true"] for r in valid])) if valid else None,
                full_recovery=sum(r["map_correct"] for r in valid) if available else None,
                mean_missing=float(np.mean([r["missing"] for r in valid])) if valid else None,
                mean_extra=float(np.mean([r["extra"] for r in valid])) if valid else None,
                mean_abs_DR_kwh=float(np.mean([r["execution"]["original_device_audit"]["absolute_DR_kwh"] for r in valid])) if valid else None,
                mean_clipping_fraction=float(np.mean([r["execution"]["clipping_fraction"] for r in valid])) if valid else None)
            for beta in BETAS:
                rows.append(dict(row,beta=beta,correct_unique=sum(r["decisions"][str(beta)]["correct_unique"] for r in valid) if available else None,
                    wrong_unique=sum(r["decisions"][str(beta)]["wrong_unique"] for r in valid) if available else None,
                    target_pass=sum(r["decisions"][str(beta)]["target_pass"] for r in valid) if available else None))
            if strategy=="no_dr" or not valid:continue
            baseline={r["replicate"]:r for r in groups.get((*key,"no_dr"),[]) if r["status"]=="complete"}
            matched=[r for r in valid if r["replicate"] in baseline]
            delta=np.array([r["p_target"]-baseline[r["replicate"]]["p_target"] for r in matched])
            gains=np.array([np.asarray(r["final_major_log_odds"])-np.asarray(baseline[r["replicate"]]["final_major_log_odds"]) for r in matched])
            for j,column in zip(entry["competitors"],gains.T,strict=True):
                interval=means_interval(column);pinterval=means_interval(delta)
                paired.append(dict(id=key[0],target=key[1],strategy=strategy,target_correct=correct,paths=len(matched),competitor=j,
                    mean_actual_log_odds_gain=interval["mean"],log_gain_mean_interval95=interval["mean_interval95"],
                    mean_posterior_gain=pinterval["mean"],posterior_gain_mean_interval95=pinterval["mean_interval95"],
                    posterior_positive_paths=int(np.sum(delta>0)),posterior_negative_paths=int(np.sum(delta<0)),
                    interval_scope="descriptive paired t mean interval, not a distribution-free guarantee"))
    model=[]
    for row in io.read(run/"confirmation/model_results.json"):
        entry=entries[row["id"],row["target"]];correct=entry["target_topology_id"]==evaluated[row["id"]]["truth_id"]
        for beta in BETAS:
            available=row["status"]=="complete"
            model.append(dict(id=row["id"],target=row["target"],strategy=row["strategy"],target_correct=correct,beta=beta,
                available=available,paths=row.get("paths"),mean_target_posterior=row.get("posterior",{}).get("mean"),
                posterior_standard_error=row.get("posterior",{}).get("standard_error"),
                target_pass_rate=row.get("pass_rates",{}).get(str(beta),{}).get("probability"),normalized_fee=row.get("normalized_fee")))
    increments=[]
    for row in io.read(run/"confirmation/model_results.json"):
        for g in row.get("increments",[]):
            increments.append(dict(id=row["id"],target=row["target"],strategy=row["strategy"],**g))
    return rows,paired,model,increments


def planning_rows(run):
    result=[]
    for version,folder in [("original_integer_devices",FIRST),("convex_device_subset",run)]:
        for row in io.read(folder/"planning_results.json"):
            scale=io.read(folder/"planning"/(row["id"]+f"__target{row['target']}")/"input.json")["fee_scale"]
            for method in (["admm","direct"] if version=="original_integer_devices" else ["admm"]):
                value=row.get(method,{})
                fee=value.get("fee") if method=="direct" else (value.get("best") or{}).get("fee")
                seed=value.get("seed",{})
                result.append(dict(version=version,id=row["id"],target=row["target"],method=method,
                    status=row["status"],qualified=value.get("qualified",False),normalized_fee=fee/scale if fee is not None else None,
                    normalized_seed_fee=seed.get("fee")/scale if seed.get("fee") is not None else None,
                    total_seconds=value.get("total_seconds"),seed_generation_seconds=seed.get("total_seconds"),
                    first_feasible_seconds=value.get("first_feasible_seconds"),
                    retained_origin=(value.get("best") or{}).get("origin"),retained_start=(value.get("best") or{}).get("start"),
                    native_status=value.get("status") if method=="direct" else None,native_gap=value.get("gap"),
                    normalized_original_domain_lower_bound=value.get("lower_bound")/scale if value.get("lower_bound") is not None else None,
                    global_optimality_proved=value.get("global_optimality_proved",False)))
    return result


def publish(run,destination):
    require(not destination.exists(),"Existing public report preserved")
    out=run/"confirmation";require((out/"actual_complete.json").exists(),"Finish physical stage first")
    protocol=io.read(out/"protocol.json");actual=io.read(out/"actual_results.json")
    receipt=audit(run,out,protocol,actual)
    comparison,paired,model,increments=tables(run,protocol,actual);planning=planning_rows(run)
    destination.mkdir(parents=True)
    for name,table in [("planning.csv",planning),("model.csv",model),("model_evidence.csv",increments),
                       ("actual.csv",comparison),("actual_paired.csv",paired)]:write_csv(destination/name,table)
    truth_rows=[r for r in comparison if r["target_correct"] and r["beta"]==.9]
    current=io.read(run/"planning_results.json")
    successful=[r for r in current if r["status"]=="complete" and r["admm"]["qualified"]]
    lines=["# 期望证据DR与ADMM：有界开发测试", "",
        "本批实现并测试了期望对数后验赔率增量规划。原整数设备版4个条件目标均未找到合格报价；同预算直接QCQP也均未找到报价。声明的凸设备子域版在Flynn的两个条件目标取得合格中间点，Paper两个目标仍无报价。两输入都是已有开发输入，不能据此声称普遍加速或泛化效果。", "",
        "## 算法实际做了什么", "",
        "ADMM分别更新设备计划、各竞争者的球外投影和一致性乘子。原设备域含整数，第一版设备步骤为限时小型MIQP。第二版只搜索直接满足原能力、名义不截断且电池方向固定的凸子域，设备步骤才是凸QP。每份方案均重放原控制器并用完整冻结H独立复算增量。", "",
        "期望增强量事前固定为log(2)。对受约束竞争者，这表示模型内平均对数赔率增加至少log(2)，不保证每条路径或完整恢复增加。背景功率误差、尺度、历史和参数未改；实际随机执行和AC偏差另测。", "",
        "## 报价、初值和停止状态", ""]
    for row in successful:
        value=row["admm"];seed=value["seed"];best=value["best"]
        zero=next(a for a in value["attempts"] if a["start"]=="zero")
        improvement=1-best["fee"]/seed["fee"] if seed.get("fee") else None
        lines.append(f"- {row['id']}，目标{row['target']}：零起点分支首次可行约{zero['first_feasible_seconds']:.2f}秒（含该分支模型构造，不含另一分支／支持平面初值生成）；支持平面初值生成{seed['total_seconds']:.2f}秒；完整双起点流程{value['total_seconds']:.2f}秒。保留{best['start']}分支第{best['retained_iteration']}轮可行点，较支持平面初值费用降低{improvement:.2%}。")
    lines += ["", "两个Flynn目标是同一输入、同一候选对的相反条件方向，所得策略相同，不计作两个独立正例。支持平面初值是单独的凸QP，初值来源和生成成本均记录。零起点分支也独立找到可行点，不能把所有成功归因于已有可行初值。", "",
        "**所有实际ADMM起点都到时间上限，未完成迭代收敛。保留的是独立核验通过的中间可行点。没有全局最优证明。** 凸子域以及支持平面问题的界不作为原QCQP下界；原域直接求解结果和失败全部见[报价表](planning.csv)。Paper无报价不写成0费用可行方案。", "",
        "## 独立确认结果", "",
        "声明模型每目标2048条新路径；交流潮流模拟每输入32条新日。下表只看目标正确的101冻结库，门槛90%；费用以该输入原固定全幅轮换费用归一化。", "",
        "| 输入 | 策略 | 模型平均后验 | 模型达标率 | AC平均后验 | AC正确唯一／32 | AC完整恢复／32 | 归一化费用 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    lookup={(r["id"],r["target"],r["strategy"]):r for r in model if r["beta"]==.9}
    for row in truth_rows:
        m=lookup[row["id"],row["target"],row["strategy"]]
        if not row["available"]:
            lines.append(f"| {row['id']} | {row['strategy']} | 未执行 | 未执行 | 未执行 | 未执行 | 未执行 | 无合格报价 |")
        else:
            lines.append(f"| {row['id']} | {row['strategy']} | {m['mean_target_posterior']:.4%} | {m['target_pass_rate']:.4%} | {row['mean_target_posterior']:.4%} | {row['correct_unique']}/32 | {row['full_recovery']}/32 | {row['normalized_fee']:.6g} |")
    lines += ["", "模型中的解析增量和配对样本均值见[模型证据表](model_evidence.csv)。理论只对规划竞争者、声明目标生成模型和名义交付成立；实际AC参数偏差、量测及乘性交付未被这条期望等式全部保证。", "",
        "实际配对变化（95%区间为描述性配对t均值区间，非分布无关保证）：", ""]
    for row in paired:
        if row["target_correct"] and row["strategy"]=="admm":
            gain_ci=[round(v,4) for v in row['log_gain_mean_interval95']]
            posterior_ci=[round(100*v,4) for v in row['posterior_gain_mean_interval95']]
            includes_zero=gain_ci[0]<=0<=gain_ci[1]
            lines.append(f"- {row['id']}：平均对数赔率增量{row['mean_actual_log_odds_gain']:.6f}，95%均值区间{gain_ci}（{'包含0' if includes_zero else '不包含0'}）；平均后验增量{100*row['mean_posterior_gain']:.4f}个百分点，95%均值区间{posterior_ci}个百分点；{row['posterior_positive_paths']}/{row['paths']}路径后验提高、{row['posterior_negative_paths']}/{row['paths']}降低。")
    lines += ["", "**AC改善幅度小，完整恢复数没有增加。** 正确唯一输出只多1/32，不能据此宣称稳定恢复收益。实际平均证据增量的95%均值区间包含0，当前32路径尚未确认实际平均证据稳定增加。", "",
        "错误条件目标也完整保留：它在自身假设生成模型下可增强证据，但原物理网络不随假设改变，实际目标后验可以降低。两目标与策略共用的背景和量测不是独立样本。", "",
        "合同支付仍是冻结名义费用；执行费用函数值只作损耗代理。轮换名义费用匹配偏差在私有冻结协议中保留，表中按其实际名义费用计。", "",
        f"独立审计复算{receipt['physical_scored_rows']}行全部后验权重、结构错漏及四门槛决定，最大概率差{receipt['maximum_independent_probability_error']:.3g}。实际记录状态：{dict(Counter(r['status'] for r in actual))}；无原始P/Q/V、拟合参数或报价轨迹上传。", "",
        "## 已实现、已验证与未完成", "",
        "已实现：精确期望增量、分享式ADMM、球外闭式更新、原设备和凸子域更新、初值与费用记录。已验证：解析成功／失败例、原执行器回放、两个开发输入的模型和AC配对检查。未完成：非凸整数情形收敛证明、原域全局最优、Paper合格报价、独立新输入上的算法对比。", "",
        "下一步优先检查原设备更新如何降低每轮成本，以及Paper是受凸子域限制还是局部协调停滞；用同一原问题比较，保留无DR、同费轮换和全部失败。当前阶段不修改辨识器。"]
    (destination/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    summary=dict(status="bounded_development_complete",created_utc=utc(),original_domain_ADMM_quotes=0,
        original_domain_direct_quotes=0,convex_subset_quotes=len(successful),independent_positive_inputs=1,
        converged_ADMM_starts=0,global_optimality_proved=False,physical_statuses=dict(Counter(r["status"] for r in actual)),
        independent_audit=receipt,new_identifier_fits=0,new_parameter_fits=0)
    io.write(run/"summary.json",summary)
    io.write(destination/"summary.json",{k:v for k,v in summary.items() if k!="independent_audit"})
    plots(destination,truth_rows,model,paired,current)
    io.write(destination/"RECEIPT.json",dict(created_utc=utc(),protocol_sha256=sha(out/"protocol.json"),
        audit_sha256=sha(out/"independent_audit.json"),files={p.name:sha(p) for p in destination.iterdir() if p.is_file()},private_data_published=False))
    io.write(run/"complete.json",dict(status="bounded_development_complete",created_utc=utc(),public_report=str(destination/"REPORT.md"),
        no_new_global_optimality_claim=True,source=io.ref(__file__)))
    print({k:v for k,v in summary.items() if k!="independent_audit"},flush=True)


def plots(destination,actual,model,paired,planning):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"svg.fonttype":"none"})
    identity="main__input054__101";target=6;strategies=["no_dr","rotation","admm"]
    a={r["strategy"]:r for r in actual if r["id"]==identity and r["target"]==target}
    m={r["strategy"]:r for r in model if r["id"]==identity and r["target"]==target and r["beta"]==.9}
    fig,axes=plt.subplots(1,3,figsize=(13,3.8));x=np.arange(3)
    axes[0].bar(x-.18,[m[s]["mean_target_posterior"] for s in strategies],.36,label="Declared model")
    axes[0].bar(x+.18,[a[s]["mean_target_posterior"] for s in strategies],.36,label="Independent AC")
    axes[0].set(xticks=x,xticklabels=strategies,ylim=(0,1),title="Mean target posterior");axes[0].legend(frameon=False,fontsize=8)
    axes[1].bar(x-.18,[m[s]["target_pass_rate"] for s in strategies],.36,label="Model")
    axes[1].bar(x+.18,[a[s]["correct_unique"]/32 for s in strategies],.36,label="AC")
    axes[1].set(xticks=x,xticklabels=strategies,ylim=(0,1),title="Posterior >= 90%")
    selected=next(r for r in planning if r["id"]==identity and r["target"]==target)["admm"]
    seed=selected["seed"]["fee"];zero=next(r for r in selected["attempts"] if r["start"]=="zero")
    axes[2].bar(np.arange(3),[1.,zero["retained_fee"]/seed,selected["best"]["fee"]/seed])
    axes[2].set(xticks=np.arange(3),xticklabels=["Convex seed","Zero-start\nADMM","Seed-start\nADMM"],ylim=(0,1.1),title="Fee / initial feasible seed fee")
    for ax in axes:ax.grid(axis="y",alpha=.2)
    fig.suptitle("Flynn development input: feasible iterates retained; no convergence or global-optimum claim",fontsize=11)
    fig.tight_layout()
    for suffix in ("png","svg"):
        p=destination/("outcomes."+suffix);fig.savefig(p,dpi=180,bbox_inches="tight")
        if suffix=="svg":p.write_text("\n".join(line.rstrip() for line in p.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",type=Path,required=True);parser.add_argument("--public",type=Path,required=True)
    args=parser.parse_args();publish(args.run.resolve(),args.public.resolve())


if __name__=="__main__":
    main()
