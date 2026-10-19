# 原域68源的只读检查与Git换行恢复验证

独立工具：`ev_original_source_restore.py`；不加入科学freeze。默认只读，仅显式 `--restore-line-endings` 才允许恢复换行差异。

验证使用 pilot_v2 的外部freeze与直接 `name -> {sha256, bytes_base64}` 源码快照。支持唯一的路径映射：普通名称对应 `research_experiments/mobile_ev_mle/`；规定的 `../mobile_ev_probing/` 对应兄弟目录；`rnj_wzzt_core/` 对应仓库根。其他父路径、Windows/大小写别名、重复目标、链接和路径逃逸均拒绝。

| 检查 | 原字节一致 | 仅CRLF/LF不同 | 真内容变化 | 实际恢复文件 |
|---|---:|---:|---:|---:|
| 冻结C目录，只读检查 | 68 | 0 | 0 | 0 |
| 实际Git对象导出的新D目录 | 27 | 41 | 0 | 0 |
| 仅在新D目录显式恢复后 | 68 | 0 | 0 | 41 |
| 冻结C目录，完成后再只读检查 | 68 | 0 | 0 | 0 |

Git提交：`e989336f4b39c6a40e7ca4c3bbdac07e9a3c46f3`。验证从该提交的68个真实blob导出，未拿C当前文件代替Git对象。导出目录为 `D:/0-github_workspace/Topo/outputs/ev_original_stage/original_sources_git_e989336f4b/`。逐个Git对象ID、blob字节SHA、前后检查记录、独立验证脚本原字节均见同名JSON。

任何一个源缺失、存在真实内容差异、快照/外部hash不符或路径不安全时，整批恢复被拒绝。不会创建缺失科学文件或覆盖真正编辑。每次恢复先验证全部目标，再只写换行不同的文件，并重算原字节SHA。

安全测试：23通过、0失败；另2个实际OS链接创建测试因Windows权限跳过（symlink1314、hardlink5）。无需OS链接权限的确定性linklike和解析后越界检查均通过。没有运行拟合或选择器。

## 使用

在指定仓库根目录默认检查：

```powershell
& 'D:/apps/miniconda3/envs/Topo/python.exe' -m research_experiments.mobile_ev_mle.ev_original_source_restore
```

在新Git检出目录确认仅换行不同后，显式允许恢复：

```powershell
& 'D:/apps/miniconda3/envs/Topo/python.exe' -m research_experiments.mobile_ev_mle.ev_original_source_restore --snapshot 'research_experiments/mobile_ev_mle/evidence/ev_original_domain/pilot_v2/source_snapshot.json.gz' --freeze 'research_experiments/mobile_ev_mle/evidence/ev_original_domain/pilot_v2/freeze.json' --restore-line-endings
```

工具恢复快照中的完整原字节，包括原有混合换行。默认检查出现换行漂移时返回非零，使用者可查看报告；真实内容改变时即使指定恢复选项也拒绝整批写入。本次冻结C科学源始终未写。
