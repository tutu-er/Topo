# 冻结源码的便携核对与换行恢复

`ev_fixed_train_source_restore.py`支持本轮78源和79源快照。它不加入已冻结science manifest。快照仅作字节数据，先核对自身SHA及外部freeze，再转为临时旧格式，复用`ev_original_source_restore.py`的路径、别名、链接及整批预检。没有拟合、仿真或选择器调用。

在仓库根目录执行，默认只读检查源码：

```powershell
python -m research_experiments.mobile_ev_mle.ev_fixed_train_source_restore --directory development_v2 --report outputs/weak8_source_readonly.json
python -m research_experiments.mobile_ev_mle.ev_fixed_train_source_restore --directory rich32 --report outputs/rich32_source_readonly.json
```

省略`--directory`默认`development_v2`。`weak8`是同义入口；`rich8/rich32/rich96`映射到`evidence/ev_fixed_train/training_support_dev_v1/`下对应目录，也可传完整目录。报告使用新文件名，已有报告不覆盖。

Git签出后若仅CRLF/LF改变了精确源码哈希，默认报告`line_ending_only_count>0`，不修改源码，并以退出码2提示未精确匹配。显式恢复：

```powershell
python -m research_experiments.mobile_ev_mle.ev_fixed_train_source_restore --directory development_v2 --restore-line-endings --report outputs/weak8_source_restored.json
python -m research_experiments.mobile_ev_mle.ev_fixed_train_source_restore --directory rich32 --restore-line-endings --report outputs/rich32_source_restored.json
```

只有整批不存在缺失源码和实质内容改变时才执行恢复。恢复前再次检查全部路径、文件身份和字节；恢复目标为快照原字节。任何缺失或真实修改都阻止整批换行写入。原freeze和新schema快照的哈希始终保持；源码缺失时不会创建文件。

检查另一个导出仓库时，同时指定源根与仓库根：

```powershell
python -m research_experiments.mobile_ev_mle.ev_fixed_train_source_restore --directory D:/export/Topo/research_experiments/mobile_ev_mle/evidence/ev_fixed_train/development_v2 --repository-root D:/export/Topo --source-root D:/export/Topo/research_experiments/mobile_ev_mle --report D:/export/Topo/outputs/source_readonly.json
```

临时适配文件只在目标仓库`outputs/ev_source_restore_temporary/`内创建并清理；可用`--temporary-root`指定仓库内另一个临时目录。科学源码默认没有写入。合法冻结路径包括本目录相对文件、`../mobile_ev_probing/`及`rnj_wzzt_core/`；其它上层穿越与Windows路径别名继续拒绝。

本次验证：

- `source_restore_D_weak8_readonly.json`：78/78精确一致，源码写入0。
- `source_restore_D_rich32_readonly.json`：79/79精确一致，源码写入0。
- `source_restore_unit.xml`：4项小夹具测试通过，覆盖默认只读、显式混合换行恢复、坏字节/缺失阻止整批写入、快照/freeze不变及路径安全。

测试临时根显式放在D阶段目录：

```powershell
python -m pytest research_experiments/mobile_ev_mle/test_ev_fixed_train_source_restore.py -q --basetemp D:/0-github_workspace/Topo/outputs/ev_train_validation_stage_20261015/outputs/pytest_fixed_train_source_restore_fresh
```

便携副本重新执行测试时选择自己的新临时目录；上述D路径只是本次验证位置。

## 异常报告修订与最终工具版本

首版工具SHA为`e013bb1e6235b132fec03d42745f40a422fd81200dcbfe243d8e1e1e885a1fff`；原`source_restore_unit.xml`及两份不带`v2`的只读核验保留。首版CLI异常出口统一报告写入0；若显式恢复过程中发生写后异常，真实写入数可能未知。

修正版SHA为`3e437f70f621ff6c847bfcb7a852828def43cffbe00a2417024d40a811479a57`。现在恢复请求的异常出口报告`source_writes: null`，表示未知；默认只读的异常出口仍为0。正常安全恢复和只读逻辑没有改变。

修正版证据：`source_restore_unit_v2.xml`四项测试通过（含两种CLI异常计数）；`source_restore_D_weak8_readonly_v2.json`为78/78、`source_restore_D_rich32_readonly_v2.json`为79/79精确一致，两者源码写入均0，原freeze与snapshot哈希均不变。Git对象导出与后续交付使用这个修正版工具。
