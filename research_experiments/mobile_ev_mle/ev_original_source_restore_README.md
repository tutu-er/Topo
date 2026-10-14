# 原域源码字节检查与换行恢复

工具 `ev_original_source_restore.py` 默认只读。`--expected-source-count` 仅允许68或70，默认68；外部freeze键、源码快照键、解码字节SHA与安全目标路径须整批一致。

在Topo仓库根目录检查68源主研究版本：

```powershell
& 'D:/apps/miniconda3/envs/Topo/python.exe' -m research_experiments.mobile_ev_mle.ev_original_source_restore --expected-source-count 68 --snapshot 'research_experiments/mobile_ev_mle/evidence/ev_original_domain/pilot_v2/source_snapshot.json.gz' --freeze 'research_experiments/mobile_ev_mle/evidence/ev_original_domain/pilot_v2/freeze.json'
```

检查70源位置消融版本：

```powershell
& 'D:/apps/miniconda3/envs/Topo/python.exe' -m research_experiments.mobile_ev_mle.ev_original_source_restore --expected-source-count 70 --snapshot 'research_experiments/mobile_ev_mle/evidence/ev_original_domain/location_ablation_v1/source_snapshot.json.gz' --freeze 'research_experiments/mobile_ev_mle/evidence/ev_original_domain/location_ablation_v1/freeze.json'
```

先按compact归档说明恢复相应证据目录。在新的Git检出目录确认只有CRLF/LF差异后，可在相应命令末尾显式追加 `--restore-line-endings`。工具只恢复原快照字节；整批中任何真实修改、缺失、路径/别名/链接问题都会阻止恢复。未指定恢复选项时不写科学文件。检查与原字节不符时返回非零，可通过 `--report 新文件.json` 保存详细记录。

外圈CLI v2验证：68/68与70/70当前C源码均原字节一致，两次均只读、0恢复。测试27通过、2个实际Windows链接创建测试因权限跳过；确定性linklike及越界拒绝测试通过。记录在 `evidence/ev_original_domain/source_restore_cli_v2_validation.json`，含当前与前版工具字节和哈希。

原 `original_source_restore_validation.json/.md` 的真实Git对象恢复验证保留未改：当时从Git导出68源，41个换行差异仅在新D目录恢复；C科学源没有写入。v2只增加CLI68/70选择，`main`之前的审计/恢复API字节保持相同。
