# stress_asset_v2 的可恢复证据

用途：stress；独立正式评价计数为0。此目录与 evaluation_v2 的12个独立评价历史分开。

选择 5 个文件：freeze、68源源码快照、2 个 trial、成功的 review_full；不压入CSV、checkpoint或重复summary。

原文件合计 8,113,200 bytes；solid tar.xz 为 3,116,052 bytes（38.41%）。
归档SHA-256：`5f92f6020aff777ec0a54049cb65d90699c2e579961412f7616b18bbd604a55b`。

已在全新的D目录 `D:\0-github_workspace\Topo\outputs\ev_original_stage\stress_asset_v2_payloads_restored` 恢复，5/5 个文件与raw全字节一致；源码快照gzip原字节也一致。
打包16.44s，恢复5.64s。进程BelowNormal，xz单线程，可能与其他研究操作重叠，计时不作为独立压缩性能基准。

原raw完整保留。`stress_asset_v2_manifest.json`记录gzip元数据、payload及原文件hash；`stress_asset_v2_archive_validation.json`记录实际命令、逐文件比对和验证脚本原字节。
