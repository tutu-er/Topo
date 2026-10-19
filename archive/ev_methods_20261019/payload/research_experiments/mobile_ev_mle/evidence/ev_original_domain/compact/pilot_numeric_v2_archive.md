# pilot_numeric_v2 的可恢复证据

用途：development；独立正式评价计数为0。此目录与 evaluation_v2 的12个独立评价历史分开。

选择 7 个文件：freeze、68源源码快照、4 个 trial、成功的 review_full；不压入CSV、checkpoint或重复summary。

原文件合计 23,659,138 bytes；solid tar.xz 为 8,079,564 bytes（34.15%）。
归档SHA-256：`25ac280342e38405c3ed6a0d9cc87b63819c6b5bdd59442b6bfdff8dc0890735`。

已在全新的D目录 `D:\0-github_workspace\Topo\outputs\ev_original_stage\pilot_numeric_v2_payloads_restored` 恢复，7/7 个文件与raw全字节一致；源码快照gzip原字节也一致。
打包45.81s，恢复17.89s。进程BelowNormal，xz单线程，可能与其他研究操作重叠，计时不作为独立压缩性能基准。

原raw完整保留。`pilot_numeric_v2_manifest.json`记录gzip元数据、payload及原文件hash；`pilot_numeric_v2_archive_validation.json`记录实际命令、逐文件比对和验证脚本原字节。
