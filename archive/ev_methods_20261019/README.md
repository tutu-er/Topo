# EV 方法封存（2026-10-19）

当前状态：**sealed**。研究总览见[EV入口](../../research_experiments/ev/README.md)。本目录保留封存前的文件字节、原有版本/失败证据及恢复映射；旧目录中的运行提示是历史记录。

| 文件或目录 | 内容 |
|---|---|
| `payload/research_experiments/mobile_ev_mle/` | 原MLE/EV源码、测试、协议、报告、证据和本机输出 |
| `payload/research_experiments/mobile_ev_probing/` | 仍被导入的旧AC/候选组件及v6–v9历史材料 |
| `FILE_CATALOG.csv` | 源码/测试/文档/检查点逐文件作用、函数与静态输出提示；含原/新路径和Git归属 |
| `RESULTS_INDEX.csv` | 报告、结果、复现和检查点索引 |
| `MANIFEST.csv.gz` | 两包全部文件的原路径、封存路径、字节数、SHA256、原Git归属及类别；包括缓存和原始本机文件 |
| `EXTERNAL_LOCATIONS.csv` | D盘执行副本的原路径、新分组、文件数与总字节数；迁移前后清单摘要一致 |
| `EXTERNAL_FILES.csv.gz` | D盘执行副本的逐文件原/新相对路径、类别和字节数；按原组名可恢复原布局 |
| `STOP_RECEIPT.json`，`before_stop/` | 用户确认后停止标准馈线批次的过程、已落盘28条记录的hash、未完成分母及原运行状态文件 |
| `ARCHIVE_STATE.json` | 基准提交、冻结状态、统计范围、数据位置及既有改动保护 |
| `VERIFICATION.json` | 本次归档完整性核查结果；与旧科学测试记录分开 |
| `manage_archive.py` | 仅验证/恢复文件的标准库工具；不运行研究算法 |

## 保存和版本控制

原来由Git管理的文件继续作为逐文件迁移提交，内容保持原字节；原来未提交/忽略的本机原始记录、缓存和旧文件保留在本机payload，并由清单标记，未追加上传。因此新克隆可以验证原Git文件，完整本机恢复还需要这里保留的local-only文件及D盘执行目录。全量清单不是“所有原始数据已上传”的声明。

本次对两个源码包逐文件计算迁移前后SHA256；D盘执行副本在同一卷中整体改名，迁移前后核对文件名/大小清单摘要，并另核验停止批次的28份原始记录hash。没有删除旧源码、失败、未决、checkpoint或缓存。历史绝对路径/冻结指纹保持原字节；旧复现命令应在恢复原布局后使用。

原早期归档 `../mobile_ev_pre_mle_20261012/` 保留原位，按其README另行恢复。旧LaTeX/PDF及当前`pyproject.toml`留在仓库原位；其已有改动不夹带本次提交。

## Windows换行保护

部分历史子目录的Git规则会把CRLF转换为LF，改变冻结证据的SHA。原`.gitattributes`字节逐个保存在`payload/original_git_attributes/`并由同一MANIFEST映射；原位置新增的属性包装在末尾关闭换行转换。恢复工具按映射还原原属性文件，不复制包装层。`GIT_ATTRIBUTE_OVERRIDES.json`记录该层配置，科学源码和数据未改写。提交前还核对Git暂存blob与原文件SHA，确保新克隆得到相同冻结字节。

深层历史文件需要Windows Git长路径支持。本机仓库已设置`git config --local core.longpaths true`。新设备在克隆时使用`git -c core.longpaths=true clone <仓库地址>`，或在检出前为该仓库启用同一选项；此设置只影响Git路径处理。

## 验证

在封存仓库根目录运行（只读、不导入旧算法）：

```powershell
python archive/ev_methods_20261019/manage_archive.py --verify
python archive/ev_methods_20261019/manage_archive.py --verify --all-local
```

第一条要求全部原Git文件及已存在local-only文件符合哈希；未随Git取得的本机文件单列缺失。第二条要求全量本机payload均存在。外部执行副本另外核对名称/大小与停止回执中的SHA，不把该项冒充全量外部文件的字节哈希证明。

## 恢复与查看旧代码

恢复只写入**新的空目录**，按清单复制原路径，不覆盖任何已有文件、不启动算法，也不移动封存原件：

```powershell
python archive/ev_methods_20261019/manage_archive.py --restore C:/path/to/new-empty-ev-restore
python archive/ev_methods_20261019/manage_archive.py --restore C:/path/to/new-empty-ev-restore --all-local
```

默认复制原Git文件；`--all-local`要求并复制全部本机文件。工具在新目录写入`RESTORE_RECEIPT.json`。恢复得到`research_experiments/mobile_ev_mle/`与相邻`mobile_ev_probing/`，仍需匹配的仓库依赖、输入和环境；基准科学版本为`ARCHIVE_STATE.json`记录的封存前提交。对特定旧批次，优先按原证据包的freeze/source snapshot和对应REPRODUCE恢复完整执行环境。

D盘执行目录已经按 `EXTERNAL_LOCATIONS.csv` 集中到`outputs/ev_archive_20261019/runs/`或`checks/`。需要原绝对路径时，先检查旧目标为空、进程已停止，再按表将选定目录恢复原名；历史脚本可能在原目录写结果，复算宜另用空输出目录。标准馈线批次保持封存，重启、续跑或新科学实验须另行授权。
