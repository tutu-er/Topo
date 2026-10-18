# 数据归属与复现文件

`evidence/inputs/prepared.json.gz`是从SimBench 1.6.3官方数据库提取的选定窗口，保留原生15分钟P/Q、PV形状和来源信息；选窗、映射、预测开发规则由本项目登记。该数据提取及派生数据库按[ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/)提供，数据库内容按[DbCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/)提供。版权归属及官方声明见`official/SimBench_1.6.3_LICENSE.txt`，逐原文件哈希见`evidence/inputs/source_acquisition.json`。数据的许可不把本项目研究源码自动变成数据库。

CIGRE数值来自固定版本pandapower v3.2.1；官方源文件及其版权、BSD许可随`official/`保留，仅静态解析核对，不执行下载代码。

case33bw来自固定版本MATPOWER 7.1及其列明的Baran–Wu论文。MATPOWER官方明确：软件采用BSD，case数据文件不归入该BSD许可；参见`official/MATPOWER_7.1_LICENSE.txt`。本项目保留原case来源、原表及数值比对，不将case数据标作本项目原创或擅自标BSD。

paper11/paper15来自本项目已有自建树，只作为论文风格兼容测试，不附作者原始算例标签。SINTEF/CINELDI本轮只作来源审查，没有把单位未确认的原始负荷数据混入正式AC输入。
