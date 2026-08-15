# 企业分布式 LLM 与电网扩容需求研究

本项目研究：在提供相同 AI 服务的前提下，企业私有/离线/本地部署 LLM，相比集中式超大规模数据中心，能否以及在什么条件下降低发电、输电、变电与配电扩容需求。

## 当前判断

- 核心研究对象不是单纯的“用电量”，而是 AI 负荷的**位置、同时率、峰值时段、接入电压等级与可调节性**。
- 分布式方案可能复用企业现有配电容量、分布式光伏、储能和备用电源，减少单点大容量并网；但也可能因设备利用率较低、PUE 较高和节点冗余而增加总用电量。
- 因此合理的主问题是：在同等 AI 服务、可靠性、时延和隐私约束下，分布式与混合式部署能避免多少不同电压等级的新增容量及投资？
- 新增行业级扩展：对 4–5 家龙头集团及其异地生产企业，比较工厂分散、集团池化、若干区域算力节点与单一行业大型数据中心的计算及电网扩容需求。
- 当前下一阶段转向全国制造业尺度：按“省级地区×制造业大类×企业规模/组织类型”覆盖31个制造业大类，路线图见 `00_admin/national_industry_cost_analysis_roadmap.md`。
- 制造业负荷曲线已形成第一版六类原型：以244个EWELD制造业用户为中国观测锚点，并用韩国、德国、法国和FfE数据做分层对照；31行业中18个为直接映射、9个为部分映射、4个仍使用代理曲线。
- 31行业峰值筛查已改为“行业专属优先、原型回退”：26个行业使用EWELD同一ISIC曲线，钢铁和其他运输设备使用专属外部曲线，仅3个行业回退到六类原型。14 TWh中心情景的AI平均负荷约1.60 GW，按任务时序峰值约2.26 GW；尚未加入31行业原有电量的绝对缩放。
- “单个数据中心 20–70 GW”暂不作为基准事实。公开资料更支持百兆瓦至 1 GW 以上的大型接入；20–70 GW 宜表述为区域、国家或项目管线的聚合规模，除非后续找到明确项目证据。

## 目录

| 目录 | 用途 |
|---|---|
| `00_admin` | 研究章程、任务清单、决策记录 |
| `01_literature` | 文献笔记、来源台账、综述 |
| `02_data` | 原始数据、处理中数据、数据字典 |
| `03_models` | 数学模型、假设、参数与验证说明 |
| `04_cases` | 基准系统、场景和敏感性算例 |
| `05_results` | 表格、日志和可复现实验输出 |
| `06_manuscript` | 论文提纲、章节草稿和投稿材料 |
| `07_figures` | 最终图片与制图源文件 |
| `08_code` | 数据处理、优化和可视化代码 |
| `09_meetings` | 会议纪要与想法日志 |

## 从零开始运行

以下命令均在项目根目录执行。主流程默认运行31个制造业行业的无AI基准，以及IF、IG和II_1host三个架构，并完成行业与全国结果校验。基础负荷使用31行业各自的连续168小时EWELD实测代表周；不会把24小时典型日重复7次。

macOS / Linux 可用 `make`（Makefile 只是对 Snakemake 的封装）。Windows 请直接调用 Snakemake，见下文「Windows：直接使用 Snakemake」。

### 1. 创建运行环境

需要先安装Conda（Miniconda、Miniforge或Anaconda均可），然后创建名为`pypsa`的环境：

```bash
cd /path/to/distributed_LLM
conda env create -n pypsa -f environment.yml
```

如果环境已经存在，则更新依赖：

```bash
conda env update -n pypsa -f environment.yml --prune
```

无需手动激活环境；`make` 和下面的 Snakemake 命令都通过 `conda run -n pypsa` 调用它。可用以下命令检查关键程序：

```bash
conda run -n pypsa python --version
conda run -n pypsa snakemake --version
```

### 2. 检查核心输入

完整项目副本应至少包含以下三个大体积或模型就绪输入：

macOS / Linux：

```bash
test -f 02_data/raw_load_profiles/eweld/EWELD.zip
test -f 02_data/raw/curated/gd_province_avg_node_price_dayahead_actual_20211101_20260603.csv
test -f 02_data/processed/core/manufacturing_31sector_real_weeks.csv
```

三条命令均无输出即表示文件存在。Windows PowerShell 用 `Test-Path`，返回 `True` 即表示文件存在：

```powershell
Test-Path 02_data/raw_load_profiles/eweld/EWELD.zip
Test-Path 02_data/raw/curated/gd_province_avg_node_price_dayahead_actual_20211101_20260603.csv
Test-Path 02_data/processed/core/manufacturing_31sector_real_weeks.csv
```

若是通过不包含数据的代码包获得项目，需要先恢复这些输入；不要用24小时典型日替代真实周。各行业代表周的来源和代理关系记录在`02_data/processed/core/manufacturing_31sector_real_weeks.lineage.json`。

### 3. 先做任务预演

Windows 请跳到「Windows：直接使用 Snakemake」。macOS / Linux 可用 `make`。预演只构建依赖图，不运行模型或改写结果：

```bash
make dry-run
```

默认配置为`config/runs/all_industries_core.yaml`，默认使用5个核心。可以覆盖核心数：

```bash
make dry-run CORES=2
```

### 4. 运行全国核心流程

```bash
make
```

也可以显式指定环境、核心数和配置：

```bash
make ENV=pypsa CORES=5 CONFIG=config/runs/all_industries_core.yaml
```

运行中断后再次执行同一条`make`命令即可续跑。Makefile已启用`--rerun-incomplete`，Snakemake会保留已经完成且仍然有效的任务，并重新运行不完整或上游代码发生变化的任务。运行期间不要修改模型代码或配置，否则同一结果目录可能混入不同输出结构。

如需先做较小的C36单行业检查：

```bash
make CONFIG=config/runs/single_industry_core.yaml CORES=1
```

### 5. 生成扩展分析和图表

全国核心结果成功后，再运行论文后处理、国家比较和主图：

```bash
make extended-analysis
```

阶段汇报页面单独生成：

```bash
make briefing
```

敏感性分析不是默认`make`的一部分；可先查看对应任务：

```bash
make sensitivity-smoke-dry-run
make sensitivity-grid-hybrid-dry-run
```

### Windows：直接使用 Snakemake

当前 Makefile 依赖 GNU Make，并把临时目录写死为 macOS 的 `/private/tmp`，因此不要在 Windows 上使用 `make`。Windows 也不必传入 `--runtime-source-cache-path`。

先激活环境，再直接调用 `snakemake`。提示符出现 `(pypsa)` 后，不要再套一层 `conda run`：Windows 上 `conda run` 启动很慢，看起来像卡住。

```powershell
conda activate pypsa
```

工作流没有内存上限；每个优化任务占用 5 个线程。`--cores 5` 因此同一时刻只跑 1 个求解。31 行业核心流程约有 31 个基准加上 93 个情景求解，本身就会很久。第一次请先跑 C36 单行业：

```powershell
snakemake core --cores 5 --configfile config/runs/single_industry_core.yaml --dry-run
snakemake core --cores 5 --configfile config/runs/single_industry_core.yaml --rerun-incomplete
```

确认单行业能跑通后，再跑全国核心流程。机器核数更多时可加大 `--cores`（例如 10 可同时跑 2 个求解）：

```powershell
snakemake core --cores 5 --configfile config/runs/all_industries_core.yaml --dry-run
snakemake core --cores 5 --configfile config/runs/all_industries_core.yaml --rerun-incomplete
```

运行中断后再次执行同一条命令即可续跑。`--rerun-incomplete` 会保留已经完成且仍然有效的任务，并重新运行不完整或上游代码发生变化的任务。运行期间不要修改模型代码或配置，否则同一结果目录可能混入不同输出结构。

`make` 目标与 Snakemake 命令的对应关系（均假设已 `conda activate pypsa`）：

| `make` 目标 | Snakemake 命令 |
|---|---|
| `make dry-run` | `snakemake core --cores 5 --configfile config/runs/all_industries_core.yaml --dry-run` |
| `make` | `snakemake core --cores 5 --configfile config/runs/all_industries_core.yaml --rerun-incomplete` |
| `make extended-analysis` | `snakemake extended_analysis --cores 5 --configfile config/runs/all_industries_core.yaml --rerun-incomplete` |
| `make briefing` | `snakemake build_bolun_progress_briefing --cores 5 --configfile config/runs/all_industries_core.yaml --rerun-incomplete` |
| `make sensitivity-smoke-dry-run` | `snakemake single_industry_sensitivity --cores 5 --configfile config/runs/all_industries_core.yaml --dry-run` |
| `make sensitivity-grid-hybrid-dry-run` | `snakemake single_industry_grid_hybrid_sensitivity --cores 5 --configfile config/runs/all_industries_core.yaml --dry-run` |
| `make sensitivity-smoke` | `snakemake single_industry_sensitivity --cores 5 --configfile config/runs/all_industries_core.yaml --rerun-incomplete` |
| `make sensitivity-grid-hybrid` | `snakemake single_industry_grid_hybrid_sensitivity --cores 5 --configfile config/runs/all_industries_core.yaml --rerun-incomplete` |
| `make industry-cost-differences` | `snakemake core_industry_cost_differences --cores 5 --configfile config/runs/all_industries_core.yaml --rerun-incomplete` |
| `make national-cloud-center` | `snakemake national_cloud_center --cores 5 --configfile config/runs/all_industries_core.yaml --rerun-incomplete` |
| `make national-grid-comparison` | `snakemake national_grid_capacity_comparison --cores 5 --configfile config/runs/all_industries_core.yaml --rerun-incomplete` |
| `make national-no-shift-sensitivity` | `snakemake national_no_shift_sensitivity --cores 5 --configfile config/runs/all_industries_core.yaml --rerun-incomplete` |
| `make national-high-impact-sensitivity` | `snakemake national_high_impact_sensitivity --cores 5 --configfile config/runs/all_industries_core.yaml --rerun-incomplete` |

默认环境名为 `pypsa`，默认目标为 `core`，默认配置为 `config/runs/all_industries_core.yaml`，默认使用 5 个核心。

### 6. 查找结果

活动版本由`config/defaults.yaml`中的`model_version`决定。目录结构为：

```text
05_results/<model_version>/model/<industry>/<scenario>/
05_results/<model_version>/result/<industry>/<scenario>/
05_results/<model_version>/result/national/
05_results/<model_version>/result/manuscript_figures/
```

`model/`保存逐时结果、汇总表和解析后的配置；`result/`保存校验、全国汇总、分析和图表。只有对应的`validated.done.json`存在且状态通过时，情景结果才视为完成。

### 7. 直接运行单幅主线图

`08_code/build_figure1_method.py`、`build_figure1_demand_architecture.py`以及`build_figure2_...`至`build_figure5_...`均可在编辑器中直接点击运行，无需填写命令行参数。默认读取 v0.8.0 主线结果，并把图片和配套数据写入`05_results/v0.8.0/result/manuscript_figures/`。Snakemake仍可显式传入其他路径并覆盖这些默认值。

除Figure 1方法图外，直接绘图只重建图表，不替代上游优化求解；应先确保`group_architecture_core/national/`中的全国核心汇总已经生成。

### 常见问题

- `IndentationError`、`SyntaxError`或`KeyError`通常表示代码与已有输出结构不一致。停止编辑代码后重新运行同一条`make`或`snakemake`命令，让Snakemake重建受影响结果。
- `Directory cannot be locked`表示另一个Snakemake进程正在运行，或上一次进程异常退出。先确认没有其他Snakemake进程；只有确认不存在运行中的任务后，才执行：

  ```bash
  conda run -n pypsa snakemake --unlock --configfile config/runs/all_industries_core.yaml
  ```

- 若需要查看更完整的失败原因，读取终端中第一个Python traceback；最后的`CalledProcessError`通常只是Snakemake对上游异常的包装。运行日志位于`.snakemake/log/`。
- 不要同时启动两个写入同一版本目录的主流程，也不要在主流程运行期间修改`08_code/core/`、配置或Snakemake规则。

## 建议起步顺序

完整扩展方案及项目盘点见 `00_admin/full_prototype_test_plan.md`。

当前优先执行的简化版本见 `00_admin/minimum_china_prototype_plan.md`：使用中国合成参数和现有24小时代表日，先跑通本地、云端与混合部署的企业成本和社会成本比较。

1. 建立中国情景的最小参数表，包括服务器、云服务、电价、接入容量和容量投资代理。
2. 复用 `04_cases/two_user_typical_day.csv` 和 `04_cases/two_user_pv_battery_typical_day.csv` 构建两个24小时合成案例。
3. 跑通本地、云端和混合三种部署的企业成本、社会成本和新增接入容量。
4. 完成服务器利用率、云服务价格、接入余量和混合比例四组敏感性。
5. 最小原型得到有区分度的结果后，再按 `00_admin/full_prototype_test_plan.md` 扩展真实数据、全年时序和完整电网模型。
