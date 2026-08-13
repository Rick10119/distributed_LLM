# 代码目录

`run_manufacturing_ai_future_demo.mjs` 生成五个典型制造行业的未来 AI 采用、服务需求和计算资源需求情景，并输出 CSV、Excel 工作簿和图表。

`run_manufacturing_ai_task_hardware_demo.mjs` 将十三个典型制造行业的中间情景拆为六类 AI 任务，先计算 L20 等效服务 GPU 小时，再换算未来加速器小时、池化双 GPU 服务器组、中央与边缘设备平均负荷和年用电。输出为条件情景和容量下限，不包含每厂整机最小采购、N+1 冗余或电网扩容优化。

`run_manufacturing_ai_31sector_topdown.mjs` 在保留十三行业独立参数的基础上，对C29/C30/C33/C35按行业自身详细任务情景进行同任务相对暴露量校准，为其余十四行业匹配近邻生产类型模板，生成完整31行业需求权重并分配8/14/28 TWh总量。程序保留首次校准前基线，并输出参数交叉表、完整计算表、校准比较和简要发现。

建议后续结构：

- `ingest/`：数据下载与导入；
- `clean/`：清洗和特征构造；
- `models/`：解析模型和优化模型；
- `run/`：批量算例入口；
- `viz/`：图表；
- `tests/`：单位测试、能量守恒和场景一致性检查。

第一版可用 Python + PyPSA/Pyomo；若要细化配电潮流，可接 OpenDSS 或 pandapower。工具选择应在数据和模型边界确认后再固定。

当前 `industry_capacity_screen.py` 实现行业级算力池的透明容量筛查。它不执行潮流或容量扩展优化，只复现第一轮研究笔记中的总新增接入和最大单点指标。

`steel_rooftop_ai_value_screen.py` 将钢铁厂屋顶光伏上限换算为太阳时段可匹配的 IT 功率、年度能量等价 IT 功率，以及不同关键小时光伏比例下的避免扩容和投资价值敏感性。

`china_minimum_prototype.py` 读取两个企业24小时代表日和中国参数表，生成本地、云端和50/50混合部署的企业节点与数据中心负荷，并输出最大需量、筛查级扩容和企业年化直接成本。它使用合成的满载等效IT任务，不是L20实测吞吐量模型。

`china_minimum_prototype_der.py` 在上述结果上加入屋顶约束光伏和循环储能。光伏上限按“屋顶投影面积×可用比例×组件效率×实际实现比例”生成；已有DER情景的AI增量电费通过“同一DER下加入AI前后电费之差”计算，避免将原本已服务企业负荷的光伏重复记为AI收益。

`run_china_prototype_full.py` 是最小原型的统一入口，直接生成24个基础场景、100行单因素敏感性、6个连续求根阈值、330个二维参数组合和11张SVG结果图。`validate_china_prototype.py` 检查场景数量、端点模式、DER扩容单调性、云价只影响企业成本、接入余量单调性、混合比例权衡方向和补充图件完整性。

`run_typical_manufacturing_base.py` 使用EWELD机械设备制造负荷形状和token推导的2030集成AI工作量，运行一个不含敏感性的本地、云端和50%混合基础案例。`validate_typical_manufacturing_base.py` 检查任务量、服务器数、容量、成本排序、电池边界和图件完整性。

`manufacturing_ai_extended_load_screen.py` 将边缘视觉、VLM、数字孪生和生产优化加入单厂量级筛查。`group_ai_center_screen.py` 比较单厂服务器离散投资、20厂分别部署、集团集中部署和公共云。`commercial_group_office_screen.py` 独立构造800人商业办公楼负荷与AI需求，并运行20栋楼商业集团的同类比较。

`regulatory_capital_bias_screen.py` 在商业集团案例上运行十二年监管激励筛查，严格区分市场直接成本、用户收入需求、股东会计收益和股东经济净价值。该程序测试云价、本地投资、有效资产比例、准许回报差、回收滞后、审慎性剔除和技术过时，并输出阈值表与四张核心图；它不预测真实企业的选择概率。

`build_manufacturing_load_archetypes.py` 直接读取已下载的压缩包和JSON，使用EWELD设施观测构造六类工作日/周末负荷原型，并生成中国31个制造业大类到ISIC和曲线原型的显式交叉表；韩国工厂、UCI钢厂、FfE、德国匿名工厂和ELMAS只用于不同层次的对照。`validate_manufacturing_load_archetypes.py` 检查10个下载文件、31行业覆盖、六类主曲线的24小时时点和图件完整性。

`run_manufacturing_31sector_peak_screen.py`与对应校验现为历史复现脚本，只写入`05_results/archive/equal_electricity_national/`。活动核心模型从该历史表中仅抽取行业原负荷身份、归一化形状和时段，写入`02_data/processed/core/`；旧AI电量与AI功率字段不进入`v0.2.0`模型。

`run_simple_national_local_cloud_screen.py`仅用于复现归档的旧等电量扩容代理；活动全国比较由版本化Snakemake工作流生成。

`run_single_industry_service_aligned_prototype.py` 是修改后核心公式的首个单行业实现，只运行C36汽车制造业。它以相同加速器服务量比较本地与云端，使用共同任务柔性比例和截止期、循环24小时期限流、服务器额定功率、空闲功率、PUE、目标装机利用率和备用容量；调度采用“先最小化叠加峰值、再最小化AI设施峰值”的字典序目标。`validate_single_industry_service_aligned_prototype.py` 检查服务守恒、能耗守恒、额定算力上限和峰值方向。

`run_single_industry_pypsa_joint_prototype.py` 将C36进一步迁移到PyPSA 1.2.2、Linopy 0.7.0和HiGHS 1.13.1。它运行无AI基线、本地AI和绿地云端AI三个反事实，在统一平段电量价和最大需量基本电费下，联合优化连续双L20服务器组、聚合AI任务时序、屋顶光伏、两小时储能和新增接入容量；结果用无AI最优值计算增量，避免把基础负荷原本就会采用的DER归因于AI。`validate_single_industry_pypsa_joint_prototype.py` 检查服务守恒、服务器额定算力、设施功率重构、屋顶上限、增量口径和HiGHS求解标识。

`run_single_industry_enterprise_direct_cost.py` 在同一C36服务量上比较企业实际支付的三种模式：本地购买并年化、按峰值容量购买双L20云实例包月服务，以及按执行实例时购买云服务。本地成本采用相对无AI基线的服务器、附属设施、电量、最大需量、接入和DER增量；云端订阅价格视为企业付款，不重复计入提供商底层服务器、电力和电网成本。`validate_single_industry_enterprise_direct_cost.py` 检查成本分解、包月容量、按量实例时和临界价格口径。

## 当前核心工作流

根目录 `Snakefile` 是新的统一入口。`config/defaults.yaml` 保存共同参数，`config/runs/single_industry_core.yaml` 只选择C36及IF、IG、II_1host三个尺度，`config/runs/all_industries_core.yaml` 用于后续批量扩展。运行配置不再散落在脚本常量中。

主线情景的统一选择入口是 `config/scenarios/mainline.yaml`。其中 `countries.enabled` 固定生成中国与美国结果：中国在核心逐时模型中按任务路由联合优化CPU/GPU装机、在线容量和设施功率，美国读取本国需求与价格参数进行下游重估，尚不重复运行一套美国逐时核心模型；`compute_hardware.active_routing_case` 默认选择 `practice_routed`；`resource_footprint` 分别选择水耗、空间、建筑材料、省级缺水度和云端地域分配的数据表及命名情景。硬件路由现在会改变中国核心物理负荷和成本，因而切换路由配置必须触发并完成31行业乘以三个架构的核心重跑，不能只重跑成本后处理或制图。

`core/` 内部分工如下：`config.py` 负责配置合并与校验，`representative_group.py` 将行业份额和成员工厂数换算成三个等服务量尺度，`data.py` 保留并缩放六任务分辨率的刚性负荷与柔性作业，`model.py` 用PyPSA/Linopy/Gurobi在一次求解中分别优化GPU和CPU装机、在线容量、任务调度与功率，并联合计算最大需量、接入、光伏和储能，`io.py` 统一输出。服务器、电费和其他物理成本均由该次求解自下而上产生，不设口径校准项。默认求解器为Gurobi，配置中仍保留HiGHS作为显式可选项。

每个行业先运行一次无AI典型工厂基准。随后，IF、IG和II_1host三个本地部署尺度按活动任务路由分别形成GPU和CPU的未错峰到达曲线；每个硬件池以`max(110%参考容量, ceil(参考容量)+1)`作为最低装机，再在同一次正式优化中调度两类硬件并进行增量归因。全国主线仍为31次无AI基准加93次正式情景求解，共124次优化，但每个正式情景现在是双硬件联合模型。`validate_core_scenario.py` 检查服务守恒、两类装机容量规则、CPU/GPU功率加总、GPU/CPU服务器成本拆分、设施总功率、既有屋顶光伏边界、逐时电价和总成本复算。完整测试设计见 [`../00_admin/core_scenario_test_plan.md`](../00_admin/core_scenario_test_plan.md)。

状态说明（2026-08-13）：代码改造和C36--IF单行业验证已完成；31行业乘以三个架构的全国重跑及Figure 2重生成按用户要求中止。`05_results/v0.8.0`中当前已有的全国异构汇总和Figure 2不得当作联合CPU/GPU物理模型的新结果。恢复工作时应从全国核心情景开始重跑，随后依次重建中国异构汇总、跨国成本汇总和Figure 2。

已有 `pypsa` 环境时，C36可用以下命令运行：

```bash
conda run -n pypsa snakemake all --cores 1 --configfile config/runs/single_industry_core.yaml
```

新环境可先按根目录 `environment.yml` 建立，或让Snakemake使用 `workflow/envs/core_model.yaml`。每个模型版本使用独立目录：模型直接输出写入 `05_results/{model_version}/model/{industry}/{scenario}/`，校验、情景比较和结论写入 `05_results/{model_version}/result/{industry}/`。`model_version`在`config/defaults.yaml`中显式设置，并采用`vMAJOR.MINOR.PATCH`格式。

核心主流程默认读取`02_data/processed/core/manufacturing_31sector_real_weeks.csv`中的31行业连续168小时EWELD实测周。24小时行业曲线仅用于无结果导向地筛选代表周并保持行业平均负荷尺度，不再由模型重复7次。27个行业使用同行业或部分对应ISIC实测周；C16、C25、C37和C42使用生产原型相近的EWELD企业真实周代理并保留显式血缘。读取器要求完整168小时和来源字段，并拒绝24小时机械重复。

`v0.7.0`在`v0.6.1`的168小时连续代表周、三档算力效率、模型生命周期、广东逐时到户电价、分项储能成本和零接入余量信用基础上，加入分行业屋顶光伏上限。31个中国制造业行业通过显式交叉表映射到2022美国MECS三位NAICS行业，使用每家企业平均封闭建筑面积作为美国参照屋顶代理；22%组件效率、90%可用比例和80%实现比例仍为共同技术参数。直接映射、父行业映射和近邻映射分别保留证据等级。该口径来自Namin等（2023）的方法，但不是中国屋顶实测值，后续应以中国分行业建筑或遥感数据替换，并对中国转移系数做敏感性分析。旧peak和national local-cloud脚本只复现`05_results/archive/equal_electricity_national/`。全国运行使用：

美国核心成本模块暂时保持固定物理量价格反事实：保留中国任务负荷、服务器数量和AI设施用电，只替换美国服务器价格、寿命和全国工业全包电价。美国光伏、储能、最大需量和电网扩容均明确排除，且不从中国参数转移，因此该结果不是完整美国本地部署成本。边界说明见 [`../03_models/us_cost_counterfactual_boundary_notes.md`](../03_models/us_cost_counterfactual_boundary_notes.md)。

默认目标`all`只生成核心输入、测试、124次核心优化及其行业/全国校验结果。云付款、成本、负荷配合、资源足迹等后处理改由`extended_analysis`显式触发；历史无柔性消融、独立现货/PV测试和阶段汇报改由`optional_diagnostics`显式触发，不再进入默认流程。

`analyze_api_token_cost.py`读取复核后的官方 API 价格、31行业就业基线、项目既有 office/agent Token 参数、预留 GPU-IaaS 付款结果和 `CL004` 对象存储价，生成 `full_cloud_cost_v1.0.0` 完整云化企业付款场景。office/agent 走 Token API，其余四类任务走预留 GPU 云，并与 IF、IG、II 和全工作负荷预留 GPU-IaaS 基准方案比较；正式中国表只保留Alibaba Cloud和DeepSeek，不展示按量 GPU 订阅。其他厂商、轻量模型及低/基准/高预留价格保留在详细审计表。剩余 GPU 容量目前按有效服务份额分配、尚未按峰值重优化，因此不能解释为质量等效采购建议或核心科学主结果。

`prepare_us_manufacturing_ai_research.py`把 Economic Census、MOPS、BTOS、MECS 和 BERD 官方输入整理为五个稳定 CSV 及六任务参数表。`build_us_manufacturing_ai_demand.py`按“美国活动量×采用率×每个采用者每日任务单元×单任务有效服务×行业适用度”自下而上生成 NAICS3×六任务×low/base/high 需求、Token、逐时容量、本地核心成本与完整云化付款；不存在按中国总量反推的缩放乘数。中国同档用电和制造业增加值比例仅作外部量级检查。office/agent 才生成 Token，其余四任务按任务峰值采购预留 GPU，且不输出按量 GPU。`validate_us_manufacturing_ai_demand.py`检查 21 行业、六任务、三情景覆盖、采用率边界、自下而上汇总、外部量级检查、Token 边界、单一 10% 装机裕量和成本组件核对。该模块已纳入项目根目录 `Makefile` 调用的主 Snakemake 依赖图。

`analyze_us_owned_core_cost.py`和`analyze_us_full_cloud_cost.py`已分别作为主流程规则`analyze_us_owned_core_cost`与`analyze_us_full_cloud_cost`接入。它们在相同物理需求与Token总量上建立独立的美国价格环境：本地成本读取EIA全国工业电价与美国服务器BOM代理；云端将美国区Token API、AWS一年期预留GPU容量代理和S3 Standard相加。美国正式表只保留OpenAI、Anthropic和Google，不展示按量GPU；完整五厂商表作为审计输出。中美分别在各自本币价格环境内计算比值。

`run_single_industry_heterogeneous_hardware_screen.py`与`run_single_industry_heterogeneous_hardware_us_cost.py`已纳入扩展主流程。中国脚本不再自行估算本地硬件和电力后回填差额，而是读取核心联合物理求解已经输出的GPU/CPU装机、设施能耗、服务器成本、电网成本和总成本；它只补充同服务边界下的云端CPU/GPU/API付款比较。美国脚本仍只替换美国需求与价格环境。美国CPU整机核心值采用16,000美元的标准2U企业服务器公开配置代理，10,786美元与27,263.50美元分别作为低值和GPU-ready超配高值敏感性。

`materialize_sensitivity_case.py`、`summarize_single_industry_oat.py`与`workflow/rules/single_industry_sensitivity.smk`构成单行业单因素敏感性入口。对外代码名不包含行业编号，实际行业由`config/sensitivity/single_industry_oat_v1.yaml` 的`industry`字段选择，当前为C38。注册表只允许显式白名单参数覆盖，所有输出写入独立敏感性目录；无AI基准可复用，但IF、IG、II使用敏感性目录内的同代码参考运行。该流程继承核心模型求解器配置；当前为Gurobi。`make sensitivity-smoke`运行单行业敏感性，`make sensitivity-smoke-dry-run`只检查任务图。

`config/sensitivity/single_industry_grid_hybrid_v1.yaml`另定义“接入扩容—储能—云订阅”结构测试。模型内生决定每项刚性或柔性任务由本地CPU/GPU服务器还是预留云容量完成，总服务量严格守恒；云端执行不占用企业接入容量，但计入年度GPU/CPU订阅成本。本地服务器、储能与云容量共同优化，允许在纯本地和纯云之间形成混合解。零扩容且允许订阅时不强制保留本地模型副本或最低在线服务器；只有实际选择本地执行时才由算力约束带出本地装机。四组均允许储能并关闭PV，分别比较正常扩容下纯本地与混合部署、严格零扩容下云订阅替代，以及高扩容惩罚下纯本地通过储能和错峰尽量减少但不硬性禁止扩容。高惩罚是稀缺性压力测试而非现实报价；每组使用同配置重新求解的无AI基准。`make sensitivity-grid-hybrid`单独运行，默认`make results`不包含它。

## 集团单节点与跨节点灵活性测试

`config/sensitivity/c36_group_multisite_continuous_v1.yaml`定义独立于全国核心场景的集团多工厂机制测试。当前使用C36汽车制造业、5个合成成员工厂和三种企业内部部署：

- `IF`：每厂安装并仅服务本厂，GPU/CPU装机服务器组为整数；
- `IG_1host`：集团在一个固定成员工厂建设共享池，装机采用连续等效容量；
- `IG_multisite`：集团在多个成员工厂配置连续等效容量，可迁移任务在原截止窗口内联合选择执行时间和工厂。

工厂曲线优先来自不同EWELD用户；若行业可用用户数不足，则从同一用户选择不同完整周，不复制完全相同的用户—周。所有曲线按星期—小时对齐，因此是合成机制输入，不表示真实同一集团的同步观测。

测试只为核心`IG_1host`增加零基础负荷配对反事实。实际运行的四个汇总组合固定为：

1. `IF + actual_load`；
2. `IG_1host + actual_load`；
3. `IG_1host + zero_load`；
4. `IG_multisite + actual_load`。

`zero_load`保持AI服务量、释放时刻、截止时间、CPU/GPU路由、价格和定容口径不变，并重新优化，不表示工厂停产。`IG_1host zero_load - actual_load`用于计算固定承载工厂的生产负荷匹配价值；实际负荷下`IG_multisite - IG_1host`用于分析跨节点调度价值。

首次运行前建议先检查任务图，不求解：

```bash
make sensitivity-group-multisite-dry-run
```

正式运行：

```bash
make sensitivity-group-multisite CORES=5
```

等价的直接Snakemake命令为：

```bash
conda run -n pypsa snakemake single_industry_group_multisite_sensitivity \
  --cores 5 \
  --configfile config/runs/all_industries_core.yaml \
  --runtime-source-cache-path "$(mktemp -d /private/tmp/dllm_group_multisite.XXXXXX)" \
  --rerun-incomplete
```

该目标不会由默认`make results`或`snakemake core`触发。结果写入`05_results/sensitivity/v0.8.0/group_multisite/C36/`：

- `summary.csv`：四个架构—基础负荷组合的汇总；
- `hourly.csv`：逐工厂、逐小时AI执行、设施功率和购电；
- `curve_lineage.csv`：每个合成工厂的EWELD用户和完整周血缘；
- `load_alignment_value.csv`：仅IG_1host的零负荷—实际负荷配对差额；
- `metadata.json`：配置边界、变量口径、服务守恒和限制说明。

修改行业、工厂数、源ISIC、求解器或输出目录时，只编辑上述测试配置；Snakemake规则本身不保存这些实验参数。若要强制重跑，可追加`SNAKEMAKE_ARGS="--forceall"`，但这会覆盖同一`output_root`中的既有测试输出，运行前应先修改`output_root`保留旧版本。

`analyze_core_industry_cost_differences.py`只读取31行业的IF、IG和II_1host核心结果，同时报告年成本总量与单位服务成本，并将行业差异拆分为服务器、电量、接入容量、PV/储能与模型固定费用。它另外计算需求规模、CPU路由、部署碎片化、定容冗余、能耗强度和接入容量强度与单位成本的描述性Spearman关联，不将其解释为因果归因。`make industry-cost-differences`可单独生成该表；默认核心流程也会在31行业核心结果完成后自动生成。

`analyze_land_material_footprint.py`读取全国 `core_scenarios.csv`，从 `II_1host` 的31行业安装服务器组和整机额定功率重建共同 MW-IT 容量，再生成小型厂房复用以及中美大型设施“既有容量—园区扩建—绿地新建”的空间结果。材料仅由新增总楼面面积驱动，钢筋与结构钢分开；ICEF 园区级材料参数只做独立量级校验。该脚本由 Snakemake 的 `analyze_land_material_footprint` 规则调用，并输出 CSV、血缘、结论和校验完成标记。

`analyze_typical_industry_load_stacking.py`选取食品、纺织、化工、汽车和电子设备五个典型行业，比较IF工厂、IG集团和II行业集中代表节点的168小时原有负荷、AI设施负荷与叠加后负荷，并另行输出AI负荷时序图，避免小规模IF曲线在绝对叠加图中不可见。

`analyze_industry_spot_price_pv_test.py`只作为历史诊断保留；它不再是获得现货/PV边界所必需的独立场景，因为相同的广东代表周现货到户价和既有屋顶PV已经进入全部核心基准与三种架构。该诊断仍关闭新增储能，用于复核旧机制结果，只有显式请求`optional_diagnostics`时才运行。

```bash
conda run -n pypsa snakemake all --configfile config/defaults.yaml config/runs/all_industries_core.yaml --cores 4
```
