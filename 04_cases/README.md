# 算例管理

每个算例使用唯一 `case_id`，输入参数来自 `scenario_matrix.csv`。建议目录形式：`case_<id>/inputs`、`case_<id>/logs`、`case_<id>/outputs`。

首个合成算例建议采用 8760 小时聚类为代表日：一个集中式节点，对比 100–10,000 个聚合企业节点；保持 AI 年服务量、可靠性和质量完全相同。

行业级扩展见 `industry_level_compute_pool.md`，参数入口为 `industry_scenario_matrix.csv`。该算例比较工厂分散、集团池化、2–8 个行业区域节点、单一行业中心和分层混合架构，同时报告计算扩容与分电压等级的电网扩容。

受监管企业的资本偏好测试见 `regulatory_capital_bias_test_design.md`，参数入口为 `regulatory_capital_bias_parameters.csv`。该测试暂不把混合部署作为核心，使用商业集团本地AI中心与公共云作为服务和市场成本基准，分别报告市场直接成本、用户收入需求、股东会计收益和经济净收益。云端社会资源成本尚未观测，不能由云采购价替代。

钢铁厂屋顶光伏采用“技术潜力 × 实现比例”的上限约束。基准和保守技术密度以及 0%–100% 实现比例组合见 `industry_rooftop_pv_sensitivity.csv`。有真实屋顶投影面积时优先使用；缺失时以 MECS 钢铁行业封闭楼面面积作为屋顶代理，并用厂型转移系数做敏感性，输入见 `02_data/industry_roof_area_proxy.csv`，推算结果见 `steel_rooftop_pv_proxy_summary.csv`。

两个企业最小原型的DER参数见 `china_prototype_der_parameters.csv`。其光伏上限不直接输入一个任意kWp，而由合成屋顶投影面积、屋顶可用率、组件效率和实际实现比例推导。
