#!/usr/bin/env python3
"""Summarize validated 31-industry equal-service architecture results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = {"IF": "工厂侧分布式、集团专网协同（核心）", "IG": "集团集中算力池", "II_1host": "行业池单节点部署"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.validation.read_text(encoding="utf-8"))
    rows = {row["scenario"]: row for row in payload["national_totals"]}
    parameters = payload["active_parameters"]
    energy_values = [
        rows[scenario]["national_annual_ai_facility_energy_twh"]
        for scenario in ("IF", "IG", "II_1host")
    ]
    lines = [
        "# 全国31个制造业大类等服务量架构比较",
        "",
        f"模型版本：`{payload['model_version']}`。所有31个行业在三种架构下均重构相同有效服务量，以下用电、服务器和扩容结果均由架构参数与优化内生得到，不再强制等电量。",
        "",
        "| 架构 | 全国日有效服务量 | AI设施用电（TWh/年） | 等价服务器组 | 新增电网容量（MW） | 年增量总成本（十亿元） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in ("IF", "IG", "II_1host"):
        row = rows[scenario]
        lines.append(
            f"| {LABELS[scenario]} | {row['national_daily_effective_service_units']:.0f} "
            f"| {row['national_annual_ai_facility_energy_twh']:.3f} "
            f"| {row['national_installed_server_groups']:.0f} "
            f"| {row['national_incremental_grid_expansion_mw']:.1f} "
            f"| {row['national_incremental_total_cost_rmb'] / 1e9:.3f} |"
        )
    lines.extend(
        [
            "",
            f"活动参数组采用`{parameters['compute_efficiency_case']}`算力效率档，单位有效服务需要{parameters['accelerator_h_per_service_unit']:.6f} accelerator-hour，并采用{parameters['server_maximum_wall_power_kw']:.2f} kW的双L20整机满载功率。三种架构的优化用电为{min(energy_values):.3f}—{max(energy_values):.3f} TWh/年，约为外部14 TWh中心值的{min(energy_values) / rows['IF']['national_external_central_energy_twh']:.3f}—{max(energy_values) / rows['IF']['national_external_central_energy_twh']:.3f}倍。",
            "",
            f"独立参考计算为{rows['IF']['national_derived_reference_energy_twh']:.3f} TWh/年，按65%装机利用率配置服务器并让全部参考服务器持续承担在线空闲功率；它是校验量而非架构用电约束。逐行业相对14 TWh分配的±25%检查仅作诊断，全国优化总量继续以{rows['IF']['national_external_low_energy_twh']:.0f}—{rows['IF']['national_external_high_energy_twh']:.0f} TWh外部包络校验。",
            "",
            f"简化生命周期层中，模型初始化电量相对年度推理设施用电可以忽略；服务器已含存储使基准增量存储成本为零。基础运维成本随部署点数变化：IF、IG和II_1host分别为{rows['IF']['national_model_operations_cost_rmb']/1e6:.1f}、{rows['IG']['national_model_operations_cost_rmb']/1e6:.1f}和{rows['II_1host']['national_model_operations_cost_rmb']/1e6:.1f}百万元/年。这是0.25 FTE/部署点的情景结果，不是制造业观测均值。",
            "",
            "这些结果是31个行业代表性负荷桶的连续容量筛查，不是省级或物理配网规划。行业池单节点是集中度上界情景；成本排序仍受当前两小时储能成本和简单电网容量代理影响。",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
