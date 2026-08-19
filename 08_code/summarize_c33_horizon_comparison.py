#!/usr/bin/env python3
"""Compare C33 measured-week and typical-day group-architecture results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=Path, required=True)
    parser.add_argument("--week-metadata", type=Path, required=True)
    parser.add_argument("--day", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--findings", type=Path, required=True)
    parser.add_argument("--done", type=Path, required=True)
    return parser.parse_args()


def extract(path: Path, horizon: str) -> list[dict]:
    data = pd.read_csv(path, encoding="utf-8-sig").set_index(
        ["architecture", "base_load_case"]
    )
    host = data.loc[("IG_1host", "actual_load")]
    multisite = data.loc[("IG_multisite", "actual_load")]
    zero = data.loc[("IG_1host", "zero_load")]
    metrics = [
        "annual_ai_energy_cost_rmb",
        "annual_incremental_maximum_demand_cost_rmb",
        "sum_incremental_grid_peak_mw",
        "annual_incremental_total_cost_rmb",
    ]
    rows = []
    for metric in metrics:
        rows.append({
            "horizon": horizon,
            "comparison": "IG_multisite_minus_IG_1host",
            "metric": metric,
            "IG_1host_value": float(host[metric]),
            "IG_multisite_value": float(multisite[metric]),
            "difference": float(multisite[metric] - host[metric]),
            "saving_fraction": 1.0 - float(multisite[metric] / host[metric]),
        })
    for metric in ["annual_incremental_total_cost_rmb", "sum_incremental_grid_peak_mw"]:
        rows.append({
            "horizon": horizon,
            "comparison": "IG_1host_zero_minus_actual",
            "metric": metric,
            "IG_1host_value": float(host[metric]),
            "IG_multisite_value": float("nan"),
            "difference": float(zero[metric] - host[metric]),
            "saving_fraction": float("nan"),
        })
    return rows


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.week_metadata.read_text(encoding="utf-8"))
    group_share = float(metadata["group_share"])
    if not 0 < group_share <= 1:
        raise ValueError(f"Invalid representative group share: {group_share}")
    multiplier = 1.0 / group_share

    comparison = pd.DataFrame(
        extract(args.week, "measured_continuous_week")
        + extract(args.day, "typical_day")
    )
    comparison["representative_group_share"] = group_share
    comparison["industry_equivalent_multiplier"] = multiplier
    comparison["industry_equivalent_difference"] = comparison["difference"] * multiplier
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output, index=False, encoding="utf-8-sig")

    indexed = comparison.set_index(["horizon", "comparison", "metric"]).sort_index()
    week = indexed.loc[("measured_continuous_week", "IG_multisite_minus_IG_1host")]
    day = indexed.loc[("typical_day", "IG_multisite_minus_IG_1host")]
    typical_day_max_difference = float(day["difference"].abs().max())

    demand_host = week.loc[
        "annual_incremental_maximum_demand_cost_rmb", "IG_1host_value"
    ]
    demand_multisite = week.loc[
        "annual_incremental_maximum_demand_cost_rmb", "IG_multisite_value"
    ]
    demand_saving = -week.loc[
        "annual_incremental_maximum_demand_cost_rmb", "difference"
    ]
    demand_fraction = week.loc[
        "annual_incremental_maximum_demand_cost_rmb", "saving_fraction"
    ]
    grid_host = week.loc["sum_incremental_grid_peak_mw", "IG_1host_value"]
    grid_multisite = week.loc["sum_incremental_grid_peak_mw", "IG_multisite_value"]
    grid_saving = -week.loc["sum_incremental_grid_peak_mw", "difference"]
    grid_fraction = week.loc["sum_incremental_grid_peak_mw", "saving_fraction"]
    energy_saving = -week.loc["annual_ai_energy_cost_rmb", "difference"]
    energy_fraction = week.loc["annual_ai_energy_cost_rmb", "saving_fraction"]
    total_saving = -week.loc["annual_incremental_total_cost_rmb", "difference"]
    total_fraction = week.loc["annual_incremental_total_cost_rmb", "saving_fraction"]

    findings = f"""# C33跨节点灵活性与时域测试发现

## 已运行边界

本结果是C33单行业、代表集团份额{group_share:.0%}、6个成员工厂节点的机制测试。IG-1host与IG-multisite采用相同的连续等效服务器定容、AI服务量、CPU/GPU路由、任务截止窗口和价格；两者之差用于识别跨节点调度价值。连续周保留6条EWELD工厂曲线的168小时变化；24小时测试则把相同曲线分别按小时对七天取平均。

## 连续周的核心发现

C33中，IG-multisite相对IG-1host显著降低了最大需量费用和AI新增电网容量：

- 最大需量费用由{demand_host:,.0f}元/年降至{demand_multisite:,.0f}元/年，节约{demand_saving:,.0f}元/年，即{demand_fraction:.1%}；
- AI新增电网容量由{grid_host:.4f} MW降至{grid_multisite:.4f} MW，节约{grid_saving:.4f} MW（{grid_saving * 1000:.1f} kW），即{grid_fraction:.1%}；
- AI电量费用只减少{energy_saving:,.0f}元/年（{energy_fraction:.3%}），AI设施年用电量也基本不变。

因此，这个单行业案例清楚显示：跨节点灵活性的主要价值不是减少AI计算或年用电，而是利用成员工厂负荷的非同时性，将AI任务调度到边际峰值较低的节点，从而降低新增最大需量和接入容量。

约90%的降幅只适用于最大需量费和新增接入容量，不能解释为企业增量总成本下降90%。由于IG-multisite的服务器成本略高，其企业增量总成本相对IG-1host净减少{total_saving:,.0f}元/年，仅{total_fraction:.2%}；因此当前结果支持明确的电网侧容量价值，但只支持较小的企业总成本优势。

## 行业等效量

C33代表集团份额为{group_share:.0%}。若严格按照当前模型的线性恢复规则乘以{multiplier:.0f}，则对应的行业等效量为：

- 最大需量费用节约约{demand_saving * multiplier / 1e4:,.0f}万元/年；
- 新增电网容量节约约{grid_saving * multiplier:.2f} MW。

行业等效量是将同一代表集团结构复制到整个行业的模型恢复量，不是对现实C33行业集团数量、负荷相关性或可迁移比例的独立预测。

## 典型日对照

24小时典型日中，IG-multisite与IG-1host的上述差额最大绝对值为{typical_day_max_difference:.6g}；IG-1host的zero-load与actual-load配对差额也在比较表中保留。与此同时，典型日IG-1host的新增接入容量为0.810 MW，而连续周为0.171 MW。典型日平均压平了跨厂、跨日非同时性，因此不能用于估计本案例的跨节点灵活性价值；168小时连续周应保留为核心时域。

## 证据边界

这是单行业、合成集团的机制结果，不能直接写成31行业共同下降约90%。6条EWELD曲线来自匿名同行业用户或不同完整周，并非一个真实集团同一日历周的同步观测；行业等效量还假定代表集团的节约可按1%份额线性复制。后续全国5代表节点运行用于检验方向的行业覆盖和异质性，真实集团同步负荷、跨厂通信和任务迁移约束仍需独立验证。
"""
    args.findings.write_text(findings, encoding="utf-8")
    args.done.write_text(
        json.dumps({
            "status": "validated",
            "industry": "C33",
            "week_nodes": 6,
            "day_nodes": 6,
            "representative_group_share": group_share,
            "industry_equivalent_multiplier": multiplier,
            "core_horizon_recommendation": "168_hour_measured_continuous_week",
            "claim_scope": "single_industry_mechanism_test",
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
