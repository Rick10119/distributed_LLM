#!/usr/bin/env python3
"""Combine per-industry Guangdong spot-price/PV tests into comparison tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_code"))

from core.config import load_config


CASES = {"spot_with_pv"}


def build_comparison(all_cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for industry, frame in all_cases.groupby("industry_code", sort=True):
        indexed = frame.set_index("case")
        missing = CASES - set(indexed.index)
        if missing or len(indexed) != len(CASES):
            raise ValueError(
                f"{industry} must contain exactly one configured case; missing={sorted(missing)}"
            )
        spot = indexed.loc["spot_with_pv"]
        rows.append(
            {
                "model_version": spot["model_version"],
                "industry_code": industry,
                "industry_name": spot["industry_name"],
                "scenario": spot["scenario"],
                "installed_server_groups": spot["installed_server_groups"],
                "ai_facility_peak_mw": spot["ai_facility_peak_mw"],
                "ai_facility_load_factor": spot["ai_facility_load_factor"],
                "installed_capacity_margin_relative_to_peak": spot[
                    "installed_capacity_margin_relative_to_peak"
                ],
                "spot_average_incremental_electricity_bill_rmb_per_kwh": spot[
                    "average_incremental_electricity_bill_rmb_per_kwh"
                ],
                "spot_low_price_quartile_compute_share": spot[
                    "share_compute_in_lowest_price_quartile"
                ],
                "low_price_quartile_hour_share": spot["lowest_price_quartile_hour_share"],
                "low_price_compute_concentration_above_time_share": spot[
                    "share_compute_in_lowest_price_quartile"
                ]
                - spot["lowest_price_quartile_hour_share"],
                "pv_capacity_mw": spot["pv_capacity_mw"],
                "solar_compute_share": spot["share_compute_during_solar_hours"],
                "solar_hour_share": spot["solar_hour_share"],
                "solar_compute_concentration_above_time_share": spot[
                    "share_compute_during_solar_hours"
                ]
                - spot["solar_hour_share"],
            }
        )
    return pd.DataFrame(rows).sort_values("industry_code").reset_index(drop=True)


def write_findings(comparison: pd.DataFrame, output: Path) -> None:
    low_shift = comparison["low_price_compute_concentration_above_time_share"]
    load_factor = comparison["ai_facility_load_factor"]
    demand_response_count = int((low_shift.abs() >= 0.01).sum())
    lines = [
        "# 31个制造业行业集团AI中心：广东现货电价与既有光伏测试",
        "",
        f"共纳入{len(comparison)}个行业。各行业均采用集团情景、同一广东代表周、相同到户电价口径和既有光伏设置，每个行业只运行一个情景。",
        "",
        "## 汇总指标",
        "",
        f"- 现货电价与既有光伏情景下，AI设施功率负荷率行业中位数为{load_factor.median():.2%}。",
        f"- 最低价格四分位的计算占比相对25%时间占比的偏离中位数为{low_shift.median():+.2%}；绝对偏离达到1个百分点的行业有{demand_response_count}个。",
        f"- 现货情景的AI增量平均到户电价行业中位数为{comparison['spot_average_incremental_electricity_bill_rmb_per_kwh'].median():.3f}元/kWh。",
        "",
        "## 解释边界",
        "",
        "各行业使用同一广东价格周和当前统一代表工厂屋顶上限，适合比较负荷形状与任务柔性，不代表各行业实际购电合同、厂房屋顶规模或光伏存量。详细结论应结合industry_comparison.csv及逐行业图表判断。",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--defaults", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary-inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--cases-output", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path, required=True)
    parser.add_argument("--findings-output", type=Path, required=True)
    parser.add_argument("--done-output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(ROOT, args.defaults, args.config)
    frames = [pd.read_csv(path, encoding="utf-8-sig") for path in args.summary_inputs]
    all_cases = pd.concat(frames, ignore_index=True).sort_values(
        ["industry_code", "case"]
    )
    expected = set(config["selected_industries"])
    actual = set(all_cases["industry_code"])
    if actual != expected:
        raise ValueError(
            f"Industry coverage mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    if all_cases["model_version"].nunique() != 1:
        raise ValueError("All spot-price tests must use one model version")
    comparison = build_comparison(all_cases)
    args.cases_output.parent.mkdir(parents=True, exist_ok=True)
    all_cases.to_csv(args.cases_output, index=False, encoding="utf-8-sig")
    comparison.to_csv(args.comparison_output, index=False, encoding="utf-8-sig")
    write_findings(comparison, args.findings_output)
    args.done_output.write_text(
        json.dumps(
            {
                "status": "validated",
                "model_version": config["model_version"],
                "industries": sorted(expected),
                "industry_count": len(expected),
                "cases_per_industry": sorted(CASES),
                "checks": [
                    "configured industry coverage is complete",
                    "each industry has exactly one spot-price plus existing-PV case",
                    "all cases use one model version",
                    "one-row-per-industry comparison generated",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
