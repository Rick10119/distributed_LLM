#!/usr/bin/env python3
"""Measure how optimized AI profiles align with each industry's base-load curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIOS = ["IF", "IG", "II_1host"]


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def alignment_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    base = frame["base_load_mw"].to_numpy(dtype=float)
    compute = frame["ai_compute_accelerator_h"].to_numpy(dtype=float)
    facility = frame["ai_facility_power_mw"].to_numpy(dtype=float)
    combined = base + facility
    base_peak = float(np.max(base))
    ai_peak = float(np.max(facility))
    combined_peak = float(np.max(combined))
    noncoincident_credit = base_peak + ai_peak - combined_peak
    base_peak_mask = np.isclose(base, base_peak, rtol=0, atol=1e-9)
    ai_peak_index = int(np.argmax(facility))
    compute_total = float(np.sum(compute))
    if ai_peak <= 0 or compute_total <= 0:
        raise ValueError("AI power and compute must be positive")
    return {
        "compute_base_correlation": safe_correlation(base, compute),
        "facility_power_base_correlation": safe_correlation(base, facility),
        "offpeak_compute_share": float(np.sum(compute[base <= np.median(base)]))
        / compute_total,
        "compute_at_base_peak_fraction_of_compute_peak": float(
            np.mean(compute[base_peak_mask]) / np.max(compute)
        ),
        "ai_power_at_base_peak_fraction_of_ai_peak": float(
            np.mean(facility[base_peak_mask]) / ai_peak
        ),
        "base_load_at_ai_peak_fraction_of_base_peak": float(base[ai_peak_index])
        / base_peak,
        "base_peak_mw": base_peak,
        "ai_facility_peak_mw": ai_peak,
        "combined_peak_mw": combined_peak,
        "ai_caused_peak_increment_mw": combined_peak - base_peak,
        "noncoincident_peak_credit_mw": noncoincident_credit,
        "noncoincident_peak_credit_fraction_of_ai_peak": noncoincident_credit / ai_peak,
        "facility_load_factor": float(np.mean(facility)) / ai_peak,
        "facility_power_coefficient_of_variation": float(np.std(facility))
        / float(np.mean(facility)),
        "base_peak_hour_of_day": int(frame.iloc[int(np.argmax(base))]["hour_of_day"]),
        "ai_peak_hour_of_day": int(frame.iloc[ai_peak_index]["hour_of_day"]),
        "combined_peak_hour_of_day": int(
            frame.iloc[int(np.argmax(combined))]["hour_of_day"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--hourly-inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--findings-output", type=Path, required=True)
    parser.add_argument("--done-output", type=Path, required=True)
    args = parser.parse_args()

    summaries = pd.read_csv(args.summaries, encoding="utf-8-sig")
    if len(summaries) != 93 or set(summaries["scenario"]) != set(SCENARIOS):
        raise ValueError("Load-alignment analysis requires 31 industries and three scenarios")
    if set(summaries["model_version"]) != {args.model_version}:
        raise ValueError("Summary model version mismatch")
    summary_lookup = summaries.set_index(["industry_code", "scenario"])

    rows: list[dict[str, object]] = []
    for path in args.hourly_inputs:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if len(frame) != 168 or frame["hour"].nunique() != 168:
            raise ValueError(f"Expected a continuous 168-hour result: {path}")
        industry = str(frame["industry_code"].iloc[0])
        scenario = str(frame["scenario"].iloc[0])
        if (industry, scenario) not in summary_lookup.index:
            raise ValueError(f"Unexpected industry-scenario result: {industry} {scenario}")
        summary = summary_lookup.loc[(industry, scenario)]
        metrics = alignment_metrics(frame)
        multiplier = float(summary["equivalent_host_multiplier"])
        rows.append(
            {
                "model_version": args.model_version,
                "industry_code": industry,
                "industry_name": str(summary["industry_name"]),
                "scenario": scenario,
                "curve_scope": "representative_host_168h_repeated_daily_profile",
                **metrics,
                "equivalent_host_multiplier": multiplier,
                "industry_equivalent_noncoincident_peak_credit_mw": float(
                    metrics["noncoincident_peak_credit_mw"]
                )
                * multiplier,
                "industry_equivalent_incremental_grid_expansion_mw": float(
                    summary["industry_equivalent_incremental_grid_expansion_mw"]
                ),
            }
        )

    detail = pd.DataFrame(rows).sort_values(["industry_code", "scenario"])
    if len(detail) != 93 or detail.groupby("scenario")["industry_code"].nunique().min() != 31:
        raise ValueError("Load-alignment output coverage is incomplete")
    if float(detail["noncoincident_peak_credit_mw"].min()) < -1e-8:
        raise ValueError("Noncoincident peak credit cannot be materially negative")
    if not detail["offpeak_compute_share"].between(0.0, 1.0).all():
        raise ValueError("Off-peak compute shares must be bounded by zero and one")
    if not detail["noncoincident_peak_credit_fraction_of_ai_peak"].between(
        -1e-8, 1.0 + 1e-8
    ).all():
        raise ValueError("Peak-credit fractions must be bounded by zero and one")

    architecture = (
        detail.groupby("scenario", as_index=False)
        .agg(
            industries=("industry_code", "nunique"),
            median_compute_base_correlation=("compute_base_correlation", "median"),
            median_facility_power_base_correlation=(
                "facility_power_base_correlation",
                "median",
            ),
            median_offpeak_compute_share=("offpeak_compute_share", "median"),
            median_ai_power_at_base_peak_fraction=(
                "ai_power_at_base_peak_fraction_of_ai_peak",
                "median",
            ),
            median_noncoincident_peak_credit_fraction=(
                "noncoincident_peak_credit_fraction_of_ai_peak",
                "median",
            ),
            mean_noncoincident_peak_credit_fraction=(
                "noncoincident_peak_credit_fraction_of_ai_peak",
                "mean",
            ),
            industries_with_peak_credit_above_1pct=(
                "noncoincident_peak_credit_fraction_of_ai_peak",
                lambda values: int((values > 0.01).sum()),
            ),
            industries_with_peak_credit_above_10pct=(
                "noncoincident_peak_credit_fraction_of_ai_peak",
                lambda values: int((values > 0.10).sum()),
            ),
            sum_of_industry_bucket_equivalent_peak_credit_mw=(
                "industry_equivalent_noncoincident_peak_credit_mw",
                "sum",
            ),
            median_facility_load_factor=("facility_load_factor", "median"),
        )
        .set_index("scenario")
        .loc[SCENARIOS]
        .reset_index()
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output, index=False, encoding="utf-8-sig")
    architecture.to_csv(args.summary_output, index=False, encoding="utf-8-sig")

    rendered = architecture.set_index("scenario")
    top_if = detail[detail["scenario"] == "IF"].nlargest(
        5, "noncoincident_peak_credit_fraction_of_ai_peak"
    )
    top_lines = "；".join(
        f"{row.industry_code}{row.industry_name} {row.noncoincident_peak_credit_fraction_of_ai_peak:.1%}"
        for row in top_if.itertuples()
    )
    findings = f"""# 31行业AI功率与原负荷曲线配合诊断

## 指标

- `compute_base_correlation`衡量优化后任务执行量与行业原负荷的线性相关性；负值表示任务执行总体偏向原负荷较低时段。
- `offpeak_compute_share`是原负荷不高于其中位数时执行的计算份额；约50%表示没有明显集中到低谷。
- `noncoincident_peak_credit_fraction_of_ai_peak = 1-[max(L+P_AI)-max(L)]/max(P_AI)`，衡量AI峰值因与原负荷错峰而未进入合成峰值的比例。
- 所有指标基于代表点连续168小时结果。逐日曲线在代表周内重复；行业等价峰值信用之和不是全国同时峰值，不能直接解释为已避免的配网工程。

## 架构汇总

| 架构 | 任务—原负荷相关系数中位数 | 低于原负荷中位数时的计算份额 | 峰值非同时信用中位数 | 信用>10%的行业数 | 行业桶等价信用之和 |
| --- | ---: | ---: | ---: | ---: | ---: |
| IF | {rendered.loc['IF','median_compute_base_correlation']:.3f} | {rendered.loc['IF','median_offpeak_compute_share']:.1%} | {rendered.loc['IF','median_noncoincident_peak_credit_fraction']:.1%} | {int(rendered.loc['IF','industries_with_peak_credit_above_10pct'])}/31 | {rendered.loc['IF','sum_of_industry_bucket_equivalent_peak_credit_mw']:.1f} MW |
| IG | {rendered.loc['IG','median_compute_base_correlation']:.3f} | {rendered.loc['IG','median_offpeak_compute_share']:.1%} | {rendered.loc['IG','median_noncoincident_peak_credit_fraction']:.1%} | {int(rendered.loc['IG','industries_with_peak_credit_above_10pct'])}/31 | {rendered.loc['IG','sum_of_industry_bucket_equivalent_peak_credit_mw']:.1f} MW |
| II_1host | {rendered.loc['II_1host','median_compute_base_correlation']:.3f} | {rendered.loc['II_1host','median_offpeak_compute_share']:.1%} | {rendered.loc['II_1host','median_noncoincident_peak_credit_fraction']:.2%} | {int(rendered.loc['II_1host','industries_with_peak_credit_above_10pct'])}/31 | {rendered.loc['II_1host','sum_of_industry_bucket_equivalent_peak_credit_mw']:.1f} MW |

IF中相对峰值配合较好的五个行业为：{top_lines}。

## 初步解释

三种架构的任务执行与原负荷相关系数中位数均为负，但低谷执行份额仍约为50%，说明当前调度只形成温和的反相关关系。IF的峰值非同时信用明显高于IG和II；共享规模扩大后，服务器设施功率几乎变成平坦基荷，因此任务时序的变化很少继续转化为设施峰值变化。换言之，当前模型中的主要池化收益来自服务器数量和固定成本，而不是主动跟随各行业生产低谷。

这一诊断还不能归因柔性、光伏或储能各自的边际贡献；需要增加无柔性、无光伏、无储能和自然到达即执行的消融情景。负荷曲线来源中既有EWELD行业对应曲线，也有原型回退曲线，因此跨行业排名只能作为筛查线索。
"""
    args.findings_output.write_text(findings, encoding="utf-8")
    payload = {
        "status": "validated",
        "model_version": args.model_version,
        "industries": 31,
        "scenarios": SCENARIOS,
        "rows": len(detail),
        "checks": [
            "93 industry-scenario hourly profiles",
            "continuous 168-hour coverage",
            "nonnegative noncoincident peak credit",
            "bounded offpeak shares and peak-credit fractions",
            "representative-host metrics linked to industry scaling",
        ],
        "interpretation_limit": "sum_of_industry_bucket_credits_is_not_national_coincident_peak_or_physical_grid_avoidance",
    }
    args.done_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
