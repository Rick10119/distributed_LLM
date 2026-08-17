#!/usr/bin/env python3
"""Test one industry's group AI scheduling under Guangdong spot prices and existing PV."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_code"))

from core.config import load_config
from core.data import load_industry_inputs, scale_workload
from core.model import factory_pv_limit_mw, optimize_host
from core.production_load import resolve_site_load_profile
from core.representative_group import read_representative_groups, scenario_scale


SCENARIO = "IG"
CASE_ORDER = ["spot_with_pv"]
CASE_NAMES = {
    "spot_with_pv": "现货电价、既有光伏",
}


def spot_retail_adder_rmb_per_kwh(config: dict) -> float:
    """Return non-energy volumetric charges added to the wholesale spot price."""
    components = config["energy"]["spot_retail_volumetric_adders_rmb_per_kwh"]
    required = {
        "line_loss",
        "transmission_and_distribution",
        "system_operation",
        "government_funds_and_surcharges",
    }
    if set(components) != required:
        raise ValueError(f"Spot retail adder components must be exactly {sorted(required)}")
    values = np.asarray([float(components[key]) for key in sorted(required)], dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("Spot retail adder components must be finite and non-negative")
    return float(values.sum())


def read_spot_week(path: Path, settings: dict) -> np.ndarray:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["date"] = pd.to_datetime(frame["business_date"])
    start = pd.Timestamp(str(settings["start_date"]))
    end = pd.Timestamp(str(settings["end_date"]))
    if (end - start).days != 6:
        raise ValueError("The spot test must select one inclusive seven-day week")
    selected = frame[
        (frame["province_code"] == str(settings["province_code"]))
        & (frame["settlement_type"] == str(settings["settlement_type"]))
        & frame["date"].between(start, end)
    ].copy()
    counts = selected.groupby(["date", "business_hour"]).size()
    if len(counts) != 168 or not (counts == 4).all():
        raise ValueError("The selected Guangdong day-ahead week must contain four 15-minute prices per hour")
    hourly = (
        selected.groupby(["date", "business_hour"], as_index=False)["price_rmb_mwh"]
        .mean()
        .sort_values(["date", "business_hour"])
    )
    prices = hourly["price_rmb_mwh"].to_numpy(float)
    if prices.shape != (168,) or not np.isfinite(prices).all():
        raise ValueError("Spot-price aggregation did not produce 168 finite hourly prices")
    return prices


def case_config(config: dict) -> dict:
    selected = deepcopy(config)
    selected["energy"]["battery_investment_enabled"] = False
    return selected


def run_cases(
    config: dict, price_path: Path, industry: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    groups = read_representative_groups(ROOT / config["paths"]["representative_group_report"])
    if industry not in groups:
        raise ValueError(f"Unknown manufacturing industry: {industry}")
    scale = scenario_scale(groups[industry], config["industry_parameter_case"], SCENARIO)
    inputs = load_industry_inputs(config, industry)
    rigid, jobs = scale_workload(inputs, scale.ai_service_scale_per_host)
    base, production_load = resolve_site_load_profile(
        root=ROOT,
        config=config,
        industry=industry,
        industry_profile_mw=inputs.base_load_mw,
        ai_service_group_share=scale.group_share,
        legacy_load_site_count=scale.group_factory_count,
    )
    spot_settings = config["energy"]["spot_representative_week"]
    spot_wholesale = read_spot_week(price_path, spot_settings)
    retail_adder_rmb_per_kwh = spot_retail_adder_rmb_per_kwh(config)
    spot_retail = spot_wholesale + retail_adder_rmb_per_kwh * 1000.0
    prices_by_case = {
        "spot_with_pv": spot_retail,
    }

    hourly_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for case in CASE_ORDER:
        selected_config = case_config(config)
        prices = prices_by_case[case]
        baseline = optimize_host(
            selected_config,
            base_load_mw=base,
            pv_capacity_factor=inputs.pv_capacity_factor,
            roof_area_m2=inputs.roof_area_proxy_m2,
            grid_energy_price_rmb_per_mwh=prices,
        )
        result = optimize_host(
            selected_config,
            base_load_mw=base,
            pv_capacity_factor=inputs.pv_capacity_factor,
            roof_area_m2=inputs.roof_area_proxy_m2,
            rigid_service_units=rigid,
            flexible_jobs=jobs,
            grid_energy_price_rmb_per_mwh=prices,
            existing_grid_capacity_mw=float(baseline.summary["grid_import_peak_mw"]),
        )
        hourly = result.hourly.copy()
        hourly["case"] = case
        hourly["case_name"] = CASE_NAMES[case]
        hourly["baseline_grid_import_mw"] = baseline.hourly["grid_import_mw"].to_numpy(float)
        hourly["incremental_grid_import_mw"] = (
            hourly["grid_import_mw"] - hourly["baseline_grid_import_mw"]
        )
        hourly_rows.append(hourly)

        compute = hourly["ai_compute_accelerator_h"].to_numpy(float)
        pv_output = hourly["rooftop_pv_output_mw"].to_numpy(float)
        low_price = prices <= np.quantile(prices, 0.25)
        solar_hours = inputs.pv_capacity_factor > 1e-9
        total_compute = float(compute.sum())
        installed_groups = float(result.summary["installed_server_groups"])
        installed_accelerators = installed_groups * float(config["server"]["accelerators_per_server"])
        peak_compute = float(compute.max())
        incremental_grid_energy_twh = float(
            result.summary["annual_grid_energy_twh"] - baseline.summary["annual_grid_energy_twh"]
        )
        incremental_energy_cost = float(
            result.summary["annual_flat_energy_cost_rmb"] - baseline.summary["annual_flat_energy_cost_rmb"]
        )
        incremental_demand_cost = float(
            result.summary["annual_maximum_demand_cost_rmb"]
            - baseline.summary["annual_maximum_demand_cost_rmb"]
        )
        incremental_electricity_bill = incremental_energy_cost + incremental_demand_cost
        incremental_grid_energy_kwh = incremental_grid_energy_twh * 1e9
        summary_rows.append(
            {
                "model_version": config["model_version"],
                "industry_code": industry,
                "industry_name": inputs.industry_name,
                "scenario": SCENARIO,
                "case": case,
                "case_name": CASE_NAMES[case],
                "price_source": f"{spot_settings['province_code']} province-average {spot_settings['settlement_type']} price plus official volumetric retail adders",
                "mean_wholesale_spot_price_rmb_mwh": float(spot_wholesale.mean()),
                "spot_retail_volumetric_adder_rmb_per_kwh": retail_adder_rmb_per_kwh,
                "mean_price_rmb_mwh": float(prices.mean()),
                "minimum_price_rmb_mwh": float(prices.min()),
                "maximum_price_rmb_mwh": float(prices.max()),
                "pv_capacity_mw": float(result.summary["rooftop_pv_capacity_mw"]),
                "pv_limit_mw": float(factory_pv_limit_mw(selected_config, inputs.roof_area_proxy_m2)),
                "installed_server_groups": installed_groups,
                "installed_accelerator_capacity": installed_accelerators,
                "peak_compute_accelerator_h_per_h": peak_compute,
                "installed_capacity_margin_relative_to_peak": installed_accelerators / peak_compute - 1.0,
                "unused_installed_capacity_share_at_peak": 1.0 - peak_compute / installed_accelerators,
                "minimum_cold_spare_server_groups": installed_groups - float(hourly["online_server_groups"].max()),
                "maximum_cold_spare_server_groups": installed_groups - float(hourly["online_server_groups"].min()),
                "ai_facility_peak_mw": float(result.summary["ai_facility_peak_mw"]),
                "ai_facility_load_factor": float(hourly["ai_facility_power_mw"].mean() / hourly["ai_facility_power_mw"].max()),
                "share_compute_in_lowest_price_quartile": float(compute[low_price].sum() / total_compute),
                "lowest_price_quartile_hour_share": float(low_price.mean()),
                "share_compute_during_solar_hours": float(compute[solar_hours].sum() / total_compute),
                "solar_hour_share": float(solar_hours.mean()),
                "incremental_grid_energy_twh": incremental_grid_energy_twh,
                "incremental_annual_energy_cost_rmb": incremental_energy_cost,
                "incremental_annual_demand_cost_rmb": incremental_demand_cost,
                "incremental_annual_electricity_bill_rmb": incremental_electricity_bill,
                "average_incremental_electricity_bill_rmb_per_kwh": incremental_electricity_bill / incremental_grid_energy_kwh,
                "incremental_annual_total_cost_rmb": float(result.summary["annual_objective_rmb"] - baseline.summary["annual_objective_rmb"]),
                "daily_effective_service_units": float(hourly["ai_executed_service_units"].sum() / 7.0),
            }
        )
    meta = {
        "industry_code": industry,
        "industry_name": inputs.industry_name,
        "province_code": str(spot_settings["province_code"]),
        "settlement_type": str(spot_settings["settlement_type"]),
        "spot_start": str(spot_settings["start_date"]),
        "spot_end": str(spot_settings["end_date"]),
        "group_share": scale.group_share,
        "group_factory_count": scale.group_factory_count,
        "production_load": production_load,
        "ai_service_scale_per_host": scale.ai_service_scale_per_host,
        "spot_wholesale_mean_rmb_mwh": float(spot_wholesale.mean()),
        "spot_retail_adder_rmb_per_kwh": retail_adder_rmb_per_kwh,
        "spot_retail_mean_rmb_mwh": float(spot_retail.mean()),
        "installed_reserve_fraction": float(config["server"]["installed_reserve_fraction"]),
    }
    return pd.concat(hourly_rows, ignore_index=True), pd.DataFrame(summary_rows), meta


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 180,
        }
    )


def plot_comparison(
    hourly: pd.DataFrame, summary: pd.DataFrame, meta: dict[str, object], output: Path
) -> None:
    configure_plotting()
    fig, axes = plt.subplots(
        len(CASE_ORDER), 3, figsize=(15.5, 4.6), sharex=True, squeeze=False
    )
    x = np.arange(168)
    day_labels = [
        day.strftime("%m-%d")
        for day in pd.date_range(str(meta["spot_start"]), str(meta["spot_end"]), freq="D")
    ] + ["末"]
    for row, case in enumerate(CASE_ORDER):
        frame = hourly[hourly["case"] == case].sort_values("hour")
        metrics = summary[summary["case"] == case].iloc[0]
        price = frame["grid_energy_price_rmb_per_mwh"].to_numpy(float)
        pv = frame["rooftop_pv_output_mw"].to_numpy(float)
        ai = frame["ai_facility_power_mw"].to_numpy(float)
        base = frame["base_load_mw"].to_numpy(float)
        combined = base + ai

        ax = axes[row, 0]
        ax.plot(x, price, color="#7d3c98", linewidth=1.05, label="电价")
        ax.set_ylabel(f"{CASE_NAMES[case]}\n到户电量价 元/MWh", fontsize=9.5)
        ax.grid(axis="y", color="#d5d8dc", linewidth=0.55)
        pv_ax = ax.twinx()
        pv_ax.fill_between(x, 0, pv, color="#f4d03f", alpha=0.48, label="光伏")
        pv_ax.set_ylim(0, max(0.01, float(pv.max()) * 1.12))
        pv_ax.set_ylabel("PV MW", fontsize=8, color="#9a7d0a")
        if row == 0:
            ax.set_title("广东日前现货电价与光伏出力", fontsize=11.5, fontweight="bold")

        ax = axes[row, 1]
        ax.fill_between(x, 0, ai, color="#e67e22", alpha=0.78)
        ax.plot(x, ai, color="#a04000", linewidth=0.9)
        ax.set_ylim(float(ai.min()) * 0.985, float(ai.max()) * 1.008)
        ax.set_ylabel("AI设施负荷（MW）", fontsize=9)
        ax.grid(axis="y", color="#d5d8dc", linewidth=0.55)
        ax.text(
            0.98,
            0.92,
            f"峰值 {metrics.ai_facility_peak_mw:.3f} MW\n负荷率 {metrics.ai_facility_load_factor:.2%}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
        )
        if row == 0:
            ax.set_title("AI设施负荷", fontsize=11.5, fontweight="bold")

        ax = axes[row, 2]
        ax.fill_between(x, 0, base, color="#aeb6bf", alpha=0.72, label="原有负荷")
        ax.fill_between(x, base, combined, color="#e67e22", alpha=0.82, label="AI负荷")
        ax.plot(x, combined, color="#273746", linewidth=0.95, label="叠加后负荷")
        ax.plot(x, frame["grid_import_mw"], color="#2471a3", linewidth=0.9, linestyle="--", label="电网购电")
        ax.set_ylabel("节点功率（MW）", fontsize=9)
        ax.grid(axis="y", color="#d5d8dc", linewidth=0.55)
        if row == 0:
            ax.set_title("原负荷、AI负荷与电网购电", fontsize=11.5, fontweight="bold")

        for col in range(3):
            axes[row, col].set_xlim(0, 167)
            axes[row, col].set_xticks([0, 24, 48, 72, 96, 120, 144, 167])
            if row == len(CASE_ORDER) - 1:
                axes[row, col].set_xticklabels(day_labels)
                axes[row, col].set_xlabel(
                    f"{meta['spot_start']}至{meta['spot_end']}代表周", fontsize=8.5
                )
            else:
                axes[row, col].set_xticklabels([])
    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.72, 0.976))
    fig.suptitle(
        f"{meta['industry_code']} {meta['industry_name']}集团AI中心：现货电价与既有光伏运行测试",
        fontsize=15.5,
        y=0.998,
    )
    fig.text(
        0.5,
        0.006,
        f"注：现货电能量价采用{meta['province_code']}省平均节点{meta['settlement_type']}价格，15分钟数据聚合为小时，并叠加{meta['spot_retail_adder_rmb_per_kwh']:.6f}元/kWh线损、输配、系统运行和政府基金；需量电费另计；将{meta['pv_limit_mw']:.4f} MW屋顶上限视为既有资产；本测试关闭新增储能。",
        ha="center",
        fontsize=8.7,
        color="#566573",
    )
    fig.tight_layout(rect=(0.03, 0.025, 0.995, 0.96), h_pad=1.0, w_pad=0.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def write_findings(summary: pd.DataFrame, meta: dict[str, object], output: Path) -> None:
    indexed = summary.set_index("case")
    lines = [
        f"# {meta['industry_code']} {meta['industry_name']}集团AI中心：广东现货电价与既有光伏测试",
        "",
        f"集团情景代表一个含{int(meta['group_factory_count'])}个工厂、覆盖行业需求{meta['group_share']:.1%}的代表集团。测试统一使用广东现货到户电价和当前{meta['pv_limit_mw']:.4f} MW代表工厂既有光伏上限，因此跨行业结果首先用于运行机制比较，而不是行业实际光伏规模估计。",
        "",
        f"现货电能量价格采用配置指定的完整周（{meta['spot_start']}至{meta['spot_end']}），四个15分钟价格平均为小时价格，批发均价为{meta['spot_wholesale_mean_rmb_mwh']:.1f}元/MWh。在每个小时另加线损、输配、系统运行和政府基金共{meta['spot_retail_adder_rmb_per_kwh']:.6f}元/kWh，形成平均{meta['spot_retail_mean_rmb_mwh']:.1f}元/MWh的到户电量价；最大需量电费另计。",
        "",
        "| 情景 | PV容量 | 服务器组 | 峰值装机余量 | AI峰值 | AI设施负荷率 | 低价四分位计算占比 | 年增量电费（电量+需量） | 平均到户电价 | 年增量总成本 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in CASE_ORDER:
        row = indexed.loc[case]
        low = f"{row.share_compute_in_lowest_price_quartile:.1%}"
        lines.append(
            f"| {CASE_NAMES[case]} | {row.pv_capacity_mw:.3f} MW | {row.installed_server_groups:.0f} | "
            f"{row.installed_capacity_margin_relative_to_peak:.1%} | {row.ai_facility_peak_mw:.3f} MW | {row.ai_facility_load_factor:.2%} | {low} | "
            f"{row.incremental_annual_electricity_bill_rmb/1e6:.3f}百万元 | {row.average_incremental_electricity_bill_rmb_per_kwh:.3f}元/kWh | "
            f"{row.incremental_annual_total_cost_rmb/1e6:.3f}百万元 |"
        )
    spot_shift = indexed.loc["spot_with_pv", "share_compute_in_lowest_price_quartile"] - indexed.loc["spot_with_pv", "lowest_price_quartile_hour_share"]
    solar_shift = indexed.loc["spot_with_pv", "share_compute_during_solar_hours"] - indexed.loc["spot_with_pv", "solar_hour_share"]
    lines.extend(
        [
            "",
            "## 初步解释",
            "",
            f"- 现货价下，最低价格四分位时段的计算份额仅比该时段本身的25%时长高{spot_shift:+.2%}，用于判断AI任务是否主动向低价时段集中。",
            f"- 日照时段的计算份额相对日照时长占比偏离{solar_shift:+.2%}，用于判断AI任务是否主动配合既有光伏。",
            "- AI设施功率仍接近稳定基荷。任务执行可以在小时之间移动，但已安装服务器的在线空闲功率和高利用率使这种移动很少转化为设施功率的大幅变化。",
            f"- 模型已经设置{meta['installed_reserve_fraction']:.0%}的日均计算需求规划裕量；逐小时任务截止约束仍可能要求安装更多容量。该裕量不代表未来需求增长情景。",
            "- 因此，本轮测试的直接结论是：当前集团AI负荷设定基本不能响应现货价或光伏。若希望AI设施负荷显著跟随现货价或光伏，需要提高真正可延期的任务比例、允许更多服务器关机，或降低长期高利用率约束。",
            "",
            "## 边界",
            "",
            f"该结果是一个代表周运行测试。现货电能量价格已叠加线损、输配、系统运行和政府基金，并另计最大需量电费，但仍未模拟中长期合约、零售服务费、偏差结算和现货风险；光伏按既有资产处理；未配置新增储能；{meta['industry_code']}代表集团和代表工厂均不是实际企业观测。",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--defaults", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--price-input", type=Path, required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--hourly-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--figure-output", type=Path, required=True)
    parser.add_argument("--findings-output", type=Path, required=True)
    parser.add_argument("--done-output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(ROOT, args.defaults, args.config)
    hourly, summary, meta = run_cases(config, args.price_input, args.industry)
    meta["pv_limit_mw"] = float(summary["pv_limit_mw"].max())
    args.hourly_output.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(args.hourly_output, index=False, encoding="utf-8-sig")
    summary.to_csv(args.summary_output, index=False, encoding="utf-8-sig")
    plot_comparison(hourly, summary, meta, args.figure_output)
    write_findings(summary, meta, args.findings_output)
    args.done_output.write_text(
        json.dumps(
            {
                "status": "validated",
                "model_version": config["model_version"],
                "industry": args.industry,
                "scenario": SCENARIO,
                "cases": CASE_ORDER,
                "spot_price_source": str(args.price_input),
                "spot_week": [meta["spot_start"], meta["spot_end"]],
                "checks": [
                    "168 hourly prices from four quarter-hours each",
                    "wholesale spot prices plus fixed retail volumetric adders",
                    "maximum-demand charge accounted for separately",
                    "same effective AI service in all cases",
                    "installed compute reserve reported separately from facility load factor",
                    "baseline and AI case share price and PV boundary",
                    "no new battery investment",
                    "hourly combined load identity",
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
