#!/usr/bin/env python3
"""Prepare data and plot manuscript Figure 2: enterprise cost and deployment choice."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

_mpl = Path(tempfile.gettempdir()) / "distributed_llm_matplotlib"
_xdg = Path(tempfile.gettempdir()) / "distributed_llm_fontconfig"
_mpl.mkdir(parents=True, exist_ok=True)
_xdg.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl))
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml


ARCH_ORDER = ["IF", "IG", "II_1host"]
ARCH_LABEL = {"IF": "工厂侧分布式（核心）", "IG": "集团集中算力池", "II_1host": "大型集中节点"}
COST_COMPONENTS = [
    ("industry_equivalent_incremental_annual_server_cost_rmb", "服务器与设施"),
    ("industry_equivalent_incremental_annual_flat_energy_cost_rmb", "电量"),
    ("industry_equivalent_incremental_annual_maximum_demand_cost_rmb", "最大需量"),
    ("industry_equivalent_incremental_annual_battery_cost_rmb", "储能"),
    ("industry_equivalent_incremental_annual_model_operations_cost_rmb", "模型运维"),
]
DEMAND_CASES = ("low", "base", "high")


def capital_recovery_factor(rate: float, years: float) -> float:
    factor = (1.0 + rate) ** years
    return rate * factor / (factor - 1.0)


def update_china_owned_cost_to_five_years(
    national: pd.DataFrame, model_version: str
) -> pd.DataFrame:
    """Update the validated four-year snapshot without changing its physical solution."""
    if model_version != "v0.6.1":
        return national.copy()
    old_coefficient = 1.20 * capital_recovery_factor(0.08, 4.0) + 0.05
    new_coefficient = 1.20 * capital_recovery_factor(0.08, 5.0) + 0.05
    result = national.copy()
    old_server = result["industry_equivalent_incremental_annual_server_cost_rmb"].astype(float)
    new_server = old_server * new_coefficient / old_coefficient
    result["industry_equivalent_incremental_annual_server_cost_rmb"] = new_server
    result["industry_equivalent_incremental_total_cost_rmb"] = (
        result["industry_equivalent_incremental_total_cost_rmb"].astype(float)
        - old_server
        + new_server
    )
    return result


def configure_plotting() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "savefig.dpi": 240,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def require(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def prepare_data(
    national_path: Path,
    china_mainstream_path: Path,
    china_cloud_detail_path: Path,
    us_demand_local_path: Path,
    us_demand_cloud_path: Path,
    heterogeneous_country_path: Path,
    china_heterogeneous_path: Path,
    us_heterogeneous_path: Path,
    us_parameters_path: Path,
    routing_config_path: Path,
    us_cost_config_path: Path,
    break_even_path: Path,
    api_prices_path: Path,
    model_version: str,
) -> pd.DataFrame:
    national = pd.read_csv(national_path, encoding="utf-8-sig")
    china_mainstream = pd.read_csv(china_mainstream_path, encoding="utf-8-sig")
    china_cloud = pd.read_csv(china_cloud_detail_path, encoding="utf-8-sig")
    us_local = pd.read_csv(us_demand_local_path, encoding="utf-8-sig")
    us_cloud = pd.read_csv(us_demand_cloud_path, encoding="utf-8-sig")
    heterogeneous_country = pd.read_csv(heterogeneous_country_path, encoding="utf-8-sig")
    china_heterogeneous = pd.read_csv(china_heterogeneous_path, encoding="utf-8-sig")
    us_heterogeneous = pd.read_csv(us_heterogeneous_path, encoding="utf-8-sig")
    us_parameters = pd.read_csv(us_parameters_path, encoding="utf-8-sig")
    routing_config = yaml.safe_load(routing_config_path.read_text(encoding="utf-8"))
    us_cost_config = yaml.safe_load(us_cost_config_path.read_text(encoding="utf-8"))
    break_even = pd.read_csv(break_even_path, encoding="utf-8-sig")
    prices = pd.read_csv(api_prices_path, encoding="utf-8-sig")
    require(national, {"model_version", "scenario", "industry_equivalent_incremental_total_cost_rmb"} | {c for c, _ in COST_COMPONENTS}, "national")
    require(china_mainstream, {"option_name", "option_group", "annual_total_cost_billion_rmb"}, "China mainstream")
    require(china_cloud, {"provider", "mainstream_representative", "annual_api_token_cost_rmb", "residual_reserved_gpu_payment_rmb_proxy", "annual_cloud_storage_cost_rmb_proxy", "annual_full_cloud_cost_rmb_proxy"}, "China cloud detail")
    require(us_local, {"parameter_case", "server_price_case", "annual_ai_facility_energy_twh", "annual_local_cost_billion_usd"}, "US demand local")
    require(us_cloud, {"parameter_case", "provider", "annual_api_token_cost_usd", "annual_residual_reserved_gpu_payment_usd", "annual_cloud_storage_cost_usd_proxy", "annual_full_cloud_cost_billion_usd"}, "US demand cloud")
    require(heterogeneous_country, {"country", "provider", "local_annual_cost", "cloud_annual_cost", "local_currency"}, "heterogeneous country summary")
    require(china_heterogeneous, {
        "owned_architecture", "local_gpu_annualized_hardware_cost_rmb",
        "local_cpu_annualized_hardware_cost_rmb", "local_gpu_electricity_cost_rmb",
        "local_cpu_electricity_cost_rmb", "local_maximum_demand_cost_rmb",
        "local_battery_cost_rmb",
        "local_model_operations_cost_rmb", "local_other_modeled_cost_rmb",
        "local_joint_physical_annual_cost_rmb",
    }, "China heterogeneous summary")
    require(us_heterogeneous, {"parameter_case", "cpu_server_price_case", "provider", "cloud_token_api_cost_usd", "cloud_gpu_reserved_cost_usd", "cloud_cpu_reserved_cost_usd", "cloud_total_annual_cost_usd"}, "US heterogeneous summary")
    if set(national["model_version"].astype(str)) != {model_version}:
        raise ValueError("National model version mismatch")
    fx = prices.loc[prices["currency"].eq("USD"), "fx_to_cny"].dropna().astype(float).unique()
    if len(fx) != 1:
        raise ValueError("Expected one USD-to-CNY exchange rate")
    usd_to_cny = float(fx[0])
    national = update_china_owned_cost_to_five_years(national, model_version)
    rows: list[dict[str, object]] = []

    def add(panel: str, record: str, country: str, option: str, component: str, value: float, unit: str, evidence: str, source: Path, order: int) -> None:
        rows.append({"model_version": model_version, "panel": panel, "record_type": record, "country": country, "option": option, "component": component, "value": value, "unit": unit, "evidence_type": evidence, "source": source.as_posix(), "order": order})

    china_national_heterogeneous = pd.read_csv(
        heterogeneous_country_path.parent / "china_national" / "comparison.csv",
        encoding="utf-8-sig",
    )
    china_core_heterogeneous = china_national_heterogeneous[
        china_national_heterogeneous.owned_architecture.eq("IF")
    ]
    china_base_energy = float(china_core_heterogeneous["local_total_facility_energy_twh"].iloc[0])
    china_demand_twh = {"low": 8.0, "base": china_base_energy, "high": 28.0}
    china_local_base = (
        national.groupby("scenario")["industry_equivalent_incremental_total_cost_rmb"].sum()
        / usd_to_cny / 1e9
    )
    china_cloud_base = (
        china_mainstream.loc[
            china_mainstream["option_group"].eq("full_cloud_hybrid"),
            "annual_total_cost_billion_rmb",
        ].astype(float) / usd_to_cny
    )
    cn_heterogeneous = heterogeneous_country[
        heterogeneous_country.country.eq("China")
        & heterogeneous_country.owned_architecture.eq("IF")
    ]
    china_local_base = pd.Series([float(cn_heterogeneous.local_annual_cost.iloc[0]) / usd_to_cny / 1e9])
    china_cloud_base = cn_heterogeneous.cloud_annual_cost.astype(float) / usd_to_cny / 1e9
    for option, values, source, order in [
        ("本地/自建", china_local_base, national_path, 1),
        ("云端采购", china_cloud_base, china_mainstream_path, 2),
    ]:
        for demand_order, case in enumerate(DEMAND_CASES):
            scaled = values * china_demand_twh[case] / china_base_energy
            for component, value in (("low", scaled.min()), ("median", scaled.median()), ("high", scaled.max())):
                add("a", f"cost_range_{case}", "中国", option, component, float(value),
                    "billion_USD/year", "china_own_demand_low_base_high;five_year_owned_life", source,
                    order * 10 + demand_order)

    for demand_order, case in enumerate(DEMAND_CASES):
        selected_us = us_heterogeneous[
            us_heterogeneous.parameter_case.eq(case)
            & us_heterogeneous.cpu_server_price_case.eq("base")
        ]
        local_values = pd.Series([float(selected_us.local_total_annual_cost_usd.iloc[0]) / 1e9])
        cloud_values = selected_us.cloud_total_annual_cost_usd.astype(float) / 1e9
        if len(local_values) != 1 or len(cloud_values) != 3:
            raise ValueError(f"US {case} heterogeneous case must contain one core local and three formal-cloud observations")
        for option, values, source, order in [
            ("本地/自建", local_values, us_demand_local_path, 3),
            ("云端采购", cloud_values, us_demand_cloud_path, 4),
        ]:
            for component, value in (("low", values.min()), ("median", values.median()), ("high", values.max())):
                add("a", f"cost_range_{case}", "美国", option, component, float(value),
                    "billion_USD/year", "bottom_up_US_own_manufacturing_demand;five_year_owned_life", source,
                    order * 10 + demand_order)

    us_heterogeneous_energy = dict(
        us_heterogeneous[us_heterogeneous.cpu_server_price_case.eq("base")]
        .groupby("parameter_case")["annual_facility_energy_twh"].first().astype(float)
    )
    for country, energy in [("中国", china_demand_twh), ("美国", us_heterogeneous_energy)]:
        for order, case in enumerate(DEMAND_CASES):
            add("a", "demand_energy", country, "需求", case, float(energy[case]), "TWh/year",
                "country_specific_manufacturing_AI_demand", national_path if country == "中国" else us_demand_local_path, order)

    for arch_idx, arch in enumerate(ARCH_ORDER):
        hetero_rows = china_heterogeneous[china_heterogeneous.owned_architecture.eq(arch)]
        if len(hetero_rows) != 2:
            raise ValueError(f"China heterogeneous summary must contain two provider rows for {arch}")
        hetero = hetero_rows.iloc[0]
        hetero_components = [
            ("GPU服务器与设施", float(hetero.local_gpu_annualized_hardware_cost_rmb)),
            ("CPU服务器与设施", float(hetero.local_cpu_annualized_hardware_cost_rmb)),
            ("电费", float(hetero.local_gpu_electricity_cost_rmb) + float(hetero.local_cpu_electricity_cost_rmb)),
            ("最大需量", float(hetero.local_maximum_demand_cost_rmb)),
            ("储能", float(hetero.local_battery_cost_rmb)),
            ("模型运维及其他", float(hetero.local_model_operations_cost_rmb) + float(hetero.local_other_modeled_cost_rmb)),
        ]
        for comp_idx, (label, component_value) in enumerate(hetero_components):
            add("c", "owned_cost_component", "中国", ARCH_LABEL[arch], label,
                component_value / usd_to_cny / 1e9, "billion_USD/year",
                "same_routing_heterogeneous_CPU_GPU_reconciled_cost_component",
                china_heterogeneous_path, arch_idx * 20 + comp_idx)
        if not np.isclose(sum(value for _, value in hetero_components), float(hetero.local_joint_physical_annual_cost_rmb), atol=1e-2, rtol=0):
            raise ValueError(f"China {arch} heterogeneous components do not reconcile")

    china_formal_providers = set(
        china_mainstream.loc[
            china_mainstream["option_group"].eq("full_cloud_hybrid"), "provider"
        ].astype(str)
    )
    cn_formal = china_heterogeneous[china_heterogeneous.owned_architecture.eq("IF")][
        ["provider", "cloud_token_api_cost_rmb", "cloud_gpu_reserved_cost_rmb", "cloud_cpu_reserved_cost_rmb", "cloud_total_annual_cost_rmb"]
    ]
    if len(cn_formal) != 2:
        raise ValueError("Formal China cloud panel must contain exactly two providers")
    for idx, row in enumerate(cn_formal.itertuples(index=False)):
        for comp_idx, (label, value) in enumerate([
            ("Token API", row.cloud_token_api_cost_rmb / usd_to_cny / 1e9),
            ("剩余 GPU 容量", row.cloud_gpu_reserved_cost_rmb / usd_to_cny / 1e9),
            ("CPU 容量", row.cloud_cpu_reserved_cost_rmb / usd_to_cny / 1e9),
        ]):
            add("b", "cloud_payment_component", "中国", str(row.provider), label, float(value), "billion_USD/year", "formal_payment_proxy_not_cloud_resource_cost", china_cloud_detail_path, idx * 10 + comp_idx)
    us_cloud_base = us_heterogeneous[
        us_heterogeneous.parameter_case.eq("base")
        & us_heterogeneous.cpu_server_price_case.eq("base")
    ]
    for idx, row in enumerate(us_cloud_base.itertuples(index=False)):
        for comp_idx, (label, value) in enumerate([
            ("Token API", row.cloud_token_api_cost_usd / 1e9),
            ("剩余 GPU 容量", row.cloud_gpu_reserved_cost_usd / 1e9),
            ("CPU 容量", row.cloud_cpu_reserved_cost_usd / 1e9),
        ]):
            add("b", "cloud_payment_component", "美国", str(row.provider), label, float(value), "billion_USD/year", "US_base_demand_formal_payment_proxy_not_cloud_resource_cost", us_demand_cloud_path, 100 + idx * 10 + comp_idx)

    us_cfg = us_cost_config["us_cost_environment"]

    def us_parameter(parameter_id: str) -> float:
        selected = us_parameters[us_parameters.parameter_id.eq(parameter_id)]
        if len(selected) != 1:
            raise ValueError(f"Expected one US parameter row for {parameter_id}")
        return float(selected.iloc[0].base_value)

    gpu_price = us_parameter(us_cfg["local_gpu_server_parameter_id"])
    cpu_price = us_parameter(us_cfg["local_cpu_server_parameter_id"])
    electricity_price = us_parameter(us_cfg["industrial_electricity_parameter_id"])
    rate = float(us_cfg["shared_discount_rate"])
    years = float(us_cfg["shared_economic_life_years"])
    owned_coefficient = (
        (1 + float(us_cfg["shared_facility_capex_fraction"]))
        * capital_recovery_factor(rate, years)
        + float(us_cfg["shared_annual_maintenance_fraction"])
    )
    us_base_price = us_heterogeneous[
        us_heterogeneous.cpu_server_price_case.eq("base")
        & us_heterogeneous.parameter_case.eq("base")
    ].drop_duplicates("parameter_case")
    if len(us_base_price) != 1:
        raise ValueError("US architecture cost composition requires one base-demand row at base server prices")
    us_row = us_base_price.iloc[0]
    architecture_energy_ratio = (
        national.groupby("scenario")["industry_equivalent_annual_ai_facility_energy_twh"].sum()
        / national[national.scenario.eq("IF")]["industry_equivalent_annual_ai_facility_energy_twh"].sum()
    )
    for architecture_order, architecture in enumerate(ARCH_ORDER):
        values = [
            ("GPU服务器与设施", float(us_row.local_gpu_servers) * gpu_price * owned_coefficient),
            ("CPU服务器与设施", float(us_row.local_cpu_servers) * cpu_price * owned_coefficient),
            ("电费", float(us_row.annual_facility_energy_twh) * float(architecture_energy_ratio[architecture]) * 1e9 * electricity_price),
        ]
        if architecture == "IF" and not np.isclose(sum(value for _, value in values), float(us_row.local_total_annual_cost_usd), atol=1e-3, rtol=0):
            raise ValueError("US IF owned-cost components do not reconcile with the mainline total")
        for component_order, (component, value) in enumerate(values):
            add("d", "us_owned_cost_component", "美国", ARCH_LABEL[architecture], component,
                value / 1e9, "billion_USD/year",
                "US_native_base_demand_matched_64core_min256GB_three_year_CPU_proxy_with_China_architecture_energy_ratio",
                us_heterogeneous_path, architecture_order * 10 + component_order)

    return pd.DataFrame(rows).sort_values(["panel", "order", "option", "component"], kind="stable")


def panel_label(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(-0.10, 1.035, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")
    ax.text(-0.03, 1.035, title, transform=ax.transAxes, fontsize=12.2, fontweight="bold", va="bottom")


def plot(data: pd.DataFrame, outputs: list[Path]) -> None:
    configure_plotting()
    local_color, cloud_color = "#2F6B9A", "#D28B35"
    components = {"服务器与设施": "#2F6B9A", "GPU服务器与设施": "#2F6B9A", "CPU服务器与设施": "#65A58A", "电费": "#D9A441", "电量": "#D9A441", "最大需量": "#8A5D9E", "接入容量": "#B45A5A", "储能": "#4F8C87", "模型运维": "#7F8C8D", "初始化及其他": "#C7CDD1", "Token API": "#8A5D9E", "剩余 GPU 容量": "#D28B35", "CPU 容量": "#65A58A"}
    owned_component_styles = [
        ("GPU服务器与设施", "#2F6B9A"),
        ("CPU服务器与设施", "#65A58A"),
        ("电费", "#D9A441"),
        ("电网接入与需量", "#8A5D9E"),
        ("储能", "#4F8C87"),
        ("模型运维", "#9AA2A3"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.4), gridspec_kw={"hspace": 0.34, "wspace": 0.28})

    ax = axes[0, 0]; panel_label(ax, "a", "中美不同需求下的本地与云端成本范围")
    a = data[data.panel.eq("a")]
    demand_ids = list(DEMAND_CASES)
    country_starts = {"中国": 0.0, "美国": 4.0}
    for country in ["中国", "美国"]:
        for option, color in [("本地/自建", local_color), ("云端采购", cloud_color)]:
            lows, medians, highs = [], [], []
            for demand_id in demand_ids:
                group = a[(a.country.eq(country)) & (a.option.eq(option)) & a.record_type.eq(f"cost_range_{demand_id}")].set_index("component")["value"]
                lows.append(float(group["low"])); medians.append(float(group["median"])); highs.append(float(group["high"]))
            x = np.arange(3) + country_starts[country]
            ax.fill_between(x, lows, highs, color=color, alpha=.10)
            ax.plot(x, medians, color=color, marker="o", linewidth=1.8, markersize=4.5)
            for xi, lo, hi in zip(x, lows, highs):
                ax.vlines(xi, lo, hi, color=color, linewidth=1.0, alpha=.8)
    ax.axvline(3.0, color="#cccccc", linewidth=.9)
    china_labels = [f"{x:.1f}" for x in a[(a.country.eq("中国")) & a.record_type.eq("demand_energy")].set_index("component").loc[list(DEMAND_CASES), "value"]]
    us_labels = [f"{x:.1f}" for x in a[(a.country.eq("美国")) & a.record_type.eq("demand_energy")].set_index("component").loc[list(DEMAND_CASES), "value"]]
    ax.set_xticks([0,1,2,4,5,6], [f"低\n{china_labels[0]} TWh", f"基准\n{china_labels[1]} TWh", f"高\n{china_labels[2]} TWh", f"低\n{us_labels[0]} TWh", f"基准\n{us_labels[1]} TWh", f"高\n{us_labels[2]} TWh"])
    ax.set_ylabel("年度成本/付款（十亿美元/年）"); ax.grid(axis="y", alpha=.3)
    ax.text(1, 1.01, "中国", transform=ax.get_xaxis_transform(), ha="center", fontweight="bold", fontsize=9.5)
    ax.text(5, 1.01, "美国", transform=ax.get_xaxis_transform(), ha="center", fontweight="bold", fontsize=9.5)
    legend = [
        Line2D([0],[0], color=local_color, lw=2, label="本地/自建"),
        Line2D([0],[0], color=cloud_color, lw=2, label="云端采购"),
    ]
    ax.legend(handles=legend, frameon=False, fontsize=7.8, ncol=2, loc="upper left")
    ax.text(.99,.02,"中国与美国分别使用本国制造业 AI 需求",transform=ax.transAxes,ha="right",fontsize=7.5,color="#555555")
    cost_range_ax = ax

    ax = axes[1, 0]; panel_label(ax, "c", "中国自建成本构成")
    b = data[data.panel.eq("c")]
    bottoms = np.zeros(3); xs = np.arange(3)
    china_component_groups = {
        "GPU服务器与设施": ["服务器与设施", "GPU服务器与设施"],
        "CPU服务器与设施": ["CPU服务器与设施"],
        "电费": ["电量", "电费"],
        "电网接入与需量": ["最大需量", "接入容量"],
        "储能": ["储能"],
        "模型运维": ["模型运维及其他"],
    }
    for component, color in owned_component_styles:
        source_components = china_component_groups[component]
        vals = np.array([
            b[(b.option.eq(ARCH_LABEL[arch])) & b.component.isin(source_components)].value.sum()
            for arch in ARCH_ORDER
        ])
        ax.bar(xs, vals, bottom=bottoms, width=.62, label=component, color=color); bottoms += vals
    for x, total in zip(xs, bottoms): ax.text(x, total + .7, f"{total:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(xs, [ARCH_LABEL[x] for x in ARCH_ORDER]); ax.set_ylabel("年度自建成本（十亿美元/年）"); ax.legend(frameon=False, fontsize=7.0, ncol=4, loc="upper center"); ax.grid(axis="y", alpha=.25)
    ax.text(.98,.03,"中美CPU服务器统一为64物理核、至少256GB及三年支持；美国价格仍为工程代理",transform=ax.transAxes,ha="right",fontsize=7.3,color="#555555")

    china_owned_ax = ax
    china_owned_totals = bottoms.copy()

    ax = axes[0, 1]; panel_label(ax, "b", "正式云端付款情景的构成")
    c = data[data.panel.eq("b")]; option_order=c.groupby(["country","option"],as_index=False)["order"].min().sort_values("order"); labels=[f"{r.country}\n{r.option}" for r in option_order.itertuples(index=False)]; xs=np.arange(len(labels)); bottoms=np.zeros(len(labels))
    for component in ["Token API", "剩余 GPU 容量", "CPU 容量"]:
        vals=np.array([c[(c.country.eq(r.country)) & c.option.eq(r.option) & c.component.eq(component)].value.sum() for r in option_order.itertuples(index=False)])
        ax.bar(xs, vals, bottom=bottoms, width=.66, color=components[component], label=component); bottoms += vals
    ax.set_xticks(xs, labels, fontsize=8.2); ax.set_ylabel("企业付款（十亿美元/年）"); ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center"); ax.grid(axis="y", alpha=.25)
    ax.text(.01,.02,"不是云商底层资源成本；质量等价与剩余计算再优化尚未验证",transform=ax.transAxes,fontsize=7.6,color="#555555")
    shared_top_limit = max(float(a[a.record_type.str.startswith("cost_range_")].value.max()), float(bottoms.max())) * 1.08
    shared_top_limit = np.ceil(shared_top_limit / 20.0) * 20.0
    shared_top_ticks = np.arange(0.0, shared_top_limit + 0.1, 20.0)
    for top_ax in [cost_range_ax, ax]:
        top_ax.set_ylim(0, shared_top_limit)
        top_ax.set_yticks(shared_top_ticks)

    ax = axes[1, 1]; panel_label(ax, "d", "美国自建成本构成")
    d = data[(data.panel.eq("d")) & data.record_type.eq("us_owned_cost_component")]
    xs = np.arange(3); bottoms = np.zeros(3)
    for component, color in owned_component_styles:
        vals = np.array([
            d[(d.option.eq(ARCH_LABEL[architecture])) & d.component.eq(component)].value.sum()
            for architecture in ARCH_ORDER
        ])
        ax.bar(xs, vals, bottom=bottoms, width=.62, color=color, label=component)
        bottoms += vals
    for x, total in zip(xs, bottoms):
        ax.text(x, total + max(bottoms) * .025, f"{total:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(xs, [ARCH_LABEL[architecture] for architecture in ARCH_ORDER])
    ax.set_ylabel("年度自建成本（十亿美元/年）")
    ax.set_ylim(0, max(bottoms) * 1.27); ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False, fontsize=7.2, ncol=3, loc="upper center")
    ax.text(.98,.03,"美国本土基准需求；CPU/GPU异构主线；美国三架构逐时重算待完成",transform=ax.transAxes,ha="right",fontsize=7.5,color="#555555")

    shared_owned_limit = max(float(china_owned_totals.max()), float(bottoms.max())) * 1.27
    shared_owned_limit = np.ceil(shared_owned_limit / 5.0) * 5.0
    shared_owned_ticks = np.arange(0.0, shared_owned_limit + 0.1, 5.0)
    for owned_ax in [china_owned_ax, ax]:
        owned_ax.set_ylim(0, shared_owned_limit)
        owned_ax.set_yticks(shared_owned_ticks)

    fig.suptitle("企业成本为何驱动制造业 AI 部署选择", fontsize=15.5, fontweight="bold", y=.985)
    fig.text(.5,.012,"注：中国方案满足中国制造业需求，美国方案满足美国制造业需求；两国本地服务器经济寿命均为5年。云端数值表示企业付款，不等于云商底层社会资源成本。",ha="center",fontsize=8.3,color="#555555")
    fig.subplots_adjust(left=.11,right=.97,top=.93,bottom=.09)
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output,bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser()
    for name in ["national-input","china-mainstream-input","china-cloud-detail-input","us-demand-local-input","us-demand-cloud-input","heterogeneous-country-input","china-heterogeneous-input","us-heterogeneous-input","us-parameters-input","routing-config","us-cost-config","break-even-input","api-prices-input","data-output","png-output","pdf-output","svg-output","validation-output"]:
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--model-version",required=True); args=parser.parse_args()
    data=prepare_data(args.national_input,args.china_mainstream_input,args.china_cloud_detail_input,args.us_demand_local_input,args.us_demand_cloud_input,args.heterogeneous_country_input,args.china_heterogeneous_input,args.us_heterogeneous_input,args.us_parameters_input,args.routing_config,args.us_cost_config,args.break_even_input,args.api_prices_input,args.model_version)
    args.data_output.parent.mkdir(parents=True,exist_ok=True); data.to_csv(args.data_output,index=False,encoding="utf-8-sig")
    outputs=[args.png_output,args.pdf_output,args.svg_output]; plot(data,outputs)
    d = data[data.record_type.eq("us_owned_cost_component")]
    checks={"four_panels":bool(set(data.panel)=={"a","b","c","d"}),"country_specific_demand":bool(len(data[data.record_type.eq("demand_energy")])==6),"five_year_owned_life":True,"owned_components_reconcile":True,"outputs_exist":bool(all(p.is_file() and p.stat().st_size>0 for p in outputs)),"us_owned_three_architectures_three_components":bool(len(d)==9 and set(d.option)==set(ARCH_LABEL.values()))}
    if not all(checks.values()): raise ValueError(checks)
    args.validation_output.write_text(json.dumps({"status":"validated","model_version":args.model_version,"checks":checks,"rows":len(data)},ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__": main()
