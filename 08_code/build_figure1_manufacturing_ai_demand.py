#!/usr/bin/env python3
"""Build the iterative Figure 1 manufacturing-AI demand draft.

The figure intentionally excludes the deployment-architecture schematic.  It
combines observed 2023 and scenario 2030 AI adoption, low/base/high effective-
service scenarios, industry demand composition, and two national-scale shares.

Run without arguments from anywhere:

    python3 08_code/build_figure1_manufacturing_ai_demand.py

Costs are intentionally excluded and reserved for Figure 2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "distributed_llm_matplotlib")
)

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:
    # macOS system Python in this workspace is intentionally minimal.  Re-run
    # with the existing project analysis runtime so `python3 script.py` remains
    # a one-command workflow for figure iteration.
    analysis_python = Path("/opt/anaconda3/bin/python")
    if analysis_python.exists() and os.environ.get("FIGURE1_PYTHON_REEXEC") != "1":
        os.environ["FIGURE1_PYTHON_REEXEC"] = "1"
        os.execv(str(analysis_python), [str(analysis_python), *sys.argv])
    raise


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = PROJECT_ROOT / "05_results/v0.8.0/result/manuscript_figures"

TASK_ORDER = [
    "office",
    "agent",
    "vision",
    "maintenance",
    "scheduling",
    "simulation",
]
TASK_LABELS = {
    "office": "办公知识",
    "agent": "业务Agent",
    "vision": "VLM复核",
    "maintenance": "预测维护",
    "scheduling": "生产排程",
    "simulation": "研发仿真",
}
TASK_COLORS = {
    "office": "#8BA6B4",
    "agent": "#E7A84B",
    "vision": "#D66B5D",
    "maintenance": "#9A78B5",
    "scheduling": "#54A69A",
    "simulation": "#356F9F",
}

ARCH_ORDER = ["IF", "IG_1host", "IG_multisite"]
ARCH_LABELS = {
    "IF": "逐厂独立",
    "IG_1host": "集团单节点",
    "IG_multisite": "集团多节点",
}

NATIONAL_ADOPTION_2030_SCENARIO = 0.480046

SHORT_INDUSTRY_NAMES = {
    "C13": "农副食品",
    "C14": "食品",
    "C15": "酒饮料茶",
    "C16": "烟草",
    "C17": "纺织",
    "C18": "纺织服装",
    "C19": "皮革制鞋",
    "C20": "木材加工",
    "C21": "家具",
    "C22": "造纸",
    "C23": "印刷",
    "C24": "文体用品",
    "C25": "石油煤炭加工",
    "C26": "化学制品",
    "C27": "医药",
    "C28": "化纤",
    "C29": "橡胶塑料",
    "C30": "非金属矿物",
    "C31": "黑色冶炼",
    "C32": "有色冶炼",
    "C33": "金属制品",
    "C34": "通用设备",
    "C35": "专用设备",
    "C36": "汽车",
    "C37": "运输设备",
    "C38": "电气机械",
    "C39": "电子设备",
    "C40": "仪器仪表",
    "C41": "其他制造",
    "C42": "废弃资源",
    "C43": "设备修理",
}


def _odds(value: float) -> float:
    return value / (1.0 - value)


def _inverse_odds(value: float) -> float:
    return value / (1.0 + value)


def prepare_data(
    adoption_path: Path,
    service_path: Path,
    growth_anchor_path: Path,
) -> pd.DataFrame:
    adoption = pd.read_csv(adoption_path, encoding="utf-8-sig")
    service = pd.read_csv(service_path, encoding="utf-8-sig")
    growth_anchors = pd.read_csv(growth_anchor_path, encoding="utf-8-sig")

    if len(adoption) != 31 or adoption["industry_code"].nunique() != 31:
        raise ValueError("Adoption input must cover 31 manufacturing industries")
    if set(service["parameter_case"]) != {"low", "base", "high"}:
        raise ValueError("Effective-service input must contain low/base/high cases")
    expected_growth_anchors = {
        ("national_computing_center_electricity", 2025),
        ("national_computing_center_electricity", 2030),
        ("manufacturing_share_of_national_ai", 2025),
        ("manufacturing_share_of_national_ai", 2030),
    }
    actual_growth_anchors = set(
        zip(growth_anchors["measure"], growth_anchors["year"].astype(int))
    )
    if not expected_growth_anchors.issubset(actual_growth_anchors):
        raise ValueError("Growth-anchor input is missing a required measure/year row")

    rows: list[dict[str, object]] = []

    for item in adoption.itertuples(index=False):
        rows.append(
            {
                "panel": "a",
                "category": item.industry_code,
                "label": item.industry_name_cn,
                "series": "2023行业观测",
                "value": float(item.any_ai_adoption_2023),
                "weight": float(item.above_size_firms_2023),
                "unit": "fraction",
                "status": "observed",
            }
        )

    national_2023 = float(
        (adoption["any_ai_adoption_2023"] * adoption["above_size_firms_2023"]).sum()
        / adoption["above_size_firms_2023"].sum()
    )
    national_odds_multiplier = _odds(NATIONAL_ADOPTION_2030_SCENARIO) / _odds(
        national_2023
    )
    for item in adoption.itertuples(index=False):
        adoption_2030 = _inverse_odds(
            _odds(float(item.any_ai_adoption_2023))
            * national_odds_multiplier ** float(item.template_diffusion_speed)
        )
        rows.append(
            {
                "panel": "a",
                "category": item.industry_code,
                "label": item.industry_name_cn,
                "series": "2030基准情景",
                "value": adoption_2030,
                "weight": float(item.above_size_firms_2023),
                "unit": "fraction",
                "status": "scenario_from_2023_anchor_and_diffusion_speed",
            }
        )

    for item in service.itertuples(index=False):
        rows.append(
            {
                "panel": "b",
                "category": f"2030_{item.parameter_case}",
                "label": item.task_name_cn,
                "series": item.task_id,
                "value": float(item.effective_service_units_day),
                "unit": "effective_service_units/day",
                "status": "scenario",
            }
        )
        if item.parameter_case == "base":
            reconstructed_2023 = float(item.reconstructed_2023_service_units_day)
            interpolated_2025 = reconstructed_2023 * (
                float(item.effective_service_units_day) / reconstructed_2023
            ) ** (2.0 / 7.0)
            for category, value, status in (
                ("2023_reconstructed", reconstructed_2023, "reconstructed_2023"),
                ("2025_path", interpolated_2025, "log_linear_path_interpolation"),
            ):
                rows.append(
                    {
                        "panel": "b",
                        "category": category,
                        "label": item.task_name_cn,
                        "series": item.task_id,
                        "value": value,
                        "unit": "effective_service_units/day",
                        "status": status,
                    }
                )
            rows.append(
                {
                    "panel": "c",
                    "category": item.industry_code,
                    "label": item.industry_name_cn,
                    "series": item.task_id,
                    "value": float(item.effective_service_units_day),
                    "unit": "effective_service_units/day",
                    "status": "scenario",
                }
            )

    for item in growth_anchors.itertuples(index=False):
        rows.append(
            {
                "panel": "d",
                "category": str(int(item.year)),
                "label": item.measure,
                "series": item.measure,
                "value": float(item.value),
                "low": float(item.low),
                "high": float(item.high),
                "unit": item.unit,
                "status": item.status,
            }
        )

    # Allocate national computing-centre electricity to manufacturing AI using
    # the application-share anchors.  This is an explicit allocation proxy,
    # not a metered manufacturing-electricity observation.
    national_power = growth_anchors[
        growth_anchors["measure"].eq("national_computing_center_electricity")
    ].set_index("year")
    manufacturing_share = growth_anchors[
        growth_anchors["measure"].eq("manufacturing_share_of_national_ai")
    ].set_index("year")
    for year in (2025, 2030):
        power = national_power.loc[year]
        share = manufacturing_share.loc[year]
        rows.append(
            {
                "panel": "d",
                "category": str(year),
                "label": "manufacturing_ai_allocated_electricity",
                "series": "manufacturing_ai_allocated_electricity",
                "value": float(power["value"] * share["value"]),
                "low": float(power["low"] * share["low"]),
                "high": float(power["high"] * share["high"]),
                "unit": "TWh/year",
                "status": "allocation_proxy_from_national_power_and_application_share",
            }
        )

    output = pd.DataFrame(rows)
    output.insert(0, "model_version", "v0.8.0")
    return output


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(colors="#333333")
    ax.grid(axis="x", color="#D7D7D7", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)


def plot(data: pd.DataFrame, outputs: list[Path]) -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 13,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11.2))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # a: paired observed 2023 and scenario 2030 adoption across 31 industries.
    adoption = data[data["panel"].eq("a")]
    observed = (
        adoption[adoption["series"].eq("2023行业观测")]
        .set_index("category")
        .sort_values("value")
    )
    scenario = adoption[adoption["series"].eq("2030基准情景")].set_index("category")
    y = np.arange(len(observed))
    values_2023 = observed["value"].to_numpy()
    values_2030 = scenario.loc[observed.index, "value"].to_numpy()
    ax_a.hlines(y, values_2023, values_2030, color="#CDD3D7", linewidth=1.1)
    ax_a.scatter(
        values_2023,
        y,
        s=24,
        color="#7F9DA9",
        edgecolor="white",
        linewidth=0.4,
        label="2023观测",
        zorder=3,
    )
    ax_a.scatter(
        values_2030,
        y,
        s=28,
        color="#D8793A",
        marker="D",
        edgecolor="white",
        linewidth=0.4,
        label="2030基准情景",
        zorder=3,
    )
    national_2023 = float(
        (observed["value"] * observed["weight"]).sum() / observed["weight"].sum()
    )
    national_2030 = float(
        (scenario["value"] * scenario["weight"]).sum() / scenario["weight"].sum()
    )
    ax_a.axvline(national_2023, color="#3D5F7D", linestyle="--", linewidth=1.0)
    ax_a.axvline(national_2030, color="#B65E29", linestyle="--", linewidth=1.0)
    ax_a.text(
        national_2023 - 0.004,
        len(observed) - 0.2,
        f"全国 {national_2023:.1%}",
        ha="right",
        va="top",
        color="#3D5F7D",
        fontsize=10,
    )
    ax_a.text(
        national_2030 + 0.004,
        len(observed) - 0.2,
        f"全国 {national_2030:.1%}",
        ha="left",
        va="top",
        color="#A85B2A",
        fontsize=10,
    )
    ax_a.set_xlim(0.10, 0.72)
    ax_a.set_ylim(-0.8, len(observed) - 0.1)
    ax_a.set_yticks(
        y,
        [f"{code} {SHORT_INDUSTRY_NAMES[code]}" for code in observed.index],
        fontsize=9.5,
    )
    ticks = np.arange(0.10, 0.71, 0.10)
    ax_a.set_xticks(ticks, [f"{v:.0%}" for v in ticks])
    ax_a.set_xlabel("报告应用至少一种AI的规上企业比例")
    ax_a.set_title("a  行业AI采用：2023观测与2030情景", loc="left", fontweight="bold", fontsize=15)
    ax_a.legend(frameon=False, loc="lower right", fontsize=10)
    _style_axis(ax_a)
    ax_a.grid(axis="y", color="#E3E3E3", linewidth=0.35, alpha=0.55)

    # b: 2023 reconstruction, 2025 path value, and 2030 scenarios.
    demand = data[data["panel"].eq("b")]
    cases = [
        "2023_reconstructed",
        "2025_path",
        "2030_low",
        "2030_base",
        "2030_high",
    ]
    case_labels = ["2023\n重建", "2025\n路径值", "2030\n低", "2030\n基准", "2030\n高"]
    bottoms = np.zeros(len(cases))
    for task in TASK_ORDER:
        values = np.array(
            [
                demand[demand["category"].eq(case) & demand["series"].eq(task)]["value"].sum()
                / 1e6
                for case in cases
            ]
        )
        ax_b.bar(
            np.arange(len(cases)),
            values,
            bottom=bottoms,
            width=0.66,
            color=TASK_COLORS[task],
            label=TASK_LABELS[task],
        )
        bottoms += values
    for x, total in enumerate(bottoms):
        ax_b.text(x, total + 3.0, f"{total:.1f}", ha="center", fontweight="bold")
    ax_b.set_xticks(np.arange(len(cases)), case_labels)
    ax_b.set_ylabel("有效服务量（百万单位/日）")
    ax_b.set_title("b  制造业AI需求增长与2030情景", loc="left", fontweight="bold", fontsize=15)
    ax_b.legend(
        frameon=False,
        ncol=3,
        fontsize=10,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.99),
    )
    ax_b.set_ylim(0, max(bottoms) * 1.18)
    early_inset = ax_b.inset_axes([0.045, 0.43, 0.34, 0.27])
    early_bottom = np.zeros(2)
    for task in TASK_ORDER:
        early_values = np.array(
            [
                demand[
                    demand["category"].eq(case) & demand["series"].eq(task)
                ]["value"].sum()
                / 1e6
                for case in cases[:2]
            ]
        )
        early_inset.bar(
            np.arange(2),
            early_values,
            bottom=early_bottom,
            width=0.62,
            color=TASK_COLORS[task],
        )
        early_bottom += early_values
    early_inset.set_xticks([0, 1], ["2023\n重建", "2025\n路径值"], fontsize=8.5)
    early_inset.set_ylim(0, max(early_bottom) * 1.23)
    early_inset.set_yticks([0, 1, 2, 3])
    early_inset.tick_params(axis="y", labelsize=8)
    early_inset.set_title("早期需求放大（百万单位/日）", fontsize=9, pad=2)
    early_inset.spines["top"].set_visible(False)
    early_inset.spines["right"].set_visible(False)
    early_inset.grid(axis="y", color="#DDDDDD", linewidth=0.4, alpha=0.6)
    early_inset.set_axisbelow(True)
    _style_axis(ax_b)
    ax_b.grid(axis="x", visible=False)

    # c: top industries by absolute demand, with remaining industries aggregated.
    industry = data[data["panel"].eq("c")]
    totals = industry.groupby("category")["value"].sum().sort_values(ascending=False)
    top_codes = totals.head(11).index.tolist()
    top = industry[industry["category"].isin(top_codes)].copy()
    other = (
        industry[~industry["category"].isin(top_codes)]
        .groupby("series", as_index=False)["value"]
        .sum()
    )
    other["category"] = "OTHER"
    other["label"] = "其他20行业"
    combined = pd.concat([top, other], ignore_index=True)
    order = (
        combined.groupby("category")["value"].sum().sort_values().index.tolist()
    )
    labels = [
        "其他20行业"
        if code == "OTHER"
        else f"{code} {SHORT_INDUSTRY_NAMES[code]}"
        for code in order
    ]
    y = np.arange(len(order))
    left = np.zeros(len(order))
    for task in TASK_ORDER:
        values = np.array(
            [
                combined[
                    combined["category"].eq(code) & combined["series"].eq(task)
                ]["value"].sum()
                / 1e6
                for code in order
            ]
        )
        ax_c.barh(y, values, left=left, color=TASK_COLORS[task], height=0.72)
        left += values
    ax_c.set_yticks(y, labels, fontsize=9.5)
    ax_c.set_xlabel("基准有效服务量（百万单位/日）")
    ax_c.set_title("c  行业需求规模与任务结构", loc="left", fontweight="bold", fontsize=15)
    top_three_share = float(totals.head(3).sum() / totals.sum())
    ax_c.text(
        0.99,
        0.02,
        f"前三行业占全国需求 {top_three_share:.1%}",
        transform=ax_c.transAxes,
        ha="right",
        va="bottom",
        color="#3D5F7D",
        fontweight="bold",
    )
    _style_axis(ax_c)

    # d: same-unit electricity comparison.  The series starts in 2025 so that
    # every national value uses the current "computing center" boundary.
    growth = data[data["panel"].eq("d")]
    ax_d.set_title("d  全国算力中心与制造业AI用电增长", loc="left", fontweight="bold", fontsize=15)
    dc = growth[
        growth["series"].eq("national_computing_center_electricity")
    ].set_index("category")
    manufacturing_power = growth[
        growth["series"].eq("manufacturing_ai_allocated_electricity")
    ].set_index("category")
    share_growth = growth[
        growth["series"].eq("manufacturing_share_of_national_ai")
    ].set_index("category")
    years = np.array([2025, 2030])
    categories = [str(year) for year in years]
    dc_base = np.array([float(dc.loc[year, "value"]) for year in categories])
    dc_low = np.array([float(dc.loc[year, "low"]) for year in categories])
    dc_high = np.array([float(dc.loc[year, "high"]) for year in categories])
    manufacturing_base = np.array(
        [float(manufacturing_power.loc[year, "value"]) for year in categories]
    )
    manufacturing_low = np.array(
        [float(manufacturing_power.loc[year, "low"]) for year in categories]
    )
    manufacturing_high = np.array(
        [float(manufacturing_power.loc[year, "high"]) for year in categories]
    )
    share_current = float(share_growth.loc["2025", "value"])
    share_2030 = float(share_growth.loc["2030", "value"])
    share_low = float(share_growth.loc["2030", "low"])
    share_high = float(share_growth.loc["2030", "high"])

    dc_line, = ax_d.plot(
        years,
        dc_base,
        color="#486F98",
        linewidth=2.2,
        linestyle="--",
        label="全国算力中心用电",
        zorder=3,
    )
    ax_d.scatter([2025], [dc_base[0]], s=52, color="#486F98", zorder=5)
    dc_scenario = ax_d.errorbar(
        [2030],
        [dc_base[1]],
        yerr=[[dc_base[1] - dc_low[1]], [dc_high[1] - dc_base[1]]],
        fmt="o",
        markersize=6.5,
        markerfacecolor="white",
        markeredgecolor="#486F98",
        color="#486F98",
        capsize=6,
        elinewidth=2.0,
        label="2030全国范围",
        zorder=5,
    )
    manufacturing_line, = ax_d.plot(
        years,
        manufacturing_base,
        color="#D8793A",
        linewidth=2.2,
        linestyle="--",
        marker="D",
        markersize=5.5,
        label="制造业AI对应用电",
        zorder=3,
    )
    manufacturing_scenario = ax_d.errorbar(
        [2030],
        [manufacturing_base[1]],
        yerr=[
            [manufacturing_base[1] - manufacturing_low[1]],
            [manufacturing_high[1] - manufacturing_base[1]],
        ],
        fmt="D",
        markersize=5.5,
        color="#D8793A",
        capsize=6,
        elinewidth=2.0,
        label="2030制造业范围",
        zorder=5,
    )

    ax_d.set_xlim(2024.5, 2030.5)
    ax_d.set_xticks(years)
    ax_d.set_xlabel("年份")
    ax_d.set_ylabel("用电量（TWh/年，对数轴）")
    ax_d.set_yscale("log")
    ax_d.set_ylim(5, 1000)
    ax_d.set_yticks([5, 10, 20, 50, 100, 200, 500, 1000])
    ax_d.set_yticklabels(["5", "10", "20", "50", "100", "200", "500", "1000"])
    ax_d.grid(axis="y", color="#D7D7D7", linewidth=0.6, alpha=0.65)
    ax_d.grid(axis="x", visible=False)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.spines["left"].set_color("#777777")
    ax_d.spines["bottom"].set_color("#777777")

    ax_d.annotate(
        "2025披露值 170",
        xy=(2025, dc_base[0]),
        xytext=(8, 9),
        textcoords="offset points",
        color="#486F98",
        fontsize=9.5,
    )
    ax_d.annotate(
        "2030全国 400–800\n中点600",
        xy=(2030, dc_base[1]),
        xytext=(-10, 10),
        textcoords="offset points",
        ha="right",
        color="#486F98",
        fontsize=9.5,
    )
    ax_d.annotate(
        f"2025制造业≈{manufacturing_base[0]:.1f}\n（{share_current:.0%}分摊）",
        xy=(2025, manufacturing_base[0]),
        xytext=(8, 9),
        textcoords="offset points",
        color="#B65E29",
        fontsize=9.5,
    )
    ax_d.annotate(
        f"2030制造业 {manufacturing_low[1]:.0f}–{manufacturing_high[1]:.0f}\n中点情景{manufacturing_base[1]:.0f}（份额{share_2030:.0%}）",
        xy=(2030, manufacturing_base[1]),
        xytext=(-10, -10),
        textcoords="offset points",
        ha="right",
        va="top",
        color="#B65E29",
        fontsize=9.5,
    )
    ax_d.legend(
        [dc_line, dc_scenario, manufacturing_line, manufacturing_scenario],
        ["全国算力中心用电", "2030全国范围", "制造业AI对应用电", "2030制造业范围"],
        frameon=False,
        fontsize=9.5,
        ncol=2,
        loc="upper left",
    )
    fig.tight_layout(rect=[0.02, 0.02, 0.99, 0.965], h_pad=2.2, w_pad=2.2)

    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the four-panel iterative Figure 1 manufacturing-AI demand draft."
    )
    parser.add_argument(
        "--adoption-input",
        type=Path,
        default=PROJECT_ROOT / "02_data/china_manufacturing_ai_31sector_proxy_parameters.csv",
    )
    parser.add_argument(
        "--service-input",
        type=Path,
        default=PROJECT_ROOT
        / "02_data/processed/effective_service/manufacturing_ai_effective_service_2030.csv",
    )
    parser.add_argument(
        "--growth-anchor-input",
        type=Path,
        default=PROJECT_ROOT
        / "02_data/china_data_center_manufacturing_ai_growth_anchors.csv",
    )
    parser.add_argument(
        "--data-output",
        type=Path,
        default=FIGURE_ROOT / "figure1_manufacturing_ai_demand_draft_data.csv",
    )
    parser.add_argument(
        "--png-output",
        type=Path,
        default=FIGURE_ROOT / "figure1_manufacturing_ai_demand_draft.png",
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=FIGURE_ROOT / "figure1_manufacturing_ai_demand_draft.pdf",
    )
    parser.add_argument(
        "--svg-output",
        type=Path,
        default=FIGURE_ROOT / "figure1_manufacturing_ai_demand_draft.svg",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=FIGURE_ROOT / "figure1_manufacturing_ai_demand_draft.metadata.json",
    )
    args = parser.parse_args()

    data = prepare_data(
        args.adoption_input,
        args.service_input,
        args.growth_anchor_input,
    )
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.data_output, index=False, encoding="utf-8-sig")
    plot(data, [args.png_output, args.pdf_output, args.svg_output])
    metadata = {
        "artifact_id": "figure1-cn-demand-20260817-interim-approved",
        "status": "interim_approved_for_figure_development",
        "reviewed_on": "2026-08-17",
        "review_note": "User accepted the current information structure and visual direction for now; this is not the final manuscript figure.",
        "model_version": "v0.8.0",
        "panels": [
            "adoption_2023_vs_2030",
            "demand_scenarios",
            "industry_demand",
            "national_computing_center_and_manufacturing_ai_electricity",
        ],
        "national_electricity_boundary": "2025 observed national computing-center electricity; 2030 external 400-800 TWh scenario range",
        "manufacturing_electricity_boundary": "allocation scenario from national computing-center electricity multiplied by manufacturing AI application share; not metered electricity",
        "architecture_schematic_included": False,
        "cost_panel_included": False,
        "cost_panel_destination": "Figure 2",
        "outputs": {
            "data": str(args.data_output),
            "png": str(args.png_output),
            "pdf": str(args.pdf_output),
            "svg": str(args.svg_output),
        },
    }
    args.metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Figure written to {args.png_output}")


if __name__ == "__main__":
    main()
