#!/usr/bin/env python3
"""Prepare data and plot manuscript Figure 1: demand and deployment architecture."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

_matplotlib_cache = Path(tempfile.gettempdir()) / "distributed_llm_matplotlib"
_fontconfig_cache = Path(tempfile.gettempdir()) / "distributed_llm_fontconfig"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
_fontconfig_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_fontconfig_cache))

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
import yaml


TASK_ORDER = ["office", "agent", "vision", "maintenance", "scheduling", "simulation"]
TASK_SHORT = {
    "office": "办公知识",
    "agent": "流程 Agent",
    "vision": "VLM 复核",
    "maintenance": "预测维护",
    "scheduling": "生产排程",
    "simulation": "研发仿真",
}
ARCHITECTURES = [
    ("IF", "工厂侧分布式（核心）", "数据与实时任务留厂；集团专网协同"),
    ("IG", "集团集中算力池", "成员工厂共享一个集团节点"),
    ("II_1host", "大型集中节点", "行业单节点集中上界；不是实际云厂商位置"),
]


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def prepare_data(service_path: Path, national_path: Path, heterogeneous_path: Path, routing_config_path: Path, model_version: str) -> pd.DataFrame:
    service = pd.read_csv(service_path, encoding="utf-8-sig")
    national = pd.read_csv(national_path, encoding="utf-8-sig")
    heterogeneous = pd.read_csv(heterogeneous_path, encoding="utf-8-sig")
    heterogeneous = heterogeneous[heterogeneous.owned_architecture.eq("IF")].copy()
    routing_config = yaml.safe_load(routing_config_path.read_text(encoding="utf-8"))
    routing_case = routing_config["active_core_routing_case"]
    routing = routing_config["routing_cases"][routing_case]
    require_columns(
        service,
        {
            "industry_code",
            "industry_name_cn",
            "parameter_case",
            "task_id",
            "task_name_cn",
            "effective_service_units_day",
            "evidence_status",
        },
        "effective-service table",
    )
    require_columns(
        national,
        {
            "model_version",
            "scenario",
            "external_energy_low_twh",
            "external_energy_central_twh",
            "external_energy_high_twh",
            "industry_equivalent_annual_ai_facility_energy_twh",
        },
        "national core summary",
    )
    if set(national["model_version"].astype(str)) != {model_version}:
        raise ValueError("National summary does not match requested model version")
    if set(national["scenario"].astype(str)) != {"IF", "IG", "II_1host"}:
        raise ValueError("National summary must contain all three deployment architectures")

    rows: list[dict[str, object]] = []
    source_service = service_path.as_posix()
    source_national = national_path.as_posix()

    base = service[service["parameter_case"].eq("base")].copy()
    task_demand = base.groupby(["task_id", "task_name_cn"], as_index=False)["effective_service_units_day"].sum()
    for order, row in enumerate(task_demand.set_index("task_id").loc[TASK_ORDER].reset_index().itertuples(index=False), start=1):
        cpu_fraction = float(routing.get(row.task_id, 0.0))
        for hardware, fraction in [("GPU", 1-cpu_fraction), ("CPU", cpu_fraction)]:
            rows.append(
                {
                    "panel": "a", "record_type": "task_hardware_routing", "order": order,
                    "item_id": row.task_id, "item_label": row.task_name_cn,
                    "group_id": hardware, "group_label": hardware,
                    "value": float(row.effective_service_units_day) * fraction,
                    "unit": "effective_service_units/day", "secondary_value": cpu_fraction,
                    "secondary_unit": "CPU_service_fraction",
                    "evidence_type": "structural_hardware_routing_scenario", "source": routing_config_path.as_posix(),
                }
            )

    for order, (architecture, label, note) in enumerate(ARCHITECTURES, start=1):
        rows.append(
            {
                "panel": "b",
                "record_type": "deployment_architecture",
                "order": order,
                "item_id": architecture,
                "item_label": label,
                "group_id": architecture,
                "group_label": note,
                "value": np.nan,
                "unit": "not_applicable",
                "secondary_value": np.nan,
                "secondary_unit": "not_applicable",
                "evidence_type": "counterfactual_architecture",
                "source": source_national,
            }
        )

    external_columns = ["external_energy_low_twh", "external_energy_central_twh", "external_energy_high_twh"]
    external_by_architecture = national.groupby("scenario")[external_columns].sum()
    if not np.allclose(
        external_by_architecture.to_numpy(float),
        external_by_architecture.iloc[0].to_numpy(float),
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError("Industry allocations of the external energy scenarios do not conserve common national totals")
    external = external_by_architecture.iloc[0]
    scenario_specs = [
        ("low", "低需求", "external_energy_low_twh"),
        ("central", "基准需求", "external_energy_central_twh"),
        ("high", "高需求", "external_energy_high_twh"),
    ]
    hetero = heterogeneous.drop_duplicates("owned_architecture")
    if len(hetero) != 1:
        raise ValueError("Heterogeneous national table must provide one physical owned configuration")
    gpu_base = float(hetero.iloc[0].local_gpu_facility_energy_twh)
    cpu_base = float(hetero.iloc[0].local_cpu_facility_energy_twh)
    hetero_total = gpu_base + cpu_base
    energy_specs = [("low","低需求",float(external.external_energy_low_twh)),("central","外部基准",float(external.external_energy_central_twh)),("model","异构模型基准",hetero_total),("high","高需求",float(external.external_energy_high_twh))]
    for order,(scenario,label,total) in enumerate(energy_specs,start=1):
        for hardware,value in [("GPU设施电量",total*gpu_base/hetero_total),("CPU设施电量",total*cpu_base/hetero_total)]:
            rows.append({"panel":"c","record_type":"energy_component","order":order,"item_id":scenario,"item_label":label,"group_id":hardware,"group_label":hardware,"value":value,"unit":"TWh/year","secondary_value":total,"secondary_unit":"TWh/year total","evidence_type":"heterogeneous_model_or_scaled_external_scenario","source":heterogeneous_path.as_posix() if scenario=="model" else source_national})

    if len(base) != 31 * 6 or base["industry_code"].nunique() != 31:
        raise ValueError("Base service table must contain 31 industries times six tasks")
    totals = base.groupby("industry_code")["effective_service_units_day"].transform("sum")
    national_total = float(base["effective_service_units_day"].sum())
    base["within_industry_task_share"] = base["effective_service_units_day"] / totals
    base["national_service_share"] = base["effective_service_units_day"] / national_total
    task_rank = {task: i + 1 for i, task in enumerate(TASK_ORDER)}
    industry_total = base.groupby("industry_code")["effective_service_units_day"].sum().sort_values(ascending=False)
    industry_rank = {code: i + 1 for i, code in enumerate(industry_total.index)}
    for row in base.itertuples(index=False):
        rows.append(
            {
                "panel": "d", "record_type": "industry_task_composition",
                "order": industry_rank[row.industry_code] * 10 + task_rank[row.task_id],
                "item_id": row.task_id, "item_label": row.task_name_cn,
                "group_id": row.industry_code, "group_label": row.industry_name_cn,
                "value": float(row.within_industry_task_share),
                "unit": "share_of_industry_effective_service",
                "secondary_value": float(row.effective_service_units_day),
                "secondary_unit": "effective_service_units/day",
                "evidence_type": str(row.evidence_status), "source": source_service,
            }
        )

    output = pd.DataFrame(rows)
    output.insert(0, "model_version", model_version)
    return output.sort_values(["panel", "order", "group_id", "item_id"], kind="stable")


def panel_label(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(-0.055, 1.04, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")
    ax.text(0.0, 1.04, title, transform=ax.transAxes, fontsize=12.5, fontweight="bold", va="bottom")


def draw_panel_a(ax: plt.Axes, data: pd.DataFrame, palette: list[str]) -> None:
    panel_label(ax, "a", "六类任务需求及 CPU/GPU 路由")
    task_labels = [TASK_SHORT[t] for t in TASK_ORDER]
    totals = data.groupby("item_id")["value"].sum().reindex(TASK_ORDER)
    shares = totals / totals.sum() * 100
    gpu = np.array([data[(data.item_id.eq(t)) & data.group_id.eq("GPU")].value.sum() for t in TASK_ORDER]) / totals.to_numpy() * shares.to_numpy()
    cpu = shares.to_numpy() - gpu
    x = np.arange(len(TASK_ORDER))
    ax.bar(x,gpu,width=.68,color="#2F6B9A",label="GPU任务")
    ax.bar(x,cpu,bottom=gpu,width=.68,color="#65A58A",label="CPU任务")
    ax.set_xticks(x,task_labels,rotation=25,ha="right",fontsize=8.2)
    ax.set_ylabel("全国有效服务需求占比（%）")
    ax.grid(axis="y",alpha=.25); ax.legend(frameon=False,ncol=2,fontsize=8,loc="upper right")
    cpu_routes = data[(data.panel.eq("a")) & (data.group_id.eq("CPU")) & data.value.gt(0)]
    routing_panel = data[data.panel.eq("a")]
    route_text = "，".join(
        f"{row.item_label}{row.value / routing_panel[routing_panel.item_id.eq(row.item_id)].value.sum():.0%}"
        for row in cpu_routes.itertuples(index=False)
    )
    ax.text(.01,.98,f"CPU路由：{route_text}；其余任务为GPU",transform=ax.transAxes,va="top",fontsize=7.7,color="#555555")


def draw_node(ax: plt.Axes, x: float, y: float, radius: float, color: str, label: str = "") -> None:
    ax.add_patch(Circle((x, y), radius, facecolor="white", edgecolor=color, linewidth=1.4))
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5)


def draw_panel_b(ax: plt.Axes, data: pd.DataFrame, palette: list[str]) -> None:
    ax.set_axis_off()
    panel_label(ax, "b", "相同有效服务的三种部署反事实")
    rows = data.sort_values("order")
    ys = [0.76, 0.50, 0.24]
    for idx, (row, y) in enumerate(zip(rows.itertuples(index=False), ys)):
        color = palette[idx]
        ax.text(0.02, y + 0.08, row.item_label, fontsize=10.5, fontweight="bold", color=color, va="center")
        factories = [0.32, 0.42, 0.52, 0.62]
        for x in factories:
            draw_node(ax, x, y, 0.027, color, "厂")
        if row.item_id == "IF":
            for x in factories:
                ax.add_patch(Rectangle((x - 0.018, y - 0.085), 0.036, 0.025, facecolor=color, alpha=0.72, linewidth=0))
        elif row.item_id == "IG":
            draw_node(ax, 0.78, y, 0.045, color, "集团")
            for x in factories:
                ax.plot([x + 0.028, 0.735], [y, y], color=color, linewidth=0.8, alpha=0.8)
        else:
            draw_node(ax, 0.82, y, 0.057, color, "集中\n节点")
            for x in factories:
                ax.plot([x + 0.028, 0.763], [y, y], color=color, linewidth=0.8, alpha=0.8)
        ax.text(0.98, y - 0.075, row.group_label, ha="right", va="center", fontsize=8.2, color="#555555")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.08, 0.92)


def draw_panel_c(ax: plt.Axes, data: pd.DataFrame, palette: list[str]) -> None:
    panel_label(ax, "c", "需求情景的 CPU/GPU 设施电量构成")
    ids=["low","central","model","high"]; labels=["低需求","外部基准","异构模型基准","高需求"]
    x=np.arange(4); bottoms=np.zeros(4)
    for component,color in [("GPU设施电量","#2F6B9A"),("CPU设施电量","#65A58A")]:
        vals=np.array([data[(data.item_id.eq(i)) & data.group_id.eq(component)].value.sum() for i in ids])
        ax.bar(x,vals,bottom=bottoms,width=.62,color=color,label=component); bottoms+=vals
    for xi,total in zip(x,bottoms): ax.text(xi,total+.55,f"{total:.1f}",ha="center",fontsize=8.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("年度设施电量（TWh/年）")
    ax.set_ylim(0, max(bottoms) * 1.18)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False,fontsize=8,loc="upper left")
    ax.text(0.99, 0.03, "外部情景按异构模型的CPU/GPU电量比例缩放", transform=ax.transAxes, ha="right", fontsize=7.6, color="#555555")


def draw_panel_d(ax: plt.Axes, data: pd.DataFrame, cmap: colors.Colormap) -> None:
    panel_label(ax, "d", "31行业需求及 CPU 可路由占比（基准情景）")
    pivot = data.pivot(index=["group_id", "group_label"], columns="item_id", values="value")
    totals = data.groupby(["group_id", "group_label"])["secondary_value"].sum().sort_values(ascending=False)
    pivot = pivot.loc[totals.index, TASK_ORDER]
    matrix = pivot.to_numpy(float) * 100
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(50, float(np.nanmax(matrix))))
    labels = [f"{code} {name}" for code, name in pivot.index]
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=7.0)
    ax.set_xticks(np.arange(len(TASK_ORDER)), [TASK_SHORT[t] for t in TASK_ORDER], rotation=28, ha="right", fontsize=8.4)
    ax.tick_params(length=0)
    for x in np.arange(-0.5, len(TASK_ORDER), 1):
        ax.axvline(x, color="white", linewidth=0.45, alpha=0.75)
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.026, pad=0.018)
    cbar.set_label("行业内有效服务占比（%）", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    cpu_fractions={"maintenance":.5,"scheduling":1.0}
    cpu_share=np.array([sum(pivot.loc[idx,t]*cpu_fractions.get(t,0) for t in TASK_ORDER)*100 for idx in pivot.index])
    for yi,share in enumerate(cpu_share):
        ax.text(len(TASK_ORDER)+.08,yi,f"CPU {share:.0f}%",va="center",fontsize=6.5,color="#4E8B68")
    ax.set_xlim(-.5,len(TASK_ORDER)+.95)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_figure(data: pd.DataFrame, outputs: list[Path]) -> None:
    configure_plotting()
    palette = ["#2F6B9A", "#D28B35", "#4E8B68", "#8A5D9E"]
    cmap = colors.LinearSegmentedColormap.from_list("task_share", ["#F6F7F8", "#9FC1D7", "#2F6B9A"])
    fig = plt.figure(figsize=(14.2, 10.8), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.78, 1.52], width_ratios=[1.23, 0.77], hspace=0.28, wspace=0.25)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[1, 0])
    draw_panel_a(ax_a, data[data["panel"].eq("a")], palette)
    draw_panel_b(ax_b, data[data["panel"].eq("b")], palette)
    draw_panel_c(ax_c, data[data["panel"].eq("c")], palette)
    draw_panel_d(ax_d, data[data["panel"].eq("d")], cmap)
    fig.suptitle("制造业 AI 需求、异构计算路由与部署架构", fontsize=15.5, fontweight="bold", y=0.985)
    fig.text(0.5, 0.012, "注：CPU/GPU路由为结构性情景，不是观测硬件份额；异构模型基准设施电量为15.067 TWh/年。大型集中节点是空间集中上界。", ha="center", fontsize=8.5, color="#555555")
    fig.subplots_adjust(left=0.14, right=0.965, top=0.94, bottom=0.065)
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def validate_outputs(data: pd.DataFrame, model_version: str, outputs: list[Path]) -> dict[str, object]:
    panel_counts = data.groupby("panel").size().to_dict()
    checks = {
        "panels_a_to_d_present": set(panel_counts) == {"a", "b", "c", "d"},
        "six_tasks_times_two_hardware_routes": panel_counts.get("a") == 12,
        "three_architectures": panel_counts.get("b") == 3,
        "four_energy_cases_times_two_components": panel_counts.get("c") == 8,
        "31_industries_times_six_tasks": panel_counts.get("d") == 186,
        "industry_task_shares_sum_to_one": bool(np.allclose(
            data[data["panel"].eq("d")].groupby("group_id")["value"].sum().to_numpy(float), 1.0, atol=1e-9
        )),
        "all_figure_files_exist": all(path.is_file() and path.stat().st_size > 0 for path in outputs),
    }
    if not all(checks.values()):
        raise ValueError(f"Figure 1 validation failed: {checks}")
    return {"status": "validated", "model_version": model_version, "checks": checks, "panel_rows": panel_counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-input", type=Path, required=True)
    parser.add_argument("--national-input", type=Path, required=True)
    parser.add_argument("--heterogeneous-input", type=Path, required=True)
    parser.add_argument("--routing-config", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--data-output", type=Path, required=True)
    parser.add_argument("--png-output", type=Path, required=True)
    parser.add_argument("--pdf-output", type=Path, required=True)
    parser.add_argument("--svg-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    args = parser.parse_args()

    data = prepare_data(args.service_input, args.national_input, args.heterogeneous_input, args.routing_config, args.model_version)
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.data_output, index=False, encoding="utf-8-sig")
    figure_outputs = [args.png_output, args.pdf_output, args.svg_output]
    plot_figure(data, figure_outputs)
    validation = validate_outputs(data, args.model_version, figure_outputs)
    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
