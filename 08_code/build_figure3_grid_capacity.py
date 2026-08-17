#!/usr/bin/env python3
"""Build Figure 3: temporal and cross-factory AI-load coordination."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(
    Path(tempfile.gettempdir()) / "distributed_llm_matplotlib"
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "distributed_llm_cache")
)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = PROJECT_ROOT / "05_results/v0.8.0/result/manuscript_figures"
CORE_ROOT = PROJECT_ROOT / "05_results/v0.8.0/result/group_architecture_core"
DEFAULT_INDUSTRIES = [f"C{i}" for i in range(13, 44)]
DAY_CENTERS = [12, 36, 60, 84, 108, 132, 156]
DAY_LABELS = ["一", "二", "三", "四", "五", "六", "日"]
DAY_BOUNDARIES = [24, 48, 72, 96, 120, 144]


def prepare(root: Path, industries: list[str], version: str) -> pd.DataFrame:
    """Prepare the exact temporal and node-level values displayed in Figure 3."""
    focus = "C36" if "C36" in industries else industries[0]
    hourly = pd.read_csv(root / focus / "hourly.csv", encoding="utf-8-sig")
    one_actual = hourly[
        hourly.architecture.eq("IG_1host")
        & hourly.base_load_case.eq("actual_load")
    ].copy()
    one_zero = hourly[
        hourly.architecture.eq("IG_1host")
        & hourly.base_load_case.eq("zero_load")
    ].copy()
    multisite = hourly[
        hourly.architecture.eq("IG_multisite")
        & hourly.base_load_case.eq("actual_load")
    ].copy()
    if len(one_actual) != 168 or len(one_zero) != 168:
        raise ValueError(f"{focus}: IG_1host Figure 3 series must each contain 168 hours")
    decomposition_columns = [
        "ai_fixed_overhead_power_mw",
        "unshiftable_ai_active_power_mw",
        "shiftable_ai_active_power_mw",
    ]
    missing_decomposition = [
        column for column in decomposition_columns if column not in hourly.columns
    ]
    if missing_decomposition:
        raise ValueError(
            "Figure 3c requires rerun hourly power decomposition columns: "
            + ", ".join(missing_decomposition)
        )
    node_counts = multisite.groupby("factory_id").hour.nunique()
    if len(node_counts) < 2 or not node_counts.eq(168).all():
        raise ValueError(f"{focus}: incomplete IG_multisite node-hour coverage")
    rows: list[dict[str, object]] = []
    for source, label in [
        (one_actual, "actual_load"),
        (one_zero, "zero_load_reoptimized"),
    ]:
        for row in source.sort_values("hour").itertuples(index=False):
            common = {
                "model_version": version,
                "panel": "a",
                "industry": focus,
                "architecture": "IG_1host",
                "base_load_case": label,
                "factory_id": row.factory_id,
                "hour": int(row.hour),
            }
            if label == "actual_load":
                rows.append(
                    {
                        **common,
                        "metric": "base_load_mw",
                        "value": float(row.base_load_mw),
                        "unit": "MW",
                    }
                )
            rows.append(
                {
                    **common,
                    "metric": "ai_facility_power_mw",
                    "value": float(row.ai_facility_power_mw),
                    "unit": "MW",
                }
            )
            for metric in decomposition_columns:
                rows.append(
                    {
                        **common,
                        "metric": metric,
                        "value": float(getattr(row, metric)),
                        "unit": "MW",
                    }
                )

    for factory_id, node in multisite.groupby("factory_id", sort=True):
        node = node.sort_values("hour")
        peak = float(node.base_load_mw.max())
        ai_peak = float(node.ai_facility_power_mw.max())
        if peak <= 0:
            raise ValueError(f"{focus}/{factory_id}: non-positive production-load peak")
        for row in node.itertuples(index=False):
            common = {
                "model_version": version,
                "panel": "b",
                "industry": focus,
                "architecture": "IG_multisite",
                "base_load_case": "actual_load",
                "factory_id": factory_id,
                "hour": int(row.hour),
            }
            rows.extend(
                [
                    {
                        **common,
                        "metric": "base_load_fraction_of_node_weekly_peak",
                        "value": float(row.base_load_mw / peak),
                        "unit": "fraction",
                    },
                    {
                        **common,
                        "metric": "ai_facility_power_mw",
                        "value": float(row.ai_facility_power_mw),
                        "unit": "MW",
                    },
                    {
                        **common,
                        "metric": "ai_facility_power_fraction_of_node_weekly_peak",
                        "value": (
                            float(row.ai_facility_power_mw / ai_peak)
                            if ai_peak > 0
                            else 0.0
                        ),
                        "unit": "fraction",
                    },
                ]
            )
    return pd.DataFrame(rows)


def _series(
    data: pd.DataFrame,
    *,
    panel: str,
    metric: str,
    base_load_case: str | None = None,
) -> np.ndarray:
    selected = data[data.panel.eq(panel) & data.metric.eq(metric)]
    if base_load_case is not None:
        selected = selected[selected.base_load_case.eq(base_load_case)]
    return selected.sort_values("hour").value.to_numpy(dtype=float)


def _heatmap_matrix(data: pd.DataFrame, metric: str) -> tuple[list[str], np.ndarray]:
    selected = data[data.panel.eq("b") & data.metric.eq(metric)]
    pivot = selected.pivot(index="factory_id", columns="hour", values="value").sort_index()
    if pivot.isna().any().any() or pivot.shape[1] != 168:
        raise ValueError(f"incomplete Figure 3 heatmap matrix for {metric}")
    return pivot.index.tolist(), pivot.to_numpy(dtype=float)


def _style_time_axis(ax: plt.Axes, *, show_labels: bool) -> None:
    ax.set_xlim(-0.5, 167.5)
    ax.set_xticks(DAY_CENTERS, DAY_LABELS if show_labels else [])
    for boundary in DAY_BOUNDARIES:
        ax.axvline(boundary - 0.5, color="#D3D8DC", lw=0.8, zorder=0)
    ax.grid(axis="y", alpha=0.18)


def plot(data: pd.DataFrame, svg: Path, png: Path) -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "font.size": 9,
        }
    )
    production = _series(
        data, panel="a", metric="base_load_mw", base_load_case="actual_load"
    )
    ai_actual = _series(
        data,
        panel="a",
        metric="ai_facility_power_mw",
        base_load_case="actual_load",
    )
    ai_zero = _series(
        data,
        panel="a",
        metric="ai_facility_power_mw",
        base_load_case="zero_load_reoptimized",
    )
    ai_overhead = _series(
        data,
        panel="a",
        metric="ai_fixed_overhead_power_mw",
        base_load_case="actual_load",
    )
    ai_unshiftable = _series(
        data,
        panel="a",
        metric="unshiftable_ai_active_power_mw",
        base_load_case="actual_load",
    )
    ai_shiftable = _series(
        data,
        panel="a",
        metric="shiftable_ai_active_power_mw",
        base_load_case="actual_load",
    )
    if not np.allclose(
        ai_actual, ai_overhead + ai_unshiftable + ai_shiftable, rtol=0, atol=1e-8
    ):
        raise ValueError("Figure 3c power components do not reconstruct total AI power")
    nodes, base_relative = _heatmap_matrix(
        data, "base_load_fraction_of_node_weekly_peak"
    )
    ai_nodes, ai_relative = _heatmap_matrix(
        data, "ai_facility_power_fraction_of_node_weekly_peak"
    )
    if ai_nodes != nodes:
        raise ValueError("Figure 3 heatmaps use inconsistent node order")
    host_nodes = data.loc[data.panel.eq("a"), "factory_id"].dropna().unique().tolist()
    if len(host_nodes) != 1:
        raise ValueError(f"Figure 3 requires one IG_1host node, found {host_nodes}")
    host_node = str(host_nodes[0])
    node_ai_peaks = (
        data[data.panel.eq("b") & data.metric.eq("ai_facility_power_mw")]
        .groupby("factory_id").value.max()
    )
    zero_ai_nodes = node_ai_peaks[node_ai_peaks <= 1e-12].index.tolist()

    x = np.arange(168)
    fig = plt.figure(figsize=(14.2, 7.6))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.18],
        height_ratios=[1.12, 1.0],
        wspace=0.18,
        hspace=0.25,
    )
    ax_prod = fig.add_subplot(grid[0, 0])
    ax_ai = fig.add_subplot(grid[1, 0], sharex=ax_prod)
    ax_base_heat = fig.add_subplot(grid[0, 1])
    ax_ai_heat = fig.add_subplot(grid[1, 1], sharex=ax_base_heat)

    ax_prod.fill_between(x, 0, production, color="#AABBC8", alpha=0.52)
    ax_prod.plot(x, production, color="#365F78", lw=1.35)
    ax_prod.set_ylim(0, production.max() * 1.08)
    ax_prod.set_ylabel("生产负荷（MW）")
    ax_prod.set_title(
        f"a  单节点时间协同（{host_node}）",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )
    _style_time_axis(ax_prod, show_labels=False)

    ax_ai.stackplot(
        x,
        ai_overhead,
        ai_unshiftable,
        ai_shiftable,
        colors=["#B7BEC4", "#D46F21", "#5A9BC4"],
        alpha=0.82,
        labels=["服务器基础功率", "不可平移任务", "可平移任务"],
    )
    ax_ai.plot(x, ai_actual, color="#6F4C2F", lw=1.05, label="配合生产负荷：总功率")
    ax_ai.plot(
        x,
        ai_zero,
        color="#3D8B85",
        lw=1.35,
        ls=(0, (4, 3)),
        label="无生产负荷：总功率",
    )
    peak_hour = int(np.argmax(production))
    for ax in [ax_prod, ax_ai]:
        ax.axvline(peak_hour, color="#7A6A58", lw=0.9, ls=(0, (2, 2)))
    ax_ai.scatter(
        [peak_hour, peak_hour],
        [ai_zero[peak_hour], ai_actual[peak_hour]],
        s=22,
        color=["#3D8B85", "#D46F21"],
        zorder=4,
    )
    ax_ai.annotate(
        f"生产峰值小时\n{ai_zero[peak_hour]:.2f} → {ai_actual[peak_hour]:.2f} MW",
        xy=(peak_hour, ai_actual[peak_hour]),
        xytext=(peak_hour + 9, 5.25),
        arrowprops={"arrowstyle": "-", "color": "#777", "lw": 0.8},
        fontsize=8,
        color="#444",
    )
    ax_ai.set_ylim(0, max(ai_actual.max(), ai_zero.max()) * 1.12)
    ax_ai.set_ylabel("AI设施功率（MW）")
    ax_ai.set_xlabel("星期")
    ax_ai.set_title(
        f"c  可平移与不可平移AI负荷（{host_node}）",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )
    ax_ai.legend(loc="upper right", frameon=False, fontsize=7.4, ncol=2)
    _style_time_axis(ax_ai, show_labels=True)

    blue_map = LinearSegmentedColormap.from_list(
        "production_relative", ["#F7FAFC", "#B8CDDA", "#345F78"]
    )
    orange_map = LinearSegmentedColormap.from_list(
        "ai_power", ["#FFF9F2", "#F1B46F", "#C95D16"]
    )
    base_image = ax_base_heat.imshow(
        base_relative,
        aspect="auto",
        interpolation="nearest",
        cmap=blue_map,
        vmin=0,
        vmax=1,
        extent=(-0.5, 167.5, len(nodes) - 0.5, -0.5),
    )
    ai_image = ax_ai_heat.imshow(
        ai_relative,
        aspect="auto",
        interpolation="nearest",
        cmap=orange_map,
        vmin=0,
        vmax=1,
        extent=(-0.5, 167.5, len(nodes) - 0.5, -0.5),
    )
    for ax in [ax_base_heat, ax_ai_heat]:
        ax.set_yticks(np.arange(len(nodes)), nodes)
        ax.set_ylabel("调度节点")
        for boundary in DAY_BOUNDARIES:
            ax.axvline(boundary - 0.5, color="white", lw=1.0, alpha=0.9)
        ax.set_xlim(-0.5, 167.5)

    ax_base_heat.set_xticks(DAY_CENTERS, [])
    ax_base_heat.set_title("b  多节点空间—时间协同", loc="left", fontweight="bold", fontsize=11)
    ax_base_heat.set_xlabel("生产负荷 / 本节点连续代表周峰值")
    base_cbar = fig.colorbar(
        base_image,
        ax=ax_base_heat,
        orientation="horizontal",
        pad=0.15,
        fraction=0.055,
        aspect=35,
    )
    base_cbar.set_ticks([0, 0.5, 1.0])
    base_cbar.set_ticklabels(["0", "50%", "100%"])

    ax_ai_heat.set_xticks(DAY_CENTERS, DAY_LABELS)
    ax_ai_heat.set_xlabel("星期")
    ax_ai_heat.set_title(
        "d  多节点AI相对负荷", loc="left", fontweight="bold", fontsize=11
    )
    ai_cbar = fig.colorbar(
        ai_image,
        ax=ax_ai_heat,
        orientation="horizontal",
        pad=0.15,
        fraction=0.055,
        aspect=35,
    )
    ai_cbar.set_ticks([0, 0.5, 1.0])
    ai_cbar.set_ticklabels(["0", "50%", "100%"])
    ai_cbar.set_label("AI设施功率 / 本节点连续代表周最大AI功率")

    fig.suptitle(
        "生产负荷与AI服务器负荷的时间和空间协同",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    zero_note = (
        f"；{', '.join(zero_ai_nodes)}的AI功率全周为0"
        if zero_ai_nodes
        else ""
    )
    fig.text(
        0.5,
        0.012,
        f"注：C36汽车制造业代表集团，连续168小时；左侧承载节点为{host_node}。"
        "多节点中的5个节点分别使用所在代表工厂的未聚合生产负荷，"
        "不把其他生产基地的电表或最大需量并入承载节点。"
        "c将配合生产负荷时的AI设施功率拆为服务器基础功率、不可平移任务活动功率和可平移任务活动功率；"
        f"生产负荷和AI负荷均按各节点自身周峰值分别归一化{zero_note}。",
        ha="center",
        fontsize=8,
        color="#555",
    )
    fig.subplots_adjust(top=0.92, bottom=0.10, left=0.07, right=0.98)
    svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Figure 3 from the v0.8.0 group-architecture hourly outputs."
    )
    parser.add_argument("--core-root", type=Path, default=CORE_ROOT)
    parser.add_argument("--industries", nargs="+", default=DEFAULT_INDUSTRIES)
    parser.add_argument("--model-version", default="v0.8.0")
    parser.add_argument(
        "--data-output", type=Path, default=FIGURE_ROOT / "figure3_grid_capacity_data.csv"
    )
    parser.add_argument(
        "--svg-output", type=Path, default=FIGURE_ROOT / "figure3_grid_capacity.svg"
    )
    parser.add_argument(
        "--png-output", type=Path, default=FIGURE_ROOT / "figure3_grid_capacity.png"
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=FIGURE_ROOT / "figure3_grid_capacity.validated.done.json",
    )
    args = parser.parse_args()
    focus = "C36" if "C36" in args.industries else args.industries[0]
    data = prepare(args.core_root, args.industries, args.model_version)
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.data_output, index=False, encoding="utf-8-sig")
    plot(data, args.svg_output, args.png_output)
    args.validation_output.write_text(
        json.dumps(
            {
                "status": "validated",
                "focus_industry": focus,
                "horizon_hours": 168,
                "panels": ["single_host_temporal_coordination", "multisite_heatmaps"],
                "architectures": ["IG_1host", "IG_multisite"],
                "zero_load_label": "AI_only_reoptimized_counterfactual",
                "panel_c_power_decomposition": [
                    "ai_fixed_overhead_power_mw",
                    "unshiftable_ai_active_power_mw",
                    "shiftable_ai_active_power_mw",
                ],
                "panel_c_decomposition_reconstructs_total_AI_facility_power": True,
                "node_dimension_preserved": True,
                "electrical_load_aggregation_at_AI_nodes": False,
                "multisite_AI_deployment_points_equal_routing_nodes": True,
                "ai_heatmap_normalization": "separate_weekly_maximum_within_each_node",
                "node_peak_bar_in_figure": False,
                "capacity_waterfall_in_figure": False,
                "II_1host_in_figure": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Figure 3 written to {args.svg_output} and {args.png_output}")


if __name__ == "__main__":
    main()
