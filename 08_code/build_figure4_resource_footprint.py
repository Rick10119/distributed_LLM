#!/usr/bin/env python3
"""Build manuscript Figure 4: on-site water, space, land, and materials."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

cache = Path(tempfile.gettempdir()) / "distributed_llm_matplotlib"
xdg_cache = Path(tempfile.gettempdir()) / "distributed_llm_fontconfig"
cache.mkdir(parents=True, exist_ok=True)
xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml


COLORS = {
    "local": "#2F6B9A", "cloud": "#D28B35", "china": "#D28B35", "us": "#4E8B68",
    "gfa": "#8A5D9E", "land": "#65A58A", "concrete": "#8AA6A3", "steel": "#A8734F",
}
DEMAND_CASES = ("low", "base", "high")
DEMAND_LABELS = {"low": "低", "base": "中", "high": "高"}


def configure() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "figure.dpi": 130,
    })


def panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(-.10, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=14)
    ax.text(-.03, 1.04, title, transform=ax.transAxes, fontweight="bold", fontsize=12)


def prepare(
    scenario_registry_path: Path,
    water_path: Path,
    space_path: Path,
    materials_path: Path,
    china_heterogeneous_path: Path,
    us_heterogeneous_path: Path,
    national_path: Path,
    model_version: str,
) -> pd.DataFrame:
    registry = yaml.safe_load(scenario_registry_path.read_text(encoding="utf-8"))
    water_case = registry["resource_footprint"]["water"]
    water = pd.read_csv(water_path, encoding="utf-8-sig")
    space = pd.read_csv(space_path, encoding="utf-8-sig")
    materials = pd.read_csv(materials_path, encoding="utf-8-sig")
    china = pd.read_csv(china_heterogeneous_path, encoding="utf-8-sig")
    china = china[china.owned_architecture.eq("IF")].copy()
    us_local = pd.read_csv(us_heterogeneous_path, encoding="utf-8-sig")
    national = pd.read_csv(national_path, encoding="utf-8-sig")
    mode = water_case["comparison_modes"]
    required_water_modes = {mode["local"], mode["china_cloud"], mode["us_cloud"]}
    if not required_water_modes <= set(water["comparison_mode"]):
        raise ValueError("Selected water scenario is not covered by the water parameter table")
    versioned_inputs = {
        "space postprocessing": space,
        "material postprocessing": materials,
        "national physical summary": national,
    }
    for label, frame in versioned_inputs.items():
        if "model_version" not in frame.columns:
            raise ValueError(f"{label} is missing model_version")
        versions = set(frame["model_version"].dropna().astype(str))
        if versions != {model_version}:
            raise ValueError(
                f"Figure 4 expected {model_version} for {label}, got {sorted(versions)}"
            )
    cn_core = china.iloc[0]
    cn_base_facility_twh = float(cn_core["local_total_facility_energy_twh"])
    cn_base_capacity_mw = (float(cn_core["local_gpu_server_groups_industry_equivalent"]) * 1.50 + float(cn_core["local_cpu_server_groups_industry_equivalent"]) * 0.78) / 1000.0
    cn_facility_twh = {"low": 8.0, "base": cn_base_facility_twh, "high": 28.0}
    cn_capacity_mw = {case: cn_base_capacity_mw * energy / cn_base_facility_twh for case, energy in cn_facility_twh.items()}
    us_physical = us_local[us_local["cpu_server_price_case"].eq("base")].copy()
    if set(us_physical["parameter_case"]) != set(DEMAND_CASES):
        raise ValueError("US native heterogeneous demand requires low/base/high cases")
    physical_columns = ["local_gpu_servers", "local_cpu_servers", "annual_facility_energy_twh"]
    if us_physical.groupby("parameter_case")[physical_columns].nunique().max().max() != 1:
        raise ValueError("US heterogeneous physical results must be provider invariant")
    us_physical = us_physical.drop_duplicates("parameter_case").set_index("parameter_case")
    us_facility_twh = {case: float(us_physical.loc[case, "annual_facility_energy_twh"]) for case in DEMAND_CASES}
    us_capacity_mw = {
        case: (float(us_physical.loc[case, "local_gpu_servers"]) * 1.50 + float(us_physical.loc[case, "local_cpu_servers"]) * 0.78) / 1000.0
        for case in DEMAND_CASES
    }
    country_inputs = {
        "china": {"prefix": "CN", "facility": cn_facility_twh, "capacity": cn_capacity_mw, "source": china_heterogeneous_path},
        "us": {"prefix": "US", "facility": us_facility_twh, "capacity": us_capacity_mw, "source": us_heterogeneous_path},
    }
    if set(national["scenario"]) != {"IF", "IG", "II_1host"} or len(national) != 93:
        raise ValueError("Figure 4 requires the complete national physical summary")
    cn_distributed_grid_mw = float(national[national["scenario"].eq("IF")]["industry_equivalent_incremental_grid_expansion_mw"].sum())
    cn_cloud_grid_mw = float(national[national["scenario"].eq("II_1host")]["industry_equivalent_incremental_grid_expansion_mw"].sum())
    intensities = water.set_index("comparison_mode")["site_water_use_l_per_kwh_it"]
    rows: list[dict[str, object]] = []
    for country, inputs in country_inputs.items():
        for demand_case in DEMAND_CASES:
            it_energy = inputs["facility"][demand_case] / 1.30
            for deployment, comparison_mode in [
                ("local", mode["local"]),
                ("cloud", mode[f"{country}_cloud"]),
            ]:
                key = f"{country}_{deployment}"
                intensity = float(intensities[comparison_mode])
                rows.append({"panel": "a", "case": key, "demand_case": demand_case, "label": deployment, "metric": "it_energy", "value": it_energy, "unit": "TWh-IT/year", "source": inputs["source"].as_posix()})
                rows.append({"panel": "a", "case": key, "demand_case": demand_case, "label": deployment, "metric": "water_intensity", "value": intensity, "unit": "L/kWh-IT", "source": water_path.as_posix()})
                rows.append({"panel": "a", "case": key, "demand_case": demand_case, "label": deployment, "metric": "annual_water", "value": it_energy * intensity * 1e6, "unit": "m3/year", "source": water_path.as_posix()})
            capacity_scale = inputs["capacity"][demand_case] / cn_capacity_mw["base"]
            for deployment, base_value, evidence in [
                ("local", cn_distributed_grid_mw, "china_IF_optimized_grid_capacity_scaled_by_country_demand"),
                ("cloud", cn_cloud_grid_mw, "china_II_1host_optimized_grid_capacity_scaled_by_country_demand"),
            ]:
                rows.append({"panel": "a_grid", "case": f"{country}_{deployment}", "demand_case": demand_case, "label": deployment, "metric": "grid_connection_capacity_mw", "value": base_value * capacity_scale, "unit": "MW", "source": f"{national_path.as_posix()}::{evidence}"})
    indexed = space.set_index("space_case_id")
    for country, inputs in country_inputs.items():
        ref = indexed.loc[f"{inputs['prefix']}_LARGE_GREENFIELD"]
        for demand_case in DEMAND_CASES:
            target_capacity = inputs["capacity"][demand_case]
            scale = target_capacity / float(ref["common_installed_it_capacity_mw"])
            rows.append({"panel": "capacity", "case": country, "demand_case": demand_case, "label": country, "metric": "installed_it_capacity_mw", "value": target_capacity, "unit": "MW-IT", "source": inputs["source"].as_posix()})
            for metric in ["required_gross_floor_area_m2", "total_site_area_m2"]:
                rows.append({"panel": "b", "case": country, "demand_case": demand_case, "label": country, "metric": metric, "value": float(ref[metric]) * scale, "unit": "m2", "source": space_path.as_posix()})
            for realization in ["EXISTING", "CAMPUS", "GREENFIELD"]:
                item = indexed.loc[f"{inputs['prefix']}_LARGE_{realization}"]
                for metric in ["new_gross_floor_area_m2", "new_land_conversion_m2"]:
                    rows.append({"panel": "c", "case": country, "demand_case": demand_case, "label": realization.lower(), "metric": metric, "value": float(item[metric]) * scale, "unit": "m2", "source": space_path.as_posix()})
    for row in materials[materials["space_case_id"].isin(["CN_LARGE_GREENFIELD", "US_LARGE_GREENFIELD"])].itertuples(index=False):
        country = "china" if row.country == "China" else "us"
        for demand_case in DEMAND_CASES:
            target_capacity = country_inputs[country]["capacity"][demand_case]
            scale = target_capacity / float(row.common_installed_it_capacity_mw)
            for metric in ["concrete_m3", "rebar_t", "structural_steel_t"]:
                rows.append({"panel": "d", "case": country, "demand_case": demand_case, "label": row.archetype_id, "metric": metric, "value": float(getattr(row, metric)) * scale, "unit": "m3_or_t", "source": materials_path.as_posix()})
    return pd.DataFrame(rows)


def value(data: pd.DataFrame, panel_id: str, case: str, metric: str, label: str | None = None, demand_case: str | None = None) -> float:
    selected = data[data.panel.eq(panel_id) & data.case.eq(case) & data.metric.eq(metric)]
    if label is not None:
        selected = selected[selected.label.eq(label)]
    if demand_case is not None:
        selected = selected[selected.demand_case.eq(demand_case)]
    if len(selected) != 1:
        raise ValueError(f"Expected one value for {panel_id}/{case}/{metric}/{label}/{demand_case}; got {len(selected)}")
    return float(selected.value.iloc[0])


def plot(data: pd.DataFrame, svg: Path, png: Path | None) -> None:
    configure()
    fig = plt.figure(figsize=(13.2, 7.0))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.18], wspace=.28)
    operating_grid = outer[0].subgridspec(2, 1, hspace=.12)
    operating_axes = [fig.add_subplot(operating_grid[i]) for i in range(2)]
    build_grid = outer[1].subgridspec(4, 1, hspace=.10)
    build_axes = [fig.add_subplot(build_grid[i]) for i in range(4)]
    demand_x = np.array([0, 1, 2, 4, 5, 6], dtype=float)
    country_cases = [("china", "中国", [0, 1, 2]), ("us", "美国", [4, 5, 6])]
    demand_ticks = []
    for country, country_label, _ in country_cases:
        for demand_case in DEMAND_CASES:
            facility = value(data, "capacity", country, "installed_it_capacity_mw", demand_case=demand_case)
            demand_ticks.append(f"{DEMAND_LABELS[demand_case]}\n{facility/1e3:.2f} GW-IT")

    def country_headers(ax: plt.Axes) -> None:
        ax.axvline(3, color="#cccccc", linewidth=.9)
        ax.text(1, 1.01, "中国", transform=ax.get_xaxis_transform(), ha="center", fontweight="bold", fontsize=9.5)
        ax.text(5, 1.01, "美国", transform=ax.get_xaxis_transform(), ha="center", fontweight="bold", fontsize=9.5)

    panel(operating_axes[0], "a", "IF基准与大型集中式云的运行和接入绝对量")
    for axis_index,(op_ax,panel_id,metric,ylabel,scale) in enumerate([
        (operating_axes[0],"a","annual_water","现场用水（百万m³/年）",1e6),
        (operating_axes[1],"a_grid","grid_connection_capacity_mw","电网接入容量（GW）",1e3),
    ]):
        for deployment,label,color in [("local","IF（工厂侧分布式）",COLORS["local"]),("cloud","大型集中式云",COLORS["cloud"])]:
            for country,_,xs in country_cases:
                vals=np.array([value(data,panel_id,f"{country}_{deployment}",metric,demand_case=d)/scale for d in DEMAND_CASES])
                op_ax.plot(xs,vals,color=color,marker="o",linewidth=1.9,markersize=4.8,label=label if country=="china" else None)
        op_ax.set_xlim(-.35,6.35); op_ax.set_ylabel(ylabel,fontsize=8); op_ax.grid(axis="y",alpha=.24); country_headers(op_ax)
        if axis_index==0: op_ax.set_xticks([]); op_ax.legend(frameon=False,fontsize=7.7,ncol=2,loc="upper left")
        else: op_ax.set_xticks(demand_x,demand_ticks,fontsize=7.2)
    operating_axes[0].text(.98,.05,"本地0.050；云端0.335 L/kWh-IT",transform=operating_axes[0].transAxes,ha="right",fontsize=7.1,color="#555555")
    operating_axes[1].text(.98,.05,"本地IF与大型集中节点均为优化结果",transform=operating_axes[1].transAxes,ha="right",fontsize=7.1,color="#555555")

    panel(build_axes[0], "b", "大型集中式云相对IF增加的建设足迹")
    build_specs = [
        ("gfa", "建筑壳体（公顷）", COLORS["gfa"]),
        ("land", "土地转换（公顷）", COLORS["land"]),
        ("concrete", "混凝土（百万m³）", COLORS["concrete"]),
        ("steel", "施工钢材（十万吨）", COLORS["steel"]),
    ]
    for axis_index,(metric,ylabel,color) in enumerate(build_specs):
        subax=build_axes[axis_index]
        for country,_,xs in country_cases:
            if metric=="gfa":
                lows=np.array([value(data,"c",country,"new_gross_floor_area_m2","greenfield",d)/1e4 for d in DEMAND_CASES]); highs=lows.copy()
            elif metric=="land":
                lows=np.array([value(data,"c",country,"new_land_conversion_m2","greenfield",d)/1e4 for d in DEMAND_CASES]); highs=lows.copy()
            else:
                lows=[]; highs=[]
                for demand_case in DEMAND_CASES:
                    archetype_vals=[]
                    for archetype in ["STEEL_FRAME","RC_FRAME"]:
                        if metric=="concrete":
                            archetype_vals.append(value(data,"d",country,"concrete_m3",archetype,demand_case)/1e6)
                        else:
                            archetype_vals.append((value(data,"d",country,"rebar_t",archetype,demand_case)+value(data,"d",country,"structural_steel_t",archetype,demand_case))/1e5)
                    lows.append(min(archetype_vals)); highs.append(max(archetype_vals))
                lows=np.array(lows); highs=np.array(highs)
            midpoint=(lows+highs)/2
            subax.fill_between(xs,lows,highs,color=color,alpha=.14)
            subax.plot(xs,midpoint,color=color,marker="o",linewidth=1.8,markersize=4.4)
            if np.any(highs>lows): subax.vlines(xs,lows,highs,color=color,linewidth=.9,alpha=.8)
        subax.set_xlim(-.35,6.35); subax.set_ylabel(ylabel,fontsize=7.7); subax.grid(axis="y",alpha=.23); country_headers(subax)
        if axis_index<3: subax.set_xticks([])
        else: subax.set_xticks(demand_x,demand_ticks,fontsize=7.2)
    build_axes[2].text(.98,.08,"材料区间＝钢框架至RC框架",transform=build_axes[2].transAxes,ha="right",fontsize=7.1,color="#555555")

    fig.suptitle("以IF为基准的运行资源需求与大型云新增足迹",fontsize=15,fontweight="bold",y=.985)
    cn_capacity=value(data,"capacity","china","installed_it_capacity_mw",demand_case="base"); us_capacity=value(data,"capacity","us","installed_it_capacity_mw",demand_case="base")
    note=(
        "注：IF为统一比较基准。a为IF与大型云的绝对量；中国IF和大型集中节点接入容量均来自活动物理优化，美国为按本国需求容量缩放的估计。\n"
        f"b为绿地大型云相对IF增加的毛新建量；各国使用自身低/中/高需求，基准容量为中国{cn_capacity:,.0f} MW-IT、美国{us_capacity:,.0f} MW-IT。当前IF复用既有工厂；工厂改造材料为NR，水泥未估算。"
    )
    fig.text(.5,.018,note,ha="center",va="bottom",fontsize=7.4,color="#555555",linespacing=1.25)
    fig.subplots_adjust(left=.065,right=.992,top=.915,bottom=.105)
    svg.parent.mkdir(parents=True,exist_ok=True); fig.savefig(svg,bbox_inches="tight",pad_inches=.025)
    if png: fig.savefig(png,bbox_inches="tight",pad_inches=.025,dpi=180)
    plt.close(fig)


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--scenario-registry",type=Path,required=True); p.add_argument("--water-input",type=Path,required=True); p.add_argument("--space-input",type=Path,required=True); p.add_argument("--materials-input",type=Path,required=True); p.add_argument("--china-heterogeneous-input",type=Path,required=True); p.add_argument("--us-heterogeneous-input",type=Path,required=True); p.add_argument("--national-input",type=Path,required=True); p.add_argument("--model-version",required=True); p.add_argument("--data-output",type=Path,required=True); p.add_argument("--svg-output",type=Path,required=True); p.add_argument("--png-output",type=Path); p.add_argument("--validation-output",type=Path,required=True); a=p.parse_args()
    data=prepare(a.scenario_registry,a.water_input,a.space_input,a.materials_input,a.china_heterogeneous_input,a.us_heterogeneous_input,a.national_input,a.model_version); a.data_output.parent.mkdir(parents=True,exist_ok=True); data.to_csv(a.data_output,index=False,encoding="utf-8-sig"); plot(data,a.svg_output,a.png_output)
    checks={"country_deployment_demand_water_cases":len(data[data.panel.eq("a")])==36,"country_deployment_demand_grid_cases":len(data[data.panel.eq("a_grid")])==12,"country_specific_demand_capacities":len(data[data.panel.eq("capacity")])==6,"two_country_demand_space_cases":len(data[data.panel.eq("b")])==12,"demand_realization_rows":len(data[data.panel.eq("c")])==36,"demand_greenfield_material_rows":len(data[data.panel.eq("d")])==36,"svg_exists":a.svg_output.is_file()}
    if not all(checks.values()): raise ValueError(checks)
    a.validation_output.write_text(json.dumps({"status":"validated","model_version":a.model_version,"checks":checks},ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__": main()
