#!/usr/bin/env python3
"""Draft Figure 5: spatial allocation and concentration diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import tempfile

cache = Path(tempfile.gettempdir()) / "distributed_llm_matplotlib"
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache))

import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import BoundaryNorm, ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = PROJECT_ROOT / "05_results/v0.8.0/result/manuscript_figures"
GROUP_NATIONAL_ROOT = PROJECT_ROOT / "05_results/v0.8.0/result/group_architecture_core/national"

TILES = {
    "xinjiang": (0, 2), "gansu": (1, 2), "inner mongolia": (2, 1), "heilongjiang": (5, 0),
    "jilin": (5, 1), "liaoning": (5, 2), "beijing": (4, 2), "tianjin": (5, 3),
    "hebei": (4, 3), "shanxi": (3, 3), "shaanxi": (2, 3), "ningxia": (1, 3),
    "qinghai": (1, 4), "tibet": (0, 5), "sichuan": (2, 5), "chongqing": (3, 5),
    "henan": (4, 4), "shandong": (5, 4), "jiangsu": (6, 5), "anhui": (5, 5),
    "hubei": (4, 5), "zhejiang": (6, 6), "shanghai": (7, 5), "jiangxi": (5, 6),
    "hunan": (4, 6), "fujian": (6, 7), "guangdong": (5, 7), "guangxi": (4, 7),
    "guizhou": (3, 6), "yunnan": (2, 7), "hainan": (5, 8),
}

LABELS = {
    "inner mongolia": "内蒙古", "heilongjiang": "黑龙江", "jilin": "吉林", "liaoning": "辽宁",
    "beijing": "北京", "tianjin": "天津", "hebei": "河北", "shanxi": "山西", "shaanxi": "陕西",
    "ningxia": "宁夏", "qinghai": "青海", "tibet": "西藏", "xinjiang": "新疆", "gansu": "甘肃",
    "sichuan": "四川", "chongqing": "重庆", "henan": "河南", "shandong": "山东", "jiangsu": "江苏",
    "anhui": "安徽", "hubei": "湖北", "zhejiang": "浙江", "shanghai": "上海", "jiangxi": "江西",
    "hunan": "湖南", "fujian": "福建", "guangdong": "广东", "guangxi": "广西", "guizhou": "贵州",
    "yunnan": "云南", "hainan": "海南",
}

ADCODE_TO_PROVINCE = {
    110000: "beijing", 120000: "tianjin", 130000: "hebei", 140000: "shanxi",
    150000: "inner mongolia", 210000: "liaoning", 220000: "jilin", 230000: "heilongjiang",
    310000: "shanghai", 320000: "jiangsu", 330000: "zhejiang", 340000: "anhui",
    350000: "fujian", 360000: "jiangxi", 370000: "shandong", 410000: "henan",
    420000: "hubei", 430000: "hunan", 440000: "guangdong", 450000: "guangxi",
    460000: "hainan", 500000: "chongqing", 510000: "sichuan", 520000: "guizhou",
    530000: "yunnan", 540000: "tibet", 610000: "shaanxi", 620000: "gansu",
    630000: "qinghai", 640000: "ningxia", 650000: "xinjiang",
}

LABEL_OFFSETS = {
    "gansu": (-28, 8), "ningxia": (5, -15), "inner mongolia": (5, 6),
    "tianjin": (5, 6), "anhui": (5, -13), "guangdong": (5, -12),
    "guizhou": (5, -12), "jiangsu": (5, 6), "zhejiang": (5, -12),
    "shandong": (5, 6), "hubei": (-29, 5), "fujian": (5, -11),
}


def configure() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "figure.dpi": 130,
    })


def panel(ax, letter: str, title: str) -> None:
    ax.text(-.08, 1.05, letter, transform=ax.transAxes, fontsize=16, fontweight="bold")
    ax.text(0, 1.05, title, transform=ax.transAxes, fontsize=14, fontweight="bold")


def prepare(
    allocation_path: Path,
    core_path: Path,
    scarcity_path: Path,
    cloud_share_path: Path,
    routing_case: str,
    local_water_l_per_kwh_it: float,
    cloud_water_l_per_kwh_it: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    allocation = pd.read_csv(allocation_path, encoding="utf-8-sig")
    scarcity = pd.read_csv(scarcity_path, encoding="utf-8-sig")
    core = pd.read_csv(core_path, encoding="utf-8-sig")
    core = core[core.architecture.eq("IF")].copy()
    if len(core) != 31 or core.industry.nunique() != 31:
        raise ValueError("Figure 5 requires 31 IF rows from the active group core")
    energy = core.set_index("industry")["industry_equivalent_annual_ai_facility_energy_twh"]
    allocation["industry_ai_energy_twh"] = allocation["industry_code"].map(energy)
    allocation["province_ai_energy_twh"] = allocation["industry_ai_energy_twh"] * allocation["electricity_share_within_industry"]
    province = allocation.groupby("province", as_index=False)["province_ai_energy_twh"].sum()
    province["national_share"] = province["province_ai_energy_twh"] / province["province_ai_energy_twh"].sum()
    province = province.sort_values("national_share", ascending=False).reset_index(drop=True)
    province["rank"] = np.arange(1, len(province) + 1)
    province["cumulative_share"] = province["national_share"].cumsum()
    national_it_energy_twh = float(province["province_ai_energy_twh"].sum()) / 1.30
    province["site_water_m3_small_local"] = province["national_share"] * national_it_energy_twh * local_water_l_per_kwh_it * 1e6
    province = province.merge(scarcity, on="province", how="left", validate="one_to_one")
    province["scarcity_weighted_water_m3_world_eq"] = province["site_water_m3_small_local"] * province["aware20_nonagri_annual_m3_world_eq_per_m3"]
    # No complete national 2030 province allocation was found. Use the latest
    # nationwide observed AI-capacity geography instead: CAICT's provincial
    # in-use intelligent-compute scale (FP16), digitised from its treemap and
    # anchored to the explicitly reported Hebei share (14.8%). This is a
    # capacity-based spatial proxy, not an observed cloud-provider traffic mix.
    cloud = pd.read_csv(cloud_share_path, encoding="utf-8-sig")
    if set(cloud["province"]) != set(LABELS):
        raise ValueError("Cloud AI-capacity proxy must cover all 31 provinces")
    if not np.isclose(cloud["cloud_load_share"].sum(), 1.0, atol=1e-6):
        raise ValueError("Cloud AI-capacity proxy shares must sum to one")
    cloud = cloud.merge(scarcity, on="province", how="left", validate="many_to_one")
    cloud["site_water_m3_large_cloud"] = cloud.cloud_load_share * national_it_energy_twh * cloud_water_l_per_kwh_it * 1e6
    cloud["scarcity_weighted_water_m3_world_eq"] = cloud.site_water_m3_large_cloud * cloud.aware20_nonagri_annual_m3_world_eq_per_m3
    return province, cloud


def load_map(map_path: Path) -> gpd.GeoDataFrame:
    china = gpd.read_file(map_path)
    if "adcode" not in china.columns:
        raise ValueError("Figure 5 requires the complete province_shapes/CHN_full_adm base map")
    china["adcode"] = pd.to_numeric(china["adcode"], errors="coerce").astype("Int64")
    china["province"] = china["adcode"].map(ADCODE_TO_PROVINCE)
    return china.to_crs("+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs")


def projected_extent(china: gpd.GeoDataFrame, lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> tuple[float, float, float, float]:
    corners = gpd.GeoSeries.from_xy(
        [lon_min, lon_max], [lat_min, lat_max], crs="EPSG:4326"
    ).to_crs(china.crs)
    return corners.x.iloc[0], corners.x.iloc[1], corners.y.iloc[0], corners.y.iloc[1]


def draw_province_map(
    ax: plt.Axes,
    china: gpd.GeoDataFrame,
    data: pd.DataFrame,
    water_col: str,
    max_water_m3: float,
    scarcity_cmap,
    scarcity_norm: BoundaryNorm,
) -> None:
    # CHN_full_adm includes the 31 mainland province-level regions used by the
    # model, plus Taiwan, Hong Kong, Macao and the South China Sea boundary line.
    # Keep the full national base visible; only the 31 modeled provinces receive bubbles.
    china.plot(ax=ax, color="#F1F1F1", edgecolor="white", linewidth=.55)
    locations = china[china.province.notna()][["province", "geometry"]].copy()
    locations["point"] = locations.geometry.representative_point()
    mapped = data.merge(locations[["province", "point"]], on="province", how="left", validate="one_to_one")
    mapped = mapped[mapped[water_col].gt(0)].copy()
    top_names = set(mapped.nlargest(7, water_col)["province"])
    for row in mapped.sort_values(water_col).itertuples():
        x, y = row.point.x, row.point.y
        water_m3 = float(getattr(row, water_col))
        scarcity = float(row.aware20_nonagri_annual_m3_world_eq_per_m3)
        ax.scatter(x, y, s=1850*water_m3/max_water_m3, color=scarcity_cmap(scarcity_norm(scarcity)),
                   edgecolor="#555", linewidth=.55, zorder=3)
        if row.province in top_names:
            offset = LABEL_OFFSETS.get(row.province, (4, 5))
            ax.annotate(f"{LABELS[row.province]} {water_m3/1e3:.0f}", (x, y), xytext=offset,
                        textcoords="offset points", fontsize=9, color="#222", zorder=4)
    x0, x1, y0, y1 = projected_extent(china, 72.5, 16.5, 136.0, 54.5)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    taiwan = china.loc[china.adcode.eq(710000), "geometry"]
    if not taiwan.empty:
        point = taiwan.iloc[0].representative_point()
        ax.annotate("台湾\n未纳入31省样本", (point.x, point.y), xytext=(7, 0),
                        textcoords="offset points", fontsize=8.5, color="#666", va="center")
    # Preserve the South China Sea boundary element in a compact inset rather
    # than shrinking the mainland/Taiwan analytical map.
    inset = ax.inset_axes([.02, .08, .14, .24])
    china.plot(ax=inset, color="#F1F1F1", edgecolor="#999", linewidth=.35)
    ix0, ix1, iy0, iy1 = projected_extent(china, 105.0, 2.5, 125.0, 25.0)
    inset.set_xlim(ix0, ix1); inset.set_ylim(iy0, iy1)
    inset.set_axis_off(); inset.set_aspect("equal")
    for spine in inset.spines.values():
        spine.set_visible(True); spine.set_color("#999"); spine.set_linewidth(.45)
    ax.set_axis_off()
    ax.set_aspect("equal")


def plot(province: pd.DataFrame, cloud: pd.DataFrame, map_path: Path, svg: Path, png: Path) -> None:
    configure()
    china = load_map(map_path)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), gridspec_kw={"hspace": .30, "wspace": .23})
    scarcity_bounds = [0, 1, 3, 5, 10, 20, 80.1]
    scarcity_cmap = ListedColormap(["#FFF7BC", "#FEE391", "#FEC44F", "#FE9929", "#E31A1C", "#800026"])
    scarcity_norm = BoundaryNorm(scarcity_bounds, scarcity_cmap.N, clip=True)
    shared_max_water_m3 = max(
        province["site_water_m3_small_local"].max(),
        cloud["site_water_m3_large_cloud"].max(),
    )

    ax = axes[0, 0]; panel(ax, "a", "本地部署：全部省份的估算取水量与水稀缺度")
    draw_province_map(ax, china, province, "site_water_m3_small_local", shared_max_water_m3, scarcity_cmap, scarcity_norm)
    scarcity_sm = plt.cm.ScalarMappable(norm=scarcity_norm, cmap=scarcity_cmap)
    cb=fig.colorbar(scarcity_sm, ax=ax, fraction=.035, pad=.01)
    cb.set_ticks([.5, 2, 4, 7.5, 15, 50])
    cb.set_ticklabels(["<1", "1–3", "3–5", "5–10", "10–20", "≥20"])
    cb.set_label("AWARE2.0水稀缺CF（分级色标）", fontsize=10)
    ax.text(.02,.02,"气泡面积＝省级估算取水量（与c同尺度）；数字为前7省千m³/年",transform=ax.transAxes,fontsize=9,color="#555")

    ax = axes[0, 1]; panel(ax, "b", "本地部署：前7省份的估算取水量与水稀缺度")
    top = province.nlargest(7,"site_water_m3_small_local").sort_values("site_water_m3_small_local")
    cols=[scarcity_cmap(scarcity_norm(v)) for v in top.aware20_nonagri_annual_m3_world_eq_per_m3]
    vals=top.site_water_m3_small_local/1e3
    ax.barh([LABELS[x] for x in top.province], vals, color=cols, alpha=.9)
    ax.set_xlabel("估算取水量（千m³/年）；越红表示越缺水"); ax.grid(axis="x", alpha=.25)
    ax.set_xlim(0, 800)
    for i,(v,cf,weighted) in enumerate(zip(vals,top.aware20_nonagri_annual_m3_world_eq_per_m3,top.scarcity_weighted_water_m3_world_eq/1e6)):
        ax.text(790,i,f"CF={cf:.1f}｜加权={weighted:.2f}百万m³-we",ha="right",va="center",fontsize=9)

    ax = axes[1, 0]; panel(ax, "c", "云服务部署：全国智算容量代理下的取水量与水稀缺度")
    draw_province_map(ax, china, cloud, "site_water_m3_large_cloud", shared_max_water_m3, scarcity_cmap, scarcity_norm)
    ax.text(.02,.02,"气泡面积＝省级估算取水量（与a同尺度）；数字为前7省千m³/年",transform=ax.transAxes,fontsize=9,color="#555")
    scarcity_cb = fig.colorbar(scarcity_sm, ax=ax, fraction=.035, pad=.01)
    scarcity_cb.set_ticks([.5, 2, 4, 7.5, 15, 50])
    scarcity_cb.set_ticklabels(["<1", "1–3", "3–5", "5–10", "10–20", "≥20"])
    scarcity_cb.set_label("AWARE2.0水稀缺CF（分级色标）", fontsize=10)

    ax = axes[1, 1]; panel(ax, "d", "云服务部署：前7省份的估算取水量与水稀缺度")
    top=cloud.nlargest(7,"site_water_m3_large_cloud").sort_values("site_water_m3_large_cloud"); vals=top.site_water_m3_large_cloud/1e3
    cols=[scarcity_cmap(scarcity_norm(v)) for v in top.aware20_nonagri_annual_m3_world_eq_per_m3]
    ax.barh([LABELS[x] for x in top.province],vals,color=cols)
    ax.set_xlabel("估算取水量（千m³/年）；越红表示越缺水");ax.grid(axis="x",alpha=.25)
    ax.set_xlim(0, 800)
    for i,(v,share,cf,weighted) in enumerate(zip(vals,top.cloud_load_share,top.aware20_nonagri_annual_m3_world_eq_per_m3,top.scarcity_weighted_water_m3_world_eq/1e6)):
        ax.text(790,i,f"{share:.0%}｜CF={cf:.1f}｜加权={weighted:.1f}百万m³-we",ha="right",va="center",fontsize=9)

    fig.subplots_adjust(left=.045,right=.985,top=.93,bottom=.055)
    svg.parent.mkdir(parents=True,exist_ok=True); fig.savefig(svg,bbox_inches="tight",pad_inches=.025); fig.savefig(png,bbox_inches="tight",pad_inches=.025,dpi=170); plt.close(fig)


def main() -> None:
    p=argparse.ArgumentParser(description="Build Figure 5; no arguments use the v0.8.0 mainline paths."); p.add_argument("--scenario-registry",type=Path,default=PROJECT_ROOT/"config/scenarios/mainline.yaml"); p.add_argument("--routing-config",type=Path,default=PROJECT_ROOT/"config/compute_hardware/cpu_gpu_routing_v1.yaml"); p.add_argument("--allocation",type=Path,default=PROJECT_ROOT/"02_data/processed/resource_footprint/province_industry_ai_allocation.csv"); p.add_argument("--core-input",type=Path,default=GROUP_NATIONAL_ROOT/"core_scenarios.csv"); p.add_argument("--scarcity",type=Path,default=PROJECT_ROOT/"02_data/processed/resource_footprint/china_province_aware20_nonagri_annual.csv"); p.add_argument("--cloud-share",type=Path,default=PROJECT_ROOT/"02_data/processed/resource_footprint/china_province_cloud_ai_capacity_share_caict2025.csv"); p.add_argument("--map",type=Path,default=PROJECT_ROOT/"02_data/raw/province_shapes/CHN_full_adm/CHN_full_adm.shp"); p.add_argument("--data-output",type=Path,default=FIGURE_ROOT/"figure5_spatial_concentration_data.csv"); p.add_argument("--cloud-data-output",type=Path,default=FIGURE_ROOT/"figure5_cloud_spatial_scenario_data.csv"); p.add_argument("--svg-output",type=Path,default=FIGURE_ROOT/"figure5_spatial_concentration.svg"); p.add_argument("--png-output",type=Path,default=FIGURE_ROOT/"figure5_spatial_concentration.png"); a=p.parse_args()
    registry=yaml.safe_load(a.scenario_registry.read_text(encoding="utf-8")); routing=yaml.safe_load(a.routing_config.read_text(encoding="utf-8")); routing_case=registry["compute_hardware"]["active_routing_case"]
    if routing.get("active_core_routing_case") != routing_case: raise ValueError("Scenario registry and routing config disagree")
    water_case=registry["resource_footprint"]["water"]; water_path=Path(water_case["parameter_file"]); water_path=water_path if water_path.is_absolute() else PROJECT_ROOT/water_path; water=pd.read_csv(water_path,encoding="utf-8-sig").set_index("comparison_mode"); modes=water_case["comparison_modes"]
    province,cloud=prepare(a.allocation,a.core_input,a.scarcity,a.cloud_share,routing_case,float(water.loc[modes["local"],"site_water_use_l_per_kwh_it"]),float(water.loc[modes["china_cloud"],"site_water_use_l_per_kwh_it"])); a.data_output.parent.mkdir(parents=True,exist_ok=True); province.to_csv(a.data_output,index=False,encoding="utf-8-sig"); cloud.to_csv(a.cloud_data_output,index=False,encoding="utf-8-sig"); plot(province,cloud,a.map,a.svg_output,a.png_output)
    print(f"Figure 5 written to {a.svg_output}")

if __name__ == "__main__": main()
