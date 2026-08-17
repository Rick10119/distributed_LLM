#!/usr/bin/env python3
"""Build Figure 4 from the active local and independent cloud mainline outputs."""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "distributed_llm_matplotlib")
)

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = PROJECT_ROOT / "05_results/v0.8.0/result/manuscript_figures"
GROUP_NATIONAL_ROOT = (
    PROJECT_ROOT / "05_results/v0.8.0/result/group_architecture_core/national"
)
GROUP_CORE_ROOT = PROJECT_ROOT / "05_results/v0.8.0/result/group_architecture_core"
CLOUD_ROOT = PROJECT_ROOT / "05_results/sensitivity/v0.8.0/national_cloud_center_v1"
GPU_MAX_WALL_POWER_KW = 1.50
CPU_MAX_WALL_POWER_KW = 0.78
CN_GFA_M2_PER_MW_IT = 1100.0
CN_SITE_M2_PER_MW_IT = 313.0
NATIONAL_DEV_ZONE_BUILT_LAND_HA = 414_900.0
NATIONAL_DEV_ZONE_INDUSTRIAL_LAND_SHARE = 0.4759
NATIONAL_DEV_ZONE_INDUSTRIAL_FAR = 0.99
RACK_USABLE_HEIGHT_U = 42.0
SERVER_HEIGHT_U = 2.0
LOCAL_FLOOR_SCENARIOS = {
    "compact": {"rack_fill": 0.75, "noncompute_rack_share": 0.10, "m2_per_rack": 5.0},
    "central": {"rack_fill": 0.60, "noncompute_rack_share": 0.10, "m2_per_rack": 9.0},
    "spacious": {"rack_fill": 0.50, "noncompute_rack_share": 0.20, "m2_per_rack": 14.0},
}


def local_technical_floor_m2(server_groups: float, scenario: str) -> float:
    """Screen occupied technical floor from 2U servers and a 42U rack."""
    params = LOCAL_FLOOR_SCENARIOS[scenario]
    compute_servers_per_rack = (
        RACK_USABLE_HEIGHT_U / SERVER_HEIGHT_U * params["rack_fill"]
    )
    compute_racks = server_groups / compute_servers_per_rack
    total_racks = compute_racks * (1.0 + params["noncompute_rack_share"])
    return total_racks * params["m2_per_rack"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_industry_summaries(root: Path) -> str:
    """Hash the ordered industry summary files used for the zero-load result."""
    digest = hashlib.sha256()
    paths = sorted(root.glob("C*/summary.csv"))
    if len(paths) != 31:
        raise ValueError(f"Expected 31 industry summaries, found {len(paths)}")
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def zero_load_national_grid_capacity_mw(
    root: Path, core: pd.DataFrame
) -> float:
    """Scale the 31 IG-1host zero-production-load runs to national equivalents."""
    multipliers = (
        core[core["architecture"].eq("IG_1host")]
        .set_index("industry")["industry_equivalent_multiplier"]
        .astype(float)
    )
    values: dict[str, float] = {}
    for path in sorted(root.glob("C*/summary.csv")):
        industry = path.parent.name
        summary = pd.read_csv(path, encoding="utf-8-sig")
        selected = summary[
            summary["architecture"].eq("IG_1host")
            & summary["base_load_case"].eq("zero_load")
        ]
        if len(selected) != 1:
            raise ValueError(
                f"{industry}: expected one IG_1host zero-load row, found {len(selected)}"
            )
        values[industry] = float(selected.iloc[0]["sum_incremental_grid_peak_mw"])
    if set(values) != set(multipliers.index):
        raise ValueError("Zero-load industry coverage does not match the national core table")
    return sum(values[industry] * multipliers.loc[industry] for industry in values)


def parse_water_modes(registry_path: Path) -> dict[str, str]:
    text = registry_path.read_text(encoding="utf-8")
    block = re.search(
        r"comparison_modes:\s*\n\s+local:\s*(\S+)\s*\n\s+china_cloud:\s*(\S+)",
        text,
    )
    if block is None:
        raise ValueError("Cannot locate active local and China-cloud water modes")
    return {"local": block.group(1), "china_cloud": block.group(2)}


def prepare(
    core_path: Path,
    zero_load_root: Path,
    cloud_path: Path,
    registry_path: Path,
    water_path: Path,
    version: str,
) -> pd.DataFrame:
    core = pd.read_csv(core_path, encoding="utf-8-sig")
    cloud = pd.read_csv(cloud_path, encoding="utf-8-sig")
    water = pd.read_csv(water_path, encoding="utf-8-sig").set_index(
        "comparison_mode"
    )
    modes = parse_water_modes(registry_path)

    expected = {"IF", "IG_1host", "IG_multisite"}
    if set(core["architecture"]) != expected:
        raise ValueError("Figure 4 received an obsolete or incomplete architecture table")
    if len(cloud) != 1 or cloud.iloc[0]["architecture"] != "CLOUD_ALL_1HOST":
        raise ValueError("Figure 4 requires the independent national cloud counterfactual")

    rows: list[dict[str, object]] = []
    local_wue = float(water.loc[modes["local"], "site_water_use_l_per_kwh_it"])
    for architecture in ["IF", "IG_1host", "IG_multisite"]:
        selected = core[core["architecture"].eq(architecture)]
        installed_server_groups = float(
            selected["industry_equivalent_installed_gpu_server_groups"].sum()
            + selected["industry_equivalent_installed_cpu_server_groups"].sum()
        )
        energy = float(
            selected["industry_equivalent_annual_ai_facility_energy_twh"].sum()
        )
        rows.append(
            {
                "model_version": version,
                "deployment": architecture,
                "facility_energy_twh": energy,
                "grid_capacity_mw": float(
                    selected["industry_equivalent_sum_incremental_grid_peak_mw"].sum()
                ),
                "water_m3": energy / 1.30 * local_wue * 1e6,
                "wue_l_per_kwh_it": local_wue,
                "installed_server_groups": installed_server_groups,
                "occupied_technical_floor_low_m2": local_technical_floor_m2(
                    installed_server_groups, "compact"
                ),
                "occupied_technical_floor_central_m2": local_technical_floor_m2(
                    installed_server_groups, "central"
                ),
                "occupied_technical_floor_high_m2": local_technical_floor_m2(
                    installed_server_groups, "spacious"
                ),
                "equivalent_land_central_m2_at_dev_zone_far": local_technical_floor_m2(
                    installed_server_groups, "central"
                )
                / NATIONAL_DEV_ZONE_INDUSTRIAL_FAR,
                "share_of_national_dev_zone_industrial_land_central": (
                    local_technical_floor_m2(installed_server_groups, "central")
                    / NATIONAL_DEV_ZONE_INDUSTRIAL_FAR
                    / (
                        NATIONAL_DEV_ZONE_BUILT_LAND_HA
                        * 10_000.0
                        * NATIONAL_DEV_ZONE_INDUSTRIAL_LAND_SHARE
                    )
                ),
                "new_gfa_m2": 0.0,
                "land_conversion_m2": 0.0,
                "evidence": "31_industry_group_core",
            }
        )

    rows.append(
        {
            "model_version": version,
            "deployment": "IG_1host_zero_load",
            "facility_energy_twh": pd.NA,
            "grid_capacity_mw": zero_load_national_grid_capacity_mw(
                zero_load_root, core
            ),
            "water_m3": pd.NA,
            "wue_l_per_kwh_it": pd.NA,
            "installed_server_groups": pd.NA,
            "occupied_technical_floor_low_m2": pd.NA,
            "occupied_technical_floor_central_m2": pd.NA,
            "occupied_technical_floor_high_m2": pd.NA,
            "equivalent_land_central_m2_at_dev_zone_far": pd.NA,
            "share_of_national_dev_zone_industrial_land_central": pd.NA,
            "new_gfa_m2": pd.NA,
            "land_conversion_m2": pd.NA,
            "evidence": "31_industry_IG_1host_zero_production_load_reoptimized",
        }
    )

    cloud_row = cloud.iloc[0]
    cloud_energy = float(cloud_row["annual_ai_facility_energy_twh"])
    cloud_wue = float(
        water.loc[modes["china_cloud"], "site_water_use_l_per_kwh_it"]
    )
    installed_it_capacity_mw = (
        float(cloud_row["installed_gpu_server_groups"]) * GPU_MAX_WALL_POWER_KW
        + float(cloud_row["installed_cpu_server_groups"]) * CPU_MAX_WALL_POWER_KW
    ) / 1000.0
    rows.append(
        {
            "model_version": version,
            "deployment": "CLOUD_ALL_1HOST",
            "facility_energy_twh": cloud_energy,
            "grid_capacity_mw": float(cloud_row["incremental_grid_expansion_mw"]),
            "water_m3": cloud_energy / 1.22 * cloud_wue * 1e6,
            "wue_l_per_kwh_it": cloud_wue,
            "installed_server_groups": float(cloud_row["installed_gpu_server_groups"])
            + float(cloud_row["installed_cpu_server_groups"]),
            "occupied_technical_floor_low_m2": installed_it_capacity_mw
            * CN_GFA_M2_PER_MW_IT,
            "occupied_technical_floor_central_m2": installed_it_capacity_mw
            * CN_GFA_M2_PER_MW_IT,
            "occupied_technical_floor_high_m2": installed_it_capacity_mw
            * CN_GFA_M2_PER_MW_IT,
            "equivalent_land_central_m2_at_dev_zone_far": pd.NA,
            "share_of_national_dev_zone_industrial_land_central": pd.NA,
            "new_gfa_m2": installed_it_capacity_mw * CN_GFA_M2_PER_MW_IT,
            "land_conversion_m2": installed_it_capacity_mw * CN_SITE_M2_PER_MW_IT,
            "evidence": "independent_national_cloud_counterfactual",
        }
    )
    return pd.DataFrame(rows)


def load_fonts() -> dict[str, ImageFont.FreeTypeFont]:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    ]
    font_path = next((path for path in candidates if path.exists()), None)
    if font_path is None:
        raise FileNotFoundError("A Chinese-capable system font is required")
    return {
        "title": ImageFont.truetype(str(font_path), 42),
        "panel": ImageFont.truetype(str(font_path), 32),
        "label": ImageFont.truetype(str(font_path), 25),
        "small": ImageFont.truetype(str(font_path), 22),
        "note": ImageFont.truetype(str(font_path), 19),
    }


def plot(data: pd.DataFrame, svg_path: Path, png_path: Path) -> None:
    label_lines = {
        "IF": ["逐厂独立"],
        "IG_1host_zero_load": ["集团单节点", "未配合生产负荷"],
        "IG_1host": ["集团单节点", "配合生产负荷"],
        "IG_multisite": ["集团多节点"],
        "CLOUD_ALL_1HOST": ["绿地大型云"],
    }
    palette = {
        "IF": "#31688E",
        "IG_1host_zero_load": "#A6B8B3",
        "IG_1host": "#4C9A88",
        "IG_multisite": "#78B36A",
        "CLOUD_ALL_1HOST": "#7B4F9D",
    }
    width, height = 2400, 880
    background, foreground, grid_color = "#FFFFFF", "#1B1F24", "#D9DDE3"
    fonts = load_fonts()
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{background}"/>',
    ]

    def add_text(
        x: float,
        y: float,
        text: str,
        font_key: str,
        fill: str = foreground,
        anchor: str = "start",
    ) -> None:
        font = fonts[font_key]
        if anchor == "middle":
            box = draw.textbbox((0, 0), text, font=font)
            draw_x = x - (box[2] - box[0]) / 2
        elif anchor == "end":
            box = draw.textbbox((0, 0), text, font=font)
            draw_x = x - (box[2] - box[0])
        else:
            draw_x = x
        draw.text((draw_x, y), text, font=font, fill=fill)
        size = {"title": 42, "panel": 32, "label": 25, "small": 22, "note": 19}[font_key]
        weight = "600" if font_key in {"title", "panel"} else "400"
        svg.append(
            f'<text x="{x}" y="{y + size}" fill="{fill}" font-family="Arial Unicode MS, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{escape(text)}</text>'
        )

    def add_line(x1: float, y1: float, x2: float, y2: float, fill: str, line_width: int = 2) -> None:
        draw.line((x1, y1, x2, y2), fill=fill, width=line_width)
        svg.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{fill}" stroke-width="{line_width}"/>'
        )

    def add_rect(x: float, y: float, rect_width: float, rect_height: float, fill: str) -> None:
        draw.rectangle((x, y, x + rect_width, y + rect_height), fill=fill)
        svg.append(
            f'<rect x="{x}" y="{y}" width="{rect_width}" height="{rect_height}" fill="{fill}"/>'
        )

    def draw_waterfall_panel(
        left: int,
        title: str,
        rows: list[dict[str, object]],
        axis_title: str,
        maximum: float,
        ticks: list[float],
    ) -> None:
        plot_left, plot_top = left + 90, 275
        plot_width, plot_height = 880, 405
        add_text(left, 145, title, "panel")
        for tick in ticks:
            y = plot_top + plot_height * (1.0 - tick / maximum)
            add_line(plot_left, y, plot_left + plot_width, y, grid_color, 2)
            add_text(plot_left - 18, y - 14, f"{tick:g}", "small", "#59616B", "end")
        add_line(plot_left, plot_top, plot_left, plot_top + plot_height, foreground, 2)
        add_line(
            plot_left,
            plot_top + plot_height,
            plot_left + plot_width,
            plot_top + plot_height,
            foreground,
            2,
        )

        group_width = plot_width / len(rows)
        bar_width = 104
        for index, row in enumerate(rows):
            center_x = plot_left + group_width * (index + 0.5)
            value = float(row["value"])
            previous = 0.0 if index == 0 else float(rows[index - 1]["value"])
            lower, upper = min(previous, value), max(previous, value)
            y_upper = plot_top + plot_height * (1.0 - upper / maximum)
            y_lower = plot_top + plot_height * (1.0 - lower / maximum)
            bar_height = max(3.0, y_lower - y_upper)
            add_rect(
                center_x - bar_width / 2,
                y_upper,
                bar_width,
                bar_height,
                str(row["color"]),
            )
            if index > 0:
                previous_x = plot_left + group_width * (index - 0.5) + bar_width / 2
                connector_y = plot_top + plot_height * (1.0 - previous / maximum)
                add_line(previous_x, connector_y, center_x - bar_width / 2, connector_y, "#8A9199", 2)
            endpoint_y = plot_top + plot_height * (1.0 - value / maximum)
            add_text(center_x, max(plot_top - 6, endpoint_y - 34), f"{value:.3f}", "small", foreground, "middle")
            if index > 0:
                delta = value - previous
                delta_label = f"{delta:+.3f}"
                delta_y = (y_upper + y_lower) / 2 - 13
                if bar_height < 45:
                    delta_y = y_lower + 5
                add_text(center_x, delta_y, delta_label, "note", "#59616B", "middle")
            lines = list(row["label_lines"])
            first_y = 690 if len(lines) > 1 else 705
            for line_index, line in enumerate(lines):
                add_text(
                    center_x,
                    first_y + line_index * 34,
                    str(line),
                    "label" if line_index == 0 else "note",
                    foreground,
                    "middle",
                )
        add_text(plot_left, 237, axis_title, "small", "#59616B")
        add_text(
            plot_left + plot_width / 2,
            778,
            "相邻情景差额；仅‘未配合→配合生产负荷’为严格配对识别",
            "note",
            "#59616B",
            "middle",
        )

    add_text(width / 2, 44, "本地部署降低相同制造业AI服务的新增基础设施需求", "title", foreground, "middle")

    grid_cloud = float(
        data.loc[data["deployment"].eq("CLOUD_ALL_1HOST"), "grid_capacity_mw"].iloc[0]
    )
    grid_rows: list[dict[str, object]] = []
    grid_order = [
        "CLOUD_ALL_1HOST",
        "IG_1host_zero_load",
        "IG_1host",
        "IF",
        "IG_multisite",
    ]
    grid_data = data.set_index("deployment").loc[grid_order].reset_index()
    for row in grid_data.itertuples(index=False):
        value = float(row.grid_capacity_mw) / 1000.0
        grid_rows.append(
            {
                "label_lines": label_lines[row.deployment],
                "value": value,
                "color": palette[row.deployment],
            }
        )
    draw_waterfall_panel(
        80,
        "a  新增电网接入容量的情景路径",
        grid_rows,
        "新增接入容量（GW）",
        2.2,
        [0, 0.5, 1.0, 1.5, 2.0],
    )

    local = data[
        data["deployment"].isin(["IF", "IG_1host", "IG_multisite"])
    ]
    cloud = data[data["deployment"].eq("CLOUD_ALL_1HOST")].iloc[0]
    local_water = float(local["water_m3"].mean()) / 1e6
    cloud_water = float(cloud["water_m3"]) / 1e6
    local_floor = float(local["occupied_technical_floor_central_m2"].mean()) / 1e6
    local_floor_low = float(local["occupied_technical_floor_low_m2"].mean()) / 1e6
    local_floor_high = float(local["occupied_technical_floor_high_m2"].mean()) / 1e6
    cloud_gfa = float(cloud["new_gfa_m2"]) / 1e6
    cloud_land = float(cloud["land_conversion_m2"]) / 1e6
    normalized_rows = [
        {"label": "现场水", "unit": "百万m³/年", "local": local_water, "cloud": cloud_water},
        {
            "label": "占用技术楼面",
            "unit": "百万m²",
            "local": local_floor,
            "cloud": cloud_gfa,
            "local_low": local_floor_low,
            "local_high": local_floor_high,
        },
        {"label": "土地转换", "unit": "百万m²", "local": 0.0, "cloud": cloud_land},
    ]
    panel_left, plot_left, plot_top = 1270, 1390, 275
    plot_width, plot_height = 850, 405
    add_text(panel_left, 145, "b  资源与建设需求（大型云 = 100%）", "panel")
    add_rect(1840, 205, 26, 18, palette["IF"])
    add_text(1878, 198, "本地工厂节点", "small")
    add_rect(2070, 205, 26, 18, palette["CLOUD_ALL_1HOST"])
    add_text(2108, 198, "绿地大型云", "small")
    for tick in [0, 25, 50, 75, 100]:
        y = plot_top + plot_height * (1.0 - tick / 100.0)
        add_line(plot_left, y, plot_left + plot_width, y, grid_color, 2)
        add_text(plot_left - 18, y - 14, f"{tick}%", "small", "#59616B", "end")
    add_line(plot_left, plot_top, plot_left, plot_top + plot_height, foreground, 2)
    add_line(plot_left, plot_top + plot_height, plot_left + plot_width, plot_top + plot_height, foreground, 2)
    group_width = plot_width / len(normalized_rows)
    bar_width = 76
    for index, row in enumerate(normalized_rows):
        label, unit = str(row["label"]), str(row["unit"])
        local_abs, cloud_abs = float(row["local"]), float(row["cloud"])
        center = plot_left + group_width * (index + 0.5)
        values = [local_abs / cloud_abs * 100.0 if cloud_abs else 0.0, 100.0]
        colors = [palette["IF"], palette["CLOUD_ALL_1HOST"]]
        absolute_values = [local_abs, cloud_abs]
        for series_index, (offset, value, color, absolute) in enumerate(
            zip([-48, 48], values, colors, absolute_values)
        ):
            x = center + offset - bar_width / 2
            bar_height = plot_height * value / 100.0
            if series_index == 0 and "local_low" in row:
                low = float(row["local_low"]) / cloud_abs * 100.0
                high = float(row["local_high"]) / cloud_abs * 100.0
                whisker_x = x + bar_width / 2
                y_low = plot_top + plot_height * (1.0 - low / 100.0)
                y_high = plot_top + plot_height * (1.0 - min(high, 100.0) / 100.0)
                add_line(whisker_x, y_high, whisker_x, y_low, foreground, 4)
                add_line(whisker_x - 22, y_high, whisker_x + 22, y_high, foreground, 4)
                add_line(whisker_x - 22, y_low, whisker_x + 22, y_low, foreground, 4)
                if high > 100.0:
                    add_text(whisker_x, plot_top - 35, f"上限{high:.0f}%", "note", "#59616B", "middle")
            if value > 0:
                add_rect(x, plot_top + plot_height - bar_height, bar_width, bar_height, color)
                label_y = max(plot_top - 6, plot_top + plot_height - bar_height - 34)
            else:
                add_line(x, plot_top + plot_height - 2, x + bar_width, plot_top + plot_height - 2, color, 6)
                label_y = plot_top + plot_height - 34
            suffix = "*" if value == 0 else ""
            add_text(x + bar_width / 2, label_y, f"{absolute:.3f}{suffix}", "small", foreground, "middle")
        add_text(center, 705, label, "label", foreground, "middle")
        add_text(center, 738, f"（{unit}）", "note", "#59616B", "middle")
        if label == "占用技术楼面":
            add_text(center, 774, "折算土地占比≈0.06%", "note", "#59616B", "middle")
    add_text(plot_left, 237, "相对绿地大型云需求", "small", "#59616B")

    add_text(
        width / 2,
        835,
        "注：a为相邻情景路径而非单因素贡献分解，首步同时改变汇聚尺度、PUE与规划边界；技术楼面误差线为规范锚定范围；按国家级开发区工业容积率0.99折算，本地中央值约116 ha，占其工业用地约0.06%；*0表示由原有厂区内部消化。",
        "note",
        "#59616B",
        "middle",
    )
    svg.append("</svg>")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    image.save(png_path, format="PNG", dpi=(300, 300))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the two-panel Figure 4 from v0.8.0 mainline outputs."
    )
    parser.add_argument(
        "--core-input",
        type=Path,
        default=GROUP_NATIONAL_ROOT / "core_scenarios.csv",
    )
    parser.add_argument(
        "--zero-load-root",
        type=Path,
        default=GROUP_CORE_ROOT,
    )
    parser.add_argument(
        "--cloud-input", type=Path, default=CLOUD_ROOT / "summary.csv"
    )
    parser.add_argument(
        "--scenario-registry",
        type=Path,
        default=PROJECT_ROOT / "config/scenarios/mainline.yaml",
    )
    parser.add_argument(
        "--water-input",
        type=Path,
        default=PROJECT_ROOT
        / "02_data/processed/resource_footprint/small_china_us_water_baseline_comparison.csv",
    )
    parser.add_argument("--model-version", default="v0.8.0")
    parser.add_argument(
        "--data-output",
        type=Path,
        default=FIGURE_ROOT / "figure4_resource_footprint_data.csv",
    )
    parser.add_argument(
        "--svg-output",
        type=Path,
        default=FIGURE_ROOT / "figure4_resource_footprint.svg",
    )
    parser.add_argument(
        "--png-output",
        type=Path,
        default=FIGURE_ROOT / "figure4_resource_footprint.png",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=FIGURE_ROOT / "figure4_resource_footprint.validated.done.json",
    )
    args = parser.parse_args()

    input_paths = {
        "core": args.core_input,
        "cloud": args.cloud_input,
        "scenario_registry": args.scenario_registry,
        "water": args.water_input,
    }
    data = prepare(
        args.core_input,
        args.zero_load_root,
        args.cloud_input,
        args.scenario_registry,
        args.water_input,
        args.model_version,
    )
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.data_output, index=False, encoding="utf-8-sig")
    plot(data, args.svg_output, args.png_output)

    validation = {
        "status": "validated",
        "model_version": args.model_version,
        "panels": ["incremental_grid_capacity", "normalized_water_technical_floor_land"],
        "local_floor_method": "2U servers; 42U racks; compact-central-spacious standard-anchored screening range",
        "II_1host_in_figure": False,
        "IG_1host_zero_production_load_in_panel_a": True,
        "zero_load_definition": "same AI service and constraints; host production load set to zero; AI schedule reoptimized",
        "cloud_source": "independent_national_cloud_scenario",
        "input_sha256": {name: sha256(path) for name, path in input_paths.items()},
        "zero_load_industry_summaries_sha256": sha256_industry_summaries(
            args.zero_load_root
        ),
    }
    args.validation_output.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Figure 4 written to {args.svg_output}")


if __name__ == "__main__":
    main()
