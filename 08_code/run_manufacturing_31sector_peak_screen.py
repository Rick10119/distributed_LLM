"""HISTORICAL equal-electricity peak screen across 31 sectors.

This script only reproduces archived pre-v0.2.0 screening artifacts. Active
national results use the versioned Snakemake equal-service workflow.

Curve hierarchy:
1. exact/specific external curve where it materially improves sector identity;
2. China EWELD curve for the closest ISIC Rev.4 division;
3. six-archetype EWELD fallback.

The analysis reports absolute AI peaks and a representative-weekday combined
manufacturing peak.  The 2030 sector electricity baseline comes from the local
d_energy.csv REFERENCE scenario and is therefore a scenario scale anchor, not
an official observed statistical table.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "02_data"
RESULTS = ROOT / "05_results"
LEGACY_RESULTS = RESULTS / "archive" / "equal_electricity_national"
FIGURES = LEGACY_RESULTS / "figures"
sys.path.insert(0, str(ROOT / "08_code"))

import build_manufacturing_load_archetypes as load_builder  # noqa: E402


TASKS = ["office", "agent", "vision", "maintenance", "scheduling", "simulation"]
TASK_BASE_GPU_H = {
    "office": 10.0,
    "agent": 12.0,
    "vision": 4.0,
    "maintenance": 8.0,
    "scheduling": 7.0,
    "simulation": 16.0,
}
TEMPORAL_SCENARIOS = {
    "task_timed": "按任务时序",
    "flat": "完全平滑",
    "production_synchronous": "完全跟随生产",
}
ENERGY_SCENARIOS = {
    "lower_8twh": "lower_8twh_allocation_twh",
    "central_14twh": "central_14twh_allocation_twh",
    "upper_28twh": "upper_28twh_allocation_twh",
}
ARCHETYPE_CN = load_builder.ARCHETYPE_CN
COLORS = load_builder.ARCHETYPE_COLORS


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if not math.isfinite(mean) or mean <= 0:
        raise ValueError("Profile mean must be positive")
    return values / mean


def median_profiles(rows: list[dict], key_name: str) -> dict[tuple[object, str], np.ndarray]:
    grouped: dict[tuple[object, str], list[np.ndarray]] = defaultdict(list)
    for row in rows:
        for day_type, values in row["profiles"].items():
            if float(np.mean(values)) > 0:
                grouped[(row[key_name], day_type)].append(normalize(values))
    return {key: normalize(np.median(np.vstack(values), axis=0)) for key, values in grouped.items()}


def ffe_branch_profile(branch_id: int) -> np.ndarray:
    payload = json.loads((DATA / "raw_load_profiles" / "ffe" / "id_opendata_59_year_2017.json").read_text(encoding="utf-8"))
    item = next(row for row in payload if int(row["internal_id_1"]) == branch_id)
    timestamps = pd.date_range("2017-01-01 00:00", periods=8760, freq="h")
    frame = pd.DataFrame({"time": timestamps, "value": item["values"]})
    frame = frame[frame["time"].dt.dayofweek < 5].copy()
    return normalize(frame.groupby(frame["time"].dt.hour)["value"].median().reindex(range(24)).to_numpy())


def build_curve_library() -> tuple[dict[int, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, Counter]:
    eweld_rows, archetype_profiles = load_builder.read_eweld()
    korea_rows, _ = load_builder.read_korea()
    uci_rows, _ = load_builder.read_uci_steel()
    isic_profiles = median_profiles(eweld_rows, "isic_division")
    isic_weekday = {int(key[0]): value for key, value in isic_profiles.items() if key[1] == "weekday"}
    archetype_weekday = {
        archetype: normalize(values)
        for (archetype, day_type), values in archetype_profiles.items()
        if day_type == "weekday"
    }
    isic_counts = Counter(row["isic_division"] for row in eweld_rows)

    steel_profiles = []
    for row in korea_rows:
        if row["user"].startswith("Steel_"):
            steel_profiles.append(normalize(row["profiles"]["weekday"]))
    for row in uci_rows:
        steel_profiles.append(normalize(row["profiles"]["weekday"]))
    dedicated_steel = normalize(np.median(np.vstack(steel_profiles), axis=0))
    dedicated_transport = ffe_branch_profile(5)
    return isic_weekday, archetype_weekday, dedicated_steel, dedicated_transport, isic_counts


def build_curve_selection(
    crosswalk: pd.DataFrame,
    isic_profiles: dict[int, np.ndarray],
    archetype_profiles: dict[str, np.ndarray],
    steel: np.ndarray,
    transport: np.ndarray,
    isic_counts: Counter,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    curves = {}
    rows = []
    for record in crosswalk.to_dict("records"):
        code = record["china_code"]
        isic = int(record["closest_isic_rev4_division"])
        archetype = record["primary_archetype"]
        if code == "C31":
            curve = steel
            source_type = "dedicated_external_facility"
            source_id = "Korea_Steel_1_2_plus_UCI_steel"
            sample_count = 3
            evidence_grade = "C"
            rationale = "黑色金属使用三家专属钢铁企业曲线，行业身份更精确，但属于跨国小样本。"
        elif code == "C37":
            curve = transport
            source_type = "dedicated_external_sector_model"
            source_id = "FfE_transport_equipment_2017"
            sample_count = 1
            evidence_grade = "C"
            rationale = "EWELD缺少ISIC 30，使用FfE运输设备行业曲线，属于德国合成行业曲线。"
        elif isic in isic_profiles:
            curve = isic_profiles[isic]
            source_type = "china_eweld_isic_specific"
            source_id = f"EWELD_ISIC_{isic:02d}"
            sample_count = int(isic_counts[isic])
            evidence_grade = "B" if sample_count >= 5 else "C"
            rationale = "使用与中国行业最接近的EWELD ISIC大类设施曲线。"
        else:
            curve = archetype_profiles[archetype]
            source_type = "six_archetype_fallback"
            source_id = f"EWELD_archetype_{archetype}"
            sample_count = 0
            evidence_grade = "D"
            rationale = "尚无同一ISIC或更专属曲线，回退到生产过程相近的六类原型。"
        curves[code] = normalize(curve)
        rows.append(
            {
                "industry_code": code,
                "industry_name_cn": record["china_industry"],
                "selected_curve_source_type": source_type,
                "selected_curve_source_id": source_id,
                "selected_curve_sample_count": sample_count,
                "curve_evidence_grade": evidence_grade,
                "primary_archetype": archetype,
                "closest_isic_rev4_division": isic,
                "selection_rationale": rationale,
            }
        )
    return curves, rows


def parse_hourly_workload_shapes() -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(DATA / "future_manufacturing_ai_workload_model.csv", encoding="utf-8-sig")
    frame = frame[(frame["record_type"] == "hourly_curve") & (frame["scenario"] == "2030_integrated")].copy()
    frame = frame.sort_values("hour")

    def extract(name: str) -> np.ndarray:
        values = []
        pattern = re.compile(rf"(?:^|; )?{re.escape(name)}=([0-9.]+)")
        for text in frame["derivation_or_definition"].astype(str):
            match = pattern.search(text)
            values.append(float(match.group(1)) if match else 0.0)
        return np.asarray(values)

    office = normalize(extract("human_tasks") + 0.05)
    agent = normalize(extract("transaction_tasks") + extract("batch_tasks") + 0.10)
    return office, agent


def fixed_task_shapes(base: np.ndarray, office: np.ndarray, agent: np.ndarray) -> dict[str, np.ndarray]:
    scheduling = np.full(24, 0.20)
    for hour, value in {5: 1.2, 6: 1.6, 7: 1.0, 13: 1.0, 14: 1.5, 15: 1.0, 21: 1.0, 22: 1.5, 23: 1.2}.items():
        scheduling[hour] = value
    simulation = np.full(24, 0.10)
    simulation[[0, 1, 2, 3, 4, 5, 22, 23]] = 1.0
    return {
        "office": office,
        "agent": agent,
        "vision": normalize(base),
        "maintenance": normalize(0.75 * normalize(base) + 0.25),
        "scheduling": normalize(scheduling),
        "simulation": normalize(simulation),
    }


def build_ai_profile(
    base: np.ndarray,
    parameter: dict,
    edge_share: float,
    temporal_scenario: str,
    office_shape: np.ndarray,
    agent_shape: np.ndarray,
) -> np.ndarray:
    if temporal_scenario == "flat":
        return np.ones(24)
    if temporal_scenario == "production_synchronous":
        return normalize(base)
    shapes = fixed_task_shapes(base, office_shape, agent_shape)
    weights = np.array(
        [TASK_BASE_GPU_H[task] * float(parameter[f"{task}_multiplier"]) for task in TASKS],
        dtype=float,
    )
    central = sum(weights[i] * shapes[task] for i, task in enumerate(TASKS)) / weights.sum()
    # Edge devices are assumed to follow the production-linked vision shape.
    combined = (1 - edge_share) * normalize(central) + edge_share * normalize(base)
    return normalize(combined)


def circular_hour_gap(left: int, right: int) -> int:
    raw = abs(left - right)
    return min(raw, 24 - raw)


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def shape_metrics(base: np.ndarray, ai: np.ndarray) -> dict[str, float]:
    base_peak_hour = int(np.argmax(base))
    ai_peak_hour = int(np.argmax(ai))
    base_peak = float(np.max(base))
    ai_peak = float(np.max(ai))
    result = {
        "base_peak_factor": base_peak,
        "base_peak_hour": base_peak_hour,
        "ai_peak_factor": ai_peak,
        "ai_peak_hour": ai_peak_hour,
        "peak_hour_gap": circular_hour_gap(base_peak_hour, ai_peak_hour),
        "ai_at_base_peak_fraction_of_ai_peak": float(ai[base_peak_hour] / ai_peak),
        "base_at_ai_peak_fraction_of_base_peak": float(base[ai_peak_hour] / base_peak),
        "base_ai_shape_correlation": pearson(base, ai),
    }
    for ratio in (0.01, 0.05, 0.10, 0.25):
        combined = base + ratio * ai
        combined_peak = float(np.max(combined))
        increment = combined_peak - base_peak
        tag = str(int(ratio * 100))
        result[f"combined_peak_index_aiavg_{tag}pct_of_baseavg"] = combined_peak
        result[f"incremental_peak_per_ai_average_{tag}pct"] = increment / ratio
        result[f"combined_peak_hour_aiavg_{tag}pct"] = int(np.argmax(combined))
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def line_points(values: np.ndarray, x: float, y: float, w: float, h: float, ymin: float, ymax: float) -> str:
    return " ".join(
        f"{x + w * hour / 23:.1f},{y + h * (ymax - float(value)) / (ymax - ymin):.1f}"
        for hour, value in enumerate(values)
    )


def build_example_svg(hourly: pd.DataFrame, selection: pd.DataFrame) -> None:
    codes = ["C13", "C22", "C31", "C34", "C37", "C39"]
    display_titles = {
        "C13": "C13 农副食品加工",
        "C22": "C22 造纸和纸制品",
        "C31": "C31 黑色金属冶炼",
        "C34": "C34 通用设备",
        "C37": "C37 其他运输设备",
        "C39": "C39 电子设备",
    }
    source_labels = {
        "china_eweld_isic_specific": "EWELD中国行业专属",
        "dedicated_external_facility": "海外钢铁设施专属",
        "dedicated_external_sector_model": "FfE运输设备专属",
        "six_archetype_fallback": "六类原型回退",
    }
    width, height = 1200, 790
    origins = [(55, 105), (425, 105), (795, 105), (55, 425), (425, 425), (795, 425)]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;fill:#263238}.small{font-size:12px;fill:#607d8b}.title{font-size:24px;font-weight:650}.panel{font-size:16px;font-weight:600}</style>',
        '<text x="52" y="42" class="title">六个典型行业：原生产负荷与按任务时序AI负荷</text>',
        '<text x="52" y="66" class="small">两条曲线均按各自平均负荷归一化；用于判断峰值时段与同时性，不表示AI与原负荷绝对比例</text>',
    ]
    for code, (x, y) in zip(codes, origins):
        subset = hourly[(hourly["industry_code"] == code) & (hourly["temporal_scenario"] == "task_timed")].sort_values("hour")
        base = subset["base_normalized_load"].to_numpy()
        ai = subset["ai_normalized_load"].to_numpy()
        row = selection[selection["industry_code"] == code].iloc[0]
        title = display_titles[code]
        source = source_labels[row["selected_curve_source_type"]]
        panel_w, panel_h = 350, 275
        px, py, pw, ph = x + 42, y + 45, panel_w - 58, panel_h - 82
        ymin, ymax = 0, max(float(base.max()), float(ai.max())) + 0.15
        parts.append(f'<rect x="{x}" y="{y}" width="{panel_w}" height="{panel_h}" rx="12" fill="#fafbfc" stroke="#dfe5e8"/>')
        parts.append(f'<text x="{x + 16}" y="{y + 24}" class="panel">{title}</text>')
        parts.append(f'<text x="{x + 16}" y="{y + 42}" class="small">{source}</text>')
        for tick in (0, 6, 12, 18, 23):
            tx = px + pw * tick / 23
            parts.append(f'<line x1="{tx:.1f}" y1="{py}" x2="{tx:.1f}" y2="{py + ph}" stroke="#edf0f2"/>')
            parts.append(f'<text x="{tx:.1f}" y="{py + ph + 17}" text-anchor="middle" class="small">{tick}</text>')
        mean_y = py + ph * (ymax - 1) / (ymax - ymin)
        parts.append(f'<line x1="{px}" y1="{mean_y:.1f}" x2="{px + pw}" y2="{mean_y:.1f}" stroke="#b0bec5" stroke-dasharray="3 4"/>')
        parts.append(f'<polyline points="{line_points(base, px, py, pw, ph, ymin, ymax)}" fill="none" stroke="#455a64" stroke-width="3"/>')
        parts.append(f'<polyline points="{line_points(ai, px, py, pw, ph, ymin, ymax)}" fill="none" stroke="#d97706" stroke-width="3"/>')
        parts.append(f'<line x1="{x + 18}" y1="{y + panel_h - 13}" x2="{x + 38}" y2="{y + panel_h - 13}" stroke="#455a64" stroke-width="3"/><text x="{x + 44}" y="{y + panel_h - 9}" class="small">原负荷</text>')
        parts.append(f'<line x1="{x + 105}" y1="{y + panel_h - 13}" x2="{x + 125}" y2="{y + panel_h - 13}" stroke="#d97706" stroke-width="3"/><text x="{x + 131}" y="{y + panel_h - 9}" class="small">AI负荷</text>')
    parts.append("</svg>")
    (FIGURES / "manufacturing_31sector_peak_example_profiles.svg").write_text("\n".join(parts), encoding="utf-8")


def build_top_sector_svg(summary: pd.DataFrame) -> None:
    top = summary.sort_values("central_ai_peak_mw", ascending=False).head(15).sort_values("central_ai_peak_mw")
    width, height = 1180, 690
    left, top_y, bar_h, gap = 350, 75, 14, 35
    max_peak = float(top["central_ai_peak_mw"].max())
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;fill:#263238}.small{font-size:12px;fill:#607d8b}.title{font-size:24px;font-weight:650}.label{font-size:13px}</style>',
        '<text x="42" y="38" class="title">2030年中心情景：AI平均负荷与任务时序峰值</text>',
        '<text x="42" y="59" class="small">14 TWh制造业AI用电情景；峰值只表示AI自身，不包含原有制造业绝对负荷</text>',
    ]
    for i, row in enumerate(top.to_dict("records")):
        y = top_y + i * gap
        label = f'{row["industry_code"]} {row["industry_name_cn"]}'
        avg_w = 700 * float(row["central_ai_average_mw"]) / max_peak
        peak_w = 700 * float(row["central_ai_peak_mw"]) / max_peak
        parts.append(f'<text x="42" y="{y + 12}" class="label">{label}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{peak_w:.1f}" height="{bar_h}" rx="7" fill="#f0ad4e" opacity="0.45"/>')
        parts.append(f'<rect x="{left}" y="{y + 3}" width="{avg_w:.1f}" height="{bar_h - 6}" rx="4" fill="#2a9d8f"/>')
        parts.append(f'<text x="{left + peak_w + 8:.1f}" y="{y + 12}" class="small">峰值 {row["central_ai_peak_mw"]:.1f} MW</text>')
    parts.append(f'<rect x="42" y="{height - 48}" width="18" height="8" rx="4" fill="#2a9d8f"/><text x="68" y="{height - 39}" class="small">平均负荷</text>')
    parts.append(f'<rect x="145" y="{height - 51}" width="18" height="14" rx="7" fill="#f0ad4e" opacity="0.45"/><text x="171" y="{height - 39}" class="small">峰值负荷</text>')
    parts.append("</svg>")
    (FIGURES / "manufacturing_31sector_ai_average_peak.svg").write_text("\n".join(parts), encoding="utf-8")


def build_aggregate_svg(aggregate: pd.DataFrame) -> None:
    central = aggregate[aggregate["energy_scenario"] == "central_14twh"]
    width, height = 1100, 520
    x, y, w, h = 85, 90, 940, 330
    ymax = float(central["aggregate_ai_load_mw"].max()) * 1.08
    colors = {"task_timed": "#d97706", "flat": "#2a9d8f", "production_synchronous": "#6d597a"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;fill:#263238}.small{font-size:12px;fill:#607d8b}.title{font-size:24px;font-weight:650}</style>',
        '<text x="50" y="40" class="title">全国制造业AI负荷的三种时序边界</text>',
        '<text x="50" y="63" class="small">31行业、14 TWh中心情景；未考虑跨时区或需求响应调度</text>',
    ]
    for frac in (0, 0.25, 0.5, 0.75, 1):
        yy = y + h * (1 - frac)
        parts.append(f'<line x1="{x}" y1="{yy:.1f}" x2="{x + w}" y2="{yy:.1f}" stroke="#edf0f2"/>')
        parts.append(f'<text x="{x - 10}" y="{yy + 4:.1f}" text-anchor="end" class="small">{ymax * frac:.0f}</text>')
    for hour in (0, 6, 12, 18, 23):
        xx = x + w * hour / 23
        parts.append(f'<text x="{xx:.1f}" y="{y + h + 22}" text-anchor="middle" class="small">{hour}</text>')
    legend_x = 620
    for scenario, label in TEMPORAL_SCENARIOS.items():
        subset = central[central["temporal_scenario"] == scenario].sort_values("hour")
        values = subset["aggregate_ai_load_mw"].to_numpy()
        points = line_points(values, x, y, w, h, 0, ymax)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[scenario]}" stroke-width="4" stroke-linejoin="round"/>')
        parts.append(f'<line x1="{legend_x}" y1="54" x2="{legend_x + 24}" y2="54" stroke="{colors[scenario]}" stroke-width="4"/><text x="{legend_x + 31}" y="58" class="small">{label}</text>')
        legend_x += 145
    parts.append('<text x="25" y="260" transform="rotate(-90 25 260)" class="small">AI负荷（MW）</text>')
    parts.append("</svg>")
    (FIGURES / "manufacturing_31sector_aggregate_ai_profiles.svg").write_text("\n".join(parts), encoding="utf-8")


def write_findings(summary: pd.DataFrame, selection: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    direct_counts = selection["selected_curve_source_type"].value_counts().to_dict()
    task_aggregate = aggregate[(aggregate["energy_scenario"] == "central_14twh") & (aggregate["temporal_scenario"] == "task_timed")]
    flat_aggregate = aggregate[(aggregate["energy_scenario"] == "central_14twh") & (aggregate["temporal_scenario"] == "flat")]
    sync_aggregate = aggregate[(aggregate["energy_scenario"] == "central_14twh") & (aggregate["temporal_scenario"] == "production_synchronous")]
    aggregate_peak = float(task_aggregate["aggregate_ai_load_mw"].max())
    aggregate_peak_hour = int(task_aggregate.loc[task_aggregate["aggregate_ai_load_mw"].idxmax(), "hour"])
    average_total = float(task_aggregate["aggregate_ai_load_mw"].mean())
    flat_peak = float(flat_aggregate["aggregate_ai_load_mw"].max())
    sync_peak = float(sync_aggregate["aggregate_ai_load_mw"].max())
    baseline_average = float(task_aggregate["aggregate_base_load_mw"].mean())
    baseline_peak = float(task_aggregate["aggregate_base_load_mw"].max())
    combined_peak = float(task_aggregate["aggregate_combined_load_mw"].max())
    combined_peak_hour = int(task_aggregate.loc[task_aggregate["aggregate_combined_load_mw"].idxmax(), "hour"])
    highest = summary.sort_values("central_ai_peak_mw", ascending=False).head(10)
    coincidence = summary.sort_values("incremental_peak_per_ai_average_5pct", ascending=False).head(8)
    specialized = summary.reindex(summary["selected_vs_archetype_peak_factor_delta"].abs().sort_values(ascending=False).index).head(8)
    lines = [
        "# 31个制造业行业AI负荷与行业峰值叠加筛查",
        "",
        "## 结论先行",
        "",
        f"在14 TWh中心情景下，31行业AI平均负荷合计为 {average_total:.1f} MW。按六类任务时序叠加后，AI自身合计峰值为 {aggregate_peak:.1f} MW，峰值出现在 {aggregate_peak_hour}:00，峰均比为 {aggregate_peak / average_total:.2f}。完全平滑情景的峰值为 {flat_peak:.1f} MW，完全跟随生产情景为 {sync_peak:.1f} MW。",
        "",
        f"本地 d_energy.csv 的2030年REFERENCE情景给出31行业制造业用电量合计4799.4 TWh，对应平均负荷 {baseline_average:.1f} MW。以各行业代表工作日曲线缩放后，原负荷合计峰值为 {baseline_peak:.1f} MW；加入任务时序AI后，合计峰值为 {combined_peak:.1f} MW，峰值仍在 {combined_peak_hour}:00，增加 {combined_peak - baseline_peak:.1f} MW（{(combined_peak / baseline_peak - 1) * 100:.3f}%）。",
        "",
        "这里的绝对值是情景筛查结果：d_energy 是REFERENCE情景模型数据，不是《中国能源统计年鉴》观测表；各行业典型工作日也被假定在全国同一小时并发，不能把该峰值直接解释为全国电网实际系统峰值。",
        "",
        "## 曲线选择覆盖",
        "",
        f"- {direct_counts.get('china_eweld_isic_specific', 0)} 个行业使用中国EWELD同一ISIC大类曲线。",
        f"- {direct_counts.get('dedicated_external_facility', 0)} 个行业使用专属外部设施曲线（C31钢铁）。",
        f"- {direct_counts.get('dedicated_external_sector_model', 0)} 个行业使用专属外部行业曲线（C37运输设备）。",
        f"- {direct_counts.get('six_archetype_fallback', 0)} 个行业回退到六类原型。",
        "",
        "## 中心情景AI峰值最高的行业",
        "",
        "| 行业 | AI平均负荷（MW） | AI峰值（MW） | 峰均比 | 峰值小时 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in highest.to_dict("records"):
        lines.append(f'| {row["industry_code"]} {row["industry_name_cn"]} | {row["central_ai_average_mw"]:.1f} | {row["central_ai_peak_mw"]:.1f} | {row["ai_peak_factor"]:.2f} | {int(row["ai_peak_hour"])} |')
    lines.extend(
        [
            "",
            "## 哪些行业的AI更容易增加既有峰值",
            "",
            "下表使用5%相对规模情景：AI平均负荷等于行业原平均负荷的5%。“每1 MW AI平均负荷带来的新增峰值”是形状指标，不需要知道行业原负荷的绝对MW。",
            "",
            "| 行业 | 新增峰值/AI平均负荷 | 原负荷峰值小时 | AI峰值小时 | 形状相关系数 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in coincidence.to_dict("records"):
        lines.append(f'| {row["industry_code"]} {row["industry_name_cn"]} | {row["incremental_peak_per_ai_average_5pct"]:.2f} | {int(row["base_peak_hour"])} | {int(row["ai_peak_hour"])} | {row["base_ai_shape_correlation"]:.2f} |')
    lines.extend(
        [
            "",
            "## 使用行业专属曲线是否重要",
            "",
            "与全部使用六类原型相比，以下行业的原负荷峰均比变化最大。差异较大并不表示专属曲线必然正确，而是说明原型选择可能影响最大需量结论，应在正式结果中保留两套曲线作为结构不确定性。",
            "",
            "| 行业 | 选定曲线峰均比 | 原型峰均比 | 差值 | 曲线来源 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in specialized.to_dict("records"):
        lines.append(f'| {row["industry_code"]} {row["industry_name_cn"]} | {row["base_peak_factor"]:.2f} | {row["archetype_base_peak_factor"]:.2f} | {row["selected_vs_archetype_peak_factor_delta"]:+.2f} | {row["selected_curve_source_type"]} |')
    lines.extend(
        [
            "",
            "## 方法和边界",
            "",
            "1. 行业原负荷曲线使用典型工作日，并统一归一化到平均值1。EWELD同一ISIC行业内先对企业归一化，再取企业中位数。行业绝对规模来自 d_energy.csv 的2030年REFERENCE情景：31个省级地区求和、排除export、electricity由GJ按1 TWh=360万GJ换算。",
            "2. AI曲线由办公、Agent、视觉、维护、排程和仿真六类任务组合。办公和Agent时序继承现有2030集成AI代表日，视觉和维护跟随行业生产曲线，排程集中在换班窗口，仿真集中在夜间。边缘设备负荷按现有31行业模型中的边缘能耗份额跟随生产。",
            "3. 完全平滑和完全跟随生产不是预测，而是时序边界。任务时序主情景同样是研究者构造，尚未由行业AI服务器实测曲线校准。",
            "4. C31采用三家海外钢铁设施，C37采用德国合成行业曲线；它们行业身份更专属，但地理代表性弱于EWELD中国样本。",
            "5. 目前不能把全国代表工作日峰值直接解释为同等规模的电网扩容。还需加入省区季节差异、行业空间分布、现有接入余量、区域同时率、光伏储能和网络约束。",
            "",
            "## 下一步",
            "",
            "先确认 d_energy.csv 的模型名称、版本和原始出处，并用《中国能源统计年鉴》或行业统计资料核对2020年分行业用电结构。之后按省份保留行业用电量，把本地部署AI分配到企业所在省，其余负荷放入候选数据中心节点，进入区域电网和社会成本模型。",
        ]
    )
    (LEGACY_RESULTS / "manufacturing_31sector_peak_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    LEGACY_RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    crosswalk = pd.read_csv(DATA / "manufacturing_load_curve_source_crosswalk.csv", encoding="utf-8-sig")
    allocation = pd.read_csv(RESULTS / "manufacturing_topdown_allocation_31sectors.csv", encoding="utf-8-sig")
    baseline = pd.read_csv(DATA / "manufacturing_31sector_electricity_baseline.csv", encoding="utf-8-sig")
    parameters = pd.read_csv(DATA / "china_manufacturing_ai_31sector_proxy_parameters.csv", encoding="utf-8-sig")
    bottomup = pd.read_csv(RESULTS / "manufacturing_ai_31sector_bottomup_weights.csv", encoding="utf-8-sig")
    archetype_frame = pd.read_csv(RESULTS / "manufacturing_load_archetype_hourly_profiles.csv", encoding="utf-8-sig")
    archetype_profiles = {
        archetype: normalize(group.sort_values("hour")["normalized_load"].to_numpy())
        for archetype, group in archetype_frame[(archetype_frame["source_role"] == "EWELD_main") & (archetype_frame["day_type"] == "weekday")].groupby("archetype")
    }

    isic_profiles, _, steel, transport, isic_counts = build_curve_library()
    curves, selection_rows = build_curve_selection(crosswalk, isic_profiles, archetype_profiles, steel, transport, isic_counts)
    write_csv(DATA / "manufacturing_31sector_curve_selection.csv", selection_rows)
    selection = pd.DataFrame(selection_rows)

    allocation_map = allocation.set_index("industry_code").to_dict("index")
    baseline_map = baseline.set_index("industry_code").to_dict("index")
    parameter_map = parameters.set_index("industry_code").to_dict("index")
    bottomup_map = bottomup.set_index("industry_code").to_dict("index")
    crosswalk_map = crosswalk.set_index("china_code").to_dict("index")
    selection_map = selection.set_index("industry_code").to_dict("index")
    office_shape, agent_shape = parse_hourly_workload_shapes()

    detail_rows = []
    hourly_rows = []
    summary_rows = []
    aggregate_accumulator: dict[tuple[str, str, int], float] = defaultdict(float)
    aggregate_base_accumulator: dict[tuple[str, str, int], float] = defaultdict(float)

    for code in crosswalk["china_code"]:
        base = curves[code]
        archetype = crosswalk_map[code]["primary_archetype"]
        fallback_base = archetype_profiles[archetype]
        parameter = parameter_map[code]
        baseline_annual_twh = float(baseline_map[code]["electricity_2030_twh"])
        baseline_average_mw = baseline_annual_twh * 1e6 / 8760
        baseline_load_mw = baseline_average_mw * base
        raw_average = float(bottomup_map[code]["raw_bottomup_average_kw_2030"])
        edge_share = float(bottomup_map[code]["edge_average_kw_2030"]) / raw_average if raw_average > 0 else 0.0
        scenario_profiles = {}
        for temporal_scenario in TEMPORAL_SCENARIOS:
            ai = build_ai_profile(base, parameter, edge_share, temporal_scenario, office_shape, agent_shape)
            scenario_profiles[temporal_scenario] = ai
            metrics = shape_metrics(base, ai)
            for energy_scenario, allocation_column in ENERGY_SCENARIOS.items():
                annual_twh = float(allocation_map[code][allocation_column])
                average_mw = annual_twh * 1e6 / 8760
                ai_load_mw = average_mw * ai
                combined_load_mw = baseline_load_mw + ai_load_mw
                row = {
                    "industry_code": code,
                    "industry_name_cn": allocation_map[code]["industry_name_cn"],
                    "energy_scenario": energy_scenario,
                    "temporal_scenario": temporal_scenario,
                    "annual_ai_twh": annual_twh,
                    "ai_average_mw": average_mw,
                    "ai_peak_mw": average_mw * metrics["ai_peak_factor"],
                    "baseline_source": "d_energy_REFERENCE_2030",
                    "baseline_annual_electricity_twh": baseline_annual_twh,
                    "baseline_average_load_mw": baseline_average_mw,
                    "baseline_peak_load_mw": float(baseline_load_mw.max()),
                    "ai_average_share_of_baseline_average_pct": average_mw / baseline_average_mw * 100,
                    "combined_peak_load_mw": float(combined_load_mw.max()),
                    "combined_peak_hour": int(np.argmax(combined_load_mw)),
                    "incremental_peak_load_mw": float(combined_load_mw.max() - baseline_load_mw.max()),
                    "incremental_peak_per_ai_average_absolute": float(
                        (combined_load_mw.max() - baseline_load_mw.max()) / average_mw
                    ),
                    "edge_energy_share": edge_share,
                    **selection_map[code],
                    **metrics,
                }
                # Remove duplicated identification fields introduced by selection map.
                row["industry_code"] = code
                row["industry_name_cn"] = allocation_map[code]["industry_name_cn"]
                detail_rows.append(row)
                for hour in range(24):
                    aggregate_accumulator[(energy_scenario, temporal_scenario, hour)] += average_mw * ai[hour]
                    aggregate_base_accumulator[(energy_scenario, temporal_scenario, hour)] += baseline_load_mw[hour]
            for hour in range(24):
                hourly_rows.append(
                    {
                        "industry_code": code,
                        "industry_name_cn": allocation_map[code]["industry_name_cn"],
                        "temporal_scenario": temporal_scenario,
                        "hour": hour,
                        "base_normalized_load": base[hour],
                        "ai_normalized_load": ai[hour],
                        "baseline_load_mw": baseline_load_mw[hour],
                    }
                )

        central = next(
            row for row in detail_rows
            if row["industry_code"] == code and row["energy_scenario"] == "central_14twh" and row["temporal_scenario"] == "task_timed"
        )
        fallback_ai = build_ai_profile(fallback_base, parameter, edge_share, "task_timed", office_shape, agent_shape)
        fallback_metrics = shape_metrics(fallback_base, fallback_ai)
        summary_rows.append(
            {
                "industry_code": code,
                "industry_name_cn": central["industry_name_cn"],
                "central_ai_annual_twh": central["annual_ai_twh"],
                "central_ai_average_mw": central["ai_average_mw"],
                "central_ai_peak_mw": central["ai_peak_mw"],
                "baseline_annual_electricity_twh": central["baseline_annual_electricity_twh"],
                "baseline_average_load_mw": central["baseline_average_load_mw"],
                "baseline_peak_load_mw": central["baseline_peak_load_mw"],
                "central_ai_average_share_of_baseline_average_pct": central["ai_average_share_of_baseline_average_pct"],
                "central_combined_peak_load_mw": central["combined_peak_load_mw"],
                "central_incremental_peak_load_mw": central["incremental_peak_load_mw"],
                "central_incremental_peak_per_ai_average_absolute": central["incremental_peak_per_ai_average_absolute"],
                "ai_peak_factor": central["ai_peak_factor"],
                "ai_peak_hour": central["ai_peak_hour"],
                "base_peak_factor": central["base_peak_factor"],
                "base_peak_hour": central["base_peak_hour"],
                "peak_hour_gap": central["peak_hour_gap"],
                "base_ai_shape_correlation": central["base_ai_shape_correlation"],
                "incremental_peak_per_ai_average_1pct": central["incremental_peak_per_ai_average_1pct"],
                "incremental_peak_per_ai_average_5pct": central["incremental_peak_per_ai_average_5pct"],
                "incremental_peak_per_ai_average_10pct": central["incremental_peak_per_ai_average_10pct"],
                "incremental_peak_per_ai_average_25pct": central["incremental_peak_per_ai_average_25pct"],
                "archetype_base_peak_factor": fallback_metrics["base_peak_factor"],
                "selected_vs_archetype_peak_factor_delta": central["base_peak_factor"] - fallback_metrics["base_peak_factor"],
                "selected_curve_source_type": central["selected_curve_source_type"],
                "selected_curve_source_id": central["selected_curve_source_id"],
                "curve_evidence_grade": central["curve_evidence_grade"],
                "primary_archetype": archetype,
                "task_parameter_status": allocation_map[code]["task_parameter_status"],
            }
        )

    aggregate_rows = [
        {
            "energy_scenario": energy,
            "temporal_scenario": temporal,
            "hour": hour,
            "aggregate_ai_load_mw": value,
            "aggregate_base_load_mw": aggregate_base_accumulator[(energy, temporal, hour)],
            "aggregate_combined_load_mw": value + aggregate_base_accumulator[(energy, temporal, hour)],
        }
        for (energy, temporal, hour), value in sorted(aggregate_accumulator.items())
    ]
    write_csv(LEGACY_RESULTS / "manufacturing_31sector_peak_screen.csv", detail_rows)
    write_csv(LEGACY_RESULTS / "manufacturing_31sector_peak_summary.csv", summary_rows)
    write_csv(LEGACY_RESULTS / "manufacturing_31sector_hourly_peak_profiles.csv", hourly_rows)
    write_csv(LEGACY_RESULTS / "manufacturing_31sector_aggregate_ai_profiles.csv", aggregate_rows)

    summary = pd.DataFrame(summary_rows)
    hourly = pd.DataFrame(hourly_rows)
    aggregate = pd.DataFrame(aggregate_rows)
    build_example_svg(hourly, selection)
    build_top_sector_svg(summary)
    build_aggregate_svg(aggregate)
    write_findings(summary, selection, aggregate)
    print(
        json.dumps(
            {
                "industries": len(summary),
                "selection_types": selection["selected_curve_source_type"].value_counts().to_dict(),
                "central_task_timed_average_mw": float(aggregate[(aggregate.energy_scenario == "central_14twh") & (aggregate.temporal_scenario == "task_timed")]["aggregate_ai_load_mw"].mean()),
                "central_task_timed_peak_mw": float(aggregate[(aggregate.energy_scenario == "central_14twh") & (aggregate.temporal_scenario == "task_timed")]["aggregate_ai_load_mw"].max()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
