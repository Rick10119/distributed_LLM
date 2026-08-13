"""Build preliminary manufacturing load archetypes from public datasets.

The script keeps three evidence layers separate:
1. EWELD facility observations form the main archetype estimates.
2. Sector-labelled Korean factories, UCI steel and FfE profiles validate shapes.
3. Anonymous German plants and ELMAS aggregates are retained as broad checks.

Outputs are normalized load shapes, not estimates of absolute Chinese demand.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "02_data" / "raw_load_profiles"
RESULTS = ROOT / "05_results"
FIGURES = RESULTS / "figures"

ARCHETYPES = [
    "food_cold_chain",
    "batch_process",
    "light_manufacturing",
    "continuous_energy_intensive",
    "discrete_equipment",
    "electronics_cleanroom",
]

ARCHETYPE_CN = {
    "food_cold_chain": "食品与冷链",
    "batch_process": "批次流程",
    "light_manufacturing": "轻工制造",
    "continuous_energy_intensive": "连续高耗能",
    "discrete_equipment": "离散装备",
    "electronics_cleanroom": "电子与洁净厂房",
}

ARCHETYPE_COLORS = {
    "food_cold_chain": "#2a9d8f",
    "batch_process": "#e9c46a",
    "light_manufacturing": "#8ab17d",
    "continuous_energy_intensive": "#e76f51",
    "discrete_equipment": "#457b9d",
    "electronics_cleanroom": "#7b2cbf",
}

# ISIC Rev.4 manufacturing divisions. This mapping is a modelling choice, not a
# classification-system crosswalk published by the data providers.
ISIC_TO_ARCHETYPE = {
    10: "food_cold_chain",
    11: "food_cold_chain",
    13: "light_manufacturing",
    14: "light_manufacturing",
    15: "light_manufacturing",
    16: "light_manufacturing",
    17: "continuous_energy_intensive",
    18: "light_manufacturing",
    20: "continuous_energy_intensive",
    21: "batch_process",
    22: "batch_process",
    23: "continuous_energy_intensive",
    24: "continuous_energy_intensive",
    25: "discrete_equipment",
    26: "electronics_cleanroom",
    27: "discrete_equipment",
    28: "discrete_equipment",
    29: "discrete_equipment",
    31: "light_manufacturing",
    32: "light_manufacturing",
    33: "discrete_equipment",
}

# Chinese GB/T 4754 divisions to the closest ISIC Rev.4 division. One-to-many
# differences are explicitly described in the output rather than hidden.
CHINA_TO_ISIC = {
    "C13": (10, "direct", "农副食品加工并入ISIC 10食品制造"),
    "C14": (10, "direct", "食品制造对应ISIC 10"),
    "C15": (11, "direct", "饮料和茶对应ISIC 11；酒类亦在该大类"),
    "C16": (12, "proxy", "EWELD制造业样本无ISIC 12；暂用批次流程代理"),
    "C17": (13, "direct", "纺织对应ISIC 13"),
    "C18": (14, "direct", "服装对应ISIC 14"),
    "C19": (15, "direct", "皮革与制鞋对应ISIC 15"),
    "C20": (16, "direct", "木材加工对应ISIC 16"),
    "C21": (31, "direct", "家具对应ISIC 31"),
    "C22": (17, "direct", "造纸对应ISIC 17"),
    "C23": (18, "direct", "印刷对应ISIC 18"),
    "C24": (32, "partial", "文教体娱用品主要并入ISIC 32其他制造"),
    "C25": (19, "proxy", "EWELD制造业样本无ISIC 19；暂用连续高耗能代理"),
    "C26": (20, "direct", "化学原料和制品对应ISIC 20"),
    "C27": (21, "direct", "医药对应ISIC 21"),
    "C28": (20, "partial", "化纤在ISIC 20内；不是EWELD路径中的C28机械制造"),
    "C29": (22, "direct", "橡胶和塑料对应ISIC 22"),
    "C30": (23, "direct", "非金属矿物制品对应ISIC 23"),
    "C31": (24, "partial", "黑色金属冶炼对应ISIC 24基本金属"),
    "C32": (24, "partial", "有色金属冶炼对应ISIC 24基本金属"),
    "C33": (25, "direct", "金属制品对应ISIC 25"),
    "C34": (28, "partial", "通用设备并入ISIC 28机械设备"),
    "C35": (28, "partial", "专用设备并入ISIC 28机械设备"),
    "C36": (29, "direct", "汽车对应ISIC 29"),
    "C37": (30, "proxy", "EWELD制造业样本无ISIC 30；暂用离散装备代理"),
    "C38": (27, "direct", "电气机械对应ISIC 27"),
    "C39": (26, "partial", "计算机通信电子并入ISIC 26"),
    "C40": (26, "partial", "仪器仪表主要并入ISIC 26"),
    "C41": (32, "partial", "其他制造对应ISIC 32"),
    "C42": (38, "proxy", "废弃资源利用属于ISIC 38而非制造业C类；暂用批次流程代理"),
    "C43": (33, "direct", "设备修理对应ISIC 33"),
}

FFE_ID_TO_NAME = {
    1: "iron_steel",
    4: "nonmetal_minerals",
    5: "transport_equipment",
    6: "machinery",
    7: "mining_quarrying",
    8: "food_tobacco",
    9: "paper_pulp_print",
    10: "wood_products_proxy",
    11: "construction_proxy",
    12: "textile_leather_proxy",
    13: "nonspecified_industry",
}

FFE_TO_ARCHETYPE = {
    1: "continuous_energy_intensive",
    4: "continuous_energy_intensive",
    5: "discrete_equipment",
    6: "discrete_equipment",
    8: "food_cold_chain",
    9: "continuous_energy_intensive",
    10: "light_manufacturing",
    12: "light_manufacturing",
}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_from_frame(
    frame: pd.DataFrame,
    time_col: str,
    value_col: str,
    min_complete_slots: int,
    expected_slots: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]] | None:
    """Return mean-normalized weekday/weekend profiles from one facility."""
    data = frame[[time_col, value_col]].copy()
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna().drop_duplicates(subset=[time_col], keep="last")
    data = data[data[value_col] >= 0].sort_values(time_col)
    if data.empty:
        return None
    data["_calendar_date"] = data[time_col].dt.date
    counts = data.groupby("_calendar_date")[value_col].count()
    good_dates = counts[counts >= min_complete_slots].index
    if len(good_dates) < 30:
        return None
    data = data[data["_calendar_date"].isin(good_dates)].copy()
    positive_mean = float(data[value_col].mean())
    if not math.isfinite(positive_mean) or positive_mean <= 0:
        return None
    data["norm"] = data[value_col] / positive_mean
    data["hour"] = data[time_col].dt.hour
    data["day_type"] = np.where(data[time_col].dt.dayofweek < 5, "weekday", "weekend")
    profiles: dict[str, np.ndarray] = {}
    for day_type in ("weekday", "weekend"):
        part = data[data["day_type"] == day_type]
        hourly = part.groupby("hour")["norm"].median().reindex(range(24))
        if hourly.notna().sum() < 24:
            continue
        profiles[day_type] = hourly.to_numpy(dtype=float)
    if "weekday" not in profiles:
        return None
    if "weekend" not in profiles:
        profiles["weekend"] = profiles["weekday"].copy()
    duration_days = (data[time_col].max() - data[time_col].min()).total_seconds() / 86400 + 1
    metrics = {
        "usable_days": int(len(good_dates)),
        "duration_days": float(duration_days),
        "daily_completeness": float(counts.clip(upper=expected_slots).sum() / (len(counts) * expected_slots)),
        "mean_value": positive_mean,
    }
    return profiles, metrics


def aggregate_user_profiles(user_rows: list[dict]) -> dict[tuple[str, str], np.ndarray]:
    grouped: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for row in user_rows:
        for day_type, values in row["profiles"].items():
            grouped[(row["archetype"], day_type)].append(values)
    output: dict[tuple[str, str], np.ndarray] = {}
    for key, values in grouped.items():
        output[key] = np.nanmedian(np.vstack(values), axis=0)
    return output


def read_eweld() -> tuple[list[dict], dict[tuple[str, str], np.ndarray]]:
    archive = RAW / "eweld" / "EWELD.zip"
    pattern = re.compile(
        r"^EWELD/Electricity Consumption/C/C(?P<isic>\d{2}) (?P<sector>.+)/(?P<user>U\d+)\.csv$"
    )
    rows: list[dict] = []
    with zipfile.ZipFile(archive) as zf:
        members = []
        for name in zf.namelist():
            match = pattern.match(name)
            if match and int(match.group("isic")) in ISIC_TO_ARCHETYPE:
                members.append((name, match))
        for index, (name, match) in enumerate(members, start=1):
            with zf.open(name) as handle:
                frame = pd.read_csv(handle)
            result = profile_from_frame(frame, "Time", "Value", min_complete_slots=80, expected_slots=96)
            if result is None:
                continue
            profiles, metrics = result
            isic = int(match.group("isic"))
            rows.append(
                {
                    "dataset": "EWELD",
                    "user": match.group("user"),
                    "isic_division": isic,
                    "isic_sector": match.group("sector"),
                    "archetype": ISIC_TO_ARCHETYPE[isic],
                    "profiles": profiles,
                    **metrics,
                }
            )
            if index % 50 == 0:
                print(f"EWELD processed {index}/{len(members)} users")
    return rows, aggregate_user_profiles(rows)


def read_korea() -> tuple[list[dict], dict[tuple[str, str], np.ndarray]]:
    archive = RAW / "korea10" / "Data_list.zip"
    site_to_archetype = {
        "Cement_1": "continuous_energy_intensive",
        "Cement_2": "continuous_energy_intensive",
        "Steel_1": "continuous_energy_intensive",
        "Steel_2": "continuous_energy_intensive",
        "Paper": "continuous_energy_intensive",
        "Forge_1": "discrete_equipment",
        "Forge_2": "discrete_equipment",
        "Metal_1": "discrete_equipment",
        "Metal_2": "discrete_equipment",
        "Metal_3": "discrete_equipment",
    }
    rows = []
    with zipfile.ZipFile(archive) as zf:
        for site, archetype in site_to_archetype.items():
            with zf.open(f"Data_list/Factories/{site}.csv") as handle:
                frame = pd.read_csv(handle)
            result = profile_from_frame(frame, "Time", "Power consumption", 1200, 1440)
            if result is None:
                continue
            profiles, metrics = result
            rows.append({"dataset": "Korea10", "user": site, "archetype": archetype, "profiles": profiles, **metrics})
    return rows, aggregate_user_profiles(rows)


def read_uci_steel() -> tuple[list[dict], dict[tuple[str, str], np.ndarray]]:
    archive = RAW / "uci_steel" / "steel_industry_energy_consumption.zip"
    with zipfile.ZipFile(archive) as zf, zf.open("Steel_industry_data.csv") as handle:
        frame = pd.read_csv(handle)
    frame = frame.rename(columns={frame.columns[0]: "date"})
    # Usage is interval energy (kWh). Multiplying by four gives average kW, but
    # normalization makes the shape identical. Keep the conversion explicit.
    frame["power_kw"] = pd.to_numeric(frame["Usage_kWh"], errors="coerce") * 4
    result = profile_from_frame(frame, "date", "power_kw", 80, 96)
    if result is None:
        return [], {}
    profiles, metrics = result
    rows = [{"dataset": "UCI_steel", "user": "DAEWOO_steel", "archetype": "continuous_energy_intensive", "profiles": profiles, **metrics}]
    return rows, aggregate_user_profiles(rows)


def read_ffe() -> dict[tuple[str, str], np.ndarray]:
    path = RAW / "ffe" / "id_opendata_59_year_2017.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    timestamps = pd.date_range("2017-01-01 00:00", periods=8760, freq="h")
    grouped: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for item in payload:
        branch_id = int(item["internal_id_1"])
        archetype = FFE_TO_ARCHETYPE.get(branch_id)
        if not archetype:
            continue
        frame = pd.DataFrame({"time": timestamps, "value": item["values"]})
        mean_value = float(frame["value"].mean())
        frame["norm"] = frame["value"] / mean_value
        frame["hour"] = frame["time"].dt.hour
        frame["day_type"] = np.where(frame["time"].dt.dayofweek < 5, "weekday", "weekend")
        for day_type in ("weekday", "weekend"):
            values = frame[frame["day_type"] == day_type].groupby("hour")["norm"].median().reindex(range(24)).to_numpy()
            grouped[(archetype, day_type)].append(values)
    return {key: np.nanmedian(np.vstack(values), axis=0) for key, values in grouped.items()}


def read_germany_anonymous() -> dict[str, np.ndarray]:
    profiles: dict[str, np.ndarray] = {}
    for path in sorted((RAW / "germany50").glob("LoadProfile_*.csv")):
        frame = pd.read_csv(path, sep=";", skiprows=1)
        frame = frame.rename(columns={frame.columns[0]: "time"})
        frame["time"] = pd.to_datetime(frame["time"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
        for column in frame.columns[1:]:
            small = frame[["time", column]].rename(columns={column: "value"})
            result = profile_from_frame(small, "time", "value", 80, 96)
            if result:
                profiles[f"{path.stem}:{column}"] = result[0]["weekday"]
    return profiles


def read_elmas_summary() -> dict:
    archive = RAW / "elmas" / "ELMAS_dataset.zip"
    with zipfile.ZipFile(archive) as zf:
        with zf.open("ELMAS_dataset/Clusters_after_manual_reclassification.csv") as handle:
            mapping = pd.read_csv(handle, sep=";", decimal=",")
        with zf.open("ELMAS_dataset/Time_series_18_clusters.csv") as handle:
            clusters = pd.read_csv(handle, sep=";", decimal=",")
    nace_col = "Class" if "Class" in mapping.columns else mapping.columns[1]
    manufacturing = mapping[mapping[nace_col].astype(str).str.match(r"^(10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33)")]
    return {
        "activities_total": int(len(mapping)),
        "manufacturing_activities": int(len(manufacturing)),
        "manufacturing_clusters": int(manufacturing["Cluster"].nunique()),
        "profile_rows": int(len(clusters)),
        "profile_columns": int(len(clusters.columns) - 1),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest() -> None:
    expected_md5 = {
        "eweld/EWELD.zip": "eea1455aff6518aae1fbfc397b178111",
        "korea10/Data_list.zip": "cc3911b3f9e1125468e864a24540cfe0",
        "korea10/Code_list.zip": "398ed0baae9c9a33466e811860f4c3b3",
        "germany50/LoadProfile_30IPs_2017.csv": "a79655288e18f84dddb242134353ab2f",
        "germany50/LoadProfile_20IPs_2016.csv": "9552cb604242bb4aa778c76d574f93e8",
        "elmas/ELMAS_dataset.zip": "fc38bad0a32e1734b3e0def13b8de6a6",
        "elmas/ELMAS_package.zip": "6d2df30d1442185d9d6715cb61482294",
    }
    source = {
        "eweld/EWELD.zip": ("EWELD", "https://doi.org/10.6084/m9.figshare.21893808.v3", "CC BY 4.0", "main_estimation"),
        "korea10/Data_list.zip": ("South Korean factories", "https://doi.org/10.1038/s41597-022-01357-8", "CC BY 4.0", "sector_validation"),
        "korea10/Code_list.zip": ("South Korean factories code", "https://doi.org/10.1038/s41597-022-01357-8", "CC BY 4.0", "documentation"),
        "germany50/LoadProfile_30IPs_2017.csv": ("German 50 plants", "https://doi.org/10.5281/zenodo.3899018", "CC BY 4.0", "anonymous_validation"),
        "germany50/LoadProfile_20IPs_2016.csv": ("German 50 plants", "https://doi.org/10.5281/zenodo.3899018", "CC BY 4.0", "anonymous_validation"),
        "elmas/ELMAS_dataset.zip": ("ELMAS", "https://doi.org/10.1038/s41597-023-02542-z", "CC BY 4.0", "sector_aggregate_validation"),
        "elmas/ELMAS_package.zip": ("ELMAS analysis package", "https://doi.org/10.1038/s41597-023-02542-z", "CC BY 4.0", "reproducibility_code"),
        "uci_steel/steel_industry_energy_consumption.zip": ("UCI steel", "https://doi.org/10.24432/C52G8C", "CC BY 4.0", "sector_validation"),
        "ffe/id_opendata_59_year_2017.json": ("FfE normalized industry profiles", "https://opendata.ffe.de/dataset/normalized-industrial-electrical-load-profiles-germany/", "CC BY 4.0", "sector_validation"),
        "ffe/id_opendata_59_metadata.json": ("FfE metadata", "https://opendata.ffe.de/dataset/normalized-industrial-electrical-load-profiles-germany/", "CC BY 4.0", "documentation"),
    }
    rows = []
    for relative, (dataset, url, license_name, use) in source.items():
        path = RAW / relative
        checksum = md5(path)
        expected = expected_md5.get(relative, "not_published")
        rows.append(
            {
                "dataset": dataset,
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "md5": checksum,
                "checksum_status": "verified" if expected == checksum else ("computed_only" if expected == "not_published" else "mismatch"),
                "license": license_name,
                "source_url": url,
                "role": use,
            }
        )
    write_csv(
        ROOT / "02_data" / "manufacturing_load_dataset_manifest.csv",
        ["dataset", "relative_path", "bytes", "md5", "checksum_status", "license", "source_url", "role"],
        rows,
    )


def write_crosswalk(eweld_rows: list[dict]) -> list[dict]:
    baseline = pd.read_csv(ROOT / "02_data" / "china_manufacturing_sector_baseline.csv", encoding="utf-8-sig")
    counts = Counter(row["isic_division"] for row in eweld_rows)
    rows = []
    for record in baseline.to_dict("records"):
        code = record["industry_code"]
        isic, mapping_strength, note = CHINA_TO_ISIC[code]
        primary = record["primary_load_archetype"]
        direct_users = counts.get(isic, 0)
        same_archetype = ISIC_TO_ARCHETYPE.get(isic) == primary
        evidence = "direct_EWELD" if direct_users > 0 and mapping_strength == "direct" and same_archetype else ("partial_EWELD" if direct_users > 0 and same_archetype else "archetype_proxy")
        rows.append(
            {
                "china_code": code,
                "china_industry": record["industry_name_cn"],
                "primary_archetype": primary,
                "secondary_archetype": record["secondary_load_archetype"],
                "closest_isic_rev4_division": isic,
                "mapping_strength": mapping_strength,
                "usable_eweld_users": direct_users,
                "curve_evidence": evidence,
                "mapping_note": note,
            }
        )
    write_csv(
        ROOT / "02_data" / "manufacturing_load_curve_source_crosswalk.csv",
        list(rows[0].keys()),
        rows,
    )
    return rows


def profile_metrics(values: np.ndarray) -> dict[str, float]:
    peak_hour = int(np.nanargmax(values))
    mean_value = float(np.nanmean(values))
    peak = float(np.nanmax(values))
    trough = float(np.nanmin(values))
    return {
        "peak_hour": peak_hour,
        "peak_to_mean": peak / mean_value,
        "trough_to_mean": trough / mean_value,
        "daily_load_factor": mean_value / peak,
    }


def write_profiles(
    main: dict[tuple[str, str], np.ndarray],
    ffe: dict[tuple[str, str], np.ndarray],
    korea: dict[tuple[str, str], np.ndarray],
    uci: dict[tuple[str, str], np.ndarray],
    eweld_rows: list[dict],
) -> None:
    profile_rows = []
    for source_name, profiles in (("EWELD_main", main), ("FfE_validation", ffe), ("Korea10_validation", korea), ("UCI_steel_validation", uci)):
        for (archetype, day_type), values in profiles.items():
            for hour, value in enumerate(values):
                profile_rows.append(
                    {
                        "source_role": source_name,
                        "archetype": archetype,
                        "archetype_cn": ARCHETYPE_CN[archetype],
                        "day_type": day_type,
                        "hour": hour,
                        "normalized_load": round(float(value), 6),
                    }
                )
    write_csv(
        RESULTS / "manufacturing_load_archetype_hourly_profiles.csv",
        ["source_role", "archetype", "archetype_cn", "day_type", "hour", "normalized_load"],
        profile_rows,
    )

    quality_rows = []
    counts = Counter(row["archetype"] for row in eweld_rows)
    division_counts: dict[str, set[int]] = defaultdict(set)
    for row in eweld_rows:
        division_counts[row["archetype"]].add(row["isic_division"])
    for archetype in ARCHETYPES:
        weekday = main.get((archetype, "weekday"))
        weekend = main.get((archetype, "weekend"))
        if weekday is None:
            continue
        metrics = profile_metrics(weekday)
        weekend_ratio = float(np.mean(weekend) / np.mean(weekday)) if weekend is not None else math.nan
        quality_rows.append(
            {
                "archetype": archetype,
                "archetype_cn": ARCHETYPE_CN[archetype],
                "usable_eweld_users": counts[archetype],
                "represented_isic_divisions": len(division_counts[archetype]),
                "weekday_peak_hour": metrics["peak_hour"],
                "weekday_peak_to_mean": round(metrics["peak_to_mean"], 3),
                "weekday_trough_to_mean": round(metrics["trough_to_mean"], 3),
                "weekday_load_factor": round(metrics["daily_load_factor"], 3),
                "weekend_to_weekday_mean": round(weekend_ratio, 3),
                "ffe_validation_available": int((archetype, "weekday") in ffe),
                "facility_validation_available": int((archetype, "weekday") in korea or (archetype, "weekday") in uci),
                "evidence_grade": "B" if counts[archetype] >= 20 and len(division_counts[archetype]) >= 2 else "C",
            }
        )
    write_csv(RESULTS / "manufacturing_load_archetype_quality.csv", list(quality_rows[0].keys()), quality_rows)

    user_quality = []
    for row in eweld_rows:
        user_quality.append(
            {
                "dataset": row["dataset"],
                "user": row["user"],
                "isic_division": row["isic_division"],
                "isic_sector": row["isic_sector"],
                "archetype": row["archetype"],
                "usable_days": row["usable_days"],
                "duration_days": round(row["duration_days"], 1),
                "daily_completeness": round(row["daily_completeness"], 4),
                "mean_value_source_units": round(row["mean_value"], 4),
            }
        )
    write_csv(RESULTS / "eweld_manufacturing_user_quality.csv", list(user_quality[0].keys()), user_quality)


def polyline(values: np.ndarray, x0: float, y0: float, width: float, height: float, ymin: float, ymax: float) -> str:
    points = []
    for hour, value in enumerate(values):
        x = x0 + width * hour / 23
        y = y0 + height * (ymax - float(value)) / (ymax - ymin)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def build_profile_svg(
    main: dict[tuple[str, str], np.ndarray],
    ffe: dict[tuple[str, str], np.ndarray],
    korea: dict[tuple[str, str], np.ndarray],
    uci: dict[tuple[str, str], np.ndarray],
) -> None:
    width, height = 1200, 780
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;fill:#263238}.small{font-size:12px;fill:#607d8b}.title{font-size:24px;font-weight:650}.panel{font-size:16px;font-weight:600}</style>',
        '<text x="52" y="42" class="title">六类制造业典型工作日负荷曲线：主样本与外部对照</text>',
        '<text x="52" y="66" class="small">各序列按自身平均负荷归一化；实线为EWELD设施观测中位数，虚线/点线仅用于形状检验</text>',
    ]
    panel_w, panel_h = 350, 270
    origins = [(55, 105), (425, 105), (795, 105), (55, 420), (425, 420), (795, 420)]
    for archetype, (x, y) in zip(ARCHETYPES, origins):
        plot_x, plot_y, plot_w, plot_h = x + 42, y + 38, panel_w - 55, panel_h - 70
        all_values = []
        for profiles in (main, ffe, korea, uci):
            values = profiles.get((archetype, "weekday"))
            if values is not None:
                all_values.extend(values.tolist())
        ymin = max(0.0, min(all_values) - 0.08) if all_values else 0.0
        ymax = max(all_values) + 0.08 if all_values else 1.5
        parts.append(f'<rect x="{x}" y="{y}" width="{panel_w}" height="{panel_h}" rx="12" fill="#fafbfc" stroke="#dfe5e8"/>')
        parts.append(f'<text x="{x + 16}" y="{y + 25}" class="panel">{ARCHETYPE_CN[archetype]}</text>')
        for tick in (0, 6, 12, 18, 23):
            tx = plot_x + plot_w * tick / 23
            parts.append(f'<line x1="{tx:.1f}" y1="{plot_y}" x2="{tx:.1f}" y2="{plot_y + plot_h}" stroke="#edf0f2"/>')
            parts.append(f'<text x="{tx:.1f}" y="{plot_y + plot_h + 18}" text-anchor="middle" class="small">{tick}</text>')
        mean_y = plot_y + plot_h * (ymax - 1.0) / (ymax - ymin)
        parts.append(f'<line x1="{plot_x}" y1="{mean_y:.1f}" x2="{plot_x + plot_w}" y2="{mean_y:.1f}" stroke="#b0bec5" stroke-dasharray="3 4"/>')
        styles = [
            (main, ARCHETYPE_COLORS[archetype], "4", "", "EWELD"),
            (ffe, "#455a64", "2", "8 5", "FfE"),
            (korea, "#f4a261", "2", "2 5", "韩国10厂"),
            (uci, "#d62828", "2", "5 3", "UCI钢厂"),
        ]
        legend_x = x + 16
        for profiles, color, stroke_w, dash, label in styles:
            values = profiles.get((archetype, "weekday"))
            if values is None:
                continue
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            points = polyline(values, plot_x, plot_y, plot_w, plot_h, ymin, ymax)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{stroke_w}" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>')
            parts.append(f'<line x1="{legend_x}" y1="{y + panel_h - 12}" x2="{legend_x + 18}" y2="{y + panel_h - 12}" stroke="{color}" stroke-width="{stroke_w}"{dash_attr}/>')
            parts.append(f'<text x="{legend_x + 23}" y="{y + panel_h - 8}" class="small">{label}</text>')
            legend_x += 75 if label != "韩国10厂" else 92
        parts.append(f'<text x="{plot_x - 8}" y="{mean_y + 4:.1f}" text-anchor="end" class="small">1.0</text>')
    parts.append('</svg>')
    (FIGURES / "manufacturing_load_archetypes_weekday.svg").write_text("\n".join(parts), encoding="utf-8")


def build_coverage_svg(crosswalk: list[dict]) -> None:
    width, height = 1180, 990
    left, top, row_h = 305, 70, 27
    status_color = {"direct_EWELD": "#2a9d8f", "partial_EWELD": "#e9c46a", "archetype_proxy": "#e76f51"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;fill:#263238}.small{font-size:12px;fill:#607d8b}.title{font-size:24px;font-weight:650}.label{font-size:13px}</style>',
        '<text x="42" y="38" class="title">中国31个制造业大类的曲线证据覆盖</text>',
        '<text x="42" y="58" class="small">绿色：可直接对应EWELD行业；黄色：分类口径为部分对应；红色：当前只能使用同类生产过程代理</text>',
    ]
    max_users = max(int(r["usable_eweld_users"]) for r in crosswalk) or 1
    for i, row in enumerate(crosswalk):
        y = top + i * row_h
        code_name = f'{row["china_code"]} {row["china_industry"]}'
        parts.append(f'<text x="42" y="{y + 14}" class="label">{code_name}</text>')
        color = status_color[row["curve_evidence"]]
        parts.append(f'<rect x="{left}" y="{y + 2}" width="16" height="16" rx="3" fill="{color}"/>')
        users = int(row["usable_eweld_users"])
        bar_w = 620 * math.sqrt(users / max_users) if users else 0
        parts.append(f'<rect x="{left + 32}" y="{y + 5}" width="{bar_w:.1f}" height="10" rx="5" fill="#78909c" opacity="0.75"/>')
        parts.append(f'<text x="{left + 42 + bar_w:.1f}" y="{y + 14}" class="small">{users} 个可用用户 · {ARCHETYPE_CN[row["primary_archetype"]]}</text>')
    parts.append('</svg>')
    (FIGURES / "manufacturing_31sector_curve_coverage.svg").write_text("\n".join(parts), encoding="utf-8")


def write_findings(
    eweld_rows: list[dict],
    crosswalk: list[dict],
    main: dict[tuple[str, str], np.ndarray],
    ffe: dict[tuple[str, str], np.ndarray],
    korea: dict[tuple[str, str], np.ndarray],
    uci: dict[tuple[str, str], np.ndarray],
    germany: dict[str, np.ndarray],
    elmas: dict,
) -> None:
    counts = Counter(row["archetype"] for row in eweld_rows)
    direct = sum(row["curve_evidence"] == "direct_EWELD" for row in crosswalk)
    partial = sum(row["curve_evidence"] == "partial_EWELD" for row in crosswalk)
    proxy = sum(row["curve_evidence"] == "archetype_proxy" for row in crosswalk)

    def shape_corr(archetype: str, comparison: dict[tuple[str, str], np.ndarray]) -> float:
        left = main[(archetype, "weekday")]
        right = comparison[(archetype, "weekday")]
        return float(np.corrcoef(left, right)[0, 1])

    corr_food_ffe = shape_corr("food_cold_chain", ffe)
    corr_light_ffe = shape_corr("light_manufacturing", ffe)
    corr_discrete_ffe = shape_corr("discrete_equipment", ffe)
    corr_cont_uci = shape_corr("continuous_energy_intensive", uci)
    corr_cont_korea = shape_corr("continuous_energy_intensive", korea)
    lines = [
        "# 制造业行业负荷曲线初步构造结果",
        "",
        "## 结论先行",
        "",
        f"六类工作日/周末原型已经可以从 EWELD 设施级实测数据中构造。按本次质量规则，共保留 {len(eweld_rows)} 个至少具有 30 个较完整观测日的制造业用户。31 个中国制造业大类中，{direct} 个可直接对应到 EWELD 的 ISIC 大类，{partial} 个只能部分对应，{proxy} 个暂时需要使用生产过程相近的原型代理。",
        "",
        "这意味着：当前曲线库足以做全国行业模型的第一轮情景测试，但还不能声称每一个中国行业都已有中国本土代表性曲线。尤其是烟草、石油炼焦、其他运输设备和废弃资源利用，仍需要中国案例或更有针对性的公开数据。",
        "",
        "## 六类主曲线的样本基础",
        "",
        "| 原型 | EWELD可用用户 | 主要行业含义 |",
        "|---|---:|---|",
    ]
    meanings = {
        "food_cold_chain": "食品、饮料及冷链相关生产",
        "batch_process": "医药、橡塑，以及烟草/资源利用代理",
        "light_manufacturing": "纺织、服装、皮革、木材、家具、印刷",
        "continuous_energy_intensive": "造纸、化工、非金属矿物和基本金属",
        "discrete_equipment": "金属制品、机械、电气、汽车、维修",
        "electronics_cleanroom": "计算机、电子、光学及仪器代理",
    }
    for archetype in ARCHETYPES:
        lines.append(f"| {ARCHETYPE_CN[archetype]} | {counts[archetype]} | {meanings[archetype]} |")
    lines.extend(
        [
            "",
            "## 当前看见的效果",
            "",
            "1. 六类曲线确实不是同一条工业平均曲线。连续高耗能和批次流程的非工作时段基荷较高、日内波动较小；离散装备、轻工制造和食品类在白天生产时段的抬升更明显。",
            "2. 外部数据没有被并入主估计。韩国水泥、钢铁、造纸、锻造/金属工厂，UCI 小型钢厂和 FfE 行业曲线仅作为形状对照，因此不会因为跨国样本规模较大而覆盖中国实测样本。",
            f"3. 德国匿名数据中有 {len(germany)} 家工厂通过基本质量筛查，可用于检查工业曲线的总体峰谷范围，但因缺少行业身份，不能支持31行业映射。",
            f"4. ELMAS 文件包含 {elmas['manufacturing_activities']} 个制造业活动类别，落入 {elmas['manufacturing_clusters']} 个聚类；它适合验证行业之间是否存在多种稳定形状，但聚合数据会平滑单厂峰值。",
            "5. 所有输出曲线都按自身平均负荷归一化。它们回答的是‘峰值在什么时候、峰均比多大、周末下降多少’，不回答某行业绝对用电规模；绝对规模仍应由中国行业电量、企业规模或接入容量另行标定。",
            "",
            "## 外部对照说明了什么",
            "",
            f"FfE 与 EWELD 的24小时形状在食品、轻工和离散装备三类上呈中等正相关，相关系数分别约为 {corr_food_ffe:.2f}、{corr_light_ffe:.2f} 和 {corr_discrete_ffe:.2f}。这支持‘日间生产时段抬升’这一方向性判断，但两者峰值时刻和峰谷幅度并不一致。",
            "",
            f"连续高耗能类的对照更分化：UCI 单一钢厂与 EWELD 原型的日内相关系数约为 {corr_cont_uci:.2f}，而韩国水泥、钢铁和造纸工厂合成曲线与 EWELD 的相关系数约为 {corr_cont_korea:.2f}。后者的夜班和连续生产特征更强，说明‘连续高耗能’仍然过宽，正式模型至少应拆分钢铁、水泥、造纸或把它们作为行业特定曲线。相关系数只用于24小时形状筛查，不是跨国代表性的统计检验。",
            "",
            "目前下载的数据中，批次流程和电子/洁净厂房缺少独立的同类设施级外部对照，因此这两类虽然可以进入原型测试，但证据等级低于已获得多源对照的类别。",
            "",
            "## 对全国模型的直接用法",
            "",
            "- 每个中国制造业大类先继承其 primary_load_archetype 的工作日和周末曲线；secondary_load_archetype 可作为结构不确定性情景。",
            "- 用行业年电量或典型企业最大需量缩放归一化曲线；如果研究最大需量计费，应保留15分钟或至少小时峰值，而不能只用平均负荷。",
            "- 将行业 AI 负荷叠加到原始曲线后，分别计算原峰值、叠加峰值、峰值发生时刻变化以及与光伏出力的重合程度。",
            "- 对使用代理曲线的行业，结果应单独标记并做替代原型测试，不能与直接对应行业使用同一置信表述。",
            "",
            "## 仍需补强的地方",
            "",
            "- EWELD 来自华南匿名用户，样本并非按全国行业、地区和企业规模分层抽样；目前只能称作中国观测锚点。",
            "- 电子与洁净厂房原型目前主要由 ISIC 26 用户形成，但缺少洁净室面积、空调系统和产线类型，不能区分半导体、消费电子和仪器仪表。",
            "- 31行业映射混合了 GB/T 4754 和 ISIC Rev.4，两套代码不能按相同数字直接解释；本次已显式建立交叉表。",
            "- 当前是日内代表曲线，没有形成季节曲线、极端日和企业规模分组。第一轮全国 demo 可以使用，正式结论需要至少增加季节和峰值日层次。",
            "",
            "## 文件",
            "",
            "- `02_data/manufacturing_load_dataset_manifest.csv`：下载文件、许可、校验值和用途。",
            "- `02_data/manufacturing_load_curve_source_crosswalk.csv`：中国31行业到ISIC和六类原型的映射。",
            "- `05_results/manufacturing_load_archetype_hourly_profiles.csv`：主曲线和外部对照曲线。",
            "- `05_results/manufacturing_load_archetype_quality.csv`：样本数、峰值小时、峰均比和周末比例。",
            "- `05_results/eweld_manufacturing_user_quality.csv`：设施级质量筛查结果。",
            "- `05_results/figures/manufacturing_load_archetypes_weekday.svg`：六类曲线效果图。",
            "- `05_results/figures/manufacturing_31sector_curve_coverage.svg`：31行业证据覆盖图。",
            "- 同名 `.png` 文件：便于直接预览和插入文档。",
            "",
            "## 方法口径",
            "",
            "EWELD 每个用户按日检查：15分钟数据中一天至少有80个有效时点，并且至少有30个这样的观测日，才进入曲线池。每个用户先按自身有效期平均负荷归一化，再计算工作日和周末每小时中位数；六类原型最后取用户之间的中位数。该方法强调稳健的日内形状，不把企业规模差异误当作行业形状差异。",
        ]
    )
    (RESULTS / "manufacturing_load_archetype_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    write_manifest()
    eweld_rows, eweld_profiles = read_eweld()
    korea_rows, korea_profiles = read_korea()
    uci_rows, uci_profiles = read_uci_steel()
    ffe_profiles = read_ffe()
    germany_profiles = read_germany_anonymous()
    elmas_summary = read_elmas_summary()
    crosswalk = write_crosswalk(eweld_rows)
    write_profiles(eweld_profiles, ffe_profiles, korea_profiles, uci_profiles, eweld_rows)
    build_profile_svg(eweld_profiles, ffe_profiles, korea_profiles, uci_profiles)
    build_coverage_svg(crosswalk)
    write_findings(eweld_rows, crosswalk, eweld_profiles, ffe_profiles, korea_profiles, uci_profiles, germany_profiles, elmas_summary)
    print(
        json.dumps(
            {
                "eweld_users": len(eweld_rows),
                "korea_users": len(korea_rows),
                "uci_users": len(uci_rows),
                "germany_anonymous_users": len(germany_profiles),
                "elmas": elmas_summary,
                "archetypes": sorted({key[0] for key in eweld_profiles}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
