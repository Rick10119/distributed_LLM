#!/usr/bin/env python3
"""Select auditable real EWELD weeks for all core manufacturing industries."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ISIC_TO_ARCHETYPE = {
    10: "food_cold_chain", 11: "food_cold_chain",
    13: "light_manufacturing", 14: "light_manufacturing",
    15: "light_manufacturing", 16: "light_manufacturing",
    17: "continuous_energy_intensive", 18: "light_manufacturing",
    20: "continuous_energy_intensive", 21: "batch_process",
    22: "batch_process", 23: "continuous_energy_intensive",
    24: "continuous_energy_intensive", 25: "discrete_equipment",
    26: "electronics_cleanroom", 27: "discrete_equipment",
    28: "discrete_equipment", 29: "discrete_equipment",
    31: "light_manufacturing", 32: "light_manufacturing",
    33: "discrete_equipment",
}


def complete_weeks(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    data = frame.copy()
    data["Time"] = pd.to_datetime(data["Time"], errors="coerce")
    data["Value"] = pd.to_numeric(data["Value"], errors="coerce")
    data = data.dropna().drop_duplicates("Time").query("Value >= 0").sort_values("Time")
    data["date"] = data["Time"].dt.normalize()
    counts = data.groupby("date")["Value"].count()
    valid_days = set(counts[counts >= 80].index)
    output = []
    for monday in sorted({date - pd.Timedelta(days=date.dayofweek) for date in valid_days}):
        dates = [monday + pd.Timedelta(days=i) for i in range(7)]
        if not all(date in valid_days for date in dates):
            continue
        selected = data[data["date"].isin(dates)].copy()
        hourly = (
            selected.set_index("Time")["Value"].resample("1h").mean()
            .reindex(pd.date_range(monday, periods=168, freq="1h"))
        )
        if hourly.notna().all() and float(hourly.mean()) > 0:
            week = hourly.rename("raw_value").reset_index()
            week.columns = ["timestamp", "raw_value"]
            output.append((monday, week))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--daily-profile", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lineage-output", type=Path, required=True)
    args = parser.parse_args()

    daily = pd.read_csv(args.daily_profile, encoding="utf-8-sig")
    crosswalk = pd.read_csv(args.crosswalk, encoding="utf-8-sig")
    rows = []
    lineage = []
    with zipfile.ZipFile(args.archive) as zf:
        member_pattern = re.compile(
            r"^EWELD/Electricity Consumption/C/C(?P<isic>\d{2}) .+/(?P<user>U\d+)\.csv$"
        )
        members_by_isic: dict[int, list[tuple[str, str]]] = {}
        for member in zf.namelist():
            match = member_pattern.match(member)
            if match:
                members_by_isic.setdefault(int(match.group("isic")), []).append(
                    (member, match.group("user"))
                )
        weeks_by_isic: dict[int, list[tuple[str, str, pd.Timestamp, pd.DataFrame]]] = {}
        for isic, members in members_by_isic.items():
            for member, user in members:
                with zf.open(member) as handle:
                    frame = pd.read_csv(handle, usecols=["Time", "Value"])
                for monday, week in complete_weeks(frame):
                    weeks_by_isic.setdefault(isic, []).append((user, member, monday, week))

        for record in crosswalk.sort_values("china_code").to_dict("records"):
            code = str(record["china_code"])
            isic = int(record["closest_isic_rev4_division"])
            industry_name = str(record["china_industry"])
            primary_archetype = str(record["primary_archetype"])
            target = daily[
                daily["industry_code"].eq(code)
                & daily["temporal_scenario"].eq("task_timed")
            ].sort_values("hour")
            if len(target) != 24:
                raise ValueError(f"Expected one 24-hour selection template for {code}")
            target_shape = target["base_normalized_load"].to_numpy(float)
            target_mean_mw = float(target["baseline_load_mw"].mean())
            direct = bool(weeks_by_isic.get(isic))
            candidate_isics = (
                [isic]
                if direct
                else sorted(
                    candidate_isic
                    for candidate_isic, archetype in ISIC_TO_ARCHETYPE.items()
                    if archetype == primary_archetype and weeks_by_isic.get(candidate_isic)
                )
            )
            candidates = []
            for source_isic in candidate_isics:
                for user, member, monday, week in weeks_by_isic[source_isic]:
                    values = week["raw_value"].to_numpy(float)
                    daily_shapes = values.reshape(7, 24)
                    if (daily_shapes.mean(axis=1) <= 0).any():
                        continue
                    daily_shapes = daily_shapes / daily_shapes.mean(axis=1, keepdims=True)
                    typical_shape = np.median(daily_shapes, axis=0)
                    rmse = float(np.sqrt(np.mean((typical_shape - target_shape) ** 2)))
                    candidates.append((rmse, source_isic, user, member, monday, week))
            if not candidates:
                raise ValueError(f"No complete EWELD week found for {code}")
            rmse, source_isic, user, member, monday, week = min(
                candidates, key=lambda item: item[0]
            )
            raw = week["raw_value"].to_numpy(float)
            scaled = raw / float(raw.mean()) * target_mean_mw
            for hour, (timestamp, raw_value, load_mw) in enumerate(
                zip(week["timestamp"], raw, scaled)
            ):
                rows.append(
                    {
                        "industry_code": code,
                        "industry_name_cn": industry_name,
                        "temporal_scenario": "task_timed",
                        "hour": hour,
                        "timestamp": pd.Timestamp(timestamp).isoformat(),
                        "baseline_load_mw": float(load_mw),
                        "base_normalized_load": float(load_mw / target_mean_mw),
                        "source_user": user,
                        "source_isic_division": source_isic,
                        "load_profile_kind": (
                            "measured_continuous_week_direct_isic"
                            if direct
                            else "measured_continuous_week_archetype_proxy"
                        ),
                    }
                )
            lineage.append(
                {
                    "industry_code": code,
                    "target_isic_division": isic,
                    "source_isic_division": source_isic,
                    "primary_archetype": primary_archetype,
                    "mapping_type": "direct_or_partial_isic" if direct else "archetype_proxy",
                    "source_user": user,
                    "source_member": member,
                    "week_start": monday.date().isoformat(),
                    "week_end": (monday + pd.Timedelta(days=6)).date().isoformat(),
                    "selection_rule": "complete_Monday_Sunday_week_minimum_RMSE_to_24h_selection_template",
                    "shape_rmse": rmse,
                    "scale_rule": "preserve_active_industry_mean_MW",
                    "candidate_weeks": len(candidates),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
    args.lineage_output.write_text(
        json.dumps({"status": "prepared", "profile_kind": "measured_continuous_week", "hours_per_industry": 168, "industry_count": len(lineage), "records": lineage}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
