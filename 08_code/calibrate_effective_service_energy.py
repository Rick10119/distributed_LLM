#!/usr/bin/env python3
"""Historical unified-multiplier calibration retained for reproducibility.

The active Snakemake workflow uses task-specific processed service inputs and
was replaced by task-level joint scenarios. Do not use this script for active results.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import itertools
import json
import math
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Candidate:
    round: str
    service_intensity_multiplier: float
    accelerator_h_per_service_unit: float
    pue: float
    installed_utilization: float
    reserve_fraction: float
    maximum_power_kw: float
    online_idle_power_kw: float
    derived_twh: float
    target_twh: float
    relative_error: float
    prior_distance: float
    score: float


PRIORS = {
    "service_intensity_multiplier": 7.5,
    "accelerator_h_per_service_unit": 1.0 / 3.0,
    "pue": 1.20,
    "installed_utilization": 0.65,
    "reserve_fraction": 0.10,
    "maximum_power_kw": 1.25,
    "online_idle_power_kw": 0.36,
}


def annual_energy_twh(
    frame: pd.DataFrame,
    *,
    service_intensity_multiplier: float,
    accelerator_h_per_service_unit: float,
    pue: float,
    installed_utilization: float,
    reserve_fraction: float,
    maximum_power_kw: float,
    online_idle_power_kw: float,
) -> float:
    total_kwh_day = 0.0
    for row in frame.itertuples():
        service_units_day = (
            float(row.sector_l20eq_gpu_h_day_2030) * service_intensity_multiplier
        )
        accelerator_h_day = service_units_day * accelerator_h_per_service_unit
        groups = math.ceil(
            accelerator_h_day
            * (1.0 + reserve_fraction)
            / (24.0 * 2.0 * installed_utilization)
        )
        it_kwh_day = (
            groups * online_idle_power_kw * 24.0
            + accelerator_h_day * (maximum_power_kw - online_idle_power_kw) / 2.0
        )
        total_kwh_day += pue * it_kwh_day
    return total_kwh_day * 365.0 / 1e9


def prior_distance(values: dict[str, float]) -> float:
    scales = {
        "service_intensity_multiplier": 3.5,
        "accelerator_h_per_service_unit": 1.0 / 12.0,
        "pue": 0.10,
        "installed_utilization": 0.10,
        "reserve_fraction": 0.05,
        "maximum_power_kw": 0.20,
        "online_idle_power_kw": 0.10,
    }
    return sum(
        ((values[key] - PRIORS[key]) / scales[key]) ** 2 for key in PRIORS
    )


def evaluate(
    frame: pd.DataFrame,
    round_name: str,
    target_twh: float,
    values: dict[str, float],
) -> Candidate:
    derived = annual_energy_twh(frame, **values)
    relative_error = abs(derived - target_twh) / target_twh
    distance = prior_distance(values)
    # A 1% target miss is comparable to a 0.1-unit prior-distance penalty.
    score = relative_error + 0.001 * distance
    return Candidate(
        round=round_name,
        derived_twh=derived,
        target_twh=target_twh,
        relative_error=relative_error,
        prior_distance=distance,
        score=score,
        **values,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--selected-output", type=Path, required=True)
    parser.add_argument("--findings-output", type=Path, required=True)
    parser.add_argument("--target-twh", type=float, default=14.0)
    parser.add_argument("--lower-twh", type=float, default=8.0)
    parser.add_argument("--upper-twh", type=float, default=28.0)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    required = {"industry_code", "sector_l20eq_gpu_h_day_2030"}
    if not required.issubset(frame.columns) or len(frame) != 31:
        raise ValueError("Expected the 31-industry bottom-up service table")

    rounds: list[Candidate] = []
    raw = dict(PRIORS)
    raw["service_intensity_multiplier"] = 1.0
    rounds.append(evaluate(frame, "R0_raw_bottom_up", args.target_twh, raw))

    demand_only: list[Candidate] = []
    for intensity in (x / 4.0 for x in range(16, 49)):
        values = dict(PRIORS)
        values["service_intensity_multiplier"] = intensity
        demand_only.append(evaluate(frame, "R1_service_depth", args.target_twh, values))
    selected_r1 = min(demand_only, key=lambda item: item.score)
    rounds.extend(demand_only)

    joint: list[Candidate] = []
    grid = itertools.product(
        [x / 4.0 for x in range(16, 49)],  # service intensity 4.00--12.00
        (0.25, 0.30, 1.0 / 3.0, 0.40, 0.50),
        (1.10, 1.20, 1.30),
        (0.55, 0.65, 0.75, 0.80),
        (0.05, 0.10, 0.15),
        ((1.05, 0.28), (1.25, 0.36), (1.50, 0.50)),
    )
    for intensity, conversion, pue, utilization, reserve, power_pair in grid:
        values = {
            "service_intensity_multiplier": intensity,
            "accelerator_h_per_service_unit": conversion,
            "pue": pue,
            "installed_utilization": utilization,
            "reserve_fraction": reserve,
            "maximum_power_kw": power_pair[0],
            "online_idle_power_kw": power_pair[1],
        }
        joint.append(evaluate(frame, "R2_joint_evidence_ranges", args.target_twh, values))
    selected_r2 = min(joint, key=lambda item: item.score)
    rounds.extend(sorted(joint, key=lambda item: item.score)[:100])

    output = pd.DataFrame(asdict(item) for item in rounds)
    args.candidates_output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.candidates_output, index=False, encoding="utf-8-sig")

    selected_payload = {
        "external_range_twh": [args.lower_twh, args.target_twh, args.upper_twh],
        "round_0": asdict(rounds[0]),
        "round_1": asdict(selected_r1),
        "round_2": asdict(selected_r2),
        "selection_rule": "minimize relative target error plus 0.001 times normalized squared departure from evidence priors",
        "interpretation": "calibrated conditional scenario, not an empirical forecast",
    }
    args.selected_output.write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    args.findings_output.write_text(
        "\n".join(
            [
                "# 有效服务量—外部用电校准记录",
                "",
                "外部8/14/28 TWh只用于校准和验证，不直接生成任务服务量。",
                "",
                f"- R0 未校准自下而上结果：{rounds[0].derived_twh:.3f} TWh/年。",
                f"- R1 只调服务深化：{selected_r1.derived_twh:.3f} TWh/年；服务强度系数 {selected_r1.service_intensity_multiplier:.2f}。",
                f"- R2 在证据范围内联合检查：{selected_r2.derived_twh:.3f} TWh/年，距14 TWh {selected_r2.relative_error:.2%}。",
                "- R2参数：服务强度系数 "
                f"{selected_r2.service_intensity_multiplier:.2f}，服务到加速器小时 {selected_r2.accelerator_h_per_service_unit:.3f}，"
                f"PUE {selected_r2.pue:.2f}，利用率 {selected_r2.installed_utilization:.2f}，"
                f"备用 {selected_r2.reserve_fraction:.2f}，整机满载/空闲功率 "
                f"{selected_r2.maximum_power_kw:.2f}/{selected_r2.online_idle_power_kw:.2f} kW。",
                "",
                "选择规则同时惩罚偏离证据先验，避免只为命中14 TWh而采用极端参数。完整候选见CSV。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
