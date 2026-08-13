"""Screen the magnitude of manufacturing AI loads omitted from the text pool.

This is a transparent magnitude calculation, not an empirical load forecast.  It
keeps ordinary computer vision at the edge and adds only complex VLM, digital
twin and surrogate-optimisation work to the central L20 pool.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "04_cases" / "typical_machinery_manufacturing_case.csv"
WORKLOAD = ROOT / "02_data" / "future_manufacturing_ai_workload_model.csv"
OUT = ROOT / "05_results" / "manufacturing_ai_extended_load_screen.csv"

# Existing local server assumptions, inherited from the prototype.
LOCAL_SERVERS = 2
GPUS_PER_SERVER = 2
SERVER_IDLE_KW = 0.42
SERVER_FULL_KW = 1.30
LOCAL_PUE = 1.60

# Added manufacturing workloads.  These are explicit screening assumptions.
VISION_STATIONS = 4
VISION_ACTIVE_HOURS = set(range(6, 22))
VISION_EDGE_ACTIVE_KW_PER_STATION = 0.040
VISION_EDGE_IDLE_KW_PER_STATION = 0.010
VISION_EDGE_AUX_FACTOR = 1.15

# A difficult image triggers a short multi-step VLM investigation.  Total GPU
# time per exception is used because image-token conversions vary by model.
VLM_GPU_SECONDS_PER_EXCEPTION = 60.0

# A live operational twin / visualisation / predictive model occupies one
# L20-equivalent GPU on average while production is active.
DIGITAL_TWIN_GPU_EQUIVALENT_ACTIVE = 1.0

# Eight daily surrogate-model / production-optimisation jobs at 0.5 GPU-hour
# each.  They are placed in the six-hour overnight scheduling window.
OPTIMISATION_GPU_H_DAY = 4.0
OPTIMISATION_HOURS = set(range(0, 6))


def read_base_gpu() -> dict[int, float]:
    with CASE.open(encoding="utf-8-sig", newline="") as handle:
        return {
            int(row["hour"]): float(row["l20_gpu_compute_h"])
            for row in csv.DictReader(handle)
        }


def read_vlm_exceptions() -> dict[int, float]:
    values: dict[int, float] = {}
    with WORKLOAD.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["record_type"] != "hourly_curve":
                continue
            text = row["derivation_or_definition"]
            match = re.search(r"vlm_escalations=([0-9.]+)", text)
            values[int(row["hour"])] = float(match.group(1)) if match else 0.0
    return values


def central_facility_power(gpu_equivalent: float) -> float:
    dynamic_kw_per_gpu = (SERVER_FULL_KW - SERVER_IDLE_KW) / GPUS_PER_SERVER
    return (
        LOCAL_SERVERS * SERVER_IDLE_KW
        + gpu_equivalent * dynamic_kw_per_gpu
    ) * LOCAL_PUE


def main() -> None:
    base_gpu = read_base_gpu()
    vlm_exceptions = read_vlm_exceptions()
    optimisation_per_hour = OPTIMISATION_GPU_H_DAY / len(OPTIMISATION_HOURS)
    rows: list[dict[str, float | int]] = []

    for hour in range(24):
        vlm_gpu = (
            vlm_exceptions[hour] * VLM_GPU_SECONDS_PER_EXCEPTION / 3600.0
        )
        twin_gpu = (
            DIGITAL_TWIN_GPU_EQUIVALENT_ACTIVE
            if hour in VISION_ACTIVE_HOURS
            else 0.0
        )
        optimisation_gpu = (
            optimisation_per_hour if hour in OPTIMISATION_HOURS else 0.0
        )
        extended_gpu = base_gpu[hour] + vlm_gpu + twin_gpu + optimisation_gpu
        edge_kw = VISION_STATIONS * (
            VISION_EDGE_ACTIVE_KW_PER_STATION
            if hour in VISION_ACTIVE_HOURS
            else VISION_EDGE_IDLE_KW_PER_STATION
        ) * VISION_EDGE_AUX_FACTOR
        rows.append(
            {
                "hour": hour,
                "base_text_agent_gpu_h": base_gpu[hour],
                "vlm_exception_gpu_h": vlm_gpu,
                "digital_twin_gpu_h": twin_gpu,
                "optimisation_gpu_h": optimisation_gpu,
                "extended_central_gpu_h": extended_gpu,
                "ordinary_vision_edge_kw": edge_kw,
                "base_central_facility_kw": central_facility_power(base_gpu[hour]),
                "extended_central_facility_kw": central_facility_power(extended_gpu),
                "extended_total_ai_kw": central_facility_power(extended_gpu) + edge_kw,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    base_gpu_day = sum(float(row["base_text_agent_gpu_h"]) for row in rows)
    vlm_gpu_day = sum(float(row["vlm_exception_gpu_h"]) for row in rows)
    twin_gpu_day = sum(float(row["digital_twin_gpu_h"]) for row in rows)
    optimisation_gpu_day = sum(float(row["optimisation_gpu_h"]) for row in rows)
    base_energy = sum(float(row["base_central_facility_kw"]) for row in rows)
    extended_energy = sum(float(row["extended_total_ai_kw"]) for row in rows)
    peak = max(float(row["extended_total_ai_kw"]) for row in rows)
    peak_gpu = max(float(row["extended_central_gpu_h"]) for row in rows)
    print(f"base_gpu_h_day={base_gpu_day:.2f}")
    print(f"vlm_gpu_h_day={vlm_gpu_day:.2f}")
    print(f"digital_twin_gpu_h_day={twin_gpu_day:.2f}")
    print(f"optimisation_gpu_h_day={optimisation_gpu_day:.2f}")
    print(f"extended_gpu_h_day={base_gpu_day + vlm_gpu_day + twin_gpu_day + optimisation_gpu_day:.2f}")
    print(f"base_facility_kwh_day={base_energy:.2f}")
    print(f"extended_total_kwh_day={extended_energy:.2f}")
    print(f"extended_peak_gpu={peak_gpu:.2f}")
    print(f"extended_peak_kw={peak:.2f}")
    print(f"installed_gpu_capacity={LOCAL_SERVERS * GPUS_PER_SERVER}")


if __name__ == "__main__":
    main()
