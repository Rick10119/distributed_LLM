"""Add roof-constrained PV and cyclic battery schedules to the China prototype.

PV capacity is derived from gross projected roof area, usable-roof fraction,
module efficiency, and a separate practical realization ratio.  The existing
two-hour battery schedules are retained but corrected to 90% round-trip
efficiency.  DER is treated as an existing whole-site resource: AI electricity
cost is measured against the same site's DER-equipped baseline, so pre-existing
PV output is not credited to AI a second time.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import china_minimum_prototype as core


ROOT = Path(__file__).resolve().parents[1]
DER_PARAMETERS = ROOT / "04_cases" / "china_prototype_der_parameters.csv"
DER_PROFILE = ROOT / "04_cases" / "two_user_pv_battery_typical_day.csv"
BASE_SUMMARY = ROOT / "05_results" / "china_prototype_screen_summary.csv"
LOAD_OUTPUT = ROOT / "05_results" / "china_prototype_der_load_profiles.csv"
SUMMARY_OUTPUT = ROOT / "05_results" / "china_prototype_der_screen_summary.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def battery_schedule(
    raw_schedule: list[float], roundtrip_efficiency: float, energy_kwh: float
) -> tuple[list[float], list[float]]:
    """Return grid-side battery power and a valid cyclic SOC trajectory.

    Positive values discharge to the site.  The original illustrative schedule
    has equal charge and discharge energy.  Scaling positive discharge by the
    round-trip efficiency makes it cyclic under symmetric charge/discharge
    efficiencies while preserving the original charging schedule.
    """

    eta = math.sqrt(roundtrip_efficiency)
    adjusted = [
        value * roundtrip_efficiency if value > 0 else value
        for value in raw_schedule
    ]
    increments = [
        (-value * eta if value < 0 else -value / eta) for value in adjusted
    ]
    cumulative = [0.0]
    for increment in increments:
        cumulative.append(cumulative[-1] + increment)
    initial_soc = -min(cumulative)
    soc = [initial_soc + value for value in cumulative]
    if abs(soc[-1] - soc[0]) > 1e-7:
        raise ValueError("Battery schedule is not cyclic after efficiency correction")
    if min(soc) < -1e-7 or max(soc) > energy_kwh + 1e-7:
        raise ValueError("Battery schedule violates energy capacity")
    return adjusted, soc[1:]


def main() -> None:
    params = core.read_parameters()
    source = core.read_profiles()
    der_parameters = {row["case"]: row for row in read_csv(DER_PARAMETERS)}
    der_profile_rows = read_csv(DER_PROFILE)
    der_profiles: dict[str, list[dict[str, str]]] = {case: [] for case in core.CASES}
    for row in der_profile_rows:
        der_profiles[row["case"]].append(row)
    baseline_summary = {
        (row["case"], row["mode"]): row for row in read_csv(BASE_SUMMARY)
    }

    days = core.number(params, "U04")
    demand_charge = core.number(params, "E05")
    valley_price = core.number(params, "E02")
    flat_price = core.number(params, "E01")
    peak_price = core.number(params, "E03")
    pv_capex_rmb_w = core.number(params, "U06")
    battery_capex_rmb_wh = core.number(params, "U07")
    grid_capex_rmb_per_kw = core.number(params, "X01") * 10_000.0 / 1_000.0

    load_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    base_load_rows = read_csv(
        ROOT / "05_results" / "china_prototype_load_profiles.csv"
    )
    by_case_mode: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in base_load_rows:
        by_case_mode.setdefault((row["case"], row["mode"]), []).append(row)

    for case, case_info in core.CASES.items():
        der = der_parameters[case]
        roof_area = float(der["gross_roof_projected_area_m2"])
        usable_fraction = float(der["roof_usable_fraction"])
        module_efficiency = float(der["module_efficiency"])
        realization = float(der["practical_realization_ratio"])
        calculated_pv_cap = (
            roof_area * usable_fraction * module_efficiency * realization
        )
        recorded_pv_cap = float(der["pv_dc_cap_kwp"])
        if abs(calculated_pv_cap - recorded_pv_cap) > 1e-6:
            raise ValueError(f"PV roof-cap calculation mismatch for {case}")

        reference_pv_kwp = float(der["reference_pv_profile_kwp"])
        profile = sorted(der_profiles[case], key=lambda row: int(row["hour"]))
        pv = [float(row["pv_kw"]) * recorded_pv_cap / reference_pv_kwp for row in profile]
        raw_battery = [
            float(row["battery_kw_positive_discharge"]) for row in profile
        ]
        battery_power = float(der["battery_power_kw"])
        battery_energy = float(der["battery_energy_kwh"])
        roundtrip_efficiency = float(der["battery_roundtrip_efficiency"])
        battery, soc = battery_schedule(
            raw_battery, roundtrip_efficiency, battery_energy
        )
        if max(abs(value) for value in battery) > battery_power + 1e-7:
            raise ValueError(f"Battery power limit violated for {case}")

        base = [float(row[case_info["base_col"]]) for row in source]
        baseline_grid = [
            max(0.0, base[t] - pv[t] - battery[t]) for t in range(24)
        ]
        baseline_peak = max(baseline_grid)

        for mode in core.MODES:
            model_rows = sorted(
                by_case_mode[(case, mode)], key=lambda row: int(row["hour"])
            )
            local_facility = [float(row["local_ai_facility_kw"]) for row in model_rows]
            cloud_facility = [float(row["cloud_ai_facility_kw"]) for row in model_rows]
            grid_with_ai = [
                max(0.0, base[t] + local_facility[t] - pv[t] - battery[t])
                for t in range(24)
            ]
            enterprise_peak = max(grid_with_ai)
            incremental_grid_energy = sum(grid_with_ai) - sum(baseline_grid)
            incremental_demand = max(0.0, enterprise_peak - baseline_peak)

            incremental_energy_cost_day = sum(
                (grid_with_ai[t] - baseline_grid[t])
                * core.tou_price(t, valley_price, flat_price, peak_price)
                for t in range(24)
            )
            annual_incremental_energy_cost = incremental_energy_cost_day * days
            annual_incremental_demand_cost = (
                incremental_demand * demand_charge * 12.0
            )

            original = baseline_summary[(case, mode)]
            local_fixed_cost = (
                float(original["annualized_local_capex_rmb"])
                + float(original["annual_local_maintenance_rmb"])
            )
            annual_cloud_monthly = float(original["annual_cloud_monthly_rmb"])
            annual_cloud_ondemand = float(original["annual_cloud_ondemand_rmb"])
            annual_direct_monthly = (
                local_fixed_cost
                + annual_incremental_energy_cost
                + annual_incremental_demand_cost
                + annual_cloud_monthly
            )
            annual_direct_ondemand = (
                local_fixed_cost
                + annual_incremental_energy_cost
                + annual_incremental_demand_cost
                + annual_cloud_ondemand
            )

            enterprise_tight_expansion = max(
                0.0, enterprise_peak - float(case_info["tight_capacity_kw"])
            )
            enterprise_spare_expansion = max(
                0.0, enterprise_peak - float(case_info["spare_capacity_kw"])
            )
            baseline_tight_expansion = max(
                0.0, baseline_peak - float(case_info["tight_capacity_kw"])
            )
            baseline_spare_expansion = max(
                0.0, baseline_peak - float(case_info["spare_capacity_kw"])
            )
            ai_induced_tight_expansion = max(
                0.0, enterprise_tight_expansion - baseline_tight_expansion
            )
            ai_induced_spare_expansion = max(
                0.0, enterprise_spare_expansion - baseline_spare_expansion
            )
            cloud_expansion = max(cloud_facility)
            total_tight_expansion = enterprise_tight_expansion + cloud_expansion
            total_spare_expansion = enterprise_spare_expansion + cloud_expansion

            for t, row in enumerate(model_rows):
                load_rows.append(
                    {
                        "case": case,
                        "mode": mode,
                        "hour": t,
                        "roof_pv_dc_cap_kwp": round(recorded_pv_cap, 6),
                        "pv_output_kw": round(pv[t], 6),
                        "battery_kw_positive_discharge": round(battery[t], 6),
                        "battery_soc_kwh": round(soc[t], 6),
                        "base_load_kw": round(base[t], 6),
                        "local_ai_facility_kw": round(local_facility[t], 6),
                        "cloud_ai_facility_kw": round(cloud_facility[t], 6),
                        "enterprise_grid_baseline_der_kw": round(
                            baseline_grid[t], 6
                        ),
                        "enterprise_grid_with_ai_der_kw": round(
                            grid_with_ai[t], 6
                        ),
                    }
                )

            pv_capex_if_new = recorded_pv_cap * 1_000.0 * pv_capex_rmb_w
            battery_capex_if_new = (
                battery_energy * 1_000.0 * battery_capex_rmb_wh
            )
            summary_rows.append(
                {
                    "case": case,
                    "mode": mode,
                    "gross_roof_projected_area_m2": roof_area,
                    "roof_usable_fraction": usable_fraction,
                    "module_efficiency": module_efficiency,
                    "practical_realization_ratio": realization,
                    "roof_pv_dc_cap_kwp": round(recorded_pv_cap, 6),
                    "pv_generation_kwh_day": round(sum(pv), 6),
                    "battery_power_kw": battery_power,
                    "battery_energy_kwh": battery_energy,
                    "battery_roundtrip_efficiency": roundtrip_efficiency,
                    "baseline_grid_peak_with_der_kw": round(baseline_peak, 6),
                    "enterprise_grid_peak_with_ai_der_kw": round(
                        enterprise_peak, 6
                    ),
                    "incremental_max_demand_kw": round(incremental_demand, 6),
                    "enterprise_incremental_grid_energy_kwh_day": round(
                        incremental_grid_energy, 6
                    ),
                    "cloud_ai_energy_kwh_day": round(sum(cloud_facility), 6),
                    "total_ai_grid_energy_kwh_day": round(
                        incremental_grid_energy + sum(cloud_facility), 6
                    ),
                    "enterprise_tight_expansion_kw": round(
                        enterprise_tight_expansion, 6
                    ),
                    "enterprise_spare_expansion_kw": round(
                        enterprise_spare_expansion, 6
                    ),
                    "ai_induced_tight_expansion_kw": round(
                        ai_induced_tight_expansion, 6
                    ),
                    "ai_induced_spare_expansion_kw": round(
                        ai_induced_spare_expansion, 6
                    ),
                    "cloud_dc_expansion_kw": round(cloud_expansion, 6),
                    "total_grid_expansion_tight_kw": round(
                        total_tight_expansion, 6
                    ),
                    "total_grid_expansion_spare_kw": round(
                        total_spare_expansion, 6
                    ),
                    "total_grid_capex_tight_rmb": round(
                        total_tight_expansion * grid_capex_rmb_per_kw, 2
                    ),
                    "total_grid_capex_spare_rmb": round(
                        total_spare_expansion * grid_capex_rmb_per_kw, 2
                    ),
                    "annual_incremental_energy_rmb": round(
                        annual_incremental_energy_cost, 2
                    ),
                    "annual_incremental_demand_rmb": round(
                        annual_incremental_demand_cost, 2
                    ),
                    "annual_enterprise_direct_monthly_rmb": round(
                        annual_direct_monthly, 2
                    ),
                    "annual_enterprise_direct_ondemand_rmb": round(
                        annual_direct_ondemand, 2
                    ),
                    "pv_capex_if_new_rmb": round(pv_capex_if_new, 2),
                    "battery_capex_if_new_rmb": round(
                        battery_capex_if_new, 2
                    ),
                }
            )

    with LOAD_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=load_rows[0].keys())
        writer.writeheader()
        writer.writerows(load_rows)
    with SUMMARY_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
