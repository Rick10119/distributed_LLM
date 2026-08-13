"""Run one base case for a typical machinery manufacturing enterprise.

The business workload is expressed as hourly L20 GPU-compute hours derived
from calls and tokens.  It is not reverse-engineered from a power curve.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import china_minimum_prototype as core


ROOT = Path(__file__).resolve().parents[1]
CASE_INPUT = ROOT / "04_cases" / "typical_machinery_manufacturing_case.csv"
PV_INPUT = ROOT / "04_cases" / "two_user_pv_battery_typical_day.csv"
RESULT_OUTPUT = ROOT / "05_results" / "typical_manufacturing_base_results.csv"
PROFILE_OUTPUT = ROOT / "05_results" / "typical_manufacturing_hourly_profiles.csv"
FIGURE_DIR = ROOT / "05_results" / "figures"

MODES = {
    "local": {"local_share": 1.0, "local_servers": 2},
    "cloud": {"local_share": 0.0, "local_servers": 0},
    "hybrid_50": {"local_share": 0.5, "local_servers": 1},
}

CONNECTION_CAPACITY_KW = 1_250.0
ROOF_AREA_M2 = 4_000.0
ROOF_USABLE_FRACTION = 0.90
MODULE_EFFICIENCY = 0.22
PV_REALIZATION_RATIO = 0.80
BATTERY_POWER_KW = 250.0
BATTERY_ENERGY_KWH = 500.0
BATTERY_ROUNDTRIP_EFFICIENCY = 0.90


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def baseline_peak_shaving_dispatch(
    load: list[float],
    pv: list[float],
    power_kw: float,
    energy_kwh: float,
    roundtrip_efficiency: float,
) -> tuple[list[float], list[float]]:
    """Minimize the baseline daily peak with grid/PV charging and cyclic SOC."""
    eta = math.sqrt(roundtrip_efficiency)
    initial_soc = 0.5 * energy_kwh
    net = [max(0.0, load[t] - pv[t]) for t in range(24)]

    def simulate(target: float) -> tuple[bool, list[float], list[float]]:
        soc = initial_soc
        schedule: list[float] = []
        soc_path: list[float] = []
        for value in net:
            if value > target:
                discharge = value - target
                if discharge > power_kw + 1e-9 or discharge / eta > soc + 1e-9:
                    return False, [], []
                soc -= discharge / eta
                schedule.append(discharge)
            else:
                charge = min(
                    power_kw,
                    target - value,
                    max(0.0, (energy_kwh - soc) / eta),
                )
                soc += charge * eta
                schedule.append(-charge)
            soc_path.append(soc)
        return soc >= initial_soc - 1e-6, schedule, soc_path

    low, high = 0.0, max(net)
    for _ in range(80):
        middle = (low + high) / 2.0
        feasible, _, _ = simulate(middle)
        if feasible:
            high = middle
        else:
            low = middle
    feasible, schedule, soc_path = simulate(high)
    if not feasible:
        raise RuntimeError("Battery peak-shaving dispatch is infeasible")
    return schedule, soc_path


def svg_mechanism(
    path: Path,
    base: list[float],
    pv: list[float],
    battery: list[float],
    local_ai: list[float],
    cloud_ai: list[float],
) -> None:
    width, height = 980, 500
    baseline_der = [max(0.0, base[t] - pv[t] - battery[t]) for t in range(24)]
    local_der = [max(0.0, base[t] + local_ai[t] - pv[t] - battery[t]) for t in range(24)]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="490" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">典型机械制造企业：负荷配合与AI增量</text>',
    ]
    panels = (
        (65, "全厂负荷与既有光储", (base, baseline_der, local_der), ("#7c8790", "#5b9a68", "#2f6f9f"), 1300.0),
        (545, "AI设施负荷（放大视图）", (local_ai, cloud_ai), ("#2f6f9f", "#d9782d"), max(max(local_ai), max(cloud_ai)) * 1.25),
    )
    for x0, title, series, colors, ymax in panels:
        y0, cw, ch = 75, 370, 300
        parts.extend([
            f'<text x="{x0+cw/2}" y="56" text-anchor="middle" font-family="sans-serif" font-size="16">{title}</text>',
            f'<line x1="{x0}" y1="{y0+ch}" x2="{x0+cw}" y2="{y0+ch}" stroke="#777"/>',
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ch}" stroke="#777"/>',
            f'<text x="{x0}" y="{y0-7}" font-family="sans-serif" font-size="12">kW</text>',
        ])
        for values, color in zip(series, colors):
            pts = []
            for h, value in enumerate(values):
                x = x0 + h / 23 * cw
                y = y0 + ch - value / ymax * ch
                pts.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="3"/>')
        if x0 == 65:
            cy = y0 + ch - CONNECTION_CAPACITY_KW / ymax * ch
            parts.append(f'<line x1="{x0}" y1="{cy:.1f}" x2="{x0+cw}" y2="{cy:.1f}" stroke="#c44e52" stroke-width="2" stroke-dasharray="7 5"/>')
        for h in (0, 6, 12, 18, 23):
            x = x0 + h / 23 * cw
            parts.append(f'<text x="{x:.1f}" y="{y0+ch+20}" text-anchor="middle" font-family="sans-serif" font-size="11">{h}</text>')
    legend = (
        ("#7c8790", "企业原负荷"),
        ("#5b9a68", "光储后基线"),
        ("#2f6f9f", "本地AI/带AI并网"),
        ("#d9782d", "云端AI设施负荷"),
        ("#c44e52", "接入容量"),
    )
    for i, (color, label) in enumerate(legend):
        x = 105 + i * 170
        dash = ' stroke-dasharray="7 5"' if color == "#c44e52" else ""
        parts.append(f'<line x1="{x}" y1="442" x2="{x+25}" y2="442" stroke="{color}" stroke-width="3"{dash}/>')
        parts.append(f'<text x="{x+32}" y="447" font-family="sans-serif" font-size="12">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_costs(path: Path, results: list[dict[str, object]]) -> None:
    width, height = 900, 440
    colors = {"local": "#2f6f9f", "cloud": "#d9782d", "hybrid_50": "#5b9a68"}
    labels = {"local": "本地", "cloud": "云端", "hybrid_50": "50%混合"}
    panels = (
        (55, "企业直接成本", "annual_enterprise_direct_best_rmb"),
        (475, "社会资源成本", "annual_social_cost_rmb"),
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="450" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">典型机械制造企业基础结果</text>',
    ]
    for x0, title, field in panels:
        y0, cw, ch = 72, 340, 285
        ymax = max(float(r[field]) for r in results) / 1e4 * 1.15
        parts.extend([
            f'<text x="{x0+cw/2}" y="55" text-anchor="middle" font-family="sans-serif" font-size="16">{title}</text>',
            f'<line x1="{x0}" y1="{y0+ch}" x2="{x0+cw}" y2="{y0+ch}" stroke="#777"/>',
            f'<text x="{x0}" y="{y0-5}" font-family="sans-serif" font-size="12">万元/年</text>',
        ])
        for i, row in enumerate(results):
            mode = str(row["mode"])
            value = float(row[field]) / 1e4
            bh = value / ymax * ch
            x = x0 + 30 + i * 105
            parts.append(f'<rect x="{x}" y="{y0+ch-bh:.1f}" width="65" height="{bh:.1f}" fill="{colors[mode]}"/>')
            parts.append(f'<text x="{x+32.5}" y="{y0+ch-bh-7:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.1f}</text>')
            parts.append(f'<text x="{x+32.5}" y="{y0+ch+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{labels[mode]}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    params = core.read_parameters()
    rows = read_csv(CASE_INPUT)
    base = [float(row["base_load_kw"]) for row in rows]
    gpu_compute = [float(row["l20_gpu_compute_h"]) for row in rows]

    pv_cap_kwp = (
        ROOF_AREA_M2
        * ROOF_USABLE_FRACTION
        * MODULE_EFFICIENCY
        * PV_REALIZATION_RATIO
    )
    pv_rows = [row for row in read_csv(PV_INPUT) if row["case"] == "office"]
    pv_rows.sort(key=lambda row: int(row["hour"]))
    pv = [float(row["pv_kw"]) * pv_cap_kwp / 600.0 for row in pv_rows]
    battery, battery_soc = baseline_peak_shaving_dispatch(
        base,
        pv,
        BATTERY_POWER_KW,
        BATTERY_ENERGY_KWH,
        BATTERY_ROUNDTRIP_EFFICIENCY,
    )

    full_power = core.number(params, "L09")
    idle_power = core.number(params, "L10")
    local_pue = core.number(params, "L17")
    cloud_pue = core.number(params, "U02")
    cloud_utilization = core.number(params, "U03")
    days = core.number(params, "U04")
    discount = core.number(params, "U01")
    server_life = core.number(params, "L13")
    grid_life = core.number(params, "U08")
    server_capex = core.number(params, "L02")
    maintenance = core.number(params, "L14")
    local_facility_capex = core.number(params, "L15")
    cloud_facility_capex = core.number(params, "U11")
    demand_charge = core.number(params, "E05")
    cloud_monthly_price = core.number(params, "C17")
    cloud_hourly_price = core.number(params, "C18")
    grid_capex_per_kw = core.number(params, "X01") * 10.0
    prices = [
        core.tou_price(
            h,
            core.number(params, "E02"),
            core.number(params, "E01"),
            core.number(params, "E03"),
        )
        for h in range(24)
    ]
    server_crf = core.capital_recovery_factor(discount, server_life)
    grid_crf = core.capital_recovery_factor(discount, grid_life)
    dynamic_power_per_gpu = (full_power - idle_power) / 2.0
    baseline_grid = [max(0.0, base[t] - pv[t] - battery[t]) for t in range(24)]

    results: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []
    local_ai_for_figure: list[float] = []
    cloud_ai_for_figure: list[float] = []

    for mode, setting in MODES.items():
        local_share = float(setting["local_share"])
        local_servers = int(setting["local_servers"])
        local_gpu = [value * local_share for value in gpu_compute]
        cloud_gpu = [value * (1.0 - local_share) for value in gpu_compute]
        installed_local_gpus = 2 * local_servers
        if local_gpu and max(local_gpu) > installed_local_gpus + 1e-9:
            raise ValueError(f"Local capacity is insufficient for {mode}")

        cloud_capacity_servers = (
            max(cloud_gpu) / (2.0 * cloud_utilization) if max(cloud_gpu) > 0 else 0.0
        )
        cloud_contract_instances = math.ceil(cloud_capacity_servers)
        local_ai = [
            (local_servers * idle_power + value * dynamic_power_per_gpu) * local_pue
            if local_servers
            else 0.0
            for value in local_gpu
        ]
        cloud_ai = [
            (cloud_capacity_servers * idle_power + value * dynamic_power_per_gpu)
            * cloud_pue
            if cloud_capacity_servers
            else 0.0
            for value in cloud_gpu
        ]
        enterprise_grid = [
            max(0.0, base[t] + local_ai[t] - pv[t] - battery[t])
            for t in range(24)
        ]
        local_energy_cost = sum(
            (enterprise_grid[t] - baseline_grid[t]) * prices[t]
            for t in range(24)
        ) * days
        cloud_energy_cost = sum(
            cloud_ai[t] * prices[t] for t in range(24)
        ) * days
        demand_increment = max(0.0, max(enterprise_grid) - max(baseline_grid))
        annual_demand_charge = demand_increment * demand_charge * 12.0
        enterprise_expansion = max(
            0.0, max(enterprise_grid) - CONNECTION_CAPACITY_KW
        )
        cloud_expansion = max(cloud_ai) if cloud_ai else 0.0
        total_expansion = enterprise_expansion + cloud_expansion

        local_compute_resource = (
            local_servers
            * server_capex
            * (1.0 + local_facility_capex)
            * server_crf
            + local_servers * server_capex * maintenance
        )
        cloud_compute_resource = (
            cloud_capacity_servers
            * server_capex
            * (1.0 + cloud_facility_capex)
            * server_crf
            + cloud_capacity_servers * server_capex * maintenance
        )
        hourly_cloud_instances = [
            math.ceil(value / (2.0 * cloud_utilization)) if value > 0 else 0
            for value in cloud_gpu
        ]
        cloud_bill_monthly = (
            cloud_contract_instances * cloud_monthly_price * 12.0
        )
        cloud_bill_ondemand = (
            sum(hourly_cloud_instances) * cloud_hourly_price * days
        )
        cloud_bill_best = min(cloud_bill_monthly, cloud_bill_ondemand)
        enterprise_grid_annual = (
            enterprise_expansion * grid_capex_per_kw * grid_crf
        )
        total_grid_annual = total_expansion * grid_capex_per_kw * grid_crf
        direct = (
            local_compute_resource
            + local_energy_cost
            + annual_demand_charge
            + cloud_bill_best
            + enterprise_grid_annual
        )
        social = (
            local_compute_resource
            + cloud_compute_resource
            + local_energy_cost
            + cloud_energy_cost
            + total_grid_annual
        )
        results.append(
            {
                "case": "typical_machinery_manufacturing_2030_integrated",
                "mode": mode,
                "local_share": local_share,
                "local_servers_2xl20": local_servers,
                "cloud_capacity_equivalent_2xl20": cloud_capacity_servers,
                "cloud_contract_instances": cloud_contract_instances,
                "daily_l20_gpu_compute_h": sum(gpu_compute),
                "peak_l20_gpu_compute_h_per_hour": max(gpu_compute),
                "local_installed_gpu_utilization_day": (
                    sum(local_gpu) / (installed_local_gpus * 24.0)
                    if installed_local_gpus
                    else 0.0
                ),
                "daily_local_ai_facility_kwh": sum(local_ai),
                "daily_cloud_ai_facility_kwh": sum(cloud_ai),
                "baseline_peak_no_der_kw": max(base),
                "baseline_grid_peak_existing_der_kw": max(baseline_grid),
                "enterprise_grid_peak_existing_der_kw": max(enterprise_grid),
                "connection_capacity_kw": CONNECTION_CAPACITY_KW,
                "incremental_max_demand_kw": demand_increment,
                "enterprise_grid_expansion_kw": enterprise_expansion,
                "cloud_grid_expansion_kw": cloud_expansion,
                "total_grid_expansion_kw": total_expansion,
                "roof_area_m2": ROOF_AREA_M2,
                "roof_pv_cap_kwp": pv_cap_kwp,
                "battery_power_kw": BATTERY_POWER_KW,
                "battery_energy_kwh": BATTERY_ENERGY_KWH,
                "annual_local_compute_resource_rmb": local_compute_resource,
                "annual_cloud_compute_resource_rmb": cloud_compute_resource,
                "annual_local_energy_rmb": local_energy_cost,
                "annual_cloud_energy_resource_rmb": cloud_energy_cost,
                "annual_demand_charge_rmb": annual_demand_charge,
                "annual_cloud_bill_monthly_rmb": cloud_bill_monthly,
                "annual_cloud_bill_ondemand_rmb": cloud_bill_ondemand,
                "annual_cloud_bill_best_rmb": cloud_bill_best,
                "annual_enterprise_grid_resource_rmb": enterprise_grid_annual,
                "annual_total_grid_resource_rmb": total_grid_annual,
                "annual_enterprise_direct_best_rmb": direct,
                "annual_social_cost_rmb": social,
            }
        )
        for t in range(24):
            profiles.append(
                {
                    "mode": mode,
                    "hour": t,
                    "base_load_kw": base[t],
                    "l20_gpu_compute_h": gpu_compute[t],
                    "local_ai_facility_kw": local_ai[t],
                    "cloud_ai_facility_kw": cloud_ai[t],
                    "pv_output_kw": pv[t],
                    "battery_kw_positive_discharge": battery[t],
                    "battery_soc_kwh": battery_soc[t],
                    "baseline_grid_existing_der_kw": baseline_grid[t],
                    "enterprise_grid_with_ai_kw": enterprise_grid[t],
                }
            )
        if mode == "local":
            local_ai_for_figure = local_ai
        if mode == "cloud":
            cloud_ai_for_figure = cloud_ai

    write_csv(RESULT_OUTPUT, results)
    write_csv(PROFILE_OUTPUT, profiles)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    svg_mechanism(
        FIGURE_DIR / "typical-manufacturing-mechanism.svg",
        base,
        pv,
        battery,
        local_ai_for_figure,
        cloud_ai_for_figure,
    )
    svg_costs(FIGURE_DIR / "typical-manufacturing-costs.svg", results)


if __name__ == "__main__":
    main()
