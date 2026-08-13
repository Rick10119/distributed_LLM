"""Run the complete 24-scenario China minimum prototype and sensitivities."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import china_minimum_prototype as core
import china_minimum_prototype_der as der_model


ROOT = Path(__file__).resolve().parents[1]
FULL_OUTPUT = ROOT / "05_results" / "china_prototype_24_scenarios.csv"
SENSITIVITY_OUTPUT = ROOT / "05_results" / "china_prototype_sensitivity.csv"
THRESHOLD_OUTPUT = ROOT / "05_results" / "china_prototype_thresholds.csv"
TWO_WAY_OUTPUT = ROOT / "05_results" / "china_prototype_two_way_sensitivity.csv"
FIGURE_DIR = ROOT / "05_results" / "figures"

MODE_SHARES = {"local": 1.0, "cloud": 0.0, "hybrid_50": 0.5}
DER_STATES = ("no_der", "existing_der")
ACCESS_STATES = ("tight", "spare")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class Prototype:
    def __init__(self) -> None:
        self.params = core.read_parameters()
        self.source = core.read_profiles()
        self.der_parameters = {
            row["case"]: row for row in read_csv(der_model.DER_PARAMETERS)
        }
        der_rows = read_csv(der_model.DER_PROFILE)
        self.der_profiles: dict[str, list[dict[str, str]]] = {
            case: [] for case in core.CASES
        }
        for row in der_rows:
            self.der_profiles[row["case"]].append(row)

        self.full_power = core.number(self.params, "L09")
        self.idle_power = core.number(self.params, "L10")
        self.local_pue = core.number(self.params, "L17")
        self.cloud_pue = core.number(self.params, "U02")
        self.cloud_utilization = core.number(self.params, "U03")
        self.reserve = core.number(self.params, "L16")
        self.days = core.number(self.params, "U04")
        self.discount = core.number(self.params, "U01")
        self.server_life = core.number(self.params, "L13")
        self.grid_life = core.number(self.params, "U08")
        self.pv_life = core.number(self.params, "U09")
        self.battery_life = core.number(self.params, "U10")
        self.server_capex = core.number(self.params, "L02")
        self.maintenance = core.number(self.params, "L14")
        self.local_facility_capex = core.number(self.params, "L15")
        self.cloud_facility_capex = core.number(self.params, "U11")
        self.demand_charge = core.number(self.params, "E05")
        self.valley_price = core.number(self.params, "E02")
        self.flat_price = core.number(self.params, "E01")
        self.peak_price = core.number(self.params, "E03")
        self.cloud_monthly_price = core.number(self.params, "C17")
        self.cloud_hourly_price = core.number(self.params, "C18")
        self.grid_capex_per_kw = core.number(self.params, "X01") * 10.0
        self.pv_capex_per_kwp = core.number(self.params, "U06") * 1_000.0
        self.battery_capex_per_kwh = core.number(self.params, "U07") * 1_000.0

    def prices(self) -> list[float]:
        return [
            core.tou_price(h, self.valley_price, self.flat_price, self.peak_price)
            for h in range(24)
        ]

    def case_series(self, case: str) -> tuple[list[float], list[float]]:
        info = core.CASES[case]
        base = [float(row[info["base_col"]]) for row in self.source]
        task = [
            float(row[info["inference_col"]])
            + float(row[info["shiftable_col"]])
            for row in self.source
        ]
        return base, task

    def der_series(
        self, case: str, der_state: str
    ) -> tuple[list[float], list[float], float, float, float]:
        if der_state == "no_der":
            return [0.0] * 24, [0.0] * 24, 0.0, 0.0, 0.0
        row = self.der_parameters[case]
        roof = float(row["gross_roof_projected_area_m2"])
        pv_cap = (
            roof
            * float(row["roof_usable_fraction"])
            * float(row["module_efficiency"])
            * float(row["practical_realization_ratio"])
        )
        reference = float(row["reference_pv_profile_kwp"])
        profile = sorted(self.der_profiles[case], key=lambda x: int(x["hour"]))
        pv = [float(x["pv_kw"]) * pv_cap / reference for x in profile]
        battery_energy = float(row["battery_energy_kwh"])
        battery, _ = der_model.battery_schedule(
            [float(x["battery_kw_positive_discharge"]) for x in profile],
            float(row["battery_roundtrip_efficiency"]),
            battery_energy,
        )
        der_capex = (
            pv_cap * self.pv_capex_per_kwp
            + battery_energy * self.battery_capex_per_kwh
        )
        der_annual = (
            pv_cap
            * self.pv_capex_per_kwp
            * core.capital_recovery_factor(self.discount, self.pv_life)
            + battery_energy
            * self.battery_capex_per_kwh
            * core.capital_recovery_factor(self.discount, self.battery_life)
        )
        return pv, battery, pv_cap, battery_energy, der_annual

    def evaluate(
        self,
        case: str,
        local_share: float,
        der_state: str,
        connection_capacity_kw: float,
        local_utilization: float | None = None,
        cloud_price_multiplier: float = 1.0,
    ) -> dict[str, float | int | str]:
        if local_utilization is None:
            local_utilization = core.number(self.params, "L12")
        base, task = self.case_series(case)
        pv, battery, pv_cap, battery_energy, der_annual = self.der_series(
            case, der_state
        )
        local_task = [value * local_share for value in task]
        cloud_task = [value * (1.0 - local_share) for value in task]
        local_servers = core.local_server_count(
            max(local_task),
            self.full_power,
            local_utilization,
            self.reserve,
        )
        cloud_physical = core.cloud_equivalent_servers(
            max(cloud_task), self.full_power, self.cloud_utilization
        )
        cloud_contract = math.ceil(cloud_physical) if cloud_physical else 0
        local_load = [
            core.facility_load(
                value,
                local_servers,
                self.idle_power,
                self.full_power,
                self.local_pue,
            )
            for value in local_task
        ]
        cloud_load = [
            core.facility_load(
                value,
                cloud_physical,
                self.idle_power,
                self.full_power,
                self.cloud_pue,
            )
            for value in cloud_task
        ]
        baseline_grid = [max(0.0, base[t] - pv[t] - battery[t]) for t in range(24)]
        ai_grid = [
            max(0.0, base[t] + local_load[t] - pv[t] - battery[t])
            for t in range(24)
        ]
        prices = self.prices()
        local_energy_cost = (
            sum((ai_grid[t] - baseline_grid[t]) * prices[t] for t in range(24))
            * self.days
        )
        demand_increment = max(0.0, max(ai_grid) - max(baseline_grid))
        demand_cost = demand_increment * self.demand_charge * 12.0

        server_crf = core.capital_recovery_factor(self.discount, self.server_life)
        local_compute_resource = (
            local_servers
            * self.server_capex
            * (1.0 + self.local_facility_capex)
            * server_crf
            + local_servers * self.server_capex * self.maintenance
        )
        cloud_compute_resource = (
            cloud_physical
            * self.server_capex
            * (1.0 + self.cloud_facility_capex)
            * server_crf
            + cloud_physical * self.server_capex * self.maintenance
        )
        cloud_energy_resource = sum(
            cloud_load[t] * prices[t] for t in range(24)
        ) * self.days

        hourly_instances = [
            math.ceil(value / (self.full_power * self.cloud_utilization))
            if value > 0
            else 0
            for value in cloud_task
        ]
        cloud_bill_monthly = (
            cloud_contract
            * self.cloud_monthly_price
            * 12.0
            * cloud_price_multiplier
        )
        cloud_bill_ondemand = (
            sum(hourly_instances)
            * self.cloud_hourly_price
            * self.days
            * cloud_price_multiplier
        )
        cloud_bill_best = min(cloud_bill_monthly, cloud_bill_ondemand)

        enterprise_expansion = max(0.0, max(ai_grid) - connection_capacity_kw)
        cloud_expansion = max(cloud_load)
        total_expansion = enterprise_expansion + cloud_expansion
        grid_crf = core.capital_recovery_factor(self.discount, self.grid_life)
        enterprise_grid_annual = (
            enterprise_expansion * self.grid_capex_per_kw * grid_crf
        )
        total_grid_annual = total_expansion * self.grid_capex_per_kw * grid_crf

        direct_monthly = (
            local_compute_resource
            + local_energy_cost
            + demand_cost
            + cloud_bill_monthly
            + enterprise_grid_annual
        )
        direct_ondemand = (
            local_compute_resource
            + local_energy_cost
            + demand_cost
            + cloud_bill_ondemand
            + enterprise_grid_annual
        )
        direct_best = (
            local_compute_resource
            + local_energy_cost
            + demand_cost
            + cloud_bill_best
            + enterprise_grid_annual
        )
        social = (
            local_compute_resource
            + cloud_compute_resource
            + local_energy_cost
            + cloud_energy_resource
            + total_grid_annual
        )
        return {
            "case": case,
            "local_share": local_share,
            "der_state": der_state,
            "connection_capacity_kw": connection_capacity_kw,
            "local_utilization": local_utilization,
            "cloud_price_multiplier": cloud_price_multiplier,
            "local_servers": local_servers,
            "cloud_physical_equivalent_servers": cloud_physical,
            "cloud_contract_instances": cloud_contract,
            "roof_pv_dc_cap_kwp": pv_cap,
            "battery_energy_kwh": battery_energy,
            "local_ai_energy_kwh_day": sum(local_load),
            "cloud_ai_energy_kwh_day": sum(cloud_load),
            "enterprise_grid_peak_kw": max(ai_grid),
            "incremental_max_demand_kw": demand_increment,
            "enterprise_grid_expansion_kw": enterprise_expansion,
            "cloud_grid_expansion_kw": cloud_expansion,
            "total_grid_expansion_kw": total_expansion,
            "annual_local_compute_resource_rmb": local_compute_resource,
            "annual_cloud_compute_resource_rmb": cloud_compute_resource,
            "annual_local_energy_rmb": local_energy_cost,
            "annual_cloud_energy_resource_rmb": cloud_energy_resource,
            "annual_demand_charge_rmb": demand_cost,
            "annual_enterprise_grid_capex_rmb": enterprise_grid_annual,
            "annual_total_grid_resource_rmb": total_grid_annual,
            "annual_cloud_bill_monthly_rmb": cloud_bill_monthly,
            "annual_cloud_bill_ondemand_rmb": cloud_bill_ondemand,
            "annual_enterprise_direct_monthly_rmb": direct_monthly,
            "annual_enterprise_direct_ondemand_rmb": direct_ondemand,
            "annual_enterprise_direct_best_rmb": direct_best,
            "annual_social_cost_existing_der_rmb": social,
            "annual_der_capex_if_new_rmb": der_annual,
            "annual_social_cost_if_new_der_rmb": social + der_annual,
            "annual_enterprise_direct_if_new_der_rmb": direct_best + der_annual,
        }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def svg_two_panel_bars(
    path: Path,
    title: str,
    data: dict[str, list[tuple[str, float]]],
    unit: str,
) -> None:
    width, height = 920, 430
    colors = {"local": "#2f6f9f", "cloud": "#d9782d", "hybrid_50": "#5b9a68"}
    labels = {"local": "本地", "cloud": "云端", "hybrid_50": "混合"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="460" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>',
    ]
    for panel, case in enumerate(("steel", "office")):
        x0 = 55 + panel * 450
        y0, chart_h, chart_w = 70, 285, 360
        values = data[case]
        maximum = max(v for _, v in values) * 1.15 or 1.0
        parts.append(
            f'<text x="{x0 + chart_w/2}" y="55" text-anchor="middle" font-family="sans-serif" font-size="16">{"制造企业" if case == "steel" else "办公/公共服务园区"}</text>'
        )
        parts.append(f'<line x1="{x0}" y1="{y0+chart_h}" x2="{x0+chart_w}" y2="{y0+chart_h}" stroke="#777"/>')
        for i, (mode, value) in enumerate(values):
            bar_h = chart_h * value / maximum
            x = x0 + 35 + i * 110
            y = y0 + chart_h - bar_h
            parts.append(f'<rect x="{x}" y="{y:.1f}" width="70" height="{bar_h:.1f}" fill="{colors[mode]}"/>')
            parts.append(f'<text x="{x+35}" y="{y-7:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.2f}</text>')
            parts.append(f'<text x="{x+35}" y="{y0+chart_h+22}" text-anchor="middle" font-family="sans-serif" font-size="13">{labels[mode]}</text>')
        parts.append(f'<text x="{x0}" y="{y0-5}" font-family="sans-serif" font-size="12">{unit}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_cloud_price_switch(path: Path, rows: list[dict[str, object]]) -> None:
    width, height = 920, 450
    colors = {"local": "#2f6f9f", "cloud": "#d9782d", "hybrid_50": "#5b9a68"}
    labels = {"local": "本地", "cloud": "云端", "hybrid_50": "混合"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="460" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">云价格变化下的企业年化直接成本</text>',
    ]
    for panel, case in enumerate(("steel", "office")):
        subset = [r for r in rows if r["case"] == case and r["dimension"] == "cloud_price_multiplier"]
        x0, y0, cw, ch = 60 + panel * 450, 70, 350, 290
        vals = [float(r["annual_enterprise_direct_best_rmb"]) / 1e6 for r in subset]
        ymin, ymax = 0.0, max(vals) * 1.1
        parts.append(f'<line x1="{x0}" y1="{y0+ch}" x2="{x0+cw}" y2="{y0+ch}" stroke="#777"/>')
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ch}" stroke="#777"/>')
        parts.append(f'<text x="{x0+cw/2}" y="53" text-anchor="middle" font-family="sans-serif" font-size="16">{"制造企业" if case == "steel" else "办公/公共服务园区"}</text>')
        for mode in MODE_SHARES:
            points=[]
            mrows=sorted([r for r in subset if r["mode"]==mode], key=lambda r: float(r["parameter_value"]))
            for r in mrows:
                mult=float(r["parameter_value"]); val=float(r["annual_enterprise_direct_best_rmb"])/1e6
                x=x0+(mult-0.5)/(1.5-0.5)*cw; y=y0+ch-(val-ymin)/(ymax-ymin)*ch
                points.append(f'{x:.1f},{y:.1f}')
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[mode]}" stroke-width="3"/>')
        for mult in (0.5,0.75,1.0,1.25,1.5):
            x=x0+(mult-0.5)*cw
            parts.append(f'<text x="{x:.1f}" y="{y0+ch+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{mult:.2g}</text>')
        parts.append(f'<text x="{x0+cw/2}" y="{y0+ch+43}" text-anchor="middle" font-family="sans-serif" font-size="12">云价格倍数</text>')
        parts.append(f'<text x="{x0}" y="{y0-5}" font-family="sans-serif" font-size="12">百万元/年</text>')
    for i, mode in enumerate(MODE_SHARES):
        x=330+i*105
        parts.append(f'<line x1="{x}" y1="420" x2="{x+25}" y2="420" stroke="{colors[mode]}" stroke-width="3"/>')
        parts.append(f'<text x="{x+32}" y="425" font-family="sans-serif" font-size="12">{labels[mode]}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_cost_breakdown(
    path: Path,
    title: str,
    rows: list[dict[str, object]],
    components: list[tuple[str, str, str]],
) -> None:
    """Two-panel stacked cost decomposition for the three deployment modes."""
    width, height = 980, 500
    mode_labels = {"local": "本地", "cloud": "云端", "hybrid_50": "混合"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="490" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>',
    ]
    for panel, case in enumerate(("steel", "office")):
        x0, y0, cw, ch = 62 + panel * 475, 76, 370, 315
        subset = {r["mode"]: r for r in rows if r["case"] == case}
        totals = [sum(float(subset[m][field]) for field, _, _ in components) / 1e6 for m in MODE_SHARES]
        ymax = max(totals) * 1.12 or 1.0
        parts.extend([
            f'<text x="{x0+cw/2}" y="57" text-anchor="middle" font-family="sans-serif" font-size="16">{"制造企业" if case == "steel" else "办公/公共服务园区"}</text>',
            f'<line x1="{x0}" y1="{y0+ch}" x2="{x0+cw}" y2="{y0+ch}" stroke="#777"/>',
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ch}" stroke="#777"/>',
            f'<text x="{x0}" y="{y0-7}" font-family="sans-serif" font-size="12">百万元/年</text>',
        ])
        for i, mode in enumerate(MODE_SHARES):
            x = x0 + 38 + i * 112
            bottom = y0 + ch
            for field, _, color in components:
                value = float(subset[mode][field]) / 1e6
                bar_h = value / ymax * ch
                bottom -= bar_h
                parts.append(f'<rect x="{x}" y="{bottom:.1f}" width="72" height="{bar_h:.1f}" fill="{color}"/>')
            parts.append(f'<text x="{x+36}" y="{bottom-7:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12">{totals[i]:.2f}</text>')
            parts.append(f'<text x="{x+36}" y="{y0+ch+22}" text-anchor="middle" font-family="sans-serif" font-size="13">{mode_labels[mode]}</text>')
    legend_x = 155
    for i, (_, label, color) in enumerate(components):
        x = legend_x + i * 155
        parts.append(f'<rect x="{x}" y="445" width="15" height="15" fill="{color}"/>')
        parts.append(f'<text x="{x+22}" y="458" font-family="sans-serif" font-size="12">{label}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_der_capacity_effect(path: Path, rows: list[dict[str, object]]) -> None:
    width, height = 980, 475
    mode_labels = {"local": "本地", "cloud": "云端", "hybrid_50": "混合"}
    colors = {"no_der": "#aeb9c2", "existing_der": "#5b9a68"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="490" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">既有光伏储能对新增接入容量的影响（容量紧张）</text>',
    ]
    for panel, case in enumerate(("steel", "office")):
        x0, y0, cw, ch = 62 + panel * 475, 76, 370, 300
        subset = {(r["mode"], r["der_state"]): r for r in rows if r["case"] == case}
        ymax = max(float(r["total_grid_expansion_kw"]) for r in subset.values()) * 1.15 or 1.0
        parts.extend([
            f'<text x="{x0+cw/2}" y="57" text-anchor="middle" font-family="sans-serif" font-size="16">{"制造企业" if case == "steel" else "办公/公共服务园区"}</text>',
            f'<line x1="{x0}" y1="{y0+ch}" x2="{x0+cw}" y2="{y0+ch}" stroke="#777"/>',
            f'<text x="{x0}" y="{y0-7}" font-family="sans-serif" font-size="12">kW</text>',
        ])
        for i, mode in enumerate(MODE_SHARES):
            group_x = x0 + 32 + i * 115
            for j, der_state in enumerate(("no_der", "existing_der")):
                value = float(subset[(mode, der_state)]["total_grid_expansion_kw"])
                bh = value / ymax * ch
                x = group_x + j * 36
                parts.append(f'<rect x="{x}" y="{y0+ch-bh:.1f}" width="30" height="{bh:.1f}" fill="{colors[der_state]}"/>')
                parts.append(f'<text x="{x+15}" y="{y0+ch-bh-6:.1f}" text-anchor="middle" font-family="sans-serif" font-size="10">{value:.1f}</text>')
            parts.append(f'<text x="{group_x+33}" y="{y0+ch+22}" text-anchor="middle" font-family="sans-serif" font-size="13">{mode_labels[mode]}</text>')
    for i, (state, label) in enumerate((("no_der", "无DER"), ("existing_der", "既有DER"))):
        x = 365 + i * 130
        parts.append(f'<rect x="{x}" y="425" width="16" height="16" fill="{colors[state]}"/>')
        parts.append(f'<text x="{x+23}" y="438" font-family="sans-serif" font-size="12">{label}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_load_matching(path: Path, model: Prototype) -> None:
    width, height = 980, 500
    colors = {"base": "#7c8790", "no_der": "#2f6f9f", "with_der": "#5b9a68", "capacity": "#c44e52"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="490" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">本地AI负荷与既有光伏储能的逐时配合</text>',
    ]
    for panel, (case, info) in enumerate(core.CASES.items()):
        base, _ = model.case_series(case)
        local = model.evaluate(case, 1.0, "existing_der", float(info["tight_capacity_kw"]))
        pv, battery, _, _, _ = model.der_series(case, "existing_der")
        _, task = model.case_series(case)
        n = int(local["local_servers"])
        local_load = [core.facility_load(v, n, model.idle_power, model.full_power, model.local_pue) for v in task]
        no_der = [base[t] + local_load[t] for t in range(24)]
        with_der = [max(0.0, no_der[t] - pv[t] - battery[t]) for t in range(24)]
        capacity = float(info["tight_capacity_kw"])
        ymax = max(max(no_der), capacity) * 1.08
        x0, y0, cw, ch = 62 + panel * 475, 72, 370, 315
        parts.extend([
            f'<text x="{x0+cw/2}" y="53" text-anchor="middle" font-family="sans-serif" font-size="16">{"制造企业" if case == "steel" else "办公/公共服务园区"}</text>',
            f'<line x1="{x0}" y1="{y0+ch}" x2="{x0+cw}" y2="{y0+ch}" stroke="#777"/>',
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ch}" stroke="#777"/>',
            f'<text x="{x0}" y="{y0-6}" font-family="sans-serif" font-size="12">kW</text>',
        ])
        series = {"base": base, "no_der": no_der, "with_der": with_der}
        for key, values in series.items():
            pts = []
            for h, value in enumerate(values):
                x = x0 + h / 23 * cw
                y = y0 + ch - value / ymax * ch
                pts.append(f'{x:.1f},{y:.1f}')
            parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[key]}" stroke-width="{3 if key != "base" else 2}"/>')
        cy = y0 + ch - capacity / ymax * ch
        parts.append(f'<line x1="{x0}" y1="{cy:.1f}" x2="{x0+cw}" y2="{cy:.1f}" stroke="{colors["capacity"]}" stroke-width="2" stroke-dasharray="7 5"/>')
        for h in (0, 6, 12, 18, 23):
            x = x0 + h / 23 * cw
            parts.append(f'<text x="{x:.1f}" y="{y0+ch+21}" text-anchor="middle" font-family="sans-serif" font-size="11">{h}</text>')
        parts.append(f'<text x="{x0+cw/2}" y="{y0+ch+42}" text-anchor="middle" font-family="sans-serif" font-size="12">小时</text>')
    legend = (("base", "企业原负荷"), ("no_der", "原负荷+本地AI"), ("with_der", "光储后并网负荷"), ("capacity", "接入容量"))
    for i, (key, label) in enumerate(legend):
        x = 195 + i * 155
        dash = ' stroke-dasharray="7 5"' if key == "capacity" else ""
        parts.append(f'<line x1="{x}" y1="456" x2="{x+25}" y2="456" stroke="{colors[key]}" stroke-width="3"{dash}/>')
        parts.append(f'<text x="{x+32}" y="461" font-family="sans-serif" font-size="12">{label}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_utilization_switch(path: Path, rows: list[dict[str, object]]) -> None:
    width, height = 980, 500
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="490" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">本地服务器利用率与成本排序</text>',
    ]
    for panel, case in enumerate(("steel", "office")):
        subset = [r for r in rows if r["case"] == case and r["dimension"] == "local_utilization"]
        local_rows = sorted([r for r in subset if r["mode"] == "local"], key=lambda r: float(r["parameter_value"]))
        cloud = next(r for r in subset if r["mode"] == "cloud")
        x0, y0, cw, ch = 62 + panel * 475, 72, 370, 315
        all_values = [float(r[k]) / 1e6 for r in local_rows for k in ("annual_enterprise_direct_best_rmb", "annual_social_cost_existing_der_rmb")]
        all_values += [float(cloud[k]) / 1e6 for k in ("annual_enterprise_direct_best_rmb", "annual_social_cost_existing_der_rmb")]
        ymin, ymax = min(all_values) * 0.88, max(all_values) * 1.08
        parts.extend([
            f'<text x="{x0+cw/2}" y="53" text-anchor="middle" font-family="sans-serif" font-size="16">{"制造企业" if case == "steel" else "办公/公共服务园区"}</text>',
            f'<line x1="{x0}" y1="{y0+ch}" x2="{x0+cw}" y2="{y0+ch}" stroke="#777"/>',
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ch}" stroke="#777"/>',
            f'<text x="{x0}" y="{y0-6}" font-family="sans-serif" font-size="12">百万元/年</text>',
        ])
        styles = (("annual_enterprise_direct_best_rmb", "#2f6f9f"), ("annual_social_cost_existing_der_rmb", "#d9782d"))
        for field, color in styles:
            pts = []
            for r in local_rows:
                u, value = float(r["parameter_value"]), float(r[field]) / 1e6
                x = x0 + (u - 0.2) / 0.6 * cw
                y = y0 + ch - (value - ymin) / (ymax - ymin) * ch
                pts.append(f'{x:.1f},{y:.1f}')
            parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="3"/>')
            benchmark = float(cloud[field]) / 1e6
            by = y0 + ch - (benchmark - ymin) / (ymax - ymin) * ch
            parts.append(f'<line x1="{x0}" y1="{by:.1f}" x2="{x0+cw}" y2="{by:.1f}" stroke="{color}" stroke-width="2" stroke-dasharray="7 5"/>')
        for u in (0.2, 0.35, 0.5, 0.65, 0.8):
            x = x0 + (u - 0.2) / 0.6 * cw
            parts.append(f'<text x="{x:.1f}" y="{y0+ch+21}" text-anchor="middle" font-family="sans-serif" font-size="11">{u:.0%}</text>')
        parts.append(f'<text x="{x0+cw/2}" y="{y0+ch+42}" text-anchor="middle" font-family="sans-serif" font-size="12">本地服务器利用率</text>')
    parts.append('<line x1="235" y1="456" x2="260" y2="456" stroke="#2f6f9f" stroke-width="3"/>')
    parts.append('<text x="268" y="461" font-family="sans-serif" font-size="12">企业直接成本</text>')
    parts.append('<line x1="375" y1="456" x2="400" y2="456" stroke="#d9782d" stroke-width="3"/>')
    parts.append('<text x="408" y="461" font-family="sans-serif" font-size="12">社会资源成本</text>')
    parts.append('<line x1="545" y1="456" x2="570" y2="456" stroke="#555" stroke-width="3"/>')
    parts.append('<text x="578" y="461" font-family="sans-serif" font-size="12">实线：本地</text>')
    parts.append('<line x1="675" y1="456" x2="700" y2="456" stroke="#555" stroke-width="2" stroke-dasharray="7 5"/>')
    parts.append('<text x="708" y="461" font-family="sans-serif" font-size="12">虚线：云端</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_hybrid_tradeoff(path: Path, rows: list[dict[str, object]]) -> None:
    width, height = 980, 500
    colors = {"direct": "#2f6f9f", "social": "#d9782d", "grid": "#5b9a68"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="490" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">本地任务比例的三重权衡（云端模式=1）</text>',
    ]
    for panel, case in enumerate(("steel", "office")):
        subset = sorted([r for r in rows if r["case"] == case and r["dimension"] == "hybrid_local_share"], key=lambda r: float(r["parameter_value"]))
        bases = {
            "direct": float(subset[0]["annual_enterprise_direct_best_rmb"]),
            "social": float(subset[0]["annual_social_cost_existing_der_rmb"]),
            "grid": float(subset[0]["total_grid_expansion_kw"]),
        }
        x0, y0, cw, ch = 62 + panel * 475, 72, 370, 315
        parts.extend([
            f'<text x="{x0+cw/2}" y="53" text-anchor="middle" font-family="sans-serif" font-size="16">{"制造企业" if case == "steel" else "办公/公共服务园区"}</text>',
            f'<line x1="{x0}" y1="{y0+ch}" x2="{x0+cw}" y2="{y0+ch}" stroke="#777"/>',
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ch}" stroke="#777"/>',
            f'<text x="{x0}" y="{y0-6}" font-family="sans-serif" font-size="12">相对云端模式</text>',
        ])
        fields = {"direct": "annual_enterprise_direct_best_rmb", "social": "annual_social_cost_existing_der_rmb", "grid": "total_grid_expansion_kw"}
        for key, field in fields.items():
            pts = []
            for r in subset:
                share = float(r["parameter_value"])
                ratio = float(r[field]) / bases[key] if bases[key] else 0.0
                x = x0 + share * cw
                y = y0 + ch - ratio / 1.6 * ch
                pts.append(f'{x:.1f},{y:.1f}')
            parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[key]}" stroke-width="3"/>')
        for share in (0, .25, .5, .75, 1):
            x = x0 + share * cw
            parts.append(f'<text x="{x:.1f}" y="{y0+ch+21}" text-anchor="middle" font-family="sans-serif" font-size="11">{share:.0%}</text>')
        parts.append(f'<text x="{x0+cw/2}" y="{y0+ch+42}" text-anchor="middle" font-family="sans-serif" font-size="12">本地任务比例</text>')
    labels = {"direct": "企业直接成本", "social": "社会资源成本", "grid": "新增接入容量"}
    for i, key in enumerate(("direct", "social", "grid")):
        x = 245 + i * 180
        parts.append(f'<line x1="{x}" y1="456" x2="{x+25}" y2="456" stroke="{colors[key]}" stroke-width="3"/>')
        parts.append(f'<text x="{x+32}" y="461" font-family="sans-serif" font-size="12">{labels[key]}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_two_way_choice_map(path: Path, rows: list[dict[str, object]]) -> None:
    """Map enterprise/social deployment alignment over utilization and cloud price."""
    width, height = 980, 520
    colors = {
        ("cloud", "cloud"): "#d7e3ec",
        ("local", "cloud"): "#e7b17f",
        ("local", "local"): "#7eb48a",
        ("cloud", "local"): "#a995bd",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="490" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">本地利用率与云价格共同决定企业选择及社会成本排序</text>',
    ]
    utils = [x / 100 for x in range(20, 91, 5)]
    prices = [x / 100 for x in range(50, 151, 10)]
    for panel, case in enumerate(("steel", "office")):
        x0, y0, cw, ch = 68 + panel * 475, 75, 360, 300
        index = {
            (float(r["local_utilization"]), float(r["cloud_price_multiplier"])): r
            for r in rows if r["case"] == case
        }
        cell_w, cell_h = cw / len(prices), ch / len(utils)
        for ui, utilization in enumerate(utils):
            for pi, price in enumerate(prices):
                row = index[(utilization, price)]
                key = (str(row["enterprise_preferred_mode"]), str(row["social_preferred_mode"]))
                x = x0 + pi * cell_w
                y = y0 + (len(utils) - 1 - ui) * cell_h
                parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w+0.3:.1f}" height="{cell_h+0.3:.1f}" fill="{colors[key]}"/>')
        parts.extend([
            f'<rect x="{x0}" y="{y0}" width="{cw}" height="{ch}" fill="none" stroke="#777"/>',
            f'<text x="{x0+cw/2}" y="56" text-anchor="middle" font-family="sans-serif" font-size="16">{"制造企业" if case == "steel" else "办公/公共服务园区"}</text>',
            f'<text x="{x0+cw/2}" y="{y0+ch+43}" text-anchor="middle" font-family="sans-serif" font-size="12">云价格倍数</text>',
            f'<text x="{x0-42}" y="{y0+ch/2}" text-anchor="middle" transform="rotate(-90 {x0-42} {y0+ch/2})" font-family="sans-serif" font-size="12">本地服务器利用率</text>',
        ])
        for price in (0.5, 0.8, 1.0, 1.2, 1.5):
            x = x0 + (price - 0.5) / 1.0 * cw
            parts.append(f'<text x="{x:.1f}" y="{y0+ch+20}" text-anchor="middle" font-family="sans-serif" font-size="11">{price:.1f}</text>')
        for utilization in (0.2, 0.4, 0.6, 0.8, 0.9):
            y = y0 + ch - (utilization - 0.2) / 0.7 * ch
            parts.append(f'<text x="{x0-9}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{utilization:.0%}</text>')
    legend = (
        (("cloud", "cloud"), "企业、社会均偏好云端"),
        (("local", "cloud"), "企业偏好本地、社会偏好云端"),
        (("local", "local"), "企业、社会均偏好本地"),
        (("cloud", "local"), "企业偏好云端、社会偏好本地"),
    )
    for i, (key, label) in enumerate(legend):
        x = 120 + (i % 2) * 410
        y = 434 + (i // 2) * 31
        parts.append(f'<rect x="{x}" y="{y}" width="17" height="17" fill="{colors[key]}"/>')
        parts.append(f'<text x="{x+25}" y="{y+14}" font-family="sans-serif" font-size="12">{label}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def bisection(function, low: float, high: float, iterations: int = 60) -> float:
    f_low, f_high = function(low), function(high)
    if f_low == 0:
        return low
    if f_high == 0:
        return high
    if f_low * f_high > 0:
        raise ValueError("Threshold is not bracketed")
    for _ in range(iterations):
        middle = (low + high) / 2.0
        f_middle = function(middle)
        if f_low * f_middle <= 0:
            high, f_high = middle, f_middle
        else:
            low, f_low = middle, f_middle
    return (low + high) / 2.0


def main() -> None:
    model = Prototype()
    full_rows: list[dict[str, object]] = []
    for case, info in core.CASES.items():
        for mode, share in MODE_SHARES.items():
            for der_state in DER_STATES:
                for access in ACCESS_STATES:
                    capacity = float(info[f"{access}_capacity_kw"])
                    row = model.evaluate(case, share, der_state, capacity)
                    row = {
                        "scenario_id": f"{case}_{mode}_{der_state}_{access}",
                        "mode": mode,
                        "access_state": access,
                        **row,
                    }
                    full_rows.append(row)
    write_csv(FULL_OUTPUT, full_rows)

    sensitivity: list[dict[str, object]] = []
    for case, info in core.CASES.items():
        tight = float(info["tight_capacity_kw"])
        for value in (0.2, 0.35, 0.5, 0.65, 0.8):
            for mode, share in MODE_SHARES.items():
                result = model.evaluate(case, share, "existing_der", tight, local_utilization=value)
                sensitivity.append({"dimension":"local_utilization","parameter_value":value,"mode":mode,**result})
        for value in (0.5, 0.75, 1.0, 1.25, 1.5):
            for mode, share in MODE_SHARES.items():
                result = model.evaluate(case, share, "existing_der", tight, cloud_price_multiplier=value)
                sensitivity.append({"dimension":"cloud_price_multiplier","parameter_value":value,"mode":mode,**result})
        base, _ = model.case_series(case)
        for value in (0.0, 0.02, 0.05, 0.10, 0.25):
            capacity = max(base) * (1.0 + value)
            for mode, share in MODE_SHARES.items():
                result = model.evaluate(case, share, "no_der", capacity)
                sensitivity.append({"dimension":"connection_headroom_fraction","parameter_value":value,"mode":mode,**result})
        for value in (0.0, 0.25, 0.5, 0.75, 1.0):
            result = model.evaluate(case, value, "existing_der", tight)
            sensitivity.append({"dimension":"hybrid_local_share","parameter_value":value,"mode":"hybrid_variable",**result})
    write_csv(SENSITIVITY_OUTPUT, sensitivity)

    threshold_rows: list[dict[str, object]] = []
    two_way_rows: list[dict[str, object]] = []
    for case, info in core.CASES.items():
        tight = float(info["tight_capacity_kw"])
        cloud_base = model.evaluate(case, 0.0, "existing_der", tight)

        def direct_gap(utilization: float) -> float:
            local = model.evaluate(case, 1.0, "existing_der", tight, local_utilization=utilization)
            return float(local["annual_enterprise_direct_best_rmb"]) - float(cloud_base["annual_enterprise_direct_best_rmb"])

        def social_gap(utilization: float) -> float:
            local = model.evaluate(case, 1.0, "existing_der", tight, local_utilization=utilization)
            return float(local["annual_social_cost_existing_der_rmb"]) - float(cloud_base["annual_social_cost_existing_der_rmb"])

        local_base = model.evaluate(case, 1.0, "existing_der", tight)

        def price_gap(multiplier: float) -> float:
            cloud = model.evaluate(case, 0.0, "existing_der", tight, cloud_price_multiplier=multiplier)
            return float(cloud["annual_enterprise_direct_best_rmb"]) - float(local_base["annual_enterprise_direct_best_rmb"])

        for threshold, value in (
            ("local_utilization_enterprise_switch", bisection(direct_gap, 0.15, 0.95)),
            ("local_utilization_social_switch", bisection(social_gap, 0.15, 0.95)),
            ("cloud_price_multiplier_enterprise_switch", bisection(price_gap, 0.25, 1.5)),
        ):
            threshold_rows.append({"case": case, "threshold": threshold, "value": value})

        for util_step in range(20, 91, 5):
            utilization = util_step / 100.0
            local = model.evaluate(case, 1.0, "existing_der", tight, local_utilization=utilization)
            for price_step in range(50, 151, 10):
                multiplier = price_step / 100.0
                cloud = model.evaluate(case, 0.0, "existing_der", tight, cloud_price_multiplier=multiplier)
                enterprise = "local" if float(local["annual_enterprise_direct_best_rmb"]) <= float(cloud["annual_enterprise_direct_best_rmb"]) else "cloud"
                social = "local" if float(local["annual_social_cost_existing_der_rmb"]) <= float(cloud["annual_social_cost_existing_der_rmb"]) else "cloud"
                two_way_rows.append({
                    "case": case,
                    "local_utilization": utilization,
                    "cloud_price_multiplier": multiplier,
                    "enterprise_preferred_mode": enterprise,
                    "social_preferred_mode": social,
                    "alignment": "aligned" if enterprise == social else "private_social_wedge",
                })
    write_csv(THRESHOLD_OUTPUT, threshold_rows)
    write_csv(TWO_WAY_OUTPUT, two_way_rows)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    chosen = [r for r in full_rows if r["der_state"]=="existing_der" and r["access_state"]=="tight"]
    direct_data={case:[(mode,float(next(r for r in chosen if r["case"]==case and r["mode"]==mode)["annual_enterprise_direct_best_rmb"])/1e6) for mode in MODE_SHARES] for case in core.CASES}
    social_data={case:[(mode,float(next(r for r in chosen if r["case"]==case and r["mode"]==mode)["annual_social_cost_existing_der_rmb"])/1e6) for mode in MODE_SHARES] for case in core.CASES}
    grid_data={case:[(mode,float(next(r for r in chosen if r["case"]==case and r["mode"]==mode)["total_grid_expansion_kw"])) for mode in MODE_SHARES] for case in core.CASES}
    svg_two_panel_bars(FIGURE_DIR/"enterprise-direct-cost.svg","企业年化直接成本：既有DER、容量紧张",direct_data,"百万元/年")
    svg_two_panel_bars(FIGURE_DIR/"social-cost.svg","社会资源成本：既有DER、容量紧张",social_data,"百万元/年")
    svg_two_panel_bars(FIGURE_DIR/"grid-expansion.svg","全社会新增接入容量：既有DER、容量紧张",grid_data,"kW")
    svg_cloud_price_switch(FIGURE_DIR/"cloud-price-switch.svg",sensitivity)
    direct_components = [
        ("annual_local_compute_resource_rmb", "本地计算资产", "#2f6f9f"),
        ("annual_local_energy_rmb", "本地电量", "#80a9c5"),
        ("annual_demand_charge_rmb", "最大需量", "#b8cedd"),
        ("annual_cloud_bill_ondemand_rmb", "云服务费", "#d9782d"),
        ("annual_enterprise_grid_capex_rmb", "企业扩容", "#8b6ca8"),
    ]
    social_components = [
        ("annual_local_compute_resource_rmb", "本地计算资产", "#2f6f9f"),
        ("annual_cloud_compute_resource_rmb", "云端计算资产", "#d9782d"),
        ("annual_local_energy_rmb", "本地电力", "#80a9c5"),
        ("annual_cloud_energy_resource_rmb", "云端电力", "#e6ad7e"),
        ("annual_total_grid_resource_rmb", "电网容量", "#5b9a68"),
    ]
    svg_cost_breakdown(FIGURE_DIR/"direct-cost-breakdown.svg", "企业年化直接成本分解：既有DER、容量紧张", chosen, direct_components)
    svg_cost_breakdown(FIGURE_DIR/"social-cost-breakdown.svg", "社会资源成本分解：既有DER、容量紧张", chosen, social_components)
    tight_rows = [r for r in full_rows if r["access_state"] == "tight"]
    svg_der_capacity_effect(FIGURE_DIR/"der-capacity-effect.svg", tight_rows)
    svg_load_matching(FIGURE_DIR/"load-matching-mechanism.svg", model)
    svg_utilization_switch(FIGURE_DIR/"utilization-switch.svg", sensitivity)
    svg_hybrid_tradeoff(FIGURE_DIR/"hybrid-tradeoff.svg", sensitivity)
    svg_two_way_choice_map(FIGURE_DIR/"two-way-choice-map.svg", two_way_rows)


if __name__ == "__main__":
    main()
