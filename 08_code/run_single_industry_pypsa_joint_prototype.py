"""Run a single-industry PyPSA/Linopy/HiGHS joint capacity-dispatch prototype.

The prototype compares three counterfactuals: factory baseline without AI,
local AI at the factory-industry bucket, and cloud AI at a greenfield data-
center bucket.  It jointly chooses continuous server groups, rooftop PV,
two-hour battery power, grid expansion, and aggregate AI task timing.  A flat
energy tariff is used; there is no time-of-use arbitrage signal.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "05_results"
CASES = ROOT / "04_cases"
sys.path.insert(0, str(ROOT / "08_code"))

import china_minimum_prototype as parameter_core  # noqa: E402
from run_single_industry_service_aligned_prototype import load_inputs  # noqa: E402
from service_aligned_flexibility_core import (  # noqa: E402
    FlexibleArrival,
    ServerTechnology,
    calibrate_service_scale_to_reference_energy,
)


ANNUAL_DAYS = 365.0
INDUSTRY_CODE = "C36"
INDUSTRY_NAME = "汽车制造业"
LOCAL_HEADROOM_SHARE = 0.0025
ROOF_AREA_PER_FIRM_M2 = 4000.0
ROOF_USABLE_FRACTION = 0.90
PV_MODULE_EFFICIENCY = 0.22
PV_REALIZATION_FRACTION = 0.80
BATTERY_DURATION_H = 2.0
BATTERY_ROUNDTRIP_EFFICIENCY = 0.90

SUMMARY_OUTPUT = RESULTS / "c36_pypsa_joint_summary.csv"
HOURLY_OUTPUT = RESULTS / "c36_pypsa_joint_hourly.csv"
FINDINGS_OUTPUT = RESULTS / "c36_pypsa_joint_findings.md"


def capital_recovery_factor(rate: float, years: float) -> float:
    return rate * (1.0 + rate) ** years / ((1.0 + rate) ** years - 1.0)


def pv_capacity_factor() -> np.ndarray:
    profile = pd.read_csv(
        CASES / "two_user_pv_battery_typical_day.csv", encoding="utf-8-sig"
    )
    values = (
        profile[profile["case"] == "steel"]
        .sort_values("hour")["pv_kw"]
        .to_numpy(dtype=float)
    )
    if values.shape != (24,) or float(np.max(values)) <= 0:
        raise ValueError("PV proxy must contain one valid 24-hour profile")
    return values / float(np.max(values))


def build_ai_inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    list[FlexibleArrival],
    float,
    int,
    float,
    float,
]:
    (
        base_load_mw,
        task_shapes,
        unscaled_daily_task_service,
        flexibility,
        reference_energy_twh,
        _,
    ) = load_inputs()
    params = parameter_core.read_parameters()
    technology = ServerTechnology(
        accelerators_per_group=2.0,
        idle_power_kw=parameter_core.number(params, "L10"),
        maximum_power_kw=parameter_core.number(params, "L09"),
    )
    unscaled_by_task: dict[str, np.ndarray] = {}
    for task_id, daily_service in unscaled_daily_task_service.items():
        shape = np.asarray(task_shapes[task_id], dtype=float)
        unscaled_by_task[task_id] = daily_service * shape / float(np.sum(shape))
    unscaled_total = sum(unscaled_by_task.values(), np.zeros(24))

    # Preserve the prior cloud-reference calibration only as the common service
    # anchor.  Joint optimization is free to produce a different facility energy.
    from service_aligned_flexibility_core import Architecture

    cloud_reference = Architecture(
        name="cloud_reference",
        pue=parameter_core.number(params, "U02"),
        target_installed_utilization=parameter_core.number(params, "U03"),
        reserve_fraction=0.10,
    )
    kappa, reference_groups, reference_daily_mwh = (
        calibrate_service_scale_to_reference_energy(
            unscaled_total,
            reference_energy_twh * 1e6 / ANNUAL_DAYS,
            technology,
            cloud_reference,
        )
    )
    scaled_by_task = {
        task_id: profile * kappa for task_id, profile in unscaled_by_task.items()
    }
    total_arrival = sum(scaled_by_task.values(), np.zeros(24))
    rigid = np.zeros(24)
    jobs: list[FlexibleArrival] = []
    for task_id, profile in scaled_by_task.items():
        setting = flexibility[task_id]
        rigid += profile * setting["R"]
        for class_name, share_key, deadline_key in (
            ("F_day", "F_day", "F_day_deadline"),
            ("F_batch", "F_batch", "F_batch_deadline"),
        ):
            arrivals = profile * setting[share_key]
            for hour, amount in enumerate(arrivals):
                if amount > 0:
                    jobs.append(
                        FlexibleArrival(
                            release_hour=hour,
                            deadline_hours=int(setting[deadline_key]),
                            amount_accelerator_h=float(amount),
                            task_id=task_id,
                            flexibility_class=class_name,
                        )
                    )
    return (
        base_load_mw,
        rigid,
        jobs,
        float(np.sum(total_arrival)),
        reference_groups,
        reference_daily_mwh,
        kappa,
    )


def add_grid_supply(
    network: pypsa.Network,
    bus: str,
    existing_capacity_mw: float,
    flat_energy_cost_rmb_mwh: float,
    grid_annual_cost_rmb_mw: float,
) -> None:
    if existing_capacity_mw > 0:
        network.add(
            "Generator",
            f"{bus}_existing_grid",
            bus=bus,
            carrier="grid",
            p_nom=existing_capacity_mw,
            marginal_cost=flat_energy_cost_rmb_mwh,
        )
    network.add(
        "Generator",
        f"{bus}_expanded_grid",
        bus=bus,
        carrier="grid",
        p_nom_extendable=True,
        capital_cost=grid_annual_cost_rmb_mw,
        marginal_cost=flat_energy_cost_rmb_mwh,
    )


def add_der(
    network: pypsa.Network,
    bus: str,
    pv_max_mw: float,
    pv_profile: np.ndarray,
    pv_annual_cost_rmb_mw: float,
    battery_annual_cost_rmb_mw: float,
) -> None:
    if pv_max_mw > 0:
        network.add(
            "Generator",
            f"{bus}_rooftop_pv",
            bus=bus,
            carrier="solar",
            p_nom_extendable=True,
            p_nom_max=pv_max_mw,
            p_max_pu=pv_profile,
            capital_cost=pv_annual_cost_rmb_mw,
            marginal_cost=0.0,
        )
    network.add(
        "StorageUnit",
        f"{bus}_battery",
        bus=bus,
        carrier="battery",
        p_nom_extendable=True,
        max_hours=BATTERY_DURATION_H,
        efficiency_store=math.sqrt(BATTERY_ROUNDTRIP_EFFICIENCY),
        efficiency_dispatch=math.sqrt(BATTERY_ROUNDTRIP_EFFICIENCY),
        cyclic_state_of_charge=True,
        capital_cost=battery_annual_cost_rmb_mw,
        marginal_cost=0.0,
    )


def run_scenario(
    scenario: str,
    base_load_mw: np.ndarray,
    rigid: np.ndarray,
    jobs: list[FlexibleArrival],
    daily_service: float,
    costs: dict[str, float],
    pv_max_mw: float,
    pv_profile: np.ndarray,
    technology: ServerTechnology,
    local_pue: float,
    cloud_pue: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    network = pypsa.Network()
    snapshots = pd.RangeIndex(24, name="snapshot")
    network.set_snapshots(snapshots)
    network.snapshot_weightings.loc[:, "objective"] = ANNUAL_DAYS
    network.snapshot_weightings.loc[:, "stores"] = 1.0
    network.snapshot_weightings.loc[:, "generators"] = 1.0
    for carrier in ("electricity", "grid", "solar", "battery", "ai"):
        network.add("Carrier", carrier)
    network.add("Bus", "factory", carrier="electricity")
    network.add("Load", "factory_base", bus="factory", p_set=base_load_mw)
    existing_factory_capacity = float(np.max(base_load_mw)) * (
        1.0 + LOCAL_HEADROOM_SHARE
    )
    add_grid_supply(
        network,
        "factory",
        existing_factory_capacity,
        costs["flat_energy"],
        costs["grid_annual"],
    )
    add_der(
        network,
        "factory",
        pv_max_mw,
        pv_profile,
        costs["pv_annual"],
        costs["battery_annual"],
    )

    ai_bus: str | None = None
    ai_link: str | None = None
    pue = 0.0
    reserve_fraction = 0.0
    server_annual_cost = 0.0
    if scenario in {"local_ai", "cloud_ai"}:
        if scenario == "local_ai":
            ai_bus = "factory"
            pue = local_pue
            reserve_fraction = 0.15
            server_annual_cost = costs["local_server_annual"]
        else:
            network.add("Bus", "cloud", carrier="electricity")
            add_grid_supply(
                network,
                "cloud",
                0.0,
                costs["flat_energy"],
                costs["grid_annual"],
            )
            add_der(
                network,
                "cloud",
                0.0,
                pv_profile,
                costs["pv_annual"],
                costs["battery_annual"],
            )
            ai_bus = "cloud"
            pue = cloud_pue
            reserve_fraction = 0.10
            server_annual_cost = costs["cloud_server_annual"]
        sink_bus = f"{ai_bus}_ai_sink"
        ai_link = f"{ai_bus}_ai_power"
        network.add("Bus", sink_bus, carrier="ai")
        network.add(
            "Link",
            ai_link,
            bus0=ai_bus,
            bus1=sink_bus,
            efficiency=1.0,
            p_nom=1e7,
            p_min_pu=0.0,
            p_max_pu=1.0,
        )
        network.add(
            "Generator",
            f"{sink_bus}_use",
            bus=sink_bus,
            carrier="ai",
            sign=-1.0,
            p_nom=1e7,
            marginal_cost=0.0,
        )

    custom: dict[str, object] = {}

    def add_ai_constraints(n: pypsa.Network, _: pd.Index) -> None:
        model = n.model
        grid_dispatch = model.variables["Generator-p"]
        for bus in ("factory", "cloud"):
            generator_names = [
                generator
                for generator in n.generators.index
                if n.generators.at[generator, "bus"] == bus
                and n.generators.at[generator, "carrier"] == "grid"
            ]
            if not generator_names:
                continue
            demand_peak = model.add_variables(
                lower=0.0, name=f"{bus}-grid-demand-peak"
            )
            model.objective += costs["demand_annual"] * demand_peak
            bus_grid_dispatch = grid_dispatch.sel(name=generator_names).sum("name")
            for hour in range(24):
                model.add_constraints(
                    bus_grid_dispatch.sel(snapshot=hour) <= demand_peak,
                    name=f"{bus}-grid-demand-peak-{hour}",
                )
        if ai_link is None:
            return
        assignments: list[tuple[int, int, str]] = []
        for job_index, job in enumerate(jobs):
            admissible = sorted(
                {
                    (job.release_hour + offset) % 24
                    for offset in range(job.deadline_hours + 1)
                }
            )
            assignments.extend(
                (job_index, hour, job.flexibility_class) for hour in admissible
            )
        assignment_index = pd.Index(range(len(assignments)), name="assignment")
        execution = model.add_variables(
            lower=0.0, coords=[assignment_index], name="AI-task-execution"
        )
        servers = model.add_variables(lower=0.0, name="AI-server-groups")
        model.objective += server_annual_cost * servers
        for job_index, job in enumerate(jobs):
            indices = [
                index
                for index, (candidate, _, _) in enumerate(assignments)
                if candidate == job_index
            ]
            model.add_constraints(
                execution.sel(assignment=indices).sum()
                == job.amount_accelerator_h,
                name=f"AI-service-{job_index}",
            )
        link_power = model.variables["Link-p"].sel(name=ai_link)
        idle_mw_per_group = pue * technology.idle_power_kw / 1000.0
        dynamic_mw_per_accelerator = (
            pue
            * technology.dynamic_power_kw
            / technology.accelerators_per_group
            / 1000.0
        )
        hourly_expressions: list[object] = []
        for hour in range(24):
            indices = [
                index
                for index, (_, candidate_hour, _) in enumerate(assignments)
                if candidate_hour == hour
            ]
            flexible = execution.sel(assignment=indices).sum()
            total_execution = flexible + float(rigid[hour])
            model.add_constraints(
                total_execution * (1.0 + reserve_fraction)
                <= technology.accelerators_per_group * servers,
                name=f"AI-nameplate-{hour}",
            )
            model.add_constraints(
                link_power.sel(snapshot=hour)
                == idle_mw_per_group * servers
                + dynamic_mw_per_accelerator * total_execution,
                name=f"AI-power-{hour}",
            )
            hourly_expressions.append(total_execution)
        custom["assignments"] = assignments
        custom["execution_variable"] = execution
        custom["server_variable"] = servers
        custom["hourly_expressions"] = hourly_expressions

    status, condition = network.optimize(
        solver_name="highs",
        extra_functionality=add_ai_constraints,
        include_objective_constant=False,
        solver_options={"log_to_console": False},
    )
    if status != "ok" or condition != "optimal":
        raise RuntimeError(f"{scenario} PyPSA/HiGHS failed: {status}, {condition}")

    server_groups = 0.0
    executed = np.zeros(24)
    ai_power = np.zeros(24)
    if ai_link is not None:
        server_groups = float(custom["server_variable"].solution)
        assignments = custom["assignments"]
        assignment_solution = np.asarray(
            custom["execution_variable"].solution, dtype=float
        )
        executed = rigid.copy()
        for value, (_, hour, _) in zip(assignment_solution, assignments):
            executed[hour] += max(0.0, float(value))
        ai_power = network.links_t.p0[ai_link].to_numpy(dtype=float)

    def generator_capacity(name: str) -> float:
        if name not in network.generators.index:
            return 0.0
        return max(0.0, float(network.generators.at[name, "p_nom_opt"]))

    def storage_capacity(name: str) -> float:
        if name not in network.storage_units.index:
            return 0.0
        return max(0.0, float(network.storage_units.at[name, "p_nom_opt"]))

    factory_grid_expansion = generator_capacity("factory_expanded_grid")
    cloud_grid_expansion = generator_capacity("cloud_expanded_grid")
    factory_pv = generator_capacity("factory_rooftop_pv")
    factory_battery_mw = storage_capacity("factory_battery")
    cloud_battery_mw = storage_capacity("cloud_battery")
    grid_energy_mwh = 0.0
    for generator in network.generators.index:
        if network.generators.at[generator, "carrier"] == "grid":
            grid_energy_mwh += float(network.generators_t.p[generator].sum()) * ANNUAL_DAYS
    annual_server_cost = server_groups * server_annual_cost
    annual_grid_cost = (
        factory_grid_expansion + cloud_grid_expansion
    ) * costs["grid_annual"]
    annual_pv_cost = factory_pv * costs["pv_annual"]
    annual_battery_cost = (
        factory_battery_mw + cloud_battery_mw
    ) * costs["battery_annual"]
    annual_grid_energy_cost = grid_energy_mwh * costs["flat_energy"]
    factory_grid_profile = sum(
        network.generators_t.p[generator]
        for generator in network.generators.index
        if network.generators.at[generator, "bus"] == "factory"
        and network.generators.at[generator, "carrier"] == "grid"
    )
    cloud_grid_profile = sum(
        (
            network.generators_t.p[generator]
            for generator in network.generators.index
            if network.generators.at[generator, "bus"] == "cloud"
            and network.generators.at[generator, "carrier"] == "grid"
        ),
        start=pd.Series(0.0, index=snapshots),
    )
    annual_demand_cost = (
        float(factory_grid_profile.max()) + float(cloud_grid_profile.max())
    ) * costs["demand_annual"]
    reconstructed_objective = (
        annual_server_cost
        + annual_grid_cost
        + annual_pv_cost
        + annual_battery_cost
        + annual_grid_energy_cost
        + annual_demand_cost
    )
    objective = float(network.objective)
    if abs(objective - reconstructed_objective) > max(10.0, objective * 1e-7):
        raise ValueError(
            f"objective reconstruction mismatch: {objective} vs {reconstructed_objective}"
        )
    summary = {
        "industry_code": INDUSTRY_CODE,
        "industry_name_cn": INDUSTRY_NAME,
        "scenario": scenario,
        "solver": "PyPSA 1.2.2 + Linopy 0.7.0 + HiGHS 1.13.1",
        "objective_scope": "annualized_capacity_and_flat_energy_cost_proxy",
        "daily_ai_service_accelerator_h": daily_service if ai_link else 0.0,
        "server_groups_2xl20_continuous": server_groups,
        "server_accelerators": server_groups * technology.accelerators_per_group,
        "ai_site_pue": pue,
        "server_reserve_fraction": reserve_fraction,
        "annual_ai_facility_energy_twh": float(ai_power.sum()) * ANNUAL_DAYS / 1e6,
        "ai_facility_peak_mw": float(ai_power.max(initial=0.0)),
        "factory_rooftop_pv_mw": factory_pv,
        "factory_rooftop_pv_limit_mw": pv_max_mw,
        "factory_battery_power_mw": factory_battery_mw,
        "factory_battery_energy_mwh": factory_battery_mw * BATTERY_DURATION_H,
        "cloud_battery_power_mw": cloud_battery_mw,
        "cloud_battery_energy_mwh": cloud_battery_mw * BATTERY_DURATION_H,
        "factory_grid_expansion_mw": factory_grid_expansion,
        "cloud_grid_expansion_mw": cloud_grid_expansion,
        "total_grid_expansion_mw": factory_grid_expansion + cloud_grid_expansion,
        "factory_grid_import_peak_mw": float(factory_grid_profile.max()),
        "cloud_grid_import_peak_mw": float(cloud_grid_profile.max()),
        "annual_grid_energy_twh": grid_energy_mwh / 1e6,
        "annual_server_cost_rmb": annual_server_cost,
        "annual_grid_capacity_cost_rmb": annual_grid_cost,
        "annual_pv_cost_rmb": annual_pv_cost,
        "annual_battery_cost_rmb": annual_battery_cost,
        "annual_flat_energy_cost_rmb": annual_grid_energy_cost,
        "annual_maximum_demand_cost_rmb": annual_demand_cost,
        "annual_objective_rmb": objective,
    }

    hourly_rows: list[dict[str, object]] = []
    for hour in range(24):
        factory_grid = sum(
            float(network.generators_t.p.at[hour, generator])
            for generator in network.generators.index
            if network.generators.at[generator, "bus"] == "factory"
            and network.generators.at[generator, "carrier"] == "grid"
        )
        cloud_grid = sum(
            float(network.generators_t.p.at[hour, generator])
            for generator in network.generators.index
            if network.generators.at[generator, "bus"] == "cloud"
            and network.generators.at[generator, "carrier"] == "grid"
        )
        factory_pv_output = (
            float(network.generators_t.p.at[hour, "factory_rooftop_pv"])
            if "factory_rooftop_pv" in network.generators.index
            else 0.0
        )
        factory_battery = (
            float(network.storage_units_t.p.at[hour, "factory_battery"])
            if "factory_battery" in network.storage_units.index
            else 0.0
        )
        cloud_battery = (
            float(network.storage_units_t.p.at[hour, "cloud_battery"])
            if "cloud_battery" in network.storage_units.index
            else 0.0
        )
        hourly_rows.append(
            {
                "scenario": scenario,
                "hour": hour,
                "factory_base_load_mw": float(base_load_mw[hour]),
                "ai_executed_accelerator_h": float(executed[hour]),
                "ai_facility_power_mw": float(ai_power[hour]),
                "factory_rooftop_pv_mw": factory_pv_output,
                "factory_battery_mw_positive_discharge": factory_battery,
                "cloud_battery_mw_positive_discharge": cloud_battery,
                "factory_grid_import_mw": factory_grid,
                "cloud_grid_import_mw": cloud_grid,
            }
        )
    return summary, hourly_rows


def main() -> None:
    (
        base_load_mw,
        rigid,
        jobs,
        daily_service,
        reference_groups,
        reference_daily_mwh,
        kappa,
    ) = build_ai_inputs()
    params = parameter_core.read_parameters()
    discount = parameter_core.number(params, "U01")
    server_capex = parameter_core.number(params, "L02")
    maintenance = parameter_core.number(params, "L14")
    costs = {
        "flat_energy": parameter_core.number(params, "E01") * 1000.0,
        "demand_annual": parameter_core.number(params, "E05") * 1000.0 * 12.0,
        "grid_annual": parameter_core.number(params, "X01")
        * 10_000.0
        * capital_recovery_factor(discount, parameter_core.number(params, "U08")),
        "pv_annual": parameter_core.number(params, "U06")
        * 1_000_000.0
        * capital_recovery_factor(discount, parameter_core.number(params, "U09")),
        "battery_annual": parameter_core.number(params, "U07")
        * 1_000_000.0
        * BATTERY_DURATION_H
        * capital_recovery_factor(discount, parameter_core.number(params, "U10")),
        "local_server_annual": server_capex
        * (1.0 + parameter_core.number(params, "L15"))
        * capital_recovery_factor(discount, parameter_core.number(params, "L13"))
        + server_capex * maintenance,
        "cloud_server_annual": server_capex
        * (1.0 + parameter_core.number(params, "U11"))
        * capital_recovery_factor(discount, parameter_core.number(params, "L13"))
        + server_capex * maintenance,
    }
    technology = ServerTechnology(
        accelerators_per_group=parameter_core.number(params, "L03"),
        idle_power_kw=parameter_core.number(params, "L10"),
        maximum_power_kw=parameter_core.number(params, "L09"),
    )
    firms = float(
        pd.read_csv(
            RESULTS / "manufacturing_ai_task_hardware_demo_task_results.csv",
            encoding="utf-8-sig",
        )
        .query("industry_code == @INDUSTRY_CODE and year == 2030")[
            "above_size_firms_2023"
        ]
        .iloc[0]
    )
    pv_max_mw = (
        firms
        * ROOF_AREA_PER_FIRM_M2
        * ROOF_USABLE_FRACTION
        * PV_MODULE_EFFICIENCY
        * PV_REALIZATION_FRACTION
        / 1000.0
    )
    pv_profile = pv_capacity_factor()
    summaries: list[dict[str, object]] = []
    hourly_rows: list[dict[str, object]] = []
    for scenario in ("baseline_no_ai", "local_ai", "cloud_ai"):
        summary, hourly = run_scenario(
            scenario,
            base_load_mw,
            rigid,
            jobs,
            daily_service,
            costs,
            pv_max_mw,
            pv_profile,
            technology,
            parameter_core.number(params, "L17"),
            parameter_core.number(params, "U02"),
        )
        summaries.append(summary)
        hourly_rows.extend(hourly)

    summary_frame = pd.DataFrame(summaries)
    baseline_objective = float(
        summary_frame.query("scenario == 'baseline_no_ai'")["annual_objective_rmb"].iloc[0]
    )
    summary_frame["incremental_vs_baseline_rmb"] = (
        summary_frame["annual_objective_rmb"] - baseline_objective
    )
    baseline_row = summary_frame.query("scenario == 'baseline_no_ai'").iloc[0]
    for field in (
        "factory_rooftop_pv_mw",
        "factory_battery_power_mw",
        "factory_battery_energy_mwh",
        "factory_grid_expansion_mw",
    ):
        summary_frame[f"incremental_{field}"] = (
            summary_frame[field] - float(baseline_row[field])
        )
    summary_frame["reference_cloud_groups_fixed_calibration"] = reference_groups
    summary_frame["reference_cloud_daily_energy_mwh"] = reference_daily_mwh
    summary_frame["service_scale_kappa"] = kappa
    summary_frame.to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8-sig")
    pd.DataFrame(hourly_rows).to_csv(HOURLY_OUTPUT, index=False, encoding="utf-8-sig")

    local = summary_frame.query("scenario == 'local_ai'").iloc[0]
    cloud = summary_frame.query("scenario == 'cloud_ai'").iloc[0]
    cost_gap = float(local["incremental_vs_baseline_rmb"] - cloud["incremental_vs_baseline_rmb"])
    findings = f"""# C36 PyPSA联合容量—调度原型

## 方法边界

- 使用 PyPSA 1.2.2、Linopy 0.7.0 和 HiGHS 1.13.1，只运行C36汽车制造业行业负荷桶。
- 运行无AI基线、本地AI和云端AI三个反事实；AI结果均与无AI基线做增量比较。
- 采用统一平段电量价，不设置峰谷价差；按最大需量基本电费计入节点峰值成本，并联合优化服务器、任务时序、屋顶光伏、两小时储能和新增接入容量。
- C36屋顶上限暂按18899家规上企业、每家4000平方米投影屋顶、90%可用率、22%组件效率和80%实现比例构造，合计上限 {pv_max_mw:.1f} MWp。它是透明的规模代理，不是汽车行业屋顶普查值。
- 云端基础情景允许两小时储能，但暂不赋予屋顶光伏潜力。

## 初步结果

| 情景 | 双L20服务器组 | AI年电量(TWh) | AI峰值(MW) | 工厂PV(MW) | 工厂电池(MW/MWh) | 云端电池(MW/MWh) | 新增接入(MW) | AI增量年成本(亿元) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 本地AI | {local['server_groups_2xl20_continuous']:.1f} | {local['annual_ai_facility_energy_twh']:.3f} | {local['ai_facility_peak_mw']:.1f} | {local['factory_rooftop_pv_mw']:.1f} | {local['factory_battery_power_mw']:.1f}/{local['factory_battery_energy_mwh']:.1f} | 0/0 | {local['total_grid_expansion_mw']:.1f} | {local['incremental_vs_baseline_rmb']/1e8:.2f} |
| 云端AI | {cloud['server_groups_2xl20_continuous']:.1f} | {cloud['annual_ai_facility_energy_twh']:.3f} | {cloud['ai_facility_peak_mw']:.1f} | {cloud['factory_rooftop_pv_mw']:.1f} | {cloud['factory_battery_power_mw']:.1f}/{cloud['factory_battery_energy_mwh']:.1f} | {cloud['cloud_battery_power_mw']:.1f}/{cloud['cloud_battery_energy_mwh']:.1f} | {cloud['total_grid_expansion_mw']:.1f} | {cloud['incremental_vs_baseline_rmb']/1e8:.2f} |

当前参数下，无AI基线已经选择铺满 {float(baseline_row['factory_rooftop_pv_mw']):.1f} MWp屋顶光伏，并建设 {float(baseline_row['factory_battery_power_mw']):.1f} MW/{float(baseline_row['factory_battery_energy_mwh']):.1f} MWh两小时储能；本地AI和云端AI均未进一步增加工厂光储，因此不能把这些投资归因于AI。储能没有峰谷电价套利收入，本轮建设完全来自降低基础负荷最大需量的价值；云端AI在可平移计算已经压平负荷后没有再建设储能。

本地AI利用工厂侧既有接入容量以及基线光储形成的净负荷条件，使AI引致新增接入保持为0；绿地云端需要 {cloud['cloud_grid_expansion_mw']:.1f} MW。与此同时，本地PUE和备用比例较高，AI年电量比云端高 {local['annual_ai_facility_energy_twh']-cloud['annual_ai_facility_energy_twh']:.3f} TWh，双L20服务器组多 {local['server_groups_2xl20_continuous']-cloud['server_groups_2xl20_continuous']:.1f} 组。综合当前年化成本，本地AI增量成本比云端高 {cost_gap/1e8:.2f} 亿元；也就是说，避免的绿地接入容量成本尚不足以抵消本地服务器与设施效率劣势。

## 解释限制

这个结果是联合容量选择的机制测试，不再把服务器台数预先固定，也不把叠加峰值本身设成唯一目标。服务器、DER和电网容量通过年化成本直接权衡。当前服务器组允许连续取值，适用于行业桶；单企业案例再恢复整数最小采购单位。零售平段电价用于企业侧资源选择代理，不等于社会边际发电成本。屋顶、光伏曲线、两小时储能以及线性扩容成本均需在正式行业研究中替换或做情景分析。
"""
    FINDINGS_OUTPUT.write_text(findings, encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(SUMMARY_OUTPUT),
                "hourly": str(HOURLY_OUTPUT),
                "findings": str(FINDINGS_OUTPUT),
                "rows": len(summary_frame),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
