"""PyPSA/Linopy model for one representative AI host factory."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
import pypsa

from core.data import FlexibleJob, read_battery_cost_parameters


@dataclass(frozen=True)
class HostResult:
    summary: dict[str, Any]
    hourly: pd.DataFrame


def capital_recovery_factor(rate: float, years: float) -> float:
    return rate * (1.0 + rate) ** years / ((1.0 + rate) ** years - 1.0)


def factory_pv_limit_mw(config: dict, roof_area_m2: float) -> float:
    factory = config["factory"]
    return (
        float(roof_area_m2)
        * float(factory["roof_usable_fraction"])
        * float(factory["pv_module_efficiency"])
        * float(factory["pv_realization_fraction"])
        / 1000.0
    )


def annual_costs(config: dict) -> dict[str, float]:
    model = config["model"]
    server = config["server"]
    energy = config["energy"]
    battery = read_battery_cost_parameters(config)
    discount = float(model["discount_rate"])
    server_capex = float(server["purchase_cost_rmb"])
    model_state = config["model_state"]
    initialization_energy_kwh = (
        float(model_state["initialization_accelerator_h_per_version"])
        * float(model_state["major_versions_per_year"])
        * float(server["maximum_wall_power_kw"])
        / float(server["accelerators_per_server"])
        * float(server["marginal_facility_multiplier"])
    )
    return {
        "flat_energy_rmb_per_mwh": float(energy["flat_grid_energy_rmb_per_kwh"]) * 1000.0,
        "demand_rmb_per_mw_year": float(energy["maximum_demand_rmb_per_kw_month"])
        * 1000.0
        * 12.0,
        "pv_rmb_per_mw_year": (
            0.0
            if energy["pv_capacity_mode"] in {"none", "existing_rooftop_at_model_limit"}
            else float(energy["pv_capex_rmb_per_w"])
            * 1_000_000.0
            * capital_recovery_factor(discount, float(energy["pv_asset_life_years"]))
        ),
        "battery_rmb_per_mw_year": battery.annualized_cost_rmb_per_mw_year(
            float(energy["battery_duration_h"]), discount
        ),
        "battery_energy_capex_rmb_per_kwh": battery.energy_capex_rmb_per_kwh,
        "battery_power_capex_rmb_per_kw": battery.power_capex_rmb_per_kw,
        "battery_energy_lifetime_years": battery.energy_lifetime_years,
        "battery_power_lifetime_years": battery.power_lifetime_years,
        "battery_power_fom_fraction_per_year": battery.power_fom_fraction_per_year,
        "server_rmb_per_group_year": server_capex
        * (1.0 + float(server["facility_capex_fraction"]))
        * capital_recovery_factor(discount, float(server["economic_life_years"]))
        + server_capex * float(server["annual_maintenance_fraction"]),
        "model_initialization_energy_kwh_per_deployment_year": initialization_energy_kwh,
        "model_initialization_rmb_per_deployment_year": initialization_energy_kwh
        * float(energy["flat_grid_energy_rmb_per_kwh"]),
        "model_operations_rmb_per_deployment_year": float(model_state["operations_fte_per_deployment"])
        * float(model_state["loaded_cost_rmb_per_fte_year"]),
        "model_storage_rmb_per_server_group_year": float(model_state["incremental_storage_capex_rmb_per_server_group"])
        * capital_recovery_factor(discount, float(model_state["storage_lifetime_years"])),
    }


def annual_server_cost_per_group(server: dict, discount_rate: float) -> float:
    capex = float(server["purchase_cost_rmb"])
    return (
        capex
        * (1.0 + float(server["facility_capex_fraction"]))
        * capital_recovery_factor(discount_rate, float(server["economic_life_years"]))
        + capex * float(server["annual_maintenance_fraction"])
    )


def add_grid(
    network: pypsa.Network,
    config: dict,
    existing_capacity_mw: float,
    costs: dict[str, float],
) -> None:
    energy = config["energy"]
    grid_limit = energy.get("grid_expansion_limit_mw")
    grid_penalty = float(
        energy.get("grid_expansion_objective_penalty_rmb_per_mw_year", 0.0)
    )
    network.add(
        "Generator",
        "existing_grid",
        bus="factory",
        carrier="grid",
        p_nom=existing_capacity_mw,
        marginal_cost=costs["flat_energy_rmb_per_mwh"],
    )
    network.add(
        "Generator",
        "expanded_grid",
        bus="factory",
        carrier="grid",
        p_nom_extendable=True,
        p_nom_max=np.inf if grid_limit is None else float(grid_limit),
        # Zero in the core case. Structural sensitivity cases may impose a
        # scarcity penalty without changing the reported physical MW metric.
        capital_cost=grid_penalty,
        marginal_cost=costs["flat_energy_rmb_per_mwh"],
    )


def add_der(
    network: pypsa.Network,
    config: dict,
    pv_capacity_factor: np.ndarray,
    costs: dict[str, float],
    roof_area_m2: float,
) -> None:
    pv_limit = factory_pv_limit_mw(config, roof_area_m2)
    pv_mode = str(config["energy"]["pv_capacity_mode"])
    if pv_mode not in {"none", "existing_rooftop_at_model_limit", "optimize_new_rooftop_pv"}:
        raise ValueError(f"Unsupported PV capacity mode: {pv_mode}")
    pv_enabled = pv_mode != "none"
    network.add(
        "Generator",
        "rooftop_pv",
        bus="factory",
        carrier="solar",
        p_nom=pv_limit if pv_mode == "existing_rooftop_at_model_limit" else 0.0,
        p_nom_extendable=pv_mode == "optimize_new_rooftop_pv",
        p_nom_max=pv_limit if pv_mode == "optimize_new_rooftop_pv" else (0.0 if not pv_enabled else np.inf),
        p_max_pu=pv_capacity_factor,
        capital_cost=costs["pv_rmb_per_mw_year"],
        marginal_cost=0.0,
    )
    energy = config["energy"]
    efficiency = math.sqrt(float(energy["battery_roundtrip_efficiency"]))
    battery_enabled = bool(energy["battery_investment_enabled"])
    fixed_battery = energy.get("battery_fixed_power_mw")
    battery_as_variable = battery_enabled or fixed_battery is not None
    network.add(
        "StorageUnit",
        "battery",
        bus="factory",
        carrier="battery",
        p_nom=0.0,
        p_nom_extendable=battery_as_variable,
        p_nom_min=0.0 if fixed_battery is None else float(fixed_battery),
        p_nom_max=np.inf if fixed_battery is None else float(fixed_battery),
        max_hours=float(energy["battery_duration_h"]),
        efficiency_store=efficiency,
        efficiency_dispatch=efficiency,
        cyclic_state_of_charge=True,
        capital_cost=costs["battery_rmb_per_mw_year"],
        marginal_cost=0.0,
    )


def resolve_existing_grid_capacity(
    base_load_mw: np.ndarray,
    existing_grid_capacity_mw: float | None,
) -> float:
    """Return the raw-load baseline capacity or an explicitly supplied net-peak boundary."""
    capacity = (
        float(np.max(base_load_mw))
        if existing_grid_capacity_mw is None
        else float(existing_grid_capacity_mw)
    )
    if not np.isfinite(capacity) or capacity < 0.0:
        raise ValueError("Existing grid capacity must be finite and non-negative")
    return capacity


def incremental_grid_capacity_mw(
    grid_import_peak_mw: float,
    matched_no_ai_peak_mw: float,
) -> float:
    """Return the physical positive net-peak increment without monetizing it."""
    values = (float(grid_import_peak_mw), float(matched_no_ai_peak_mw))
    if not all(np.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("Grid-import peaks must be finite and non-negative")
    return max(0.0, values[0] - values[1])


def optimize_host(
    config: dict,
    base_load_mw: np.ndarray,
    pv_capacity_factor: np.ndarray,
    roof_area_m2: float,
    rigid_service_units: np.ndarray | None = None,
    flexible_jobs: tuple[FlexibleJob, ...] = (),
    grid_energy_price_rmb_per_mwh: np.ndarray | None = None,
    existing_grid_capacity_mw: float | None = None,
    minimum_installed_server_groups: float | None = None,
    rigid_service_units_by_task: dict[str, np.ndarray] | None = None,
    heterogeneous_hardware: dict[str, Any] | None = None,
    minimum_installed_hardware_groups: dict[str, float] | None = None,
) -> HostResult:
    horizon_hours = int(config["model"]["horizon_hours"])
    represented_days = horizon_hours / 24.0
    base = np.asarray(base_load_mw, dtype=float)
    pv_profile = np.asarray(pv_capacity_factor, dtype=float)
    if base.shape != (horizon_hours,) or pv_profile.shape != (horizon_hours,):
        raise ValueError(f"Core host model requires {horizon_hours} hourly values")
    costs = annual_costs(config)
    grid_prices = (
        np.asarray(grid_energy_price_rmb_per_mwh, dtype=float)
        if grid_energy_price_rmb_per_mwh is not None
        else np.full(horizon_hours, costs["flat_energy_rmb_per_mwh"], dtype=float)
    )
    if grid_prices.shape != (horizon_hours,) or not np.isfinite(grid_prices).all():
        raise ValueError(f"Grid-energy prices must contain {horizon_hours} finite hourly values")
    costs["model_initialization_rmb_per_deployment_year"] = (
        costs["model_initialization_energy_kwh_per_deployment_year"]
        * float(np.mean(grid_prices))
        / 1000.0
    )
    include_ai = rigid_service_units is not None or rigid_service_units_by_task is not None
    rigid = (
        np.asarray(rigid_service_units, dtype=float)
        if rigid_service_units is not None
        else np.zeros(horizon_hours)
    )
    if rigid.shape != (horizon_hours,):
        raise ValueError(f"Rigid AI service must contain {horizon_hours} hours")
    task_rigid = {
        str(task): np.asarray(values, dtype=float)
        for task, values in (rigid_service_units_by_task or {}).items()
    }
    if task_rigid and any(values.shape != (horizon_hours,) for values in task_rigid.values()):
        raise ValueError(f"Every task-resolved rigid AI profile must contain {horizon_hours} hours")
    if task_rigid and not np.allclose(sum(task_rigid.values(), np.zeros(horizon_hours)), rigid):
        rigid = sum(task_rigid.values(), np.zeros(horizon_hours))
    heterogeneous = heterogeneous_hardware is not None
    if heterogeneous and not task_rigid:
        raise ValueError("Heterogeneous hardware requires task-resolved rigid service")

    annual_days = float(config["model"]["annualization_days"])
    annual_periods = annual_days / represented_days
    network = pypsa.Network()
    snapshots = pd.RangeIndex(horizon_hours, name="snapshot")
    network.set_snapshots(snapshots)
    network.snapshot_weightings.loc[:, "objective"] = annual_periods
    network.snapshot_weightings.loc[:, "stores"] = 1.0
    network.snapshot_weightings.loc[:, "generators"] = 1.0
    for carrier in ("electricity", "grid", "solar", "battery", "ai"):
        network.add("Carrier", carrier)
    network.add("Bus", "factory", carrier="electricity")
    network.add("Load", "factory_base", bus="factory", p_set=base)
    # The no-AI solve starts from the physical base-load peak.  Each subsequent
    # AI scenario must instead receive the optimized no-AI net grid-import peak
    # under the same PV/storage investment rules.  This prevents storage-created
    # headroom in the baseline from being credited free of charge to AI load.
    existing_capacity = resolve_existing_grid_capacity(base, existing_grid_capacity_mw)
    add_grid(network, config, existing_capacity, costs)
    network.generators_t.marginal_cost = pd.DataFrame(
        {
            "existing_grid": grid_prices,
            "expanded_grid": grid_prices,
        },
        index=snapshots,
    )
    if not np.isfinite(roof_area_m2) or roof_area_m2 <= 0.0:
        raise ValueError("Industry rooftop-area proxy must be finite and positive")
    add_der(network, config, pv_profile, costs, roof_area_m2)

    ai_link: str | None = None
    if include_ai:
        network.add("Bus", "ai_sink", carrier="ai")
        ai_link = "ai_power"
        network.add(
            "Link",
            ai_link,
            bus0="factory",
            bus1="ai_sink",
            efficiency=1.0,
            p_nom=1e7,
            p_min_pu=0.0,
            p_max_pu=1.0,
        )
        network.add(
            "Generator",
            "ai_use",
            bus="ai_sink",
            carrier="ai",
            sign=-1.0,
            p_nom=1e7,
            marginal_cost=0.0,
        )

    custom: dict[str, Any] = {}

    def add_custom_constraints(n: pypsa.Network, _: pd.Index) -> None:
        model = n.model
        grid_dispatch = model.variables["Generator-p"].sel(
            name=["existing_grid", "expanded_grid"]
        ).sum("name")
        demand_peak = model.add_variables(lower=0.0, name="factory-grid-demand-peak")
        model.objective += costs["demand_rmb_per_mw_year"] * demand_peak
        for hour in range(horizon_hours):
            model.add_constraints(
                grid_dispatch.sel(snapshot=hour) <= demand_peak,
                name=f"factory-grid-demand-peak-{hour}",
            )
        custom["demand_peak"] = demand_peak
        if not include_ai or ai_link is None:
            return

        aggregated_amounts: dict[tuple[int, int, str, str], float] = {}
        for job in flexible_jobs:
            key = (job.release_hour, job.deadline_hours, job.flexibility_class, job.task_id)
            aggregated_amounts[key] = aggregated_amounts.get(key, 0.0) + job.amount_service_units
        aggregated_jobs = [
            FlexibleJob(
                release_hour=release,
                deadline_hours=deadline,
                amount_service_units=amount,
                task_id=task_id,
                flexibility_class=class_name,
            )
            for (release, deadline, class_name, task_id), amount in sorted(aggregated_amounts.items())
        ]
        assignments: list[tuple[int, int, str, str]] = []
        for job_index, job in enumerate(aggregated_jobs):
            if job.deadline_hours >= horizon_hours:
                raise ValueError("Flexible deadline must be shorter than the continuous horizon")
            admissible = sorted(
                {
                    (job.release_hour + offset) % horizon_hours
                    for offset in range(job.deadline_hours + 1)
                }
            )
            assignments.extend(
                (job_index, hour, job.flexibility_class, job.task_id) for hour in admissible
            )
        assignment_index = pd.Index(range(len(assignments)), name="assignment")
        execution = model.add_variables(
            lower=0.0,
            coords=[assignment_index],
            name="AI-task-execution",
        )
        hybrid_cloud = config.get("hybrid_cloud", {})
        cloud_enabled = bool(hybrid_cloud.get("enabled", False))
        cloud_hardware_index = pd.Index(
            ["gpu", "cpu"] if heterogeneous else ["gpu"], name="cloud_hardware"
        )
        cloud_prices = {
            "gpu": float(hybrid_cloud.get("gpu_annual_subscription_rmb_per_group", 0.0)),
            "cpu": float(hybrid_cloud.get("cpu_annual_subscription_rmb_per_group", 0.0)),
        }
        cloud_execution = None
        cloud_reserved = None
        if cloud_enabled:
            cloud_execution = model.add_variables(
                lower=0.0,
                coords=[assignment_index],
                name="AI-cloud-task-execution",
            )
            cloud_reserved = model.add_variables(
                lower=0.0,
                coords=[cloud_hardware_index],
                name="AI-cloud-reserved-groups",
            )
            model.objective += sum(
                cloud_prices[hardware] * cloud_reserved.sel(cloud_hardware=hardware)
                for hardware in cloud_hardware_index
            )
            rigid_pairs = [
                (hour, task)
                for task, values in (
                    task_rigid.items()
                    if heterogeneous
                    else {"aggregate": rigid}.items()
                )
                for hour, value in enumerate(values)
                if float(value) > 0.0
            ]
            rigid_cloud_index = pd.Index(
                range(len(rigid_pairs)), name="rigid_cloud_assignment"
            )
            rigid_cloud_upper = np.asarray(
                [
                    float(
                        (task_rigid if heterogeneous else {"aggregate": rigid})[
                            task
                        ][hour]
                    )
                    for hour, task in rigid_pairs
                ],
                dtype=float,
            )
            cloud_rigid_execution = model.add_variables(
                lower=0.0,
                upper=rigid_cloud_upper,
                coords=[rigid_cloud_index],
                name="AI-cloud-rigid-task-execution",
            )
        else:
            rigid_pairs = []
            cloud_rigid_execution = None
        legacy_integer = bool(config["model"]["server_groups_integer"])
        installed_integer = bool(
            config["model"].get("installed_server_groups_integer", legacy_integer)
        )
        online_integer = bool(
            config["model"].get("online_server_groups_integer", legacy_integer)
        )
        hardware_names = ["gpu", "cpu"] if heterogeneous else ["gpu"]
        hardware_index = pd.Index(hardware_names, name="hardware")
        installed = model.add_variables(lower=0.0, integer=installed_integer, coords=[hardware_index], name="AI-installed-server-groups")
        floors = minimum_installed_hardware_groups or ({"gpu": minimum_installed_server_groups} if minimum_installed_server_groups is not None else {})
        for hardware, floor in floors.items():
            minimum_installed = float(floor)
            if hardware not in hardware_names or not np.isfinite(minimum_installed) or minimum_installed < 0.0:
                raise ValueError("Minimum installed hardware groups must be valid and non-negative")
            model.add_constraints(installed.sel(hardware=hardware) >= minimum_installed, name=f"AI-external-minimum-installed-{hardware}-groups")
        model_state = config["model_state"]
        # A hybrid subscription case may optimally have no local footprint at
        # all.  Requiring a permanently online local replica would make a
        # zero-headroom connection infeasible even when every task is served
        # by the cloud.  The ordinary local-only cases retain the replica and
        # bundled-storage floor unchanged.
        if bool(model_state["enabled"]) and not cloud_enabled:
            model.add_constraints(
                installed.sel(hardware="gpu") >= int(model_state["minimum_server_groups_for_model_state"]),
                name="AI-minimum-model-replica-and-vram",
            )
            model.add_constraints(
                installed.sel(hardware="gpu") * float(model_state["bundled_storage_gb_per_server_group"])
                >= float(model_state["model_storage_required_gb_per_deployment"]),
                name="AI-model-storage-capacity",
            )
        online = model.add_variables(
            lower=0.0,
            integer=online_integer,
            coords=[snapshots, hardware_index],
            name="AI-online-server-groups",
        )
        servers = {"gpu": config["server"]}
        if heterogeneous:
            servers["cpu"] = heterogeneous_hardware["cpu_server"]
        server_costs = {
            hardware: annual_server_cost_per_group(server, float(config["model"]["discount_rate"]))
            for hardware, server in servers.items()
        }
        model.objective += sum(server_costs[hardware] * installed.sel(hardware=hardware) for hardware in hardware_names)
        for job_index, job in enumerate(aggregated_jobs):
            indices = [
                index
                for index, (candidate, _, _, _) in enumerate(assignments)
                if candidate == job_index
            ]
            service_expression = execution.sel(assignment=indices).sum()
            if cloud_enabled:
                service_expression += cloud_execution.sel(assignment=indices).sum()
            model.add_constraints(
                service_expression == job.amount_service_units,
                name=f"AI-service-{job_index}",
            )
        total_flexible_service = sum(job.amount_service_units for job in aggregated_jobs)
        total_rigid_service = float(sum(values.sum() for values in task_rigid.values())) if heterogeneous else float(rigid.sum())
        if cloud_enabled:
            maximum_cloud_share = float(hybrid_cloud["maximum_cloud_service_share"])
            model.add_constraints(
                cloud_execution.sum() + cloud_rigid_execution.sum()
                <= maximum_cloud_share * (total_flexible_service + total_rigid_service),
                name="AI-cloud-maximum-service-share",
            )

        accelerator_h_per_service_unit = float(
            config["demand"]["effective_service"]["accelerator_h_per_service_unit"]
        )
        cpu_fractions = heterogeneous_hardware.get("cpu_fraction_by_task", {}) if heterogeneous else {}
        cpu_multipliers = heterogeneous_hardware.get("cpu_compute_multiplier_by_task", {}) if heterogeneous else {}
        minimum_online = (
            0 if cloud_enabled else int(config["model"]["minimum_online_servers"])
        )
        link_power = model.variables["Link-p"].sel(name=ai_link)
        hourly_expressions: list[Any] = []
        hardware_compute_expressions: dict[str, list[Any]] = {
            hardware: [] for hardware in hardware_names
        }
        cloud_compute_expressions: dict[str, list[Any]] = {
            hardware: [] for hardware in hardware_names
        }
        cloud_task_service_expressions: dict[str, list[Any]] = {}
        for hour in range(horizon_hours):
            task_service: dict[str, Any] = {}
            task_ids = sorted(set(task_rigid) | {job.task_id for job in aggregated_jobs}) if heterogeneous else ["aggregate"]
            for task in task_ids:
                indices = [index for index, (_, candidate_hour, _, candidate_task) in enumerate(assignments) if candidate_hour == hour and (candidate_task == task or not heterogeneous)]
                flexible = execution.sel(assignment=indices).sum()
                rigid_indices = [
                    index
                    for index, pair in enumerate(rigid_pairs)
                    if pair == (hour, task)
                ]
                cloud_rigid = (
                    cloud_rigid_execution.sel(
                        rigid_cloud_assignment=rigid_indices
                    ).sum()
                    if cloud_enabled
                    else 0.0
                )
                rigid_value = float(task_rigid[task][hour]) if heterogeneous else float(rigid[hour])
                task_service[task] = flexible + rigid_value - cloud_rigid
            cloud_task_service: dict[str, Any] = {}
            for task in task_ids:
                indices = [index for index, (_, candidate_hour, _, candidate_task) in enumerate(assignments) if candidate_hour == hour and (candidate_task == task or not heterogeneous)]
                rigid_indices = [
                    index
                    for index, pair in enumerate(rigid_pairs)
                    if pair == (hour, task)
                ]
                cloud_task_service[task] = (
                    cloud_execution.sel(assignment=indices).sum()
                    + cloud_rigid_execution.sel(
                        rigid_cloud_assignment=rigid_indices
                    ).sum()
                    if cloud_enabled
                    else 0.0
                )
                cloud_task_service_expressions.setdefault(task, []).append(
                    cloud_task_service[task]
                )
            total_service = sum(task_service.values())
            power_terms = []
            for hardware in hardware_names:
                server = servers[hardware]
                capacity = float(server["accelerators_per_server"]) if hardware == "gpu" else float(server["service_capacity_cpu_server_h_per_h"])
                compute = 0
                for task, service_value in task_service.items():
                    cpu_fraction = float(cpu_fractions.get(task, 0.0))
                    fraction = cpu_fraction if hardware == "cpu" else 1.0 - cpu_fraction
                    multiplier = float(cpu_multipliers.get(task, 1.0)) if hardware == "cpu" else 1.0
                    compute += service_value * accelerator_h_per_service_unit * fraction * multiplier
                hardware_compute_expressions[hardware].append(compute)
                cloud_compute = 0
                for task, service_value in cloud_task_service.items():
                    cpu_fraction = float(cpu_fractions.get(task, 0.0))
                    fraction = cpu_fraction if hardware == "cpu" else 1.0 - cpu_fraction
                    multiplier = float(cpu_multipliers.get(task, 1.0)) if hardware == "cpu" else 1.0
                    cloud_compute += service_value * accelerator_h_per_service_unit * fraction * multiplier
                cloud_compute_expressions[hardware].append(cloud_compute)
                cloud_capacity = (
                    float(hybrid_cloud.get("gpu_capacity_accelerator_h_per_h", 1.0))
                    if hardware == "gpu"
                    else float(hybrid_cloud.get("cpu_capacity_local_server_h_per_h", 1.0))
                )
                if cloud_enabled:
                    model.add_constraints(
                        cloud_compute
                        <= cloud_capacity
                        * cloud_reserved.sel(cloud_hardware=hardware),
                        name=f"AI-cloud-{hardware}-reserved-throughput-{hour}",
                    )
                online_hour = online.sel(snapshot=hour, hardware=hardware)
                installed_hardware = installed.sel(hardware=hardware)
                planning_reserve = float(server["installed_reserve_fraction"])
                dispatch_reserve = float(server.get("normal_dispatch_reserve_fraction", 0.0))
                model.add_constraints(online_hour <= installed_hardware, name=f"AI-{hardware}-online-below-installed-{hour}")
                model.add_constraints(online_hour >= (minimum_online if hardware == "gpu" else 0), name=f"AI-{hardware}-minimum-online-{hour}")
                model.add_constraints(compute <= capacity * online_hour, name=f"AI-{hardware}-online-throughput-{hour}")
                if hardware not in floors:
                    model.add_constraints(compute * (1.0 + planning_reserve) <= capacity * installed_hardware, name=f"AI-{hardware}-planning-reserve-{hour}")
                model.add_constraints(compute * (1.0 + dispatch_reserve) <= capacity * installed_hardware, name=f"AI-{hardware}-dispatch-reserve-{hour}")
                pue = float(server["marginal_facility_multiplier"])
                maximum_kw = float(server["maximum_wall_power_kw"])
                idle_kw = float(server["online_idle_wall_power_kw"])
                standby_kw = float(server["cold_spare_standby_power_kw"])
                power_terms.append(pue / 1000.0 * (standby_kw * installed_hardware + (idle_kw - standby_kw) * online_hour + (maximum_kw - idle_kw) / capacity * compute))
            model.add_constraints(link_power.sel(snapshot=hour) == sum(power_terms), name=f"AI-facility-power-{hour}")
            hourly_expressions.append(total_service)
        custom.update(
            {
                "assignments": assignments,
                "aggregated_jobs": aggregated_jobs,
                "execution": execution,
                "cloud_enabled": cloud_enabled,
                "cloud_execution": cloud_execution,
                "cloud_rigid_execution": cloud_rigid_execution,
                "rigid_pairs": rigid_pairs,
                "cloud_reserved": cloud_reserved,
                "cloud_prices": cloud_prices,
                "installed": installed,
                "online": online,
                "hourly_expressions": hourly_expressions,
                "hardware_compute_expressions": hardware_compute_expressions,
                "cloud_compute_expressions": cloud_compute_expressions,
                "cloud_task_service_expressions": cloud_task_service_expressions,
            }
        )
        custom.update({"hardware_names": hardware_names, "servers": servers, "server_costs": server_costs})

    solver = config["model"]["solver"]
    solver_name = str(solver["name"])
    solver_label = {"gurobi": "Gurobi", "highs": "HiGHS"}.get(solver_name, solver_name)
    if solver_name == "gurobi":
        solver_options = {
            "LogToConsole": int(bool(solver.get("log_to_console", False))),
            "MIPGap": float(solver.get("mip_gap", 0.001)),
            "Threads": int(solver.get("threads", 1)),
        }
    elif solver_name == "highs":
        solver_options = {
            "log_to_console": bool(solver.get("log_to_console", False)),
            "mip_rel_gap": float(solver.get("mip_gap", 0.001)),
            "threads": int(solver.get("threads", 1)),
        }
    else:
        raise ValueError(f"Unsupported core solver: {solver_name}")
    status, condition = network.optimize(
        solver_name=solver_name,
        extra_functionality=add_custom_constraints,
        include_objective_constant=False,
        solver_options=solver_options,
    )
    if status != "ok" or condition != "optimal":
        raise RuntimeError(f"PyPSA/{solver_label} failed: {status}, {condition}")

    installed_groups = 0.0
    installed_by_hardware: dict[str, float] = {}
    online_by_hardware: dict[str, np.ndarray] = {}
    online_groups = np.zeros(horizon_hours)
    executed_service = np.zeros(horizon_hours)
    ai_power = np.zeros(horizon_hours)
    hardware_power: dict[str, np.ndarray] = {}
    hardware_compute: dict[str, np.ndarray] = {}
    cloud_service = np.zeros(horizon_hours)
    cloud_reserved_by_hardware: dict[str, float] = {}
    cloud_compute: dict[str, np.ndarray] = {}
    cloud_service_by_task: dict[str, np.ndarray] = {}
    if include_ai and ai_link is not None:
        installed_solution = custom["installed"].solution
        online_solution = custom["online"].solution
        installed_by_hardware = {name: float(installed_solution.sel(hardware=name)) for name in custom["hardware_names"]}
        online_by_hardware = {name: np.asarray(online_solution.sel(hardware=name), dtype=float) for name in custom["hardware_names"]}
        installed_groups = sum(installed_by_hardware.values())
        online_groups = sum(online_by_hardware.values(), np.zeros(horizon_hours))
        for hardware in custom["hardware_names"]:
            compute = np.asarray(
                [float(expression.solution) for expression in custom["hardware_compute_expressions"][hardware]],
                dtype=float,
            )
            hardware_compute[hardware] = compute
            server = custom["servers"][hardware]
            capacity = float(server["accelerators_per_server"]) if hardware == "gpu" else float(server["service_capacity_cpu_server_h_per_h"])
            pue = float(server["marginal_facility_multiplier"])
            maximum_kw = float(server["maximum_wall_power_kw"])
            idle_kw = float(server["online_idle_wall_power_kw"])
            standby_kw = float(server["cold_spare_standby_power_kw"])
            hardware_power[hardware] = pue / 1000.0 * (
                standby_kw * installed_by_hardware[hardware]
                + (idle_kw - standby_kw) * online_by_hardware[hardware]
                + (maximum_kw - idle_kw) / capacity * compute
            )
        cloud_service_by_task = {
            task: np.asarray(
                [
                    float(expression.solution)
                    if custom["cloud_enabled"]
                    else float(expression)
                    for expression in expressions
                ],
                dtype=float,
            )
            for task, expressions in custom["cloud_task_service_expressions"].items()
        }
        executed_service = rigid.copy()
        values = np.asarray(custom["execution"].solution, dtype=float)
        for value, (_, hour, _, _) in zip(values, custom["assignments"]):
            executed_service[hour] += max(0.0, float(value))
        if custom["cloud_enabled"]:
            cloud_values = np.asarray(custom["cloud_execution"].solution, dtype=float)
            for value, (_, hour, _, _) in zip(cloud_values, custom["assignments"]):
                cloud_service[hour] += max(0.0, float(value))
            rigid_cloud_values = np.asarray(
                custom["cloud_rigid_execution"].solution, dtype=float
            )
            for value, (hour, _) in zip(rigid_cloud_values, custom["rigid_pairs"]):
                shifted = max(0.0, float(value))
                cloud_service[hour] += shifted
                executed_service[hour] -= shifted
            cloud_reserved_solution = custom["cloud_reserved"].solution
            cloud_reserved_by_hardware = {
                name: float(cloud_reserved_solution.sel(cloud_hardware=name))
                for name in custom["hardware_names"]
            }
        for hardware in custom["hardware_names"]:
            cloud_compute[hardware] = np.asarray(
                [
                    float(expression.solution)
                    if custom["cloud_enabled"]
                    else float(expression)
                    for expression in custom["cloud_compute_expressions"][hardware]
                ],
                dtype=float,
            )
        ai_power = network.links_t.p0[ai_link].to_numpy(dtype=float)

    pv_capacity = max(
        0.0,
        float(
            network.generators.at[
                "rooftop_pv",
                "p_nom_opt"
                if bool(network.generators.at["rooftop_pv", "p_nom_extendable"])
                else "p_nom",
            ]
        ),
    )
    battery_power = max(
        0.0,
        float(
            network.storage_units.at[
                "battery",
                "p_nom_opt"
                if bool(network.storage_units.at["battery", "p_nom_extendable"])
                else "p_nom",
            ]
        ),
    )
    grid_profile = network.generators_t.p[["existing_grid", "expanded_grid"]].sum(axis=1)
    grid_energy_mwh = float(grid_profile.sum()) * annual_periods
    demand_peak = float(custom["demand_peak"].solution)
    grid_expansion = incremental_grid_capacity_mw(demand_peak, existing_capacity)
    cloud_subscription_cost = sum(
        cloud_reserved_by_hardware.get(name, 0.0) * custom["cloud_prices"][name]
        for name in cloud_reserved_by_hardware
    ) if include_ai else 0.0
    server_cost_by_hardware = ({name: installed_by_hardware[name] * custom["server_costs"][name] for name in custom["hardware_names"]} if include_ai else {})
    physical_components = {
        "annual_server_cost_rmb": sum(server_cost_by_hardware.values()),
        "annual_pv_cost_rmb": pv_capacity * costs["pv_rmb_per_mw_year"],
        "annual_battery_cost_rmb": battery_power * costs["battery_rmb_per_mw_year"],
        "annual_flat_energy_cost_rmb": float(np.sum(grid_profile.to_numpy(dtype=float) * grid_prices))
        * annual_periods,
        "annual_maximum_demand_cost_rmb": demand_peak * costs["demand_rmb_per_mw_year"],
        "annual_grid_expansion_objective_penalty_rmb": grid_expansion
        * float(
            config["energy"].get(
                "grid_expansion_objective_penalty_rmb_per_mw_year", 0.0
            )
        ),
        "annual_cloud_subscription_cost_rmb": cloud_subscription_cost,
    }
    lifecycle_components = {
        "annual_model_initialization_cost_rmb": costs["model_initialization_rmb_per_deployment_year"] if include_ai else 0.0,
        "annual_model_storage_cost_rmb": installed_by_hardware.get("gpu", 0.0) * costs["model_storage_rmb_per_server_group_year"] if include_ai else 0.0,
        "annual_model_operations_cost_rmb": costs["model_operations_rmb_per_deployment_year"] if include_ai else 0.0,
    }
    network_objective = float(network.objective)
    reconstructed_network = sum(physical_components.values())
    if abs(network_objective - reconstructed_network) > max(10.0, abs(network_objective) * 1e-7):
        raise ValueError(f"Objective reconstruction mismatch: {network_objective} vs {reconstructed_network}")
    components = {**physical_components, **lifecycle_components}
    objective = network_objective + sum(lifecycle_components.values())
    initialization_energy_twh = (
        costs["model_initialization_energy_kwh_per_deployment_year"] / 1e9
        if include_ai
        else 0.0
    )

    hourly = pd.DataFrame(
        {
            "hour": range(horizon_hours),
            "day": np.arange(horizon_hours) // 24,
            "hour_of_day": np.arange(horizon_hours) % 24,
            "base_load_mw": base,
            "ai_executed_service_units": executed_service,
            "ai_cloud_service_units": cloud_service,
            "ai_compute_accelerator_h": executed_service
            * float(config["demand"]["effective_service"]["accelerator_h_per_service_unit"]),
            "installed_server_groups": installed_groups,
            "online_server_groups": online_groups,
            "ai_facility_power_mw": ai_power,
            "rooftop_pv_output_mw": network.generators_t.p["rooftop_pv"].to_numpy(dtype=float),
            "battery_mw_positive_discharge": network.storage_units_t.p["battery"].to_numpy(dtype=float),
            "battery_state_of_charge_mwh": network.storage_units_t.state_of_charge["battery"].to_numpy(dtype=float),
            "grid_import_mw": grid_profile.to_numpy(dtype=float),
            "grid_energy_price_rmb_per_mwh": grid_prices,
        }
    )
    for hardware in ("gpu", "cpu"):
        hourly[f"installed_{hardware}_server_groups"] = installed_by_hardware.get(hardware, 0.0)
        hourly[f"online_{hardware}_server_groups"] = online_by_hardware.get(
            hardware, np.zeros(horizon_hours)
        )
        hourly[f"{hardware}_compute_h"] = hardware_compute.get(
            hardware, np.zeros(horizon_hours)
        )
        hourly[f"cloud_{hardware}_compute_h"] = cloud_compute.get(
            hardware, np.zeros(horizon_hours)
        )
        hourly[f"{hardware}_facility_power_mw"] = hardware_power.get(
            hardware, np.zeros(horizon_hours)
        )
    for task, values in cloud_service_by_task.items():
        hourly[f"cloud_task_{task}_service_units"] = values
    summary: dict[str, Any] = {
        "solver": f"PyPSA + Linopy + {solver_label}",
        "include_ai": include_ai,
        "horizon_hours": horizon_hours,
        "represented_days": represented_days,
        "storage_cycle_horizon_hours": horizon_hours,
        "maximum_flexible_deadline_h": max((job.deadline_hours for job in flexible_jobs), default=0),
        "existing_grid_capacity_mw": existing_capacity,
        "installed_server_groups": installed_groups,
        "installed_gpu_server_groups": installed_by_hardware.get("gpu", 0.0) if include_ai else 0.0,
        "installed_cpu_server_groups": installed_by_hardware.get("cpu", 0.0) if include_ai else 0.0,
        "cloud_reserved_gpu_groups": cloud_reserved_by_hardware.get("gpu", 0.0),
        "cloud_reserved_cpu_groups": cloud_reserved_by_hardware.get("cpu", 0.0),
        "annual_cloud_service_units": float(cloud_service.sum()) * annual_periods,
        "cloud_service_share": float(cloud_service.sum())
        / max(float((cloud_service + executed_service).sum()), 1e-12),
        "cloud_service_by_task": {
            task: float(values.sum()) * annual_periods
            for task, values in cloud_service_by_task.items()
        },
        "annual_gpu_server_cost_rmb": server_cost_by_hardware.get("gpu", 0.0),
        "annual_cpu_server_cost_rmb": server_cost_by_hardware.get("cpu", 0.0),
        "annual_gpu_facility_energy_twh": float(np.sum(hardware_power.get("gpu", np.zeros(horizon_hours)))) * annual_periods / 1e6,
        "annual_cpu_facility_energy_twh": float(np.sum(hardware_power.get("cpu", np.zeros(horizon_hours)))) * annual_periods / 1e6,
        "annual_gpu_electricity_cost_rmb": float(np.sum(hardware_power.get("gpu", np.zeros(horizon_hours)) * grid_prices)) * annual_periods,
        "annual_cpu_electricity_cost_rmb": float(np.sum(hardware_power.get("cpu", np.zeros(horizon_hours)) * grid_prices)) * annual_periods,
        "heterogeneous_hardware_enabled": heterogeneous,
        "annual_ai_facility_energy_twh": float(np.sum(ai_power)) * annual_periods / 1e6,
        "annual_model_initialization_energy_twh": initialization_energy_twh,
        "annual_ai_facility_energy_including_initialization_twh": float(np.sum(ai_power)) * annual_periods / 1e6 + initialization_energy_twh,
        "ai_facility_peak_mw": float(np.max(ai_power, initial=0.0)),
        "rooftop_pv_capacity_mw": pv_capacity,
        "roof_area_proxy_m2": float(roof_area_m2),
        "rooftop_pv_limit_mw": factory_pv_limit_mw(config, roof_area_m2),
        "battery_power_mw": battery_power,
        "battery_energy_mwh": battery_power * float(config["energy"]["battery_duration_h"]),
        "grid_expansion_mw": grid_expansion,
        "grid_expansion_limit_mw": config["energy"].get("grid_expansion_limit_mw"),
        "grid_expansion_objective_penalty_rmb_per_mw_year": float(
            config["energy"].get(
                "grid_expansion_objective_penalty_rmb_per_mw_year", 0.0
            )
        ),
        "grid_import_peak_mw": float(np.max(grid_profile)),
        "annual_grid_energy_twh": grid_energy_mwh / 1e6,
        "mean_grid_energy_price_rmb_per_mwh": float(np.mean(grid_prices)),
        "minimum_grid_energy_price_rmb_per_mwh": float(np.min(grid_prices)),
        "maximum_grid_energy_price_rmb_per_mwh": float(np.max(grid_prices)),
        "grid_energy_price_mode": str(config["energy"]["grid_energy_price_mode"]),
        "pv_capacity_mode": str(config["energy"]["pv_capacity_mode"]),
        "server_groups_integer": bool(config["model"]["server_groups_integer"]),
        "installed_server_groups_integer": bool(
            config["model"].get(
                "installed_server_groups_integer",
                config["model"]["server_groups_integer"],
            )
        ),
        "online_server_groups_integer": bool(
            config["model"].get(
                "online_server_groups_integer",
                config["model"]["server_groups_integer"],
            )
        ),
        "planning_reserve_fraction": float(config["server"]["installed_reserve_fraction"]),
        "normal_dispatch_reserve_fraction": float(
            config["server"].get(
                "normal_dispatch_reserve_fraction",
                config["server"]["installed_reserve_fraction"],
            )
        ),
        "minimum_installed_server_groups": minimum_installed_server_groups,
        "battery_energy_capex_rmb_per_kwh": costs["battery_energy_capex_rmb_per_kwh"],
        "battery_power_capex_rmb_per_kw": costs["battery_power_capex_rmb_per_kw"],
        "battery_energy_lifetime_years": costs["battery_energy_lifetime_years"],
        "battery_power_lifetime_years": costs["battery_power_lifetime_years"],
        "battery_power_fom_fraction_per_year": costs["battery_power_fom_fraction_per_year"],
        "battery_annualized_cost_rmb_per_mw_year": costs["battery_rmb_per_mw_year"],
        "battery_investment_enabled": bool(config["energy"]["battery_investment_enabled"]),
        "annual_objective_rmb": objective,
        "network_optimization_objective_rmb": network_objective,
        "required_model_replicas": int(config["model_state"]["required_model_replicas"]) if include_ai and not bool(config.get("hybrid_cloud", {}).get("enabled", False)) else 0,
        "minimum_server_groups_for_model_state": int(config["model_state"]["minimum_server_groups_for_model_state"]) if include_ai and not bool(config.get("hybrid_cloud", {}).get("enabled", False)) else 0,
        "model_vram_gb_per_replica": float(config["model_state"]["vram_gb_per_replica"]) if include_ai else 0.0,
        "model_storage_required_gb": float(config["model_state"]["model_storage_required_gb_per_deployment"]) if include_ai and not bool(config.get("hybrid_cloud", {}).get("enabled", False)) else 0.0,
        "model_storage_available_gb": (installed_by_hardware.get("gpu", 0.0) if include_ai else 0.0) * float(config["model_state"]["bundled_storage_gb_per_server_group"]),
        **components,
    }
    return HostResult(summary=summary, hourly=hourly)
