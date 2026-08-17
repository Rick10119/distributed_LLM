#!/usr/bin/env python3
"""Factory/group/multisite test with IF-only integer installed capacity."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import linopy
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_code"))

from core.config import deep_merge, load_config
from core.data import load_industry_inputs, read_core_grid_energy_prices, scale_task_workload
from core.model import annual_server_cost_per_group
from core.production_load import resolve_site_mean_load_mw
from core.representative_group import read_representative_groups
from prepare_eweld_representative_weeks import complete_weeks


INTEGER_INSTALLED_CAPACITY_BY_ARCHITECTURE = {
    "IF": True,
    "IG_1host": False,
    "IG_multisite": False,
}


def validate_integer_boundary(experiment: dict) -> None:
    """Keep integer server installation exclusive to independent factories."""
    boundary = experiment.get("integer_boundary", {})
    for architecture, expected in INTEGER_INSTALLED_CAPACITY_BY_ARCHITECTURE.items():
        field = f"{architecture}_installed_server_groups_integer"
        configured = boundary.get(field, expected)
        if not isinstance(configured, bool) or configured is not expected:
            raise ValueError(
                f"{field} must be {str(expected).lower()}: only IF uses integer "
                "installed server groups; both group architectures use continuous "
                "equivalent capacity"
            )
    if boundary.get("online_server_groups_integer", False) is not False:
        raise ValueError("online_server_groups_integer must be false for all group-architecture cases")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--defaults", type=Path, default=Path("config/defaults.yaml"))
    parser.add_argument("--config", type=Path, default=Path("config/runs/all_industries_core.yaml"))
    parser.add_argument("--experiment", type=Path, default=Path("config/sensitivity/c36_group_multisite_continuous_v1.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("05_results/sensitivity/v0.8.0/group_multisite_continuous/C36"))
    parser.add_argument("--industry", help="Override experiment.industry; used by the 31-industry core workflow.")
    parser.add_argument(
        "--architectures",
        nargs="+",
        help="Override experiment/run-config architectures; used by IG_1host-only sensitivity screens.",
    )
    parser.add_argument("--describe-only", action="store_true", help="Report continuous variable counts without solving.")
    return parser.parse_args()


def select_factory_weeks(
    archive: Path,
    target: np.ndarray,
    factory_count: int,
    source_isic: int,
    quality: dict | None = None,
) -> tuple[list[dict], np.ndarray]:
    quality = quality or {}
    minimum_monday_ratio = float(
        quality.get("minimum_monday_to_tuesday_friday_mean_ratio", 0.0)
    )
    maximum_weekday_cv = float(quality.get("maximum_weekday_daily_mean_cv", np.inf))
    maximum_peak_to_mean = float(quality.get("maximum_weekly_peak_to_mean_ratio", np.inf))
    pattern = re.compile(rf"^EWELD/Electricity Consumption/C/C{source_isic:02d} .+/(?P<user>U\d+)\.csv$")
    candidates: dict[
        str,
        list[tuple[float, pd.Timestamp, str, np.ndarray, float, float, float, float]],
    ] = defaultdict(list)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            match = pattern.match(member)
            if not match:
                continue
            with zf.open(member) as handle:
                frame = pd.read_csv(handle, usecols=["Time", "Value"])
            for monday, week in complete_weeks(frame):
                raw = week["raw_value"].to_numpy(float)
                days = raw.reshape(7, 24)
                daily_mean = days.mean(axis=1, keepdims=True)
                if (daily_mean <= 0).any():
                    continue
                shapes = days / daily_mean
                typical = np.median(shapes, axis=0)
                rmse = float(np.sqrt(np.mean((typical - target) ** 2)))
                daily_values = daily_mean[:, 0]
                monday_ratio = float(daily_values[0] / daily_values[1:5].mean())
                weekday_cv = float(daily_values[:5].std() / daily_values[:5].mean())
                weekend_ratio = float(daily_values[5:].mean() / daily_values[:5].mean())
                peak_to_mean = float(raw.max() / raw.mean())
                if monday_ratio < minimum_monday_ratio:
                    continue
                if weekday_cv > maximum_weekday_cv:
                    continue
                if peak_to_mean > maximum_peak_to_mean:
                    continue
                candidates[match.group("user")].append(
                    (
                        rmse,
                        monday,
                        member,
                        raw,
                        monday_ratio,
                        weekday_cv,
                        weekend_ratio,
                        peak_to_mean,
                    )
                )
    if not candidates:
        raise ValueError(f"No usable EWELD users for ISIC {source_isic}")
    selected = []
    for user in sorted(candidates, key=lambda value: int(value[1:])):
        candidate = min(candidates[user], key=lambda row: (row[0], row[1]))
        selected.append((user, *candidate))
        if len(selected) == factory_count:
            break
    if len(selected) < factory_count:
        used = {(row[0], row[2]) for row in selected}
        remaining = sorted(
            (
                (user, *candidate)
                for user, weeks in candidates.items()
                for candidate in weeks
                if (user, candidate[1]) not in used
            ),
            key=lambda row: (row[1], row[2], int(row[0][1:])),
        )
        selected.extend(remaining[: factory_count - len(selected)])
    if len(selected) != factory_count:
        raise ValueError(f"Need {factory_count} distinct weeks for ISIC {source_isic}, found {len(selected)}")
    lineage = []
    curves = []
    user_counts = defaultdict(int)
    for (
        user,
        rmse,
        monday,
        member,
        raw,
        monday_ratio,
        weekday_cv,
        weekend_ratio,
        peak_to_mean,
    ) in selected:
        user_counts[user] += 1
        curves.append(raw / float(raw.mean()))
        lineage.append({
            "factory_id": f"F{len(lineage) + 1}",
            "source_user": user,
            "source_member": member,
            "week_start": monday.date().isoformat(),
            "week_end": (monday + pd.Timedelta(days=6)).date().isoformat(),
            "same_user_week_sequence": user_counts[user],
            "curve_role": "distinct_user_factory_proxy" if user_counts[user] == 1 else "same_user_distinct_week_factory_proxy",
            "shape_rmse_to_industry_core": rmse,
            "monday_to_tuesday_friday_mean_ratio": monday_ratio,
            "weekday_daily_mean_cv": weekday_cv,
            "weekend_to_weekday_mean_ratio": weekend_ratio,
            "weekly_peak_to_mean_ratio": peak_to_mean,
            "calendar_interpretation": "weekday_hour_aligned_synthetic_simultaneous_week",
        })
    return lineage, np.asarray(curves, dtype=float)


def hardware_parameters(config: dict) -> tuple[dict, dict, dict, dict]:
    routing = yaml.safe_load((ROOT / config["compute_hardware"]["routing_config"]).read_text(encoding="utf-8"))
    case = str(config["compute_hardware"].get("routing_case", routing["active_core_routing_case"]))
    cpu_fraction = {task: float(value) for task, value in routing["routing_cases"][case].items()}
    cpu_multiplier = {
        task: float(value)
        for task, value in routing["core_cpu_server_hour_per_reference_l20_accelerator_hour"].items()
        if task != "rationale"
    }
    cpu = deep_merge(dict(routing["cpu_server"]), dict(config["compute_hardware"].get("cpu_server_overrides", {})))
    price_case = str(config["compute_hardware"].get("local_cpu_price_case", "base"))
    cpu["purchase_cost_rmb"] = float(cpu["purchase_cost_cases_rmb"][price_case])
    cpu.setdefault("normal_dispatch_reserve_fraction", 0.0)
    return {"gpu": deepcopy(config["server"]), "cpu": cpu}, cpu_fraction, cpu_multiplier, {"routing_case": case, "cpu_price_case": price_case}


def balanced_factory_weights(physical_factory_count: int, modeled_node_count: int) -> np.ndarray:
    """Assign every physical factory to one representative node with near-equal integer weights."""
    if not 1 <= modeled_node_count <= physical_factory_count:
        raise ValueError("modeled_node_count must be between one and physical_factory_count")
    quotient, remainder = divmod(physical_factory_count, modeled_node_count)
    weights = np.full(modeled_node_count, quotient, dtype=int)
    weights[:remainder] += 1
    if int(weights.sum()) != physical_factory_count or (weights <= 0).any():
        raise AssertionError("Invalid representative factory weights")
    return weights


def resolve_population_count(setting: object, representative_group_count: int, label: str) -> int:
    """Resolve one physical population without coupling it to another population."""
    if setting in {None, "representative_group_case", "registry_or_representative_group_case"}:
        value = int(representative_group_count)
    else:
        value = int(setting)
    if value < 1:
        raise ValueError(f"{label} must be positive")
    return value


def aggregate_site_power_weights(
    site_power_weights: list[float], node_site_counts: np.ndarray
) -> np.ndarray:
    """Aggregate ordered physical-site power shares into representative routing nodes."""
    site_weights = np.asarray(site_power_weights, dtype=float)
    counts = np.asarray(node_site_counts, dtype=int)
    if len(site_weights) != int(counts.sum()):
        raise ValueError("Production-site power weights do not match the registered site count")
    if not np.isfinite(site_weights).all() or (site_weights <= 0).any():
        raise ValueError("Production-site power weights must be finite and positive")
    site_weights = site_weights / site_weights.sum()
    boundaries = np.concatenate(([0], np.cumsum(counts)))
    node_weights = np.asarray(
        [site_weights[boundaries[index]:boundaries[index + 1]].sum() for index in range(len(counts))]
    )
    if not np.isclose(node_weights.sum(), 1.0):
        raise AssertionError("Representative-node power shares must sum to one")
    return node_weights


def solve_architecture(
    *,
    architecture: str,
    config: dict,
    base_loads: np.ndarray,
    rigid_by_task: dict[str, np.ndarray],
    jobs: tuple,
    grid_prices: np.ndarray,
    servers: dict,
    cpu_fraction: dict,
    cpu_multiplier: dict,
    solver_name: str,
    installed_integer: bool,
    base_load_case: str,
) -> tuple[dict, pd.DataFrame]:
    sites = list(range(base_loads.shape[0]))
    hours = range(base_loads.shape[1])
    tasks = sorted(rigid_by_task)
    hardware = ["gpu", "cpu"]
    annual_periods = float(config["model"]["annualization_days"]) / (len(list(hours)) / 24.0)
    discount = float(config["model"]["discount_rate"])
    demand_rate = float(config["energy"]["maximum_demand_rmb_per_kw_month"]) * 1000.0 * 12.0
    accelerator_factor = float(config["demand"]["effective_service"]["accelerator_h_per_service_unit"])
    reserve = float(servers["gpu"]["installed_reserve_fraction"])
    n_plus_spare = int(servers["gpu"].get("n_plus_spare_server_groups", 0))
    if any(int(servers[name].get("n_plus_spare_server_groups", n_plus_spare)) != n_plus_spare for name in hardware):
        raise ValueError("GPU/CPU n_plus_spare_server_groups must match within one solve")
    if any(abs(float(servers[name]["installed_reserve_fraction"]) - reserve) > 1e-12 for name in hardware):
        raise ValueError("GPU/CPU installed_reserve_fraction must match within one solve")

    model = linopy.Model()
    site_index = pd.Index(sites, name="site")
    hardware_index = pd.Index(hardware, name="hardware")
    hour_index = pd.Index(list(hours), name="hour")
    installed = model.add_variables(lower=0.0, integer=installed_integer, coords=[site_index, hardware_index], name="installed")
    online = model.add_variables(lower=0.0, coords=[site_index, hardware_index, hour_index], name="online")
    peak = model.add_variables(lower=0.0, coords=[site_index], name="grid_peak")

    rigid_keys = [(hour, task, site) for hour in hours for task in tasks for site in sites]
    rigid_index = pd.Index(range(len(rigid_keys)), name="rigid_assignment")
    rigid_route = model.add_variables(lower=0.0, coords=[rigid_index], name="rigid_route")
    rigid_lookup: dict[tuple[int, str, int], int] = {key: idx for idx, key in enumerate(rigid_keys)}
    for hour in hours:
        for task in tasks:
            indices = [rigid_lookup[(hour, task, site)] for site in sites]
            model.add_constraints(
                rigid_route.sel(rigid_assignment=indices).sum() == float(rigid_by_task[task][hour]),
                name=f"rigid_service_{hour}_{task}",
            )

    flexible_keys: list[tuple[int, int, str, int]] = []
    by_job: dict[int, list[int]] = defaultdict(list)
    by_site_hour_task: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    for job_idx, job in enumerate(jobs):
        admissible = sorted({(job.release_hour + offset) % len(list(hours)) for offset in range(job.deadline_hours + 1)})
        for site in sites:
            for hour in admissible:
                idx = len(flexible_keys)
                flexible_keys.append((job_idx, hour, job.task_id, site))
                by_job[job_idx].append(idx)
                by_site_hour_task[(site, hour, job.task_id)].append(idx)
    flexible_index = pd.Index(range(len(flexible_keys)), name="flexible_assignment")
    flexible_route = model.add_variables(lower=0.0, coords=[flexible_index], name="flexible_route")
    for job_idx, job in enumerate(jobs):
        model.add_constraints(
            flexible_route.sel(flexible_assignment=by_job[job_idx]).sum() == float(job.amount_service_units),
            name=f"flexible_service_{job_idx}",
        )

    server_cost = {name: annual_server_cost_per_group(server, discount) for name, server in servers.items()}
    objective = sum(server_cost[name] * installed.sel(hardware=name).sum() for name in hardware)
    objective += demand_rate * peak.sum()
    power_expr: dict[tuple[int, int], object] = {}
    compute_expr: dict[tuple[int, int, str], object] = {}
    for site in sites:
        for hour in hours:
            task_service = {}
            for task in tasks:
                rigid_value = rigid_route.sel(rigid_assignment=rigid_lookup[(hour, task, site)])
                flex_indices = by_site_hour_task.get((site, hour, task), [])
                flexible_value = flexible_route.sel(flexible_assignment=flex_indices).sum() if flex_indices else 0.0
                task_service[task] = rigid_value + flexible_value
            power_terms = []
            for name in hardware:
                compute = 0
                for task, service in task_service.items():
                    cpu_share = float(cpu_fraction.get(task, 0.0))
                    fraction = cpu_share if name == "cpu" else 1.0 - cpu_share
                    multiplier = float(cpu_multiplier.get(task, 1.0)) if name == "cpu" else 1.0
                    compute += service * accelerator_factor * fraction * multiplier
                compute_expr[(site, hour, name)] = compute
                server = servers[name]
                capacity = float(server["service_capacity_cpu_server_h_per_h"] if name == "cpu" else server["accelerators_per_server"])
                online_value = online.sel(site=site, hardware=name, hour=hour)
                installed_value = installed.sel(site=site, hardware=name)
                model.add_constraints(online_value <= installed_value, name=f"online_installed_{site}_{name}_{hour}")
                model.add_constraints(compute <= capacity * online_value, name=f"online_capacity_{site}_{name}_{hour}")
                pue = float(server["marginal_facility_multiplier"])
                maximum = float(server["maximum_wall_power_kw"])
                idle = float(server["online_idle_wall_power_kw"])
                standby = float(server["cold_spare_standby_power_kw"])
                power_terms.append(pue / 1000.0 * (standby * installed_value + (idle - standby) * online_value + (maximum - idle) / capacity * compute))
            power = sum(power_terms)
            power_expr[(site, hour)] = power
            model.add_constraints(float(base_loads[site, hour]) + power <= peak.sel(site=site), name=f"grid_peak_{site}_{hour}")
            objective += annual_periods * float(grid_prices[hour]) * power
    for site in sites:
        for name in hardware:
            server = servers[name]
            capacity = float(server["service_capacity_cpu_server_h_per_h"] if name == "cpu" else server["accelerators_per_server"])
            average_compute = sum(compute_expr[(site, hour, name)] for hour in hours) / len(list(hours))
            reserve_fraction = float(server["installed_reserve_fraction"])
            spare_groups = int(server.get("n_plus_spare_server_groups", 0))
            if spare_groups < 0:
                raise ValueError("n_plus_spare_server_groups must be non-negative")
            average_required = average_compute / capacity
            model.add_constraints(
                installed.sel(site=site, hardware=name) >= (1.0 + reserve_fraction) * average_required,
                name=f"average_demand_planning_capacity_{site}_{name}",
            )
            if spare_groups:
                model.add_constraints(
                    installed.sel(site=site, hardware=name) >= average_required + spare_groups,
                    name=f"n_plus_spare_capacity_{site}_{name}",
                )
    model.objective = objective
    status, condition = model.solve(solver_name=solver_name, log_to_console=False)
    if status != "ok":
        raise RuntimeError(f"{architecture} solve failed: {status}/{condition}")

    installed_values = installed.solution.to_pandas()
    online_values = online.solution.to_series()
    peak_values = peak.solution.to_pandas()
    rigid_values = rigid_route.solution.to_numpy()
    flexible_values = flexible_route.solution.to_numpy()
    rows = []
    for site in sites:
        for hour in hours:
            service_by_task = {}
            for task in tasks:
                service = float(rigid_values[rigid_lookup[(hour, task, site)]])
                service += sum(float(flexible_values[idx]) for idx in by_site_hour_task.get((site, hour, task), []))
                service_by_task[task] = service
            compute_by_hardware = {}
            facility_power = 0.0
            for name in hardware:
                compute = 0.0
                for task, service in service_by_task.items():
                    cpu_share = float(cpu_fraction.get(task, 0.0))
                    fraction = cpu_share if name == "cpu" else 1.0 - cpu_share
                    multiplier = float(cpu_multiplier.get(task, 1.0)) if name == "cpu" else 1.0
                    compute += service * accelerator_factor * fraction * multiplier
                compute_by_hardware[name] = compute
                server = servers[name]
                capacity = float(server["service_capacity_cpu_server_h_per_h"] if name == "cpu" else server["accelerators_per_server"])
                online_count = float(online_values.loc[(site, name, hour)])
                installed_count = float(installed_values.loc[site, name])
                facility_power += float(server["marginal_facility_multiplier"]) / 1000.0 * (
                    float(server["cold_spare_standby_power_kw"]) * installed_count
                    + (float(server["online_idle_wall_power_kw"]) - float(server["cold_spare_standby_power_kw"])) * online_count
                    + (float(server["maximum_wall_power_kw"]) - float(server["online_idle_wall_power_kw"])) / capacity * compute
                )
            rows.append({
                "architecture": architecture,
                "base_load_case": base_load_case,
                "factory_id": f"F{site + 1}",
                "hour": hour,
                "base_load_mw": float(base_loads[site, hour]),
                "ai_service_units": sum(service_by_task.values()),
                "gpu_compute_h": compute_by_hardware["gpu"],
                "cpu_compute_h": compute_by_hardware["cpu"],
                "ai_facility_power_mw": facility_power,
                "grid_import_mw": float(base_loads[site, hour]) + facility_power,
            })
    hourly = pd.DataFrame(rows)
    base_peaks = base_loads.max(axis=1)
    incremental_demand_cost = float(np.sum(np.asarray(peak_values) - base_peaks) * demand_rate)
    annual_energy_cost = float(sum(row["ai_facility_power_mw"] * grid_prices[int(row["hour"])] * annual_periods for row in rows))
    annual_server_cost = float(sum(installed_values.loc[site, name] * server_cost[name] for site in sites for name in hardware))
    average_required_gpu_groups = float(hourly["gpu_compute_h"].sum()) / (len(list(hours)) * float(servers["gpu"]["accelerators_per_server"]))
    average_required_cpu_groups = float(hourly["cpu_compute_h"].sum()) / (len(list(hours)) * float(servers["cpu"]["service_capacity_cpu_server_h_per_h"]))
    average_required_total_groups = average_required_gpu_groups + average_required_cpu_groups
    installed_total_groups = float(installed_values["gpu"].sum() + installed_values["cpu"].sum())
    installed_to_average_ratio = installed_total_groups / average_required_total_groups
    summary = {
        "architecture": architecture,
        "base_load_case": base_load_case,
        "solver_status": status,
        "solver_condition": condition,
        "physical_factory_count": len(sites),
        "installed_server_groups_integer": installed_integer,
        "online_server_groups_integer": False,
        "n_plus_spare_server_groups": n_plus_spare,
        "model_state_minimum_groups_enabled": False,
        "planning_reserve_fraction": reserve,
        "installed_gpu_server_groups": float(installed_values["gpu"].sum()),
        "installed_cpu_server_groups": float(installed_values["cpu"].sum()),
        "average_required_gpu_server_groups": average_required_gpu_groups,
        "average_required_cpu_server_groups": average_required_cpu_groups,
        "average_required_total_server_groups": average_required_total_groups,
        "installed_total_server_capacity_to_average_demand_ratio": installed_to_average_ratio,
        "realized_total_installed_capacity_utilization_fraction": 1.0 / installed_to_average_ratio,
        "realized_total_capacity_redundancy_fraction_above_average": installed_to_average_ratio - 1.0,
        "annual_server_cost_rmb": annual_server_cost,
        "annual_ai_energy_cost_rmb": annual_energy_cost,
        "annual_incremental_maximum_demand_cost_rmb": incremental_demand_cost,
        "annual_incremental_total_cost_rmb": annual_server_cost + annual_energy_cost + incremental_demand_cost,
        "annual_ai_facility_energy_twh": float(hourly["ai_facility_power_mw"].sum() * annual_periods / 1e6),
        "sum_incremental_grid_peak_mw": float(np.sum(np.maximum(np.asarray(peak_values) - base_peaks, 0.0))),
        "maximum_single_site_ai_facility_power_mw": float(hourly.groupby("factory_id")["ai_facility_power_mw"].max().max()),
        "weekly_service_units": float(hourly["ai_service_units"].sum()),
    }
    return summary, hourly


def main() -> None:
    args = parse_args()
    experiment = yaml.safe_load((ROOT / args.experiment).read_text(encoding="utf-8"))
    config = load_config(ROOT, args.defaults, args.config)
    industry = str(args.industry or experiment.get("industry", ""))
    if industry not in config["selected_industries"]:
        raise ValueError(f"Industry {industry!r} is not selected by {args.config}")
    inputs = load_industry_inputs(config, industry)
    group = read_representative_groups(ROOT / config["paths"]["representative_group_report"])[industry]
    representative_group_count = group.factories(config["industry_parameter_case"])
    legacy_load_site_count = resolve_population_count(
        experiment.get(
            "production_load_site_count",
            experiment.get("factory_count", "registry_or_representative_group_case"),
        ),
        representative_group_count,
        "production_load_site_count",
    )
    share = group.share(config["industry_parameter_case"])
    mean_per_load_site, load_calibration = resolve_site_mean_load_mw(
        root=ROOT,
        config=config,
        industry=industry,
        industry_mean_load_mw=float(inputs.base_load_mw.mean()),
        ai_service_group_share=share,
        legacy_load_site_count=legacy_load_site_count,
    )
    production_load_site_count = int(load_calibration["load_site_count"])
    node_identity_rule = str(
        experiment.get("ai_deployment_node_rule", "same_as_modeled_routing_nodes")
    )
    if node_identity_rule != "same_as_modeled_routing_nodes":
        raise ValueError(
            "ai_deployment_node_rule must be same_as_modeled_routing_nodes"
        )
    requested_node_count = experiment.get("modeled_routing_node_count")
    if requested_node_count is not None:
        modeled_node_count = int(requested_node_count)
        if modeled_node_count < 1 or modeled_node_count > production_load_site_count:
            raise ValueError(
                "modeled_routing_node_count must be between 1 and production_load_site_count"
            )
        node_selection_mode = "explicit"
    else:
        node_cap = int(experiment.get("max_modeled_routing_nodes", production_load_site_count))
        if node_cap < 1:
            raise ValueError("max_modeled_routing_nodes must be positive")
        modeled_node_count = min(production_load_site_count, node_cap)
        node_selection_mode = "capped"
    production_load_site_weights = balanced_factory_weights(
        production_load_site_count, modeled_node_count
    )
    representative_cluster_power_shares = aggregate_site_power_weights(
        load_calibration["load_site_power_weights"], production_load_site_weights
    )
    target = inputs.base_load_mw.reshape(-1, 24)
    target = np.median(target / target.mean(axis=1, keepdims=True), axis=0)
    source_setting = experiment["load_curve_selection"].get("source_isic", "auto_from_core_lineage")
    if source_setting == "auto_from_core_lineage":
        core_lineage = json.loads((ROOT / config["paths"]["hourly_industry_profiles_lineage"]).read_text(encoding="utf-8"))
        source_by_industry = {
            str(row["industry_code"]): int(row["source_isic_division"])
            for row in core_lineage["records"]
        }
        source_isic = source_by_industry[industry]
    else:
        source_isic = int(source_setting)
    lineage, normalized_curves = select_factory_weeks(
        ROOT / config["paths"]["eweld_archive"],
        target,
        modeled_node_count,
        source_isic,
        experiment["load_curve_selection"].get("week_quality_by_industry", {}).get(
            industry, experiment["load_curve_selection"].get("week_quality")
        ),
    )
    for row, weight, power_share in zip(
        lineage, production_load_site_weights, representative_cluster_power_shares
    ):
        row["modeled_routing_node_id"] = row.pop("factory_id").replace("F", "R", 1)
        row["represented_factory_count"] = int(weight)
        row["represented_production_load_site_count"] = int(weight)
        row["IF_representative_site_weight"] = int(weight)
        row["representative_cluster_power_share"] = float(power_share)
        row["electrical_load_aggregation_at_AI_node"] = False
        row["ai_deployment_node_count"] = 1
    print(f"selected {industry} factory weeks from EWELD ISIC {source_isic}", flush=True)
    group_mean_load_mw = float(load_calibration["representative_group_mean_load_mw"])
    if not np.isclose(
        mean_per_load_site * production_load_site_count, group_mean_load_mw
    ):
        raise AssertionError("Production-load site mean does not reconstruct group mean load")
    representative_site_mean_by_node = (
        group_mean_load_mw * representative_cluster_power_shares / production_load_site_weights
    )
    representative_factory_loads = normalized_curves * representative_site_mean_by_node[:, None]
    host_index = int(np.argsort(representative_factory_loads.max(axis=1))[modeled_node_count // 2])
    servers, cpu_fraction, cpu_multiplier, hardware_meta = hardware_parameters(config)
    # Prefer the merged run/case config so PHY07/PHY08 overrides are not wiped by
    # the core experiment pin. Fill only missing hardware fields.
    default_planning_reserve = float(config["server"]["installed_reserve_fraction"])
    default_n_plus = int(config["server"].get("n_plus_spare_server_groups", 0))
    for server in servers.values():
        server.setdefault("installed_reserve_fraction", default_planning_reserve)
        server.setdefault("n_plus_spare_server_groups", default_n_plus)
    planning_reserve = float(servers["gpu"]["installed_reserve_fraction"])
    n_plus_spare = int(servers["gpu"].get("n_plus_spare_server_groups", default_n_plus))
    for server in servers.values():
        if abs(float(server["installed_reserve_fraction"]) - planning_reserve) > 1e-12:
            raise ValueError("GPU/CPU planning reserves must match")
        if int(server.get("n_plus_spare_server_groups", n_plus_spare)) != n_plus_spare:
            raise ValueError("GPU/CPU N+k spare counts must match")
    rigid_group, jobs_group = scale_task_workload(inputs, share)
    grid_prices = read_core_grid_energy_prices(config)
    horizon_hours = int(config["model"]["horizon_hours"])
    if horizon_hours == 24:
        normalized_day_curves = normalized_curves.reshape(modeled_node_count, 7, 24).mean(axis=1)
        representative_factory_loads = normalized_day_curves * representative_site_mean_by_node[:, None]
    elif horizon_hours != 168:
        raise ValueError("The group-multisite curve selector currently supports 24 or 168 hours")

    admissible_per_site = sum(
        len({(job.release_hour + offset) % horizon_hours for offset in range(job.deadline_hours + 1)})
        for job in jobs_group
    )
    variable_counts = {
        "production_load_site_count": production_load_site_count,
        "multisite_ai_deployment_node_count": modeled_node_count,
        "multisite_task_routing_node_count": modeled_node_count,
        "physical_factory_count": production_load_site_count,
        "modeled_routing_node_count": modeled_node_count,
        "production_load_site_weights": production_load_site_weights.tolist(),
        "IF_representative_site_counts": production_load_site_weights.tolist(),
        "representative_cluster_power_shares": representative_cluster_power_shares.tolist(),
        "electrical_load_aggregation_at_AI_nodes": False,
        "representative_host_site_mean_loads_mw": representative_site_mean_by_node.tolist(),
        "multisite_total_host_site_mean_load_mw": float(representative_site_mean_by_node.sum()),
        "representative_group_all_site_mean_load_mw": group_mean_load_mw,
        "IG_1host_candidate_node_id": f"R{host_index + 1}",
        "IG_1host_mean_base_load_mw": float(representative_site_mean_by_node[host_index]),
        "group_ai_service_share": share,
        "ai_service_routing_scope": "full_group_service_across_all_AI_deployment_routing_nodes",
        "routing_node_selection_mode": node_selection_mode,
        "IF_installed_server_integer_variables_solved": modeled_node_count * 2,
        "IF_installed_server_integer_variables_physical_equivalent": production_load_site_count * 2,
        "IG_1host_installed_server_integer_variables": 0,
        "IG_multisite_installed_server_integer_variables": 0,
        "load_profile_mode": str(config["model"]["load_profile_mode"]),
        "horizon_hours": horizon_hours,
        "online_server_variables": modeled_node_count * 2 * horizon_hours,
        "rigid_route_variables": modeled_node_count * len(rigid_group) * horizon_hours,
        "flexible_time_site_route_variables": modeled_node_count * admissible_per_site,
        "grid_peak_variables": modeled_node_count,
        "primary_continuous_variables_IG_multisite": modeled_node_count * 2 * horizon_hours + modeled_node_count * len(rigid_group) * horizon_hours + modeled_node_count * admissible_per_site + modeled_node_count,
        "integer_variables_IF_solved": modeled_node_count * 2,
    }
    if args.describe_only:
        print(json.dumps(variable_counts, ensure_ascii=False, indent=2))
        return

    selected_architectures = [
        str(name)
        for name in (args.architectures or config.get("selected_scenarios") or experiment.get("architectures") or [])
    ]
    allowed_architectures = {"IF", "IG_1host", "IG_multisite"}
    if not selected_architectures or set(selected_architectures) - allowed_architectures:
        raise ValueError(f"Unsupported group architectures: {selected_architectures}")
    validate_integer_boundary(experiment)
    cases_by_architecture = experiment["base_load_cases_by_architecture"]
    expected_pairs = {
        (architecture, case)
        for architecture in selected_architectures
        for case in cases_by_architecture[architecture]
    }

    final_summaries = []
    final_hourly = []
    additive_fields = [
        "installed_gpu_server_groups", "installed_cpu_server_groups", "annual_server_cost_rmb",
        "average_required_gpu_server_groups", "average_required_cpu_server_groups", "average_required_total_server_groups",
        "annual_ai_energy_cost_rmb", "annual_incremental_maximum_demand_cost_rmb",
        "annual_incremental_total_cost_rmb", "annual_ai_facility_energy_twh",
        "sum_incremental_grid_peak_mw", "weekly_service_units",
    ]
    if_site_summaries = []
    if_site_hourly = []
    if "IF" in selected_architectures:
        for site in range(modeled_node_count):
            rigid_site = {task: values / production_load_site_count for task, values in rigid_group.items()}
            jobs_site = tuple(
                job.__class__(job.release_hour, job.deadline_hours, job.amount_service_units / production_load_site_count, job.task_id, job.flexibility_class)
                for job in jobs_group
            )
            summary, hourly = solve_architecture(
                architecture=f"IF_F{site + 1}", config=config, base_loads=representative_factory_loads[[site]],
                rigid_by_task=rigid_site, jobs=jobs_site, grid_prices=grid_prices,
                servers=servers, cpu_fraction=cpu_fraction, cpu_multiplier=cpu_multiplier,
                solver_name=str(experiment["solver"]),
                installed_integer=INTEGER_INSTALLED_CAPACITY_BY_ARCHITECTURE["IF"],
                base_load_case="actual_load",
            )
            weight = int(production_load_site_weights[site])
            for field in additive_fields:
                summary[field] = float(summary[field]) * weight
            summary["representative_factory_weight"] = weight
            if_site_summaries.append(summary)
            for field in ["base_load_mw", "ai_service_units", "gpu_compute_h", "cpu_compute_h", "ai_facility_power_mw", "grid_import_mw"]:
                hourly[field] = hourly[field].astype(float) * weight
            hourly["factory_id"] = f"R{site + 1}"
            hourly["represented_factory_count"] = weight
            hourly["represented_production_load_site_count"] = weight
            hourly["ai_deployment_node_count"] = 1
            if_site_hourly.append(hourly)
        if_rows = pd.DataFrame(if_site_summaries)
        if_summary = {key: if_rows[key].sum() for key in additive_fields}
        if_summary.update({
            "architecture": "IF", "base_load_case": "actual_load",
            "solver_status": "ok", "solver_condition": "weighted_representative_independent_factory_models",
            "physical_factory_count": production_load_site_count,
            "production_load_site_count": production_load_site_count,
            "ai_deployment_node_count": production_load_site_count,
            "installed_server_groups_integer": True,
            "modeled_routing_node_count": modeled_node_count,
            "active_compute_node_count": modeled_node_count,
            "online_server_groups_integer": False, "n_plus_spare_server_groups": n_plus_spare,
            "model_state_minimum_groups_enabled": False,
            "planning_reserve_fraction": planning_reserve,
            "maximum_single_site_ai_facility_power_mw": float(max(row["maximum_single_site_ai_facility_power_mw"] for row in if_site_summaries)),
        })
        if_ratio = float(if_summary["installed_gpu_server_groups"] + if_summary["installed_cpu_server_groups"]) / float(if_summary["average_required_total_server_groups"])
        if_summary["installed_total_server_capacity_to_average_demand_ratio"] = if_ratio
        if_summary["realized_total_installed_capacity_utilization_fraction"] = 1.0 / if_ratio
        if_summary["realized_total_capacity_redundancy_fraction_above_average"] = if_ratio - 1.0
        final_summaries.append(if_summary)
        final_hourly.append(pd.concat(if_site_hourly, ignore_index=True).assign(architecture="IF"))

    if "IG_1host" in selected_architectures:
        for base_load_case in cases_by_architecture["IG_1host"]:
            host_load = (
                representative_factory_loads[[host_index]]
                if base_load_case == "actual_load"
                else np.zeros_like(representative_factory_loads[[host_index]])
            )
            if base_load_case not in {"actual_load", "zero_load"}:
                raise ValueError(f"Unsupported IG_1host base-load case: {base_load_case}")
            summary, hourly = solve_architecture(
                architecture="IG_1host", config=config, base_loads=host_load,
                rigid_by_task=rigid_group, jobs=jobs_group, grid_prices=grid_prices,
                servers=servers, cpu_fraction=cpu_fraction, cpu_multiplier=cpu_multiplier,
                solver_name=str(experiment["solver"]),
                installed_integer=INTEGER_INSTALLED_CAPACITY_BY_ARCHITECTURE["IG_1host"],
                base_load_case=base_load_case,
            )
            summary["physical_factory_count"] = production_load_site_count
            summary["production_load_site_count"] = production_load_site_count
            summary["ai_deployment_node_count"] = 1
            summary["modeled_routing_node_count"] = 1
            summary["active_compute_node_count"] = 1
            hourly["factory_id"] = f"R{host_index + 1}"
            hourly["represented_factory_count"] = 1
            hourly["represented_production_load_site_count"] = 1
            hourly["ai_deployment_node_count"] = 1
            final_summaries.append(summary)
            final_hourly.append(hourly)

    if "IG_multisite" in selected_architectures:
        summary, hourly = solve_architecture(
            architecture="IG_multisite", config=config, base_loads=representative_factory_loads,
            rigid_by_task=rigid_group, jobs=jobs_group, grid_prices=grid_prices,
            servers=servers, cpu_fraction=cpu_fraction, cpu_multiplier=cpu_multiplier,
            solver_name=str(experiment["solver"]),
            installed_integer=INTEGER_INSTALLED_CAPACITY_BY_ARCHITECTURE["IG_multisite"],
            base_load_case="actual_load",
        )
        summary["physical_factory_count"] = production_load_site_count
        summary["production_load_site_count"] = production_load_site_count
        summary["ai_deployment_node_count"] = modeled_node_count
        summary["modeled_routing_node_count"] = modeled_node_count
        summary["active_compute_node_count"] = modeled_node_count
        hourly["factory_id"] = hourly["factory_id"].str.replace("F", "R", n=1, regex=False)
        hourly["represented_factory_count"] = 1
        hourly["represented_production_load_site_count"] = 1
        hourly["ai_deployment_node_count"] = 1
        final_summaries.append(summary)
        final_hourly.append(hourly)

    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(final_summaries)
    summary_frame.insert(0, "industry", industry)
    summary_frame["production_load_mode"] = str(load_calibration["mode"])
    summary_frame["production_load_boundary_id"] = str(load_calibration["boundary_id"])
    summary_frame["production_activity_share"] = float(load_calibration["production_activity_share"])
    summary_frame["load_site_count"] = int(load_calibration["load_site_count"])
    summary_frame["ai_service_group_share"] = share
    summary_frame["production_load_site_count"] = production_load_site_count
    observed_pairs = set(zip(summary_frame["architecture"], summary_frame["base_load_case"]))
    if observed_pairs != expected_pairs or len(summary_frame) != len(expected_pairs):
        raise ValueError(f"Unexpected architecture/base-load cases: {observed_pairs}")
    reference_architecture = "IF" if "IF" in selected_architectures else selected_architectures[0]
    reference_cost = float(summary_frame.loc[
        summary_frame.architecture.eq(reference_architecture) & summary_frame.base_load_case.eq("actual_load"),
        "annual_incremental_total_cost_rmb",
    ].iloc[0])
    summary_frame["cost_relative_to_IF_actual_load"] = (
        summary_frame["annual_incremental_total_cost_rmb"] / reference_cost
        if reference_architecture == "IF"
        else float("nan")
    )
    summary_frame["cost_relative_to_reference_actual_load"] = summary_frame["annual_incremental_total_cost_rmb"] / reference_cost
    summary_frame.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(final_hourly, ignore_index=True).to_csv(output / "hourly.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(lineage).to_csv(output / "curve_lineage.csv", index=False, encoding="utf-8-sig")
    if "IG_1host" in selected_architectures:
        paired = summary_frame.set_index(["architecture", "base_load_case"])
        actual = paired.loc[("IG_1host", "actual_load")]
        zero = paired.loc[("IG_1host", "zero_load")]
        paired_frame = pd.DataFrame([{
            "industry": industry,
            "architecture": "IG_1host",
            "load_alignment_value_total_cost_rmb": float(zero.annual_incremental_total_cost_rmb - actual.annual_incremental_total_cost_rmb),
            "load_alignment_value_server_cost_rmb": float(zero.annual_server_cost_rmb - actual.annual_server_cost_rmb),
            "load_alignment_value_energy_cost_rmb": float(zero.annual_ai_energy_cost_rmb - actual.annual_ai_energy_cost_rmb),
            "load_alignment_value_maximum_demand_cost_rmb": float(zero.annual_incremental_maximum_demand_cost_rmb - actual.annual_incremental_maximum_demand_cost_rmb),
            "avoided_incremental_grid_peak_mw": float(zero.sum_incremental_grid_peak_mw - actual.sum_incremental_grid_peak_mw),
            "definition": "IG_1host_zero_load_reoptimized_minus_actual_load_reoptimized",
        }])
        paired_frame.to_csv(output / "load_alignment_value.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["industry", "architecture"]).to_csv(output / "load_alignment_value.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "status": "completed_IF_integer_IG_continuous_capacity_mechanism_test",
        "experiment": experiment,
        "industry": industry,
        "source_isic": source_isic,
        "group_share": share,
        "synthetic_factory_count": production_load_site_count,
        "physical_factory_count": production_load_site_count,
        "production_load_site_count": production_load_site_count,
        "modeled_routing_node_count": modeled_node_count,
        "ai_task_origin_production_site_count": production_load_site_count,
        "ai_service_routing_scope": "full_group_service_across_all_AI_deployment_routing_nodes",
        "ai_deployment_node_count_by_architecture": {
            "IF": production_load_site_count,
            "IG_1host": 1,
            "IG_multisite": modeled_node_count,
        },
        "multisite_AI_deployment_points_equal_routing_nodes": True,
        "electrical_load_aggregation_at_AI_nodes": False,
        "routing_node_selection_mode": node_selection_mode,
        "representative_factory_weights": production_load_site_weights.tolist(),
        "production_load_site_weights": production_load_site_weights.tolist(),
        "IF_representative_site_counts": production_load_site_weights.tolist(),
        "representative_cluster_power_shares": representative_cluster_power_shares.tolist(),
        "load_calibration": load_calibration,
        "IG_1host_factory_id": f"R{host_index + 1}",
        "hardware": hardware_meta,
        "variable_counts": variable_counts,
        "selected_architectures": selected_architectures,
        "planning_reserve_fraction": planning_reserve,
        "n_plus_spare_server_groups": n_plus_spare,
        "base_load_cases_by_architecture": {
            architecture: cases_by_architecture[architecture]
            for architecture in selected_architectures
        },
        "load_profile_mode": str(config["model"]["load_profile_mode"]),
        "validated_architecture_base_load_pairs": sorted(f"{architecture}__{case}" for architecture, case in observed_pairs),
        "service_conservation_relative_error": {
            f"{row['architecture']}__{row['base_load_case']}": abs(float(row["weekly_service_units"]) - float(summary_frame.iloc[0]["weekly_service_units"])) / max(float(summary_frame.iloc[0]["weekly_service_units"]), 1e-12)
            for row in final_summaries
        },
        "limitations": [
            "EWELD users are anonymous same-industry facilities, not observed members of one corporate group",
            "weeks are aligned by weekday and hour, not observed in the same calendar week",
            "all six modeled central AI task classes are allowed to route within the synthetic group",
            "flexible tasks retain their registered intraday and batch deadlines",
            "cross-site network cost, latency, data-residency restrictions, PV and batteries are excluded",
            "only IF installed GPU/CPU server groups are integer; IG_1host and IG_multisite installed capacity is continuous equivalent",
            "hourly online server groups remain continuous in all three architectures",
            "same-user distinct weeks are factory-shape proxies when an industry has too few distinct EWELD users",
            f"multisite host nodes are selected by {node_selection_mode} mode; each modeled routing node is one AI deployment point on an independent host-factory electrical boundary",
            "IF solves one integer-capacity factory per representative curve and weights its cost and capacity to reconstruct physical-factory fragmentation",
            "IF representative weights are not applied to IG_1host or IG_multisite production loads",
            "IG_1host and IG_multisite use the same unaggregated per-host production-load curves for common candidate nodes",
            "production sites without an AI host contribute to group AI-service demand but their electrical loads are not transferred to host nodes",
            "zero_load is a reoptimized AI-only counterfactual, not a factory operating condition",
            "only IG_1host receives the zero_load paired counterfactual",
            "IG_1host load-alignment value includes endogenous changes in AI timing and installed capacity permitted by that architecture",
            "model-state minimum server floors remain disabled in this mechanism prototype",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
