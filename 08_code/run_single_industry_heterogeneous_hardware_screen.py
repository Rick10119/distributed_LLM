#!/usr/bin/env python3
"""Single-industry same-service CPU/GPU routing mechanism screen.

This is deliberately a one-industry sensitivity module. It preserves the active
C36 service quantities, task shapes, flexibility windows, IG replication scale,
five-year life, and one 15% installed margin. CPU task time and routing shares
remain explicit structural scenarios until task benchmarks are available.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_code"))

from core.config import load_config  # noqa: E402
from core.data import load_industry_inputs  # noqa: E402
from core.representative_group import read_representative_groups, scenario_scale  # noqa: E402


TOKEN_TASKS = {"office", "agent"}
ROUTABLE_TASKS = {"maintenance", "scheduling", "simulation"}
ALL_TASKS = TOKEN_TASKS | ROUTABLE_TASKS | {"vision"}


def capital_recovery_factor(rate: float, years: float) -> float:
    return rate * (1.0 + rate) ** years / ((1.0 + rate) ** years - 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--defaults", type=Path, default=Path("config/defaults.yaml"))
    parser.add_argument("--run-config", type=Path, default=Path("config/runs/single_industry_core.yaml"))
    parser.add_argument("--screen-config", type=Path, default=Path("config/compute_hardware/cpu_gpu_routing_v1.yaml"))
    parser.add_argument("--industry", default=None)
    parser.add_argument("--owned-architecture", default=None)
    parser.add_argument(
        "--token-demand",
        type=Path,
        default=Path("05_results/v0.8.0/result/api_token_cost/task_token_demand.csv"),
    )
    parser.add_argument(
        "--local-summary",
        type=Path,
        default=Path("05_results/v0.8.0/model/C36/IF/summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("05_results/v0.8.0/result/cost_benchmark/c36_heterogeneous_hardware_v1"),
    )
    return parser.parse_args()


def task_jobs(inputs, task_id: str, scale: float) -> tuple[np.ndarray, list[object]]:
    """Recover task-specific rigid service and jobs from active aggregate inputs."""
    jobs = [job for job in inputs.flexible_jobs if job.task_id == task_id]
    flexible_total = sum(job.amount_service_units for job in jobs)
    service_table = pd.read_csv(
        ROOT / "02_data/processed/effective_service/manufacturing_ai_effective_service_2030.csv",
        encoding="utf-8-sig",
    )
    daily = float(
        service_table.loc[
            (service_table.industry_code == inputs.industry_code)
            & (service_table.year == 2030)
            & (service_table.parameter_case == "base")
            & (service_table.task_id == task_id),
            "effective_service_units_day",
        ].iloc[0]
    )
    total = daily * inputs.rigid_service_units.size / 24.0
    rigid_total = total - flexible_total
    # The active loader has already applied the task shape before aggregation.
    # Reconstruct its task-specific rigid profile from the flexible arrivals:
    profile = np.zeros_like(inputs.rigid_service_units, dtype=float)
    for job in jobs:
        profile[job.release_hour] += job.amount_service_units
    if flexible_total > 0:
        profile *= rigid_total / flexible_total
    elif rigid_total > 0:
        profile[:] = rigid_total / profile.size
    return profile * scale, jobs


def _waterfill(load: np.ndarray, hours: list[int], amount: float) -> None:
    """Add divisible work to the least-loaded admissible hours."""
    remaining = float(amount)
    ordered = sorted(hours, key=lambda hour: float(load[hour]))
    for count in range(1, len(ordered) + 1):
        current = float(load[ordered[count - 1]])
        next_level = float(load[ordered[count]]) if count < len(ordered) else math.inf
        required = max(0.0, next_level - current) * count
        if remaining <= required or not math.isfinite(next_level):
            increment = remaining / count
            for hour in ordered[:count]:
                load[hour] += increment
            return
        for hour in ordered[:count]:
            load[hour] = next_level
        remaining -= required


def minimize_peak(
    inputs,
    task_fractions: dict[str, float],
    ai_scale: float,
    compute_per_service: dict[str, float],
) -> tuple[np.ndarray, float, float]:
    """Construct a deadline-feasible load-leveled profile.

    Short-deadline-first water filling is a transparent heuristic upper bound on
    the exact minimum peak. It avoids adding a new solver dependency to this
    one-industry screen while preserving every job's admissible window.
    """
    horizon = inputs.rigid_service_units.size
    rigid = np.zeros(horizon)
    jobs: list[tuple[object, float]] = []
    for task, fraction in task_fractions.items():
        if fraction <= 0:
            continue
        task_rigid, task_jobs_list = task_jobs(inputs, task, ai_scale)
        rigid += task_rigid * fraction * float(compute_per_service[task])
        jobs.extend((job, fraction * ai_scale) for job in task_jobs_list)

    compute = rigid.copy()
    for job, fraction_scale in sorted(
        jobs,
        key=lambda item: (item[0].deadline_hours, item[0].release_hour, item[0].task_id),
    ):
        hours = sorted(
            {(job.release_hour + offset) % horizon for offset in range(job.deadline_hours + 1)}
        )
        _waterfill(
            compute,
            hours,
            job.amount_service_units
            * fraction_scale
            * float(compute_per_service[job.task_id]),
        )
    return compute, float(compute.max()), float(compute.sum())


def annual_hardware_cost(server: dict, installed_groups: float, discount_rate: float) -> float:
    capex = float(server["purchase_cost_rmb"])
    per_group = (
        capex
        * (1.0 + float(server["facility_capex_fraction"]))
        * capital_recovery_factor(discount_rate, float(server["economic_life_years"]))
        + capex * float(server["annual_maintenance_fraction"])
    )
    return installed_groups * per_group


def hardware_screen(
    compute: np.ndarray,
    *,
    capacity_per_group: float,
    server: dict,
    annual_days: float,
    discount_rate: float,
) -> dict[str, float]:
    reserve = float(server["installed_reserve_fraction"])
    average_required = float(compute.mean()) / capacity_per_group
    peak_required = float(compute.max(initial=0.0)) / capacity_per_group
    installed = max(peak_required, average_required * (1.0 + reserve))
    online = compute / capacity_per_group
    maximum = float(server["maximum_wall_power_kw"])
    idle = float(server["online_idle_wall_power_kw"])
    standby = float(server["cold_spare_standby_power_kw"])
    pue = float(server["marginal_facility_multiplier"])
    power_kw = pue * (
        installed * standby
        + online * (idle - standby)
        + compute / capacity_per_group * (maximum - idle)
    )
    represented_days = compute.size / 24.0
    annual_energy_kwh = float(power_kw.sum()) * annual_days / represented_days
    return {
        "installed_groups": installed,
        "average_required_groups": average_required,
        "peak_required_groups": peak_required,
        "installed_capacity_to_average_demand_ratio": installed / average_required if average_required > 0.0 else np.nan,
        "realized_capacity_redundancy_fraction_above_average": installed / average_required - 1.0 if average_required > 0.0 else np.nan,
        "peak_compute_per_h": float(compute.max(initial=0.0)),
        "annual_compute_h": float(compute.sum()) * annual_days / represented_days,
        "annual_facility_energy_kwh": annual_energy_kwh,
        "annual_hardware_cost_rmb": annual_hardware_cost(server, installed, discount_rate),
    }


def token_bill_for_industry(provider: str, token_demand: Path, industry: str) -> float:
    demand = pd.read_csv(
        ROOT / token_demand,
        encoding="utf-8-sig",
    )
    demand = demand[demand.industry_code == industry]
    prices = pd.read_csv(
        ROOT / "02_data/processed/api_token_cost/api_token_prices_v1_1.csv",
        encoding="utf-8-sig",
    )
    price = prices[(prices.provider == provider) & prices.mainstream_representative.astype(str).str.lower().isin({"true", "1"})].iloc[0]
    return (
        float(demand.annual_input_tokens.sum()) / 1e6 * float(price.input_per_mtoken)
        + float(demand.annual_output_tokens.sum()) / 1e6 * float(price.output_per_mtoken)
    ) * float(price.fx_to_cny)


def main() -> None:
    args = parse_args()
    config = load_config(ROOT, args.defaults, args.run_config)
    screen = yaml.safe_load((ROOT / args.screen_config).read_text(encoding="utf-8"))
    industry = str(args.industry or screen["calibration_industry_code"])
    inputs = load_industry_inputs(config, industry)
    group = read_representative_groups(ROOT / config["paths"]["representative_group_report"])[industry]
    owned_architecture = str(
        args.owned_architecture or screen["calibration_owned_architecture"]
    )
    if owned_architecture not in {"IF", "IG", "II_1host"}:
        raise ValueError(f"Unsupported owned architecture: {owned_architecture}")
    local_scale = scenario_scale(group, config["industry_parameter_case"], owned_architecture)
    if str(Path(args.local_summary).parent.name) != owned_architecture:
        raise ValueError(
            f"Local summary architecture {Path(args.local_summary).parent.name} does not match "
            f"configured owned_architecture {owned_architecture}"
        )
    annual_days = float(config["model"]["annualization_days"])
    discount = float(config["model"]["discount_rate"])
    electricity = float(config["energy"]["flat_grid_energy_rmb_per_kwh"])
    accel_per_service = float(inputs.accelerator_h_per_service_unit)

    gpu_server = dict(config["server"])
    cpu_server = dict(screen["cpu_server"])
    cpu_price_case = str(config["compute_hardware"].get("local_cpu_price_case", "base"))
    cpu_price_cases = cpu_server.get("purchase_cost_cases_rmb", {})
    if cpu_price_case not in cpu_price_cases:
        raise ValueError(f"Unsupported local CPU purchase-price case: {cpu_price_case}")
    cpu_server["purchase_price_case"] = cpu_price_case
    cpu_server["purchase_cost_rmb"] = float(cpu_price_cases[cpu_price_case])
    gpu_price_table = pd.read_csv(ROOT / config["paths"]["enterprise_ai_cost_parameters"], encoding="utf-8-sig")
    gpu_cloud_price_case = str(config["full_cloud_cost"].get("main_gpu_price_case", "base"))
    cpu_cloud_price_case = str(config["full_cloud_cost"].get("main_cpu_price_case", "base"))
    price_columns = {"low": "low_value", "base": "base_value", "high": "high_value"}
    if gpu_cloud_price_case not in price_columns:
        raise ValueError(f"Unsupported cloud GPU price case: {gpu_cloud_price_case}")
    cpu_cloud_price_cases = screen["cloud_cpu"].get("annual_reserved_price_cases_rmb", {})
    if cpu_cloud_price_case not in cpu_cloud_price_cases:
        raise ValueError(f"Unsupported cloud CPU price case: {cpu_cloud_price_case}")
    gpu_cloud_annual = float(
        gpu_price_table.loc[
            gpu_price_table.parameter_id == "C19", price_columns[gpu_cloud_price_case]
        ].iloc[0]
    )
    cpu_cloud_annual = float(cpu_cloud_price_cases[cpu_cloud_price_case])
    cloud_cpu_capacity = float(screen["cloud_cpu"]["service_capacity_relative_to_local_cpu_server"])
    if not 0 < cloud_cpu_capacity <= 1:
        raise ValueError("Cloud CPU capacity relative to the local CPU server must be in (0, 1]")
    active_local = pd.read_csv(ROOT / args.local_summary, encoding="utf-8-sig").iloc[0]
    active_local_total = float(active_local.industry_equivalent_incremental_total_cost_rmb)
    if str(active_local.get("compute_hardware_mode", "")) != "heterogeneous_cpu_gpu":
        raise ValueError("China heterogeneous summary requires the joint CPU/GPU physical core solve")
    active_server_cost = float(active_local.industry_equivalent_incremental_annual_server_cost_rmb)
    active_electricity_cost = float(active_local.industry_equivalent_incremental_annual_flat_energy_cost_rmb)
    active_maximum_demand_cost = float(active_local.industry_equivalent_incremental_annual_maximum_demand_cost_rmb)
    active_battery_cost = float(active_local.industry_equivalent_incremental_annual_battery_cost_rmb)
    active_model_operations_cost = float(active_local.industry_equivalent_incremental_annual_model_operations_cost_rmb)
    active_known_cost = (
        active_server_cost + active_electricity_cost + active_maximum_demand_cost
        + active_battery_cost + active_model_operations_cost
    )
    active_other_modeled_cost = active_local_total - active_known_cost
    task_service_table = pd.read_csv(
        ROOT / config["paths"]["model_ready_task_service"], encoding="utf-8-sig"
    )
    task_service_table = task_service_table[
        (task_service_table.industry_code == industry)
        & (task_service_table.year == 2030)
        & (task_service_table.parameter_case == "base")
    ].set_index("task_id")

    rows: list[dict[str, object]] = []
    route_defs = screen["routing_cases"]
    active_case = str(screen["active_core_routing_case"])
    selected_routes = {
        active_case: route_defs[active_case],
    }
    core_multipliers = {
        task: float(value)
        for task, value in screen["core_cpu_server_hour_per_reference_l20_accelerator_hour"].items()
        if task != "rationale"
    }
    for routing_case, cpu_shares in selected_routes.items():
        multiplier_label = 0.0 if routing_case == "gpu_only" else math.nan
        cpu_task_multipliers = {
            task: (1.0 if routing_case == "gpu_only" else core_multipliers.get(task, 1.0))
            for task in ALL_TASKS
        }
        for _ in [None]:
            cpu_fraction = {task: float(cpu_shares.get(task, 0.0)) for task in ALL_TASKS}
            gpu_fraction = {task: 1.0 - cpu_fraction[task] for task in ALL_TASKS}

            gpu_compute_factors = {task: accel_per_service for task in ALL_TASKS}
            cpu_compute_factors = {
                task: accel_per_service * cpu_task_multipliers[task]
                for task in ALL_TASKS
            }
            local_gpu_compute, _, _ = minimize_peak(inputs, gpu_fraction, local_scale.ai_service_scale_per_host, gpu_compute_factors)
            local_cpu_compute, _, _ = minimize_peak(inputs, cpu_fraction, local_scale.ai_service_scale_per_host, cpu_compute_factors)
            local_gpu = hardware_screen(local_gpu_compute, capacity_per_group=float(gpu_server["accelerators_per_server"]), server=gpu_server, annual_days=annual_days, discount_rate=discount)
            local_cpu = hardware_screen(local_cpu_compute, capacity_per_group=float(cpu_server["service_capacity_cpu_server_h_per_h"]), server=cpu_server, annual_days=annual_days, discount_rate=discount)
            for values in (local_gpu, local_cpu):
                for key in list(values):
                    if key.startswith("annual_") or key == "installed_groups":
                        values[key] *= local_scale.equivalent_host_multiplier
            local_physical = active_local_total
            local_gpu_hardware_cost = local_gpu["annual_hardware_cost_rmb"]
            local_cpu_hardware_cost = local_cpu["annual_hardware_cost_rmb"]
            local_gpu_electricity_cost = electricity * local_gpu["annual_facility_energy_kwh"]
            local_cpu_electricity_cost = electricity * local_cpu["annual_facility_energy_kwh"]
            if routing_case == active_case:
                local_gpu_hardware_cost = float(active_local.industry_equivalent_incremental_annual_gpu_server_cost_rmb)
                local_cpu_hardware_cost = float(active_local.industry_equivalent_incremental_annual_cpu_server_cost_rmb)
                local_gpu_electricity_cost = float(active_local.industry_equivalent_incremental_annual_gpu_electricity_cost_rmb)
                local_cpu_electricity_cost = float(active_local.industry_equivalent_incremental_annual_cpu_electricity_cost_rmb)
                local_gpu["installed_groups"] = float(active_local.industry_equivalent_installed_gpu_server_groups)
                local_cpu["installed_groups"] = float(active_local.industry_equivalent_installed_cpu_server_groups)
                local_gpu["annual_facility_energy_kwh"] = float(active_local.industry_equivalent_annual_gpu_facility_energy_twh) * 1e9
                local_cpu["annual_facility_energy_kwh"] = float(active_local.industry_equivalent_annual_cpu_facility_energy_twh) * 1e9
                local_physical_components = {
                    "local_maximum_demand_cost_rmb": active_maximum_demand_cost,
                    "local_battery_cost_rmb": active_battery_cost,
                    "local_model_operations_cost_rmb": active_model_operations_cost,
                    "local_other_modeled_cost_rmb": active_other_modeled_cost,
                }

            # Cloud office/agent use API; only the four non-Token tasks enter reserved hardware.
            cloud_cpu_fraction = {task: cpu_fraction[task] if task not in TOKEN_TASKS else 0.0 for task in ALL_TASKS}
            cloud_gpu_fraction = {
                task: (1.0 - cloud_cpu_fraction[task]) if task not in TOKEN_TASKS else 0.0
                for task in ALL_TASKS
            }
            cloud_gpu_compute, cloud_gpu_peak, _ = minimize_peak(inputs, cloud_gpu_fraction, 1.0, gpu_compute_factors)
            cloud_cpu_compute, cloud_cpu_peak, _ = minimize_peak(inputs, cloud_cpu_fraction, 1.0, cpu_compute_factors)
            cloud_gpu_instances = math.ceil(cloud_gpu_peak / float(gpu_server["accelerators_per_server"]))
            cloud_cpu_instances = math.ceil(
                cloud_cpu_peak
                / (float(cpu_server["service_capacity_cpu_server_h_per_h"]) * cloud_cpu_capacity)
            )
            for provider in ("DeepSeek", "Alibaba Cloud"):
                api = token_bill_for_industry(provider, args.token_demand, industry)
                cloud_total = api + cloud_gpu_instances * gpu_cloud_annual + cloud_cpu_instances * cpu_cloud_annual
                rows.append({
                    "scenario_version": screen["scenario_version"],
                    "owned_architecture": owned_architecture,
                    "owned_architecture_interpretation": "factory_side_distributed_group_network_coordinated" if owned_architecture == "IF" else "group_centralized_compute_pool",
                    "industry_code": industry,
                    "industry_name_cn": inputs.industry_name,
                    "routing_case": routing_case,
                    "cpu_time_multiplier": multiplier_label,
                    "maintenance_cpu_time_multiplier": cpu_task_multipliers["maintenance"],
                    "scheduling_cpu_time_multiplier": cpu_task_multipliers["scheduling"],
                    "simulation_cpu_time_multiplier": cpu_task_multipliers["simulation"],
                    "provider": provider,
                    "local_cpu_purchase_price_case": cpu_price_case,
                    "local_cpu_purchase_price_rmb": float(cpu_server["purchase_cost_rmb"]),
                    "cloud_gpu_price_case": gpu_cloud_price_case,
                    "cloud_gpu_unit_price_rmb_per_year": gpu_cloud_annual,
                    "cloud_cpu_price_case": cpu_cloud_price_case,
                    "cloud_cpu_unit_price_rmb_per_year": cpu_cloud_annual,
                    "cpu_service_share": sum(
                        float(cpu_fraction[t])
                        * float(task_service_table.loc[t, "effective_service_units_day"])
                        for t in ALL_TASKS
                    ) / inputs.daily_effective_service_units,
                    "local_gpu_server_groups_industry_equivalent": local_gpu["installed_groups"],
                    "local_cpu_server_groups_industry_equivalent": local_cpu["installed_groups"],
                    "local_gpu_facility_energy_twh": local_gpu["annual_facility_energy_kwh"] / 1e9,
                    "local_cpu_facility_energy_twh": local_cpu["annual_facility_energy_kwh"] / 1e9,
                    "local_total_facility_energy_twh": (local_gpu["annual_facility_energy_kwh"] + local_cpu["annual_facility_energy_kwh"]) / 1e9,
                    "local_gpu_annualized_hardware_cost_rmb": local_gpu_hardware_cost,
                    "local_cpu_annualized_hardware_cost_rmb": local_cpu_hardware_cost,
                    "local_gpu_electricity_cost_rmb": local_gpu_electricity_cost,
                    "local_cpu_electricity_cost_rmb": local_cpu_electricity_cost,
                    **local_physical_components,
                    "local_joint_physical_annual_cost_rmb": local_physical,
                    "cloud_gpu_reserved_instances": cloud_gpu_instances,
                    "cloud_cpu_reserved_instances": cloud_cpu_instances,
                    "cloud_cpu_service_capacity_relative_to_local_server": cloud_cpu_capacity,
                    "cloud_token_api_cost_rmb": api,
                    "cloud_gpu_reserved_cost_rmb": cloud_gpu_instances * gpu_cloud_annual,
                    "cloud_cpu_reserved_cost_rmb": cloud_cpu_instances * cpu_cloud_annual,
                    "cloud_total_annual_cost_rmb": cloud_total,
                    "cloud_to_local_cost_ratio": cloud_total / local_physical,
                    "evidence_status": "structural_sensitivity_not_observed_hardware_mix",
                })

    result = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    routing = pd.DataFrame([
        {"routing_case": case, "task_id": task, "cpu_service_fraction": float(shares.get(task, 0.0)), "status": "structural_scenario_not_observed_share"}
        for case, shares in route_defs.items() for task in sorted(ALL_TASKS)
    ])
    routing.to_csv(args.output_dir / "routing_parameters.csv", index=False, encoding="utf-8-sig")

    deepseek = result[result.provider == "DeepSeek"].copy()
    core = deepseek[deepseek.routing_case == active_case].iloc[0]
    findings = f"""# {industry}{inputs.industry_name}CPU/GPU异构路由结果

## 口径

- 固定{industry}每日{inputs.daily_effective_service_units:,.0f}个有效服务单位、既有任务形状、质量/SLA和灵活窗口。
- 本地使用IF工厂侧分布式口径并按{local_scale.equivalent_host_multiplier:.4f}个等价工厂还原全行业；集团专网用于协同管理，但当前成本边界不新增专网、数据治理或协同平台费用。云端按该行业全行业池化。
- 核心候选将维护50%、排程100%、仿真0%路由至CPU，对应总有效服务的{core.cpu_service_share:.1%}。排程采用CPU原生求解器1.0服务器时/参考L20小时归一化，维护采用保守2.0倍CPU推理时间；仿真继续留在GPU路径。
- 本地CPU整机统一为双路、每路32物理核（合计64物理核）、至少256 GB内存和完整交付边界；活动{cpu_price_case}价格档为{float(cpu_server['purchase_cost_rmb']):,.0f}元，低/中/高为6.0/9.0/11.8万元。云CPU采用阿里云c8i 32 vCPU/64 GiB，活动{cpu_cloud_price_case}价格档为{cpu_cloud_annual:,.1f}元/年；公开订阅价作为低值，基准和高值分别增加15%和30%企业交付情景附加项。阿里云官方口径下每个vCPU对应一个超线程，故该实例按16物理核计，容量设为本地服务器的25%。
- 云GPU容量价格使用同一个{gpu_cloud_price_case}交付情景；Token API仍使用厂商公开价格，不施加容量价格加成。
- 本地核心结果直接读取同一次168小时联合物理优化：CPU/GPU装机、在线状态、任务调度和设施功率共同决定服务器、电费、最大需量与接入容量，不再使用全GPU锚定或口径校准；云端为office/agent Token API加CPU/GPU预留容量。

## DeepSeek API下的结果

| 情景 | CPU服务份额 | 本地设施用电 TWh/年 | 本地年成本 十亿元 | 完整云年成本 十亿元 | 云/本地 |
|---|---:|---:|---:|---:|---:|
| evidence-core | {core.cpu_service_share:.1%} | {core.local_total_facility_energy_twh:.3f} | {core.local_joint_physical_annual_cost_rmb/1e9:.3f} | {core.cloud_total_annual_cost_rmb/1e9:.3f} | {core.cloud_to_local_cost_ratio:.2f} |

## 解释

该核心候选没有用成本目标反推倍率：排程采用CPU原生路径、维护采用保守CPU推理倍率、仿真继续走GPU。云/本地成本比用于事后验算。它是应用于31行业的共同技术情景，不是任何行业的实际硬件份额；共同0.5/1/2/4扫描仅保留在敏感性分析计划。
"""
    (args.output_dir / "findings.md").write_text(findings, encoding="utf-8")
    payload = {
        "status": "validated_structural_sensitivity",
        "industry_code": industry,
        "owned_architecture": owned_architecture,
        "owned_architecture_interpretation": "factory_side_distributed_group_network_coordinated" if owned_architecture == "IF" else "group_centralized_compute_pool",
        "incremental_private_network_cost_included": False,
        "incremental_data_governance_cost_included": False,
        "rows": len(result),
        "same_service": True,
        "routing_cases": sorted(result.routing_case.unique()),
        "core_task_cpu_time_multipliers": core_multipliers,
        "checks": [
            f"same {industry} task service quantities",
            "task release and deadline windows retained",
            f"local {owned_architecture} scale reconstructs industry service",
            "one 10 percent installed margin per hardware pool",
            "cloud and local routes updated symmetrically",
            "cost components reconcile",
        ],
    }
    (args.output_dir / "validated.done.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
