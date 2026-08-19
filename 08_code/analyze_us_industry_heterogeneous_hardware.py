#!/usr/bin/env python3
"""Recompute U.S. 21-NAICS manufacturing demand with CPU/GPU routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import pandas as pd
import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_code"))
from build_us_manufacturing_ai_demand import (  # noqa: E402
    allocate_daily_to_shape, capital_recovery_factor, parameter_value, task_shapes
)
from core.capacity import (  # noqa: E402
    average_required_server_groups,
    local_installed_capacity_floor,
)


def china_aligned_continuous_capacity(
    hourly_compute_h: np.ndarray,
    service_capacity_per_server_group_h_per_h: float,
    planning_headroom_fraction: float,
    n_plus_spare_server_groups: int,
    normal_dispatch_reserve_fraction: float,
) -> tuple[float, np.ndarray]:
    """Apply the continuous-capacity constraints used by the China core model."""
    compute = np.asarray(hourly_compute_h, dtype=float)
    if compute.ndim != 1 or len(compute) == 0 or not np.isfinite(compute).all():
        raise ValueError("Hourly compute must be a finite, non-empty one-dimensional array")
    if (compute < 0.0).any():
        raise ValueError("Hourly compute must be non-negative")
    capacity = float(service_capacity_per_server_group_h_per_h)
    dispatch_reserve = float(normal_dispatch_reserve_fraction)
    if not np.isfinite(capacity) or capacity <= 0.0:
        raise ValueError("Server-group service capacity must be positive")
    if not 0.0 <= dispatch_reserve < 1.0:
        raise ValueError("Normal dispatch reserve must be in [0, 1)")

    average_required = average_required_server_groups(
        float(compute.sum()), len(compute), capacity
    )
    planning_floor = local_installed_capacity_floor(
        average_required,
        planning_headroom_fraction,
        n_plus_spare_server_groups,
    )
    peak_dispatch_floor = float(compute.max()) * (1.0 + dispatch_reserve) / capacity
    installed = max(planning_floor, peak_dispatch_floor)
    online = compute / capacity
    return installed, online


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--demand", type=Path, required=True)
    p.add_argument("--national-task-summary", type=Path, required=True)
    p.add_argument("--defaults", type=Path, required=True)
    p.add_argument("--demand-config", type=Path, required=True)
    p.add_argument("--routing-config", type=Path, required=True)
    p.add_argument("--us-cost-config", type=Path, required=True)
    p.add_argument("--us-parameters", type=Path, required=True)
    p.add_argument("--us-api-prices", type=Path, required=True)
    p.add_argument("--compute-efficiency", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    demand = pd.read_csv(args.demand, encoding="utf-8-sig")
    national_tokens = pd.read_csv(args.national_task_summary, encoding="utf-8-sig")
    defaults = yaml.safe_load(args.defaults.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(args.demand_config.read_text(encoding="utf-8"))
    route_cfg = yaml.safe_load(args.routing_config.read_text(encoding="utf-8"))
    params = pd.read_csv(args.us_parameters, encoding="utf-8-sig")
    prices = pd.read_csv(args.us_api_prices, encoding="utf-8-sig")
    efficiency = pd.read_csv(args.compute_efficiency, encoding="utf-8-sig")
    us_cost_cfg = yaml.safe_load(args.us_cost_config.read_text(encoding="utf-8"))
    us = us_cost_cfg["us_cost_environment"]
    if int(us["local_cpu_physical_cores"]) != 64 or int(us["local_cpu_minimum_memory_gb"]) < 256:
        raise ValueError("U.S. CPU cost case must use the common 64-core, >=256GB local unit")
    if int(us["local_cpu_support_term_years"]) < 3:
        raise ValueError("U.S. CPU cost case must include at least three-year support")
    prices = prices[prices.provider.isin(us["formal_api_providers"])]
    shapes = task_shapes(ROOT / cfg["task_shapes"]["profile_file"], cfg["task_shapes"])
    routing = route_cfg["routing_cases"][route_cfg["active_core_routing_case"]]
    multipliers = route_cfg["core_cpu_server_hour_per_reference_l20_accelerator_hour"]
    selected_efficiency = efficiency[
        efficiency.efficiency_case == cfg["compute"]["efficiency_case"]
    ]
    if len(selected_efficiency) != 1:
        raise ValueError("Configured U.S. compute-efficiency case is missing or duplicated")
    accel_per_service = float(selected_efficiency.iloc[0].accelerator_h_per_service_unit)
    gpu_cfg = defaults["server"]
    cpu_cfg = route_cfg["cpu_server"]
    default_n_plus_spare = int(gpu_cfg.get("n_plus_spare_server_groups", 0))
    default_dispatch_reserve = float(gpu_cfg.get("normal_dispatch_reserve_fraction", 0.0))
    for server in (gpu_cfg, cpu_cfg):
        server.setdefault(
            "installed_reserve_fraction",
            float(gpu_cfg["installed_reserve_fraction"]),
        )
        server.setdefault("n_plus_spare_server_groups", default_n_plus_spare)
        server.setdefault("normal_dispatch_reserve_fraction", default_dispatch_reserve)
    gpu_capacity = float(gpu_cfg["accelerators_per_server"])
    cpu_capacity = float(cpu_cfg["service_capacity_cpu_server_h_per_h"])
    cloud_gpu_capacity = float(defaults["hybrid_cloud"]["gpu_capacity_accelerator_h_per_h"])
    if not np.isclose(cloud_gpu_capacity, gpu_capacity):
        raise ValueError("U.S. and China GPU cloud capacity units must match")
    annual_days = float(cfg["annualization_days"])
    gpu_purchase = parameter_value(params, us["local_gpu_server_parameter_id"], "base")
    cpu_prices = {case: parameter_value(params, us["local_cpu_server_parameter_id"], case) for case in ("low","base","high")}
    electricity = parameter_value(params, us["industrial_electricity_parameter_id"], "base")
    cloud_gpu_price = parameter_value(params, us["cloud_gpu_reserved_parameter_id"], "base")
    cloud_cpu_price = parameter_value(params, us["cloud_cpu_reserved_parameter_id"], "base")
    cloud_cpu_capacity = float(us["cloud_cpu_service_capacity_relative_to_local_cpu_server"])
    if not 0 < cloud_cpu_capacity <= 1:
        raise ValueError("U.S. cloud CPU capacity relative to local CPU server must be in (0, 1]")
    coeff = (1+float(us["shared_facility_capex_fraction"])) * capital_recovery_factor(float(us["shared_discount_rate"]), float(us["shared_economic_life_years"])) + float(us["shared_annual_maintenance_fraction"])

    rows=[]
    for (case, naics), ind in demand.groupby(["parameter_case","naics3"]):
        gpu=np.zeros(24); cpu=np.zeros(24)
        for r in ind.itertuples(index=False):
            profile=allocate_daily_to_shape(float(r.effective_service_units_day), shapes[r.task_id])
            share=float(routing.get(r.task_id,0.0))
            gpu += profile*(1-share)*accel_per_service
            cpu += profile*share*float(multipliers.get(r.task_id,1.0))*accel_per_service
        gpu_installed, gpu_online = china_aligned_continuous_capacity(
            gpu,
            gpu_capacity,
            float(gpu_cfg["installed_reserve_fraction"]),
            int(gpu_cfg["n_plus_spare_server_groups"]),
            float(gpu_cfg["normal_dispatch_reserve_fraction"]),
        )
        cpu_installed, cpu_online = china_aligned_continuous_capacity(
            cpu,
            cpu_capacity,
            float(cpu_cfg["installed_reserve_fraction"]),
            int(cpu_cfg["n_plus_spare_server_groups"]),
            float(cpu_cfg["normal_dispatch_reserve_fraction"]),
        )
        gpu_kw=float(gpu_cfg["marginal_facility_multiplier"])*(gpu_installed*float(gpu_cfg["cold_spare_standby_power_kw"])+gpu_online*(float(gpu_cfg["online_idle_wall_power_kw"])-float(gpu_cfg["cold_spare_standby_power_kw"]))+gpu/gpu_capacity*(float(gpu_cfg["maximum_wall_power_kw"])-float(gpu_cfg["online_idle_wall_power_kw"])))
        cpu_kw=float(cpu_cfg["marginal_facility_multiplier"])*(cpu_installed*float(cpu_cfg["cold_spare_standby_power_kw"])+cpu_online*(float(cpu_cfg["online_idle_wall_power_kw"])-float(cpu_cfg["cold_spare_standby_power_kw"]))+cpu/cpu_capacity*(float(cpu_cfg["maximum_wall_power_kw"])-float(cpu_cfg["online_idle_wall_power_kw"])))
        energy=(gpu_kw.sum()+cpu_kw.sum())*annual_days
        cloud_gpu=float(gpu.max())/cloud_gpu_capacity
        cloud_cpu=float(cpu.max())/(cpu_capacity*cloud_cpu_capacity)
        token_rows=national_tokens[national_tokens.parameter_case==case].set_index("task_id")
        annual_in=annual_out=0.0
        for task in ("office","agent"):
            task_total=float(demand[(demand.parameter_case==case)&(demand.task_id==task)].effective_service_units_day.sum())
            task_ind=float(ind[ind.task_id==task].effective_service_units_day.sum())
            if task_total>0:
                annual_in += float(token_rows.loc[task,"annual_input_tokens"])*task_ind/task_total
                annual_out += float(token_rows.loc[task,"annual_output_tokens"])*task_ind/task_total
        for cpu_case,cpu_purchase in cpu_prices.items():
            local=gpu_installed*gpu_purchase*coeff+cpu_installed*cpu_purchase*coeff+energy*electricity
            for price in prices.itertuples(index=False):
                api=annual_in/1e6*float(price.input_usd_per_mtoken)+annual_out/1e6*float(price.output_usd_per_mtoken)
                cloud=cloud_gpu*cloud_gpu_price+cloud_cpu*cloud_cpu_price+api
                rows.append({"parameter_case":case,"naics3":int(naics),"industry_name":ind.industry_name.iloc[0],"routing_case":route_cfg["active_core_routing_case"],"cpu_server_price_case":cpu_case,"provider":price.provider,"cpu_service_share":sum(float(r.effective_service_units_day)*float(routing.get(r.task_id,0)) for r in ind.itertuples())/float(ind.effective_service_units_day.sum()),"local_gpu_servers":gpu_installed,"local_cpu_servers":cpu_installed,"annual_facility_energy_twh":energy/1e9,"local_total_annual_cost_usd":local,"cloud_gpu_reserved_instances":cloud_gpu,"cloud_cpu_reserved_instances":cloud_cpu,"cloud_cpu_service_capacity_relative_to_local_server":cloud_cpu_capacity,"cloud_token_api_cost_usd":api,"cloud_gpu_reserved_cost_usd":cloud_gpu*cloud_gpu_price,"cloud_cpu_reserved_cost_usd":cloud_cpu*cloud_cpu_price,"cloud_total_annual_cost_usd":cloud,"cloud_to_local_cost_ratio":cloud/local,"local_savings_vs_cloud_fraction":1-local/cloud})
    detail=pd.DataFrame(rows)
    if detail.naics3.nunique()!=21: raise ValueError("Expected 21 U.S. NAICS3 industries")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    detail.to_csv(args.output_dir/"us_naics3_comparison.csv",index=False,encoding="utf-8-sig")
    numeric=["local_gpu_servers","local_cpu_servers","annual_facility_energy_twh","local_total_annual_cost_usd","cloud_gpu_reserved_instances","cloud_cpu_reserved_instances","cloud_token_api_cost_usd","cloud_gpu_reserved_cost_usd","cloud_cpu_reserved_cost_usd","cloud_total_annual_cost_usd"]
    national=detail.groupby(["parameter_case","cpu_server_price_case","provider"],as_index=False)[numeric].sum()
    national["aggregation_boundary"]="sum_of_21_NAICS3_industries_separately_sized_with_china_aligned_continuous_capacity"
    national["cloud_cpu_service_capacity_relative_to_local_server"]=cloud_cpu_capacity
    national["cloud_to_local_cost_ratio"]=national.cloud_total_annual_cost_usd/national.local_total_annual_cost_usd
    national["local_savings_vs_cloud_fraction"]=1-national.local_total_annual_cost_usd/national.cloud_total_annual_cost_usd
    national.to_csv(args.output_dir/"us_national_comparison.csv",index=False,encoding="utf-8-sig")
    base=national.query("parameter_case=='base' and cpu_server_price_case=='base'").sort_values("cloud_total_annual_cost_usd")
    lines="\n".join(f"| {r.provider} | {r.local_total_annual_cost_usd/1e9:.3f} | {r.cloud_total_annual_cost_usd/1e9:.3f} | {r.cloud_to_local_cost_ratio:.3f} | {r.local_savings_vs_cloud_fraction:.1%} |" for r in base.itertuples(index=False))
    (args.output_dir/"findings.md").write_text("# 美国21个NAICS制造行业异构硬件成本\n\n主口径为21个行业按美国本土base需求分别定容后加总；本地GPU/CPU装机采用与中国核心模型一致的连续容量约束：日均容量下限叠加规划裕量与N+k备用，并满足逐小时运行容量。\n\n| API厂商 | 本地 十亿美元/年 | 完整云 十亿美元/年 | 云/本地 | 本地节省 |\n|---|---:|---:|---:|---:|\n"+lines+"\n",encoding="utf-8")
    done={"status":"complete_validated_us_21_naics_heterogeneous_cost","industries":21,"demand_cases":sorted(detail.parameter_case.unique()),"cpu_price_cases":sorted(detail.cpu_server_price_case.unique()),"providers":sorted(detail.provider.unique()),"aggregation_boundary":"industry_sum_china_aligned_continuous_capacity","installed_capacity_rule":gpu_cfg["installed_capacity_rule"],"planning_reserve_fraction":float(gpu_cfg["installed_reserve_fraction"]),"n_plus_spare_server_groups":int(gpu_cfg["n_plus_spare_server_groups"]),"normal_dispatch_reserve_fraction":float(gpu_cfg["normal_dispatch_reserve_fraction"]),"accelerator_h_per_service_unit":accel_per_service,"compute_efficiency_case":cfg["compute"]["efficiency_case"]}
    (args.output_dir/"validated.done.json").write_text(json.dumps(done,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__": main()
