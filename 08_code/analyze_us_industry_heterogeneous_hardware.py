#!/usr/bin/env python3
"""Recompute U.S. 21-NAICS manufacturing demand with CPU/GPU routing."""

from __future__ import annotations

import argparse
import json
import math
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--demand", type=Path, required=True)
    p.add_argument("--national-task-summary", type=Path, required=True)
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
    compute = cfg["compute"]
    selected_efficiency = efficiency[
        efficiency.efficiency_case == cfg["compute"]["efficiency_case"]
    ]
    if len(selected_efficiency) != 1:
        raise ValueError("Configured U.S. compute-efficiency case is missing or duplicated")
    accel_per_service = float(selected_efficiency.iloc[0].accelerator_h_per_service_unit)
    reserve = float(compute["installed_reserve_fraction"])
    utilization = float(compute["installed_utilization"])
    pue = float(compute["marginal_facility_multiplier"])
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
        gpu_installed=float(gpu.max())/(2*utilization)*(1+reserve)
        cpu_installed=float(cpu.max())/utilization*(1+reserve)
        gpu_online=gpu/(2*utilization); cpu_online=cpu/utilization
        gpu_kw=pue*(gpu_installed*.02+gpu_online*(.36-.02)+gpu/2*(1.50-.36))
        cpu_cfg=route_cfg["cpu_server"]
        cpu_kw=pue*(cpu_installed*float(cpu_cfg["cold_spare_standby_power_kw"])+cpu_online*(float(cpu_cfg["online_idle_wall_power_kw"])-float(cpu_cfg["cold_spare_standby_power_kw"]))+cpu*(float(cpu_cfg["maximum_wall_power_kw"])-float(cpu_cfg["online_idle_wall_power_kw"])))
        energy=(gpu_kw.sum()+cpu_kw.sum())*annual_days
        cloud_gpu=math.ceil(float(gpu.max())/(2*utilization))
        cloud_cpu=math.ceil(float(cpu.max())/(utilization*cloud_cpu_capacity))
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
    national["aggregation_boundary"]="sum_of_21_NAICS3_industries_separately_sized"
    national["cloud_cpu_service_capacity_relative_to_local_server"]=cloud_cpu_capacity
    national["cloud_to_local_cost_ratio"]=national.cloud_total_annual_cost_usd/national.local_total_annual_cost_usd
    national["local_savings_vs_cloud_fraction"]=1-national.local_total_annual_cost_usd/national.cloud_total_annual_cost_usd
    national.to_csv(args.output_dir/"us_national_comparison.csv",index=False,encoding="utf-8-sig")
    base=national.query("parameter_case=='base' and cpu_server_price_case=='base'").sort_values("cloud_total_annual_cost_usd")
    lines="\n".join(f"| {r.provider} | {r.local_total_annual_cost_usd/1e9:.3f} | {r.cloud_total_annual_cost_usd/1e9:.3f} | {r.cloud_to_local_cost_ratio:.3f} | {r.local_savings_vs_cloud_fraction:.1%} |" for r in base.itertuples(index=False))
    (args.output_dir/"findings.md").write_text("# 美国21个NAICS制造行业异构硬件成本\n\n主口径为21个行业按美国本土base需求分别定容后加总。\n\n| API厂商 | 本地 十亿美元/年 | 完整云 十亿美元/年 | 云/本地 | 本地节省 |\n|---|---:|---:|---:|---:|\n"+lines+"\n",encoding="utf-8")
    done={"status":"complete_validated_us_21_naics_heterogeneous_cost","industries":21,"demand_cases":sorted(detail.parameter_case.unique()),"cpu_price_cases":sorted(detail.cpu_server_price_case.unique()),"providers":sorted(detail.provider.unique()),"aggregation_boundary":"industry_sum","accelerator_h_per_service_unit":accel_per_service,"compute_efficiency_case":cfg["compute"]["efficiency_case"]}
    (args.output_dir/"validated.done.json").write_text(json.dumps(done,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__": main()
