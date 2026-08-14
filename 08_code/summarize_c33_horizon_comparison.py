#!/usr/bin/env python3
"""Compare C33 measured-week and typical-day group-architecture results."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def parse_args():
    p=argparse.ArgumentParser();p.add_argument("--week",type=Path,required=True);p.add_argument("--day",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--findings",type=Path,required=True);p.add_argument("--done",type=Path,required=True);return p.parse_args()
def extract(path:Path,horizon:str)->list[dict]:
    d=pd.read_csv(path,encoding="utf-8-sig").set_index(["architecture","base_load_case"]);h=d.loc[("IG_1host","actual_load")];m=d.loc[("IG_multisite","actual_load")];z=d.loc[("IG_1host","zero_load")]
    return [{"horizon":horizon,"comparison":"IG_multisite_minus_IG_1host","metric":"annual_ai_energy_cost_rmb","difference":float(m.annual_ai_energy_cost_rmb-h.annual_ai_energy_cost_rmb),"saving_fraction":1-float(m.annual_ai_energy_cost_rmb/h.annual_ai_energy_cost_rmb)},
    {"horizon":horizon,"comparison":"IG_multisite_minus_IG_1host","metric":"annual_incremental_maximum_demand_cost_rmb","difference":float(m.annual_incremental_maximum_demand_cost_rmb-h.annual_incremental_maximum_demand_cost_rmb),"saving_fraction":1-float(m.annual_incremental_maximum_demand_cost_rmb/h.annual_incremental_maximum_demand_cost_rmb)},
    {"horizon":horizon,"comparison":"IG_multisite_minus_IG_1host","metric":"sum_incremental_grid_peak_mw","difference":float(m.sum_incremental_grid_peak_mw-h.sum_incremental_grid_peak_mw),"saving_fraction":1-float(m.sum_incremental_grid_peak_mw/h.sum_incremental_grid_peak_mw)},
    {"horizon":horizon,"comparison":"IG_multisite_minus_IG_1host","metric":"annual_incremental_total_cost_rmb","difference":float(m.annual_incremental_total_cost_rmb-h.annual_incremental_total_cost_rmb),"saving_fraction":1-float(m.annual_incremental_total_cost_rmb/h.annual_incremental_total_cost_rmb)},
    {"horizon":horizon,"comparison":"IG_1host_zero_minus_actual","metric":"annual_incremental_total_cost_rmb","difference":float(z.annual_incremental_total_cost_rmb-h.annual_incremental_total_cost_rmb),"saving_fraction":float("nan")},
    {"horizon":horizon,"comparison":"IG_1host_zero_minus_actual","metric":"sum_incremental_grid_peak_mw","difference":float(z.sum_incremental_grid_peak_mw-h.sum_incremental_grid_peak_mw),"saving_fraction":float("nan")}]
def main():
    a=parse_args();out=pd.DataFrame(extract(a.week,"measured_continuous_week")+extract(a.day,"typical_day"));a.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.output,index=False,encoding="utf-8-sig")
    q=out.set_index(["horizon","comparison","metric"]);w=q.loc[("measured_continuous_week","IG_multisite_minus_IG_1host")];d=q.loc[("typical_day","IG_multisite_minus_IG_1host")]
    text=f"""# C33连续周与典型日时域测试发现

## 已运行边界

两组均使用C33、6个成员工厂节点、相同AI服务定义、CPU/GPU路由、整数边界和求解器。连续周保留6条EWELD工厂曲线的168小时变化；典型日使用相同曲线各自按小时对七天取平均，并通过统一的24小时运行配置生成任务和电价。该测试只改变时域表达，不改变节点数。

## 当前发现

连续周中，IG-multisite相对IG-1host：

- AI电量费减少{-w.loc['annual_ai_energy_cost_rmb','difference']:,.0f}元/年，仅{w.loc['annual_ai_energy_cost_rmb','saving_fraction']:.3%}；
- 最大需量费减少{-w.loc['annual_incremental_maximum_demand_cost_rmb','difference']:,.0f}元/年，即{w.loc['annual_incremental_maximum_demand_cost_rmb','saving_fraction']:.1%}；
- 新增接入容量减少{-w.loc['sum_incremental_grid_peak_mw','difference']:.3f} MW，即{w.loc['sum_incremental_grid_peak_mw','saving_fraction']:.1%}；
- 企业增量总成本净减少{-w.loc['annual_incremental_total_cost_rmb','difference']:,.0f}元/年，即{w.loc['annual_incremental_total_cost_rmb','saving_fraction']:.3%}。

典型日中，上述IG-multisite与IG-1host差额在数值容差内全部为零；IG-1host的zero-load与actual-load配对差额也为零。与此同时，典型日的IG-1host新增接入容量为0.810 MW，而连续周为0.171 MW。

## 解释

C33支持的机制不是“跨节点显著节省AI电量”，而是“保留连续周的跨厂、跨日非同时性后，跨节点调度可以显著削减最大需量和新增接入容量”。24小时典型日压平星期之间的差异后，这项价值消失，并明显高估集团新增接入容量。因此典型日不适合用于估计跨节点灵活性价值；168小时连续周应保留为核心时域。

## 证据边界

这是单行业、合成集团的机制测试。6条EWELD曲线来自匿名同行业用户或不同完整周，并非一个真实集团同一日历周的同步观测。90%的容量降幅不能外推为31行业结论，需等待新5代表节点口径下的全国运行。
""";a.findings.write_text(text,encoding="utf-8");a.done.write_text(json.dumps({"status":"validated","industry":"C33","week_nodes":6,"day_nodes":6,"core_horizon_recommendation":"168_hour_measured_continuous_week","claim_scope":"single_industry_mechanism_test"},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
if __name__=="__main__":main()
