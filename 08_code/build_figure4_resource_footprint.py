#!/usr/bin/env python3
"""Build Figure 4 without using the retired II_1host scenario."""
from __future__ import annotations
import argparse,json,os,tempfile
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"distributed_llm_matplotlib"))
import matplotlib.pyplot as plt
import pandas as pd
import yaml
PROJECT_ROOT=Path(__file__).resolve().parents[1]
FIGURE_ROOT=PROJECT_ROOT/"05_results/v0.8.0/result/manuscript_figures"
GROUP_NATIONAL_ROOT=PROJECT_ROOT/"05_results/v0.8.0/result/group_architecture_core/national"
CLOUD_ROOT=PROJECT_ROOT/"05_results/sensitivity/v0.8.0/national_cloud_center_v1"
def prepare(core_path,cloud_path,registry_path,water_path,version):
    core=pd.read_csv(core_path,encoding="utf-8-sig");cloud=pd.read_csv(cloud_path,encoding="utf-8-sig");reg=yaml.safe_load(Path(registry_path).read_text(encoding="utf-8"));w=pd.read_csv(water_path,encoding="utf-8-sig").set_index("comparison_mode");m=reg["resource_footprint"]["water"]["comparison_modes"]
    if set(core.architecture)!={"IF","IG_1host","IG_multisite"}:raise ValueError("Figure 4 received an obsolete core architecture table")
    rows=[]
    for a in ["IF","IG_1host","IG_multisite"]:
        s=core[core.architecture.eq(a)];energy=float(s.industry_equivalent_annual_ai_facility_energy_twh.sum());peak=float(s.industry_equivalent_sum_incremental_grid_peak_mw.sum());intensity=float(w.loc[m["local"],"site_water_use_l_per_kwh_it"]);rows.append(dict(model_version=version,deployment=a,facility_energy_twh=energy,grid_capacity_mw=peak,water_m3=energy/1.30*intensity*1e6,evidence="31_industry_group_core"))
    c=cloud.iloc[0];energy=float(c.annual_ai_facility_energy_twh);intensity=float(w.loc[m["china_cloud"],"site_water_use_l_per_kwh_it"]);rows.append(dict(model_version=version,deployment="CLOUD_ALL_1HOST",facility_energy_twh=energy,grid_capacity_mw=float(c.incremental_grid_expansion_mw),water_m3=energy/1.22*intensity*1e6,evidence="independent_national_cloud_counterfactual"));return pd.DataFrame(rows)
def plot(d,svg,png):
    lab={"IF":"逐厂独立","IG_1host":"集团单节点","IG_multisite":"集团多节点","CLOUD_ALL_1HOST":"独立大型云反事实"};colors=["#2F6B9A","#D28B35","#4E8B68","#8A5D9E"];fig,axs=plt.subplots(1,2,figsize=(11,4.3));x=range(4);axs[0].bar(x,d.water_m3/1e6,color=colors);axs[0].set_ylabel("现场取水（百万m³/年）");axs[0].set_title("a  运行取水",loc="left",fontweight="bold");axs[1].bar(x,d.grid_capacity_mw/1e3,color=colors);axs[1].set_ylabel("接入容量（GW）");axs[1].set_title("b  接入容量及空间集中",loc="left",fontweight="bold");
    for ax in axs:ax.set_xticks(x,[lab[v] for v in d.deployment],rotation=18,ha="right");ax.grid(axis="y",alpha=.2)
    fig.suptitle("企业内部部署与独立大型云反事实的运行资源需求",fontweight="bold");fig.text(.5,.01,"注：大型云由独立全国云情景求解，不使用或替代II_1host；本图不报告尚未重建的建筑材料足迹。",ha="center",fontsize=8,color="#555");fig.tight_layout(rect=[0,.05,1,.93]);svg.parent.mkdir(parents=True,exist_ok=True);fig.savefig(svg,bbox_inches="tight");fig.savefig(png,bbox_inches="tight");plt.close(fig)
def main():
    p=argparse.ArgumentParser(description="Build Figure 4; no arguments use the v0.8.0 mainline paths.");p.add_argument("--core-input",type=Path,default=GROUP_NATIONAL_ROOT/"core_scenarios.csv");p.add_argument("--cloud-input",type=Path,default=CLOUD_ROOT/"summary.csv");p.add_argument("--scenario-registry",type=Path,default=PROJECT_ROOT/"config/scenarios/mainline.yaml");p.add_argument("--water-input",type=Path,default=PROJECT_ROOT/"02_data/processed/resource_footprint/small_china_us_water_baseline_comparison.csv");p.add_argument("--model-version",default="v0.8.0");p.add_argument("--data-output",type=Path,default=FIGURE_ROOT/"figure4_resource_footprint_data.csv");p.add_argument("--svg-output",type=Path,default=FIGURE_ROOT/"figure4_resource_footprint.svg");p.add_argument("--png-output",type=Path,default=FIGURE_ROOT/"figure4_resource_footprint.png");p.add_argument("--validation-output",type=Path,default=FIGURE_ROOT/"figure4_resource_footprint.validated.done.json");a=p.parse_args();d=prepare(a.core_input,a.cloud_input,a.scenario_registry,a.water_input,a.model_version);a.data_output.parent.mkdir(parents=True,exist_ok=True);d.to_csv(a.data_output,index=False,encoding="utf-8-sig");plot(d,a.svg_output,a.png_output);a.validation_output.write_text(json.dumps({"status":"validated","II_1host_in_figure":False,"cloud_source":"independent_national_cloud_scenario"},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Figure 4 written to {a.svg_output}")
if __name__=="__main__":main()
