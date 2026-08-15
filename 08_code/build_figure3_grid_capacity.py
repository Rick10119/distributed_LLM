#!/usr/bin/env python3
"""Build Figure 3: group-host load matching and cross-node flexibility."""
from __future__ import annotations
import argparse,json,os,tempfile
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"distributed_llm_matplotlib"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
PROJECT_ROOT=Path(__file__).resolve().parents[1]
FIGURE_ROOT=PROJECT_ROOT/"05_results/v0.8.0/result/manuscript_figures"
CORE_ROOT=PROJECT_ROOT/"05_results/v0.8.0/result/group_architecture_core"
DEFAULT_INDUSTRIES=[f"C{i}" for i in range(13,44)]

def prepare(root:Path,industries:list[str],version:str)->pd.DataFrame:
    rows=[]
    for ind in industries:
        s=pd.read_csv(root/ind/"summary.csv",encoding="utf-8-sig")
        actual=s[s.base_load_case.eq("actual_load")].set_index("architecture")
        if set(actual.index)!={"IF","IG_1host","IG_multisite"}:raise ValueError(f"{ind}: incomplete Figure 3 architectures")
        one=actual.loc["IG_1host"];multi=actual.loc["IG_multisite"]
        rows += [dict(model_version=version,panel="b",industry=ind,architecture="IG_multisite_minus_IG_1host",metric="annual_cost_difference",value=float(multi.annual_incremental_total_cost_rmb-one.annual_incremental_total_cost_rmb),unit="RMB/year"),dict(model_version=version,panel="c",industry=ind,architecture="IG_multisite_minus_IG_1host",metric="sum_grid_peak_difference",value=float(multi.sum_incremental_grid_peak_mw-one.sum_incremental_grid_peak_mw),unit="MW")]
    focus="C36" if "C36" in industries else industries[0]
    h=pd.read_csv(root/focus/"hourly.csv",encoding="utf-8-sig")
    for arch in ["IG_1host","IG_multisite"]:
        p=h[h.architecture.eq(arch)&h.base_load_case.eq("actual_load")].groupby("hour",as_index=False)[["base_load_mw","ai_facility_power_mw","grid_import_mw"]].sum()
        for r in p.itertuples(index=False):
            for metric in ["base_load_mw","ai_facility_power_mw","grid_import_mw"]:rows.append(dict(model_version=version,panel="a",industry=focus,architecture=arch,metric=metric,value=float(getattr(r,metric)),unit="MW",hour=int(r.hour)))
    return pd.DataFrame(rows)

def plot(d,svg,png):
    plt.rcParams.update({"font.sans-serif":["Arial Unicode MS","PingFang SC","DejaVu Sans"],"axes.unicode_minus":False,"svg.fonttype":"none"})
    fig,axs=plt.subplots(2,2,figsize=(13.5,8));
    p=d[d.panel.eq("a")];x=np.arange(168)
    for ax,arch,title in [(axs[0,0],"IG_1host","a  集团单节点叠加于承载工厂"),(axs[0,1],"IG_multisite","b  多节点跨厂调度后的集团总负荷")]:
        q=p[p.architecture.eq(arch)];base=q[q.metric.eq("base_load_mw")].sort_values("hour").value.to_numpy();ai=q[q.metric.eq("ai_facility_power_mw")].sort_values("hour").value.to_numpy();ax.fill_between(x,0,base,color="#90A9BC",alpha=.75,label="原始生产负荷");ax.fill_between(x,base,base+ai,color="#D27A2C",alpha=.85,label="AI设施负荷");ax.plot(x,base+ai,c="#234F70",lw=1);ax.set_title(title,loc="left",fontweight="bold");ax.set_ylabel("MW");ax.set_xlim(0,167);ax.grid(alpha=.18)
    axs[0,0].legend(frameon=False,ncol=2,fontsize=8)
    for ax in axs[0]:ax.set_xticks([12,36,60,84,108,132,156],["一","二","三","四","五","六","日"])
    for ax,panel,metric,title,xlabel in [(axs[1,0],"b","annual_cost_difference","c  跨节点灵活性的成本差","IG-multisite－IG-1host（百万元/年）"),(axs[1,1],"c","sum_grid_peak_difference","d  跨节点灵活性的接入峰值差","IG-multisite－IG-1host（MW）")]:
        q=d[d.panel.eq(panel)&d.metric.eq(metric)].sort_values("value");scale=1e6 if metric=="annual_cost_difference" else 1;vals=q.value/scale;colors=np.where(vals<=0,"#4E8B68","#C65D4B");ax.barh(q.industry,vals,color=colors);ax.axvline(0,c="#555",lw=.8);ax.set_xlabel(xlabel);ax.set_title(title,loc="left",fontweight="bold");ax.tick_params(axis="y",labelsize=6);ax.grid(axis="x",alpha=.2)
    fig.suptitle("集团单节点负荷匹配与跨工厂算力调度价值",fontsize=15,fontweight="bold");fig.text(.5,.012,"注：上排以C36为示例；下排为31行业实际负荷下的重新优化差额。负值表示多节点协调降低对应指标。",ha="center",fontsize=8,color="#555");fig.tight_layout(rect=[0,.04,1,.95]);svg.parent.mkdir(parents=True,exist_ok=True);fig.savefig(svg,bbox_inches="tight");fig.savefig(png,bbox_inches="tight");plt.close(fig)

def main():
    p=argparse.ArgumentParser(description="Build Figure 3; no arguments use the v0.8.0 mainline paths.");p.add_argument("--core-root",type=Path,default=CORE_ROOT);p.add_argument("--industries",nargs="+",default=DEFAULT_INDUSTRIES);p.add_argument("--model-version",default="v0.8.0");p.add_argument("--data-output",type=Path,default=FIGURE_ROOT/"figure3_grid_capacity_data.csv");p.add_argument("--svg-output",type=Path,default=FIGURE_ROOT/"figure3_grid_capacity.svg");p.add_argument("--png-output",type=Path,default=FIGURE_ROOT/"figure3_grid_capacity.png");p.add_argument("--validation-output",type=Path,default=FIGURE_ROOT/"figure3_grid_capacity.validated.done.json");a=p.parse_args();d=prepare(a.core_root,a.industries,a.model_version);a.data_output.parent.mkdir(parents=True,exist_ok=True);d.to_csv(a.data_output,index=False,encoding="utf-8-sig");plot(d,a.svg_output,a.png_output);a.validation_output.write_text(json.dumps({"status":"validated","industry_count":31,"architectures":["IG_1host","IG_multisite"],"II_1host_in_figure":False},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Figure 3 written to {a.svg_output}")
if __name__=="__main__":main()
