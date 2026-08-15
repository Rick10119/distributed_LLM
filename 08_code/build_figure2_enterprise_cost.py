#!/usr/bin/env python3
"""Build Figure 2: enterprise cost across active group architectures."""
from __future__ import annotations
import argparse,json,os,tempfile
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"distributed_llm_matplotlib"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ARCHS=["IF","IG_1host","IG_multisite"]
LABEL={"IF":"逐厂独立","IG_1host":"集团单节点","IG_multisite":"集团多节点协调"}
COLOR={"IF":"#2F6B9A","IG_1host":"#D28B35","IG_multisite":"#4E8B68"}
COMP=[("industry_equivalent_annual_server_cost_rmb","服务器"),("industry_equivalent_annual_ai_energy_cost_rmb","电量"),("industry_equivalent_annual_incremental_maximum_demand_cost_rmb","最大需量")]
PROJECT_ROOT=Path(__file__).resolve().parents[1]
FIGURE_ROOT=PROJECT_ROOT/"05_results/v0.8.0/result/manuscript_figures"
GROUP_NATIONAL_ROOT=PROJECT_ROOT/"05_results/v0.8.0/result/group_architecture_core/national"
def prepare(core_path:Path,align_path:Path,version:str)->pd.DataFrame:
    c=pd.read_csv(core_path,encoding="utf-8-sig"); a=pd.read_csv(align_path,encoding="utf-8-sig")
    if set(c.architecture)!=set(ARCHS) or len(c)!=93 or c.industry.nunique()!=31: raise ValueError("Figure 2 requires 31 industries and the three active architectures")
    if len(a)!=31 or set(a.architecture)!={"IG_1host"}: raise ValueError("Figure 2 requires 31 IG_1host load-alignment pairs")
    rows=[]
    if_cost = c[c.architecture.eq("IF")].set_index("industry")["annual_incremental_total_cost_rmb"]
    for arch in ARCHS:
        s=c[c.architecture.eq(arch)]
        for col,label in COMP: rows.append(dict(panel="a",industry="national",architecture=arch,metric=label,value=float(s[col].sum()),unit="RMB/year"))
        for r in s.itertuples(index=False): rows.append(dict(panel="b",industry=r.industry,architecture=arch,metric="cost_ratio_to_IF",value=float(r.annual_incremental_total_cost_rmb)/float(if_cost[r.industry]),unit="ratio"))
    col="industry_equivalent_load_alignment_value_total_cost_rmb"
    for r in a.itertuples(index=False): rows.append(dict(panel="c",industry=r.industry,architecture="IG_1host",metric="zero_minus_actual",value=float(getattr(r,col)),unit="RMB/year"))
    out=pd.DataFrame(rows);out.insert(0,"model_version",version);return out
def plot(d,outputs):
    plt.rcParams.update({"font.sans-serif":["Arial Unicode MS","PingFang SC","DejaVu Sans"],"axes.unicode_minus":False,"svg.fonttype":"none"})
    fig,axs=plt.subplots(1,3,figsize=(14,4.6));
    p=d[d.panel.eq("a")];bottom=np.zeros(3);x=np.arange(3)
    for metric,color in [("服务器","#587A9E"),("电量","#D7A85B"),("最大需量","#9A6A8A")]:
        vals=np.array([p[p.architecture.eq(a)&p.metric.eq(metric)].value.sum()/1e9 for a in ARCHS]);axs[0].bar(x,vals,bottom=bottom,label=metric,color=color);bottom+=vals
    axs[0].set_xticks(x,[LABEL[a] for a in ARCHS],rotation=15);axs[0].set_ylabel("行业等效年成本（十亿元）");axs[0].legend(frameon=False);axs[0].set_title("a  全国成本构成",loc="left",fontweight="bold")
    p=d[d.panel.eq("b")]; data=[p[p.architecture.eq(a)].value for a in ARCHS];bp=axs[1].boxplot(data,labels=[LABEL[a] for a in ARCHS],patch_artist=True,showfliers=False);[box.set_facecolor(COLOR[a]) for box,a in zip(bp["boxes"],ARCHS)];axs[1].axhline(1,c="#555",ls=":");axs[1].set_ylabel("相对逐厂独立成本");axs[1].tick_params(axis="x",rotation=15);axs[1].set_title("b  31行业相对成本",loc="left",fontweight="bold")
    p=d[d.panel.eq("c")].sort_values("value");colors=np.where(p.value>=0,"#4E8B68","#C65D4B");axs[2].barh(p.industry,p.value/1e6,color=colors);axs[2].axvline(0,c="#555",lw=.8);axs[2].set_xlabel("零原始负荷－实际负荷（百万元/年）");axs[2].set_title("c  原负荷匹配价值（IG-1host）",loc="left",fontweight="bold");axs[2].tick_params(axis="y",labelsize=6)
    for ax in axs:ax.grid(axis="y",alpha=.2)
    fig.suptitle("31行业企业内部部署成本与负荷匹配价值",fontsize=15,fontweight="bold");fig.text(.5,.01,"正的零负荷差额表示与工厂实际负荷联合优化降低了AI增量成本；该配对只对IG-1host运行。",ha="center",fontsize=8,color="#555");fig.tight_layout(rect=[0,.04,1,.94])
    for o in outputs:o.parent.mkdir(parents=True,exist_ok=True);fig.savefig(o,bbox_inches="tight")
    plt.close(fig)
def main():
    p=argparse.ArgumentParser(description="Build Figure 2; no arguments use the v0.8.0 mainline paths.");p.add_argument("--national-input",type=Path,default=GROUP_NATIONAL_ROOT/"core_scenarios.csv");p.add_argument("--alignment-input",type=Path,default=GROUP_NATIONAL_ROOT/"ig_1host_load_alignment.csv");p.add_argument("--model-version",default="v0.8.0");p.add_argument("--data-output",type=Path,default=FIGURE_ROOT/"figure2_enterprise_cost_data.csv");p.add_argument("--png-output",type=Path,default=FIGURE_ROOT/"figure2_enterprise_cost.png");p.add_argument("--pdf-output",type=Path,default=FIGURE_ROOT/"figure2_enterprise_cost.pdf");p.add_argument("--svg-output",type=Path,default=FIGURE_ROOT/"figure2_enterprise_cost.svg");p.add_argument("--validation-output",type=Path,default=FIGURE_ROOT/"figure2_enterprise_cost.validated.done.json");a=p.parse_args();d=prepare(a.national_input,a.alignment_input,a.model_version);a.data_output.parent.mkdir(parents=True,exist_ok=True);d.to_csv(a.data_output,index=False,encoding="utf-8-sig");plot(d,[a.png_output,a.pdf_output,a.svg_output]);a.validation_output.write_text(json.dumps({"status":"validated","architectures":ARCHS,"II_1host_in_figure":False,"alignment_industries":31},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Figure 2 written to {a.svg_output}")
if __name__=="__main__":main()
