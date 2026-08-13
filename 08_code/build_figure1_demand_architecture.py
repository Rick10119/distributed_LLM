#!/usr/bin/env python3
"""Build Figure 1 from the active 31-industry group-architecture core package."""
from __future__ import annotations
import argparse, json, os, tempfile
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "distributed_llm_matplotlib"))
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd
import yaml

ARCHS = ["IF", "IG_1host", "IG_multisite"]
LABELS = {"IF":"逐厂独立", "IG_1host":"集团单节点", "IG_multisite":"集团多节点协调"}
COLORS = {"IF":"#2F6B9A", "IG_1host":"#D28B35", "IG_multisite":"#4E8B68"}
TASKS = ["office","agent","vision","maintenance","scheduling","simulation"]

def prepare(service_path: Path, national_path: Path, routing_path: Path, version: str) -> pd.DataFrame:
    service=pd.read_csv(service_path,encoding="utf-8-sig"); core=pd.read_csv(national_path,encoding="utf-8-sig")
    routing_doc=yaml.safe_load(routing_path.read_text(encoding="utf-8")); case=routing_doc["active_core_routing_case"]; routing=routing_doc["routing_cases"][case]
    if set(core.architecture)!=set(ARCHS) or core.industry.nunique()!=31 or len(core)!=93: raise ValueError("Figure 1 requires the validated 31-industry three-architecture core table")
    base=service[service.parameter_case.eq("base")].copy()
    if len(base)!=186: raise ValueError("Expected 31 industries times six tasks")
    rows=[]
    task=base.groupby(["task_id","task_name_cn"],as_index=False).effective_service_units_day.sum()
    for r in task.itertuples(index=False):
        cpu=float(routing.get(r.task_id,0)); rows += [dict(panel="a",item=r.task_id,label=r.task_name_cn,group="GPU",value=float(r.effective_service_units_day)*(1-cpu)),dict(panel="a",item=r.task_id,label=r.task_name_cn,group="CPU",value=float(r.effective_service_units_day)*cpu)]
    for a in ARCHS: rows.append(dict(panel="b",item=a,label=LABELS[a],group=a,value=np.nan))
    energy=core.groupby("architecture")["industry_equivalent_annual_ai_facility_energy_twh"].sum()
    for a in ARCHS: rows.append(dict(panel="c",item=a,label=LABELS[a],group=a,value=float(energy[a])))
    totals=base.groupby("industry_code").effective_service_units_day.transform("sum")
    for r,share in zip(base.itertuples(index=False),base.effective_service_units_day/totals): rows.append(dict(panel="d",item=r.task_id,label=r.task_name_cn,group=r.industry_code,value=float(share)))
    out=pd.DataFrame(rows); out.insert(0,"model_version",version); return out

def plot(d: pd.DataFrame, outputs: list[Path]):
    plt.rcParams.update({"font.sans-serif":["Arial Unicode MS","PingFang SC","DejaVu Sans"],"axes.unicode_minus":False,"svg.fonttype":"none"})
    fig=plt.figure(figsize=(13.5,9)); gs=fig.add_gridspec(2,2,height_ratios=[.8,1.2],hspace=.32,wspace=.28)
    ax=fig.add_subplot(gs[0,0]); p=d[d.panel.eq("a")]; totals=p.groupby("item").value.sum().reindex(TASKS); x=np.arange(6); gpu=np.array([p[p.item.eq(t)&p.group.eq("GPU")].value.sum() for t in TASKS]); cpu=totals.to_numpy()-gpu
    ax.bar(x,gpu/totals.sum()*100,color="#2F6B9A",label="GPU"); ax.bar(x,cpu/totals.sum()*100,bottom=gpu/totals.sum()*100,color="#65A58A",label="CPU"); ax.set_xticks(x,[t for t in TASKS],rotation=25,ha="right"); ax.set_ylabel("全国有效服务占比（%）"); ax.legend(frameon=False); ax.set_title("a  六类任务与CPU/GPU路由",loc="left",fontweight="bold")
    ax=fig.add_subplot(gs[0,1]); ax.axis("off"); ax.set_title("b  三种企业内部部署架构",loc="left",fontweight="bold")
    for y,a in zip([.78,.5,.22],ARCHS):
        c=COLORS[a]; xs=[.15,.31,.47,.63]
        for xx in xs: ax.add_patch(Circle((xx,y),.035,fc="white",ec=c,lw=1.5)); ax.text(xx,y,"厂",ha="center",va="center",fontsize=7)
        if a=="IF":
            for xx in xs: ax.add_patch(Rectangle((xx-.025,y-.1),.05,.03,fc=c,alpha=.8))
        elif a=="IG_1host":
            ax.add_patch(Rectangle((xs[1]-.035,y-.105),.07,.035,fc=c,alpha=.85)); [ax.plot([xx,xs[1]],[y,y],c=c,lw=.8) for xx in xs]
        else:
            for xx in xs: ax.add_patch(Rectangle((xx-.025,y-.1),.05,.03,fc=c,alpha=.8))
            for x1,x2 in zip(xs[:-1],xs[1:]): ax.annotate("",(x2,y-.085),(x1,y-.085),arrowprops=dict(arrowstyle="<->",color=c,lw=.8))
        ax.text(.02,y+.08,LABELS[a],color=c,fontweight="bold");
    ax.set_xlim(0,1); ax.set_ylim(.05,.95)
    ax=fig.add_subplot(gs[1,1]); p=d[d.panel.eq("c")].set_index("item"); vals=[p.loc[a,"value"] for a in ARCHS]; ax.bar(range(3),vals,color=[COLORS[a] for a in ARCHS]); ax.set_xticks(range(3),[LABELS[a] for a in ARCHS],rotation=15); ax.set_ylabel("行业等效设施电量（TWh/年）"); ax.set_title("c  相同有效服务下的设施电量",loc="left",fontweight="bold"); ax.grid(axis="y",alpha=.25)
    ax=fig.add_subplot(gs[1,0]); p=d[d.panel.eq("d")]; pivot=p.pivot(index="group",columns="item",values="value").reindex(columns=TASKS); im=ax.imshow(pivot.to_numpy()*100,aspect="auto",cmap="Blues"); ax.set_yticks(range(31),pivot.index,fontsize=6.5); ax.set_xticks(range(6),TASKS,rotation=25,ha="right"); ax.set_title("d  31行业任务构成",loc="left",fontweight="bold"); fig.colorbar(im,ax=ax,label="行业内有效服务占比（%）",fraction=.035)
    fig.suptitle("制造业AI需求、异构计算与集团部署架构",fontweight="bold",fontsize=15); fig.text(.5,.015,"注：三架构提供相同有效服务；仅IF安装服务器组取整数，集团单节点和多节点采用连续等效容量。",ha="center",fontsize=8,color="#555")
    for o in outputs: o.parent.mkdir(parents=True,exist_ok=True); fig.savefig(o,bbox_inches="tight")
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--service-input",type=Path,required=True); p.add_argument("--national-input",type=Path,required=True); p.add_argument("--routing-config",type=Path,required=True); p.add_argument("--model-version",required=True); p.add_argument("--data-output",type=Path,required=True); p.add_argument("--png-output",type=Path,required=True); p.add_argument("--pdf-output",type=Path,required=True); p.add_argument("--svg-output",type=Path,required=True); p.add_argument("--validation-output",type=Path,required=True); a=p.parse_args()
    d=prepare(a.service_input,a.national_input,a.routing_config,a.model_version); a.data_output.parent.mkdir(parents=True,exist_ok=True); d.to_csv(a.data_output,index=False,encoding="utf-8-sig"); plot(d,[a.png_output,a.pdf_output,a.svg_output]); a.validation_output.write_text(json.dumps({"status":"validated","architectures":ARCHS,"II_1host_in_figure":False,"rows":len(d)},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
if __name__=="__main__": main()
