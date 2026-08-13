#!/usr/bin/env python3
"""Draft Figure 6: cost-grid-resource trade-offs and switching thresholds."""

from __future__ import annotations
import argparse, os, tempfile
from pathlib import Path
cache=Path(tempfile.gettempdir())/"distributed_llm_matplotlib"; cache.mkdir(parents=True,exist_ok=True); os.environ.setdefault("MPLCONFIGDIR",str(cache))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL={"IF":"#2F6B9A","IG":"#D28B35","II_1host":"#8A5D9E","CLOUD":"#4E8B68"}
LAB={"IF":"工厂分散","IG":"集团共享","II_1host":"大型集中自建","CLOUD":"大型云采购"}

def configure():
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial Unicode MS","PingFang SC","Noto Sans CJK SC","DejaVu Sans"],"axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none","figure.dpi":130})
def panel(ax,l,t): ax.text(-.09,1.05,l,transform=ax.transAxes,fontweight="bold",fontsize=14); ax.text(0,1.05,t,transform=ax.transAxes,fontweight="bold",fontsize=11.5)

def prepare(f2p:Path,f3p:Path,f4p:Path)->pd.DataFrame:
    f2=pd.read_csv(f2p); f3=pd.read_csv(f3p); f4=pd.read_csv(f4p)
    owned=f2[(f2.panel.eq("b"))].groupby("option").value.sum()
    cost={"IF":float(owned["工厂分散"]),"IG":float(owned["集团共享"]),"II_1host":float(owned["大型集中节点"])}
    cloud=f2[(f2.panel.eq("c"))&(f2.country.eq("中国"))&f2.option.str.startswith("DeepSeek")].value.sum()*7.8
    cost["CLOUD"]=float(cloud)
    g=f3[f3.panel.eq("a_b")].pivot(index="architecture",columns="metric",values="value")
    water_local=float(f4[(f4.panel.eq("a"))&(f4.case.eq("cn_local"))&(f4.metric.eq("annual_water"))].value.iloc[0])/1e6
    water_cloud=float(f4[(f4.panel.eq("a"))&(f4.case.eq("cn_cloud"))&(f4.metric.eq("annual_water"))].value.iloc[0])/1e6
    land=float(f4[(f4.panel.eq("c"))&(f4.case.eq("china"))&(f4.label.eq("greenfield"))&(f4.metric.eq("new_land_conversion_m2"))].value.iloc[0])/1e4
    rows=[]
    for a in ["IF","IG","II_1host","CLOUD"]:
        grid_arch="CLOUD" if a=="CLOUD" else a
        if grid_arch not in g.index: grid_arch="CLOUD"
        rows.append({"architecture":a,"cost_billion_rmb":cost[a],"total_grid_mw":float(g.loc[grid_arch,"total_incremental_grid_capacity_mw"]),"max_site_mw":float(g.loc[grid_arch,"maximum_single_site_incremental_capacity_mw"]),"water_million_m3":water_cloud if a in ["II_1host","CLOUD"] else water_local,"new_land_ha_greenfield":land if a in ["II_1host","CLOUD"] else 0.0})
    return pd.DataFrame(rows)

def plot(d,svg,png):
    configure(); fig,axes=plt.subplots(2,2,figsize=(13.0,8.6),gridspec_kw={"hspace":.38,"wspace":.32})
    ax=axes[0,0]; panel(ax,"a","企业成本—全国新增接入容量")
    for r in d.itertuples(): ax.scatter(r.total_grid_mw,r.cost_billion_rmb,s=95,color=COL[r.architecture],zorder=3); ax.annotate(LAB[r.architecture],(r.total_grid_mw,r.cost_billion_rmb),xytext=(5,5),textcoords="offset points",fontsize=8)
    ax.set_xlabel("全国新增接入容量（MW）"); ax.set_ylabel("企业年成本（十亿元/年）"); ax.grid(alpha=.25)
    ax.text(.03,.95,"左下方更优；云端为企业付款，\n不是云商社会资源成本",transform=ax.transAxes,va="top",fontsize=7.5,color="#555")

    ax=axes[0,1]; panel(ax,"b","成本—最大单点容量暴露")
    for r in d.itertuples(): ax.scatter(r.max_site_mw,r.cost_billion_rmb,s=95,color=COL[r.architecture]); ax.annotate(LAB[r.architecture],(r.max_site_mw,r.cost_billion_rmb),xytext=(5,5),textcoords="offset points",fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("最大单点新增容量（MW，对数）"); ax.set_ylabel("企业年成本（十亿元/年）"); ax.grid(alpha=.25,which="both")

    ax=axes[1,0]; panel(ax,"c","多资源权衡（各指标按列归一化）")
    metrics=["cost_billion_rmb","total_grid_mw","max_site_mw","water_million_m3","new_land_ha_greenfield"]
    names=["企业成本","总接入容量","最大单点","现场水","绿地新增土地"]
    vals=d.set_index("architecture")[metrics]; norm=(vals-vals.min())/(vals.max()-vals.min()).replace(0,1)
    xx=np.arange(len(metrics))
    for a in d.architecture: ax.plot(xx,norm.loc[a],marker="o",lw=1.8,color=COL[a],label=LAB[a])
    ax.set_xticks(xx,names); ax.set_ylim(-.05,1.08); ax.set_ylabel("相对影响（0低—1高）"); ax.grid(axis="y",alpha=.25); ax.legend(frameon=False,fontsize=7.5,ncol=2)
    ax.text(.02,.04,"土地采用绿地上界；本地复用为0新土地",transform=ax.transAxes,fontsize=7.2,color="#555")

    ax=axes[1,1]; panel(ax,"d","云端价格折扣决定成本切换")
    ig=float(d.set_index("architecture").loc["IG","cost_billion_rmb"]); cloud=float(d.set_index("architecture").loc["CLOUD","cost_billion_rmb"])
    frac=np.linspace(.30,1.0,120); ax.plot(frac*100,cloud*frac,color=COL["CLOUD"],lw=2.2,label="DeepSeek完整云化")
    ax.axhline(ig,color=COL["IG"],lw=2,label="集团自建成本"); be=ig/cloud*100; ax.axvline(be,color="#555",ls="--",lw=1); ax.scatter([be],[ig],color="#222",zorder=4)
    ax.text(be+1,ig+3,f"打平：保留当前付款的 {be:.1f}%",fontsize=8)
    ax.set_xlabel("云端付款保留比例（%）"); ax.set_ylabel("企业年成本（十亿元/年）"); ax.grid(alpha=.25); ax.legend(frameon=False,fontsize=8)

    fig.suptitle("Figure 6草图｜综合权衡、有效前沿与稳健性",fontsize=15,fontweight="bold",y=.985)
    fig.text(.5,.015,"注：基于v0.6.1已验证快照的探索性后处理。水和土地不货币化；云端空间与绿地建设为透明上界情景。正式相图仍需服务器利用率、PUE和云价联合敏感性。",ha="center",fontsize=8,color="#555")
    fig.subplots_adjust(left=.09,right=.96,top=.92,bottom=.10); svg.parent.mkdir(parents=True,exist_ok=True); fig.savefig(svg,bbox_inches="tight"); fig.savefig(png,bbox_inches="tight",dpi=170); plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument("--figure2",type=Path,required=True);p.add_argument("--figure3",type=Path,required=True);p.add_argument("--figure4",type=Path,required=True);p.add_argument("--data-output",type=Path,required=True);p.add_argument("--svg-output",type=Path,required=True);p.add_argument("--png-output",type=Path,required=True);a=p.parse_args();d=prepare(a.figure2,a.figure3,a.figure4);a.data_output.parent.mkdir(parents=True,exist_ok=True);d.to_csv(a.data_output,index=False,encoding="utf-8-sig");plot(d,a.svg_output,a.png_output)
if __name__=="__main__":main()
