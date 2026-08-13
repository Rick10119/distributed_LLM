#!/usr/bin/env python3
"""Draw the standalone demand-to-load accounting method diagram."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

_cache = Path(tempfile.gettempdir()) / "distributed_llm_matplotlib"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png-output", type=Path, required=True)
    parser.add_argument("--svg-output", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"], "svg.fonttype": "none"})
    steps = [
        ("制造业活动与采用率", "企业数 × 活动强度 × 采用率"),
        ("六类 AI 任务", "使用频率 × 单次任务量"),
        ("有效服务需求", "实时性、质量与并发约束"),
        ("CPU / GPU 路由", "按任务匹配计算硬件"),
        ("服务器装机", "10%裕量与 N+1 取较大者"),
        ("逐时设施负荷", "IT负荷 × 设施附加系数"),
    ]
    colors = ["#2F6B9A", "#2F6B9A", "#6B7F8C", "#65A58A", "#D28B35", "#8A5D9E"]
    fig, ax = plt.subplots(figsize=(14.0, 3.1))
    ax.set_axis_off(); xs = [0.08, .245, .41, .575, .74, .905]
    for i, ((title, note), x, color) in enumerate(zip(steps, xs, colors)):
        box = FancyBboxPatch((x-.068,.39),.136,.27,boxstyle="round,pad=.012,rounding_size=.014",facecolor="white",edgecolor=color,linewidth=1.5)
        ax.add_patch(box); ax.text(x,.555,title,ha="center",va="center",fontsize=10,fontweight="bold",color=color)
        ax.text(x,.305,note,ha="center",va="top",fontsize=8.2,color="#555555")
        if i < len(xs)-1:
            ax.add_patch(FancyArrowPatch((x+.074,.525),(xs[i+1]-.074,.525),arrowstyle="-|>",mutation_scale=11,color="#777777",linewidth=1))
    ax.text(.01,.92,"制造业 AI 需求核算与电力负荷转换方法",fontsize=15,fontweight="bold",va="top")
    ax.text(.01,.80,"活动数据与情景参数",fontsize=8.5,color="#2F6B9A")
    ax.text(.35,.80,"服务需求模型",fontsize=8.5,color="#6B7F8C")
    ax.text(.535,.80,"异构计算与容量配置",fontsize=8.5,color="#65A58A")
    ax.text(.845,.80,"电力系统接口",fontsize=8.5,color="#8A5D9E")
    ax.set_xlim(0,1); ax.set_ylim(.12,1)
    fig.subplots_adjust(left=.025,right=.99,top=.98,bottom=.05)
    for output in [args.png_output,args.svg_output]:
        output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(output,bbox_inches="tight",dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
