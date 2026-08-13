#!/usr/bin/env python3
"""Plot three load-shape-diverse IG candidates for Figure 3 selection."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

cache = Path(tempfile.gettempdir()) / "distributed_llm_matplotlib"
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache))

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("C43", "金属制品、机械和设备修理业", 0.13),
    ("C19", "皮革、毛皮、羽毛及制鞋业", 0.35),
    ("C38", "电气机械和器材制造业", 0.49),
)


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(3, 1, figsize=(13, 9.3), sharex=True)
    handles = None
    for ax, (code, name, correlation) in zip(axes, CASES):
        hourly = pd.read_csv(ROOT / f"05_results/v0.8.0/model/{code}/IG/hourly.csv", encoding="utf-8-sig").sort_values("hour")
        summary = pd.read_csv(ROOT / f"05_results/v0.8.0/model/{code}/IG/summary.csv", encoding="utf-8-sig").iloc[0]
        x = hourly["hour"].to_numpy(int)
        base = hourly["base_load_mw"].to_numpy(float)
        ai = hourly["ai_facility_power_mw"].to_numpy(float)
        total = base + ai
        base_fill = ax.fill_between(x, 0, base, color="#90A9BC", alpha=0.72, label="企业原始负荷")
        ai_fill = ax.fill_between(x, base, total, color="#D27A2C", alpha=0.86, label="新增AI服务器负荷")
        total_line = ax.plot(x, total, color="#234F70", linewidth=1.7, label="企业总负荷")[0]
        ax.axhline(float(summary["per_host_existing_grid_capacity_mw"]), color="#999", linestyle=":", linewidth=0.9)
        for boundary in range(24, 168, 24):
            ax.axvline(boundary - 0.5, color="white", linewidth=0.9, alpha=0.9)
        increment = float(summary["per_host_incremental_grid_expansion_mw"])
        ax.text(0.985, 0.86, f"与C39曲线相关系数 {correlation:.2f}\n集团IG新增容量 {increment:.2f} MW",
                transform=ax.transAxes, ha="right", va="top", color="#A14C32", fontsize=9.5,
                bbox={"facecolor":"white", "edgecolor":"none", "alpha":0.82, "pad":2.5})
        ax.set_title(f"{code} {name}｜集团集中算力池（IG）", loc="left", fontsize=11.5, fontweight="bold")
        ax.set_ylabel("负荷（MW)")
        ax.set_xlim(0, 167)
        ax.grid(alpha=0.16)
        handles = [base_fill, ai_fill, total_line]
    axes[-1].set_xticks([12,36,60,84,108,132,156])
    axes[-1].set_xticklabels(["周一","周二","周三","周四","周五","周六","周日"])
    axes[-1].set_xlabel("连续一周")
    fig.legend(handles, [h.get_label() for h in handles], ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5,0.965))
    fig.suptitle("与C39负荷形态差异较大的集团情景候选", fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5,0.012,"注：相关系数按24小时标准化原始负荷曲线计算；无光伏、无储能基准。",ha="center",fontsize=9,color="#555")
    fig.subplots_adjust(left=0.08,right=0.98,top=0.91,bottom=0.08,hspace=0.42)
    output=ROOT/"05_results/v0.8.0/result/manuscript_figures/figure3_group_shape_candidates.png"
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=190,bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
