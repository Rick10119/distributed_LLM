#!/usr/bin/env python3
"""Plot which manufacturing industries drive national grid expansion."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

cache = Path(tempfile.gettempdir()) / "distributed_llm_matplotlib"
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scenario", default="IG", choices=["IF", "IG", "II_1host"])
    parser.add_argument("--data-output", type=Path, required=True)
    parser.add_argument("--svg-output", type=Path, required=True)
    parser.add_argument("--png-output", type=Path, required=True)
    args = parser.parse_args()

    raw = pd.read_csv(args.input, encoding="utf-8-sig")
    data = raw[raw.scenario.eq(args.scenario)].copy()
    if len(data) != 31:
        raise ValueError(f"Expected 31 industries, found {len(data)}")
    data = data.sort_values(
        ["industry_equivalent_incremental_grid_expansion_mw", "per_host_incremental_grid_expansion_mw"],
        ascending=[True, True],
    )
    data["requires_expansion"] = data.industry_equivalent_incremental_grid_expansion_mw > 1e-6
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.data_output, index=False, encoding="utf-8-sig")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    })
    y = np.arange(len(data))
    labels = [f"{r.industry_code}  {r.industry_name}" for r in data.itertuples()]
    colors = np.where(data.industry_code.eq("C36"), "#B2473E", np.where(data.requires_expansion, "#D28B35", "#B9C2C8"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 11.2), sharey=True, gridspec_kw={"width_ratios": [1.45, 1], "wspace": .07})

    ax1.barh(y, data.industry_equivalent_incremental_grid_expansion_mw, color=colors, alpha=.9)
    ax1.set_yticks(y, labels, fontsize=8.2)
    for tick, code in zip(ax1.get_yticklabels(), data.industry_code):
        if code == "C36":
            tick.set_color("#B2473E"); tick.set_fontweight("bold")
    ax1.set_xlabel("行业全国等效新增接入容量（MW）")
    ax1.set_title("全国等效总量：哪些行业贡献了新增容量", fontweight="bold")
    ax1.grid(axis="x", alpha=.24)
    for yy, value in zip(y, data.industry_equivalent_incremental_grid_expansion_mw):
        if value >= 0.5:
            ax1.text(value + 1.2, yy, f"{value:.1f}", va="center", fontsize=7.4)

    ax2.barh(y, data.per_host_incremental_grid_expansion_mw, color=colors, alpha=.9)
    ax2.set_xlabel("最大代表节点新增接入容量（MW/节点）")
    ax2.set_title("单节点强度", fontweight="bold")
    ax2.grid(axis="x", alpha=.24)
    for yy, value in zip(y, data.per_host_incremental_grid_expansion_mw):
        if value >= .05:
            ax2.text(value + .06, yy, f"{value:.2f}", va="center", fontsize=7.2)

    c36_y = int(np.flatnonzero(data.industry_code.eq("C36"))[0])
    ax1.scatter([0], [c36_y], marker="D", s=42, color="#B2473E", zorder=4, clip_on=False)
    ax2.scatter([0], [c36_y], marker="D", s=42, color="#B2473E", zorder=4, clip_on=False)

    total = float(data.industry_equivalent_incremental_grid_expansion_mw.sum())
    positive = int(data.requires_expansion.sum())
    top4 = float(data.tail(4).industry_equivalent_incremental_grid_expansion_mw.sum() / total)
    c36 = data[data.industry_code.eq("C36")].iloc[0]
    fig.suptitle("集团共享部署（IG）下31个制造业行业的新增电网接入容量", fontsize=15, fontweight="bold", y=.985)
    fig.text(.5, .025, f"全国等效合计 {total:.1f} MW；{positive}/31 个行业需要扩容；前4个行业贡献 {top4:.1%}。汽车制造业（红色）为 {c36.industry_equivalent_incremental_grid_expansion_mw:.1f} MW。", ha="center", fontsize=9.3)
    fig.text(.5, .010, "注：全国等效总量=代表节点新增容量×等效节点数；灰色行业在当前负荷、光伏和储能联合优化下无需扩容。结果仍为活动全GPU物理情景。", ha="center", fontsize=8.3, color="#555555")
    fig.subplots_adjust(left=.30, right=.96, top=.94, bottom=.07)
    fig.savefig(args.svg_output, bbox_inches="tight")
    fig.savefig(args.png_output, bbox_inches="tight", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
