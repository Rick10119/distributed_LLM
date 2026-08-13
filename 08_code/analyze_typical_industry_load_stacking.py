#!/usr/bin/env python3
"""Plot representative-host base and AI load stacking for five industries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INDUSTRIES = {
    "C14": "食品制造业",
    "C17": "纺织业",
    "C26": "化学原料和化学制品制造业",
    "C36": "汽车制造业",
    "C39": "计算机、通信和其他电子设备制造业",
}
SCENARIOS = ["IF", "IG", "II_1host"]
SCENARIO_NAMES = {
    "IF": "工厂分散部署",
    "IG": "集团集中部署",
    "II_1host": "行业集中节点",
}


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 180,
        }
    )


def read_profiles(paths: list[Path], model_version: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    seen: set[tuple[str, str]] = set()
    required = {
        "industry_code",
        "scenario",
        "hour",
        "base_load_mw",
        "ai_facility_power_mw",
    }
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        if len(frame) != 168 or frame["hour"].nunique() != 168:
            raise ValueError(f"Expected a continuous 168-hour profile: {path}")
        industry = str(frame["industry_code"].iloc[0])
        scenario = str(frame["scenario"].iloc[0])
        if industry not in INDUSTRIES or scenario not in SCENARIOS:
            raise ValueError(f"Unexpected profile: {industry} {scenario}")
        seen.add((industry, scenario))
        selected = frame[
            ["industry_code", "scenario", "hour", "day", "hour_of_day", "base_load_mw", "ai_facility_power_mw"]
        ].copy()
        selected["model_version"] = model_version
        selected["industry_name"] = INDUSTRIES[industry]
        selected["scenario_name"] = SCENARIO_NAMES[scenario]
        selected["combined_load_mw"] = (
            selected["base_load_mw"] + selected["ai_facility_power_mw"]
        )
        frames.append(selected)
    expected = {(industry, scenario) for industry in INDUSTRIES for scenario in SCENARIOS}
    if seen != expected:
        raise ValueError(f"Profile coverage mismatch; missing={sorted(expected-seen)}")
    return pd.concat(frames, ignore_index=True)


def summarize(profiles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (industry, scenario), frame in profiles.groupby(["industry_code", "scenario"]):
        frame = frame.sort_values("hour")
        base = frame["base_load_mw"].to_numpy(float)
        ai = frame["ai_facility_power_mw"].to_numpy(float)
        combined = base + ai
        base_peak = float(base.max())
        ai_peak = float(ai.max())
        combined_peak = float(combined.max())
        peak_credit = base_peak + ai_peak - combined_peak
        rows.append(
            {
                "model_version": frame["model_version"].iloc[0],
                "industry_code": industry,
                "industry_name": INDUSTRIES[industry],
                "scenario": scenario,
                "scenario_name": SCENARIO_NAMES[scenario],
                "base_peak_mw": base_peak,
                "ai_peak_mw": ai_peak,
                "combined_peak_mw": combined_peak,
                "combined_peak_increment_mw": combined_peak - base_peak,
                "ai_peak_as_fraction_of_base_peak": ai_peak / base_peak,
                "noncoincident_peak_credit_mw": peak_credit,
                "noncoincident_peak_credit_fraction_of_ai_peak": peak_credit / ai_peak,
                "ai_load_factor": float(ai.mean() / ai_peak),
            }
        )
    return pd.DataFrame(rows).sort_values(["industry_code", "scenario"])


def plot_stacked(profiles: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    configure_plotting()
    fig, axes = plt.subplots(5, 3, figsize=(15.5, 16.0), sharex=True)
    x = np.arange(168)
    for row, (industry, industry_name) in enumerate(INDUSTRIES.items()):
        row_profiles = profiles[profiles["industry_code"] == industry]
        row_max = float(row_profiles["combined_load_mw"].max()) * 1.09
        for col, scenario in enumerate(SCENARIOS):
            ax = axes[row, col]
            frame = row_profiles[row_profiles["scenario"] == scenario].sort_values("hour")
            metrics = summary[
                (summary["industry_code"] == industry) & (summary["scenario"] == scenario)
            ].iloc[0]
            base = frame["base_load_mw"].to_numpy(float)
            ai = frame["ai_facility_power_mw"].to_numpy(float)
            combined = base + ai
            ax.fill_between(x, 0, base, color="#aeb6bf", alpha=0.72, label="原有负荷")
            ax.fill_between(x, base, combined, color="#e67e22", alpha=0.88, label="AI负荷")
            ax.plot(x, combined, color="#273746", linewidth=1.05, label="叠加后负荷")
            ax.set_xlim(0, 167)
            ax.set_ylim(0, row_max)
            ax.grid(axis="y", color="#d5d8dc", linewidth=0.55, alpha=0.65)
            if row == 0:
                ax.set_title(SCENARIO_NAMES[scenario], fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{industry_name}\n节点功率（MW）", fontsize=10)
            ax.text(
                0.985,
                0.94,
                f"AI峰值 {metrics.ai_peak_mw:.2f} MW\n叠加峰值 {metrics.combined_peak_mw:.2f} MW",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color="#273746",
            )
            if row == len(INDUSTRIES) - 1:
                ax.set_xticks([0, 24, 48, 72, 96, 120, 144, 167])
                ax.set_xticklabels(["周一", "周二", "周三", "周四", "周五", "周六", "周日", "末"])
                ax.set_xlabel("代表周", fontsize=9)
            else:
                ax.set_xticks([0, 24, 48, 72, 96, 120, 144])
                ax.set_xticklabels([])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("典型制造业：原有负荷与AI设施负荷叠加（代表节点，168小时）", fontsize=16, y=0.999)
    fig.text(
        0.5,
        0.006,
        "注：每个行业三列采用相同纵轴。工厂、集团和行业集中场景的AI服务规模依次扩大；原有负荷均为承载节点的代表工厂负荷。",
        ha="center",
        fontsize=9,
        color="#566573",
    )
    fig.tight_layout(rect=(0.03, 0.025, 0.995, 0.965), h_pad=1.25, w_pad=0.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_ai_only(profiles: pd.DataFrame, output: Path) -> None:
    configure_plotting()
    fig, axes = plt.subplots(5, 3, figsize=(15.5, 14.5), sharex=True)
    x = np.arange(168)
    for row, (industry, industry_name) in enumerate(INDUSTRIES.items()):
        for col, scenario in enumerate(SCENARIOS):
            ax = axes[row, col]
            frame = profiles[
                (profiles["industry_code"] == industry) & (profiles["scenario"] == scenario)
            ].sort_values("hour")
            ai = frame["ai_facility_power_mw"].to_numpy(float)
            ax.fill_between(x, 0, ai, color="#e67e22", alpha=0.82)
            ax.plot(x, ai, color="#a04000", linewidth=0.9)
            ax.set_xlim(0, 167)
            ax.set_ylim(0, float(ai.max()) * 1.14)
            ax.grid(axis="y", color="#d5d8dc", linewidth=0.55, alpha=0.65)
            if row == 0:
                ax.set_title(SCENARIO_NAMES[scenario], fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{industry_name}\nAI负荷（MW）", fontsize=10)
            ax.text(
                0.985,
                0.92,
                f"峰值 {ai.max():.3g} MW\n负荷率 {ai.mean()/ai.max():.1%}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
            )
            ax.set_xticks([0, 24, 48, 72, 96, 120, 144, 167])
            if row == len(INDUSTRIES) - 1:
                ax.set_xticklabels(["周一", "周二", "周三", "周四", "周五", "周六", "周日", "末"])
                ax.set_xlabel("代表周", fontsize=9)
            else:
                ax.set_xticklabels([])
    fig.suptitle("典型制造业AI设施负荷（各面板采用独立纵轴）", fontsize=16, y=0.998)
    fig.text(
        0.5,
        0.007,
        "独立纵轴用于显示AI负荷的时序形状，不能据此直接比较三种场景的绝对规模；绝对规模请看叠加图。",
        ha="center",
        fontsize=9,
        color="#566573",
    )
    fig.tight_layout(rect=(0.03, 0.026, 0.995, 0.97), h_pad=1.15, w_pad=0.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def write_findings(summary: pd.DataFrame, output: Path) -> None:
    pivot = summary.pivot(index=["industry_code", "industry_name"], columns="scenario")
    lines = [
        "# 典型行业原有负荷与AI负荷叠加",
        "",
        "本结果使用五个典型行业的代表节点连续168小时结果。工厂、集团和行业集中三种场景分别表示一个代表工厂承载本厂AI、一个代表工厂承载集团AI池、一个代表工厂承载全行业集中AI池。原有负荷均为承载节点的代表工厂负荷。",
        "",
        "| 行业 | IF AI峰值/原峰值 | IG AI峰值/原峰值 | II AI峰值/原峰值 | IF叠加峰值增量 | IG叠加峰值增量 | II叠加峰值增量 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (code, name) in pivot.index:
        row = pivot.loc[(code, name)]
        lines.append(
            f"| {code} {name} | "
            f"{row[('ai_peak_as_fraction_of_base_peak','IF')]:.1%} | "
            f"{row[('ai_peak_as_fraction_of_base_peak','IG')]:.1%} | "
            f"{row[('ai_peak_as_fraction_of_base_peak','II_1host')]:.1%} | "
            f"{row[('combined_peak_increment_mw','IF')]:.3f} MW | "
            f"{row[('combined_peak_increment_mw','IG')]:.3f} MW | "
            f"{row[('combined_peak_increment_mw','II_1host')]:.3f} MW |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- IF中单厂AI负荷相对原有工厂负荷较小，橙色叠加带通常很薄；但这不表示全国AI负荷小，而是需求分散到大量工厂。",
            "- IG把集团AI需求集中到一个代表工厂，AI负荷明显扩大；II将全行业需求放到一个代表节点，AI负荷通常主导该节点的叠加峰值。",
            "- AI设施负荷在IG和II中接近稳定基荷，当前调度并未形成明显的生产低谷跟随。IF的错峰空间相对较大，但绝对功率较小。",
            "- 图中叠加值是代表承载节点的筛查结果。II尤其是集中上界，不代表现实中会把全行业AI需求放到同一个数据中心。",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hourly-inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--profiles-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--stacked-figure-output", type=Path, required=True)
    parser.add_argument("--ai-figure-output", type=Path, required=True)
    parser.add_argument("--findings-output", type=Path, required=True)
    parser.add_argument("--done-output", type=Path, required=True)
    args = parser.parse_args()

    profiles = read_profiles(args.hourly_inputs, args.model_version)
    summary = summarize(profiles)
    args.profiles_output.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(args.profiles_output, index=False, encoding="utf-8-sig")
    summary.to_csv(args.summary_output, index=False, encoding="utf-8-sig")
    plot_stacked(profiles, summary, args.stacked_figure_output)
    plot_ai_only(profiles, args.ai_figure_output)
    write_findings(summary, args.findings_output)
    args.done_output.write_text(
        json.dumps(
            {
                "status": "validated",
                "model_version": args.model_version,
                "industries": list(INDUSTRIES),
                "scenarios": SCENARIOS,
                "hours_per_profile": 168,
                "profiles": 15,
                "scope": "representative_host_load_stacking_screen",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
