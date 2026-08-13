#!/usr/bin/env python3
"""Build Figure 3: stacked original and AI load in the no-DER core baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

cache = Path(tempfile.gettempdir()) / "distributed_llm_matplotlib"
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INDUSTRIES = ("C38", "C30", "C39")
INDUSTRY_LABELS = {
    "C38": "电气机械和器材制造业",
    "C30": "非金属矿物制品业",
    "C39": "计算机、通信和电子设备制造业",
}
SERIES = ("IF", "IG")
SERIES_LABELS = {
    "IF": "工厂侧分布式（IF，核心）",
    "IG": "集团集中算力池（IG）",
}
COLORS = {
    "original": "#90A9BC",
    "grid": "#234F70",
    "ai": "#D27A2C",
}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial Unicode MS",
                "PingFang SC",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "figure.dpi": 130,
        }
    )


def read_week(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"hour", "day", "hour_of_day", "grid_import_mw"}
    if not required.issubset(frame.columns) or len(frame) != 168:
        raise ValueError(f"Expected one complete 168-hour model output: {path}")
    week = frame.sort_values("hour")
    if list(week["hour"]) != list(range(168)):
        raise ValueError(f"Representative week is incomplete: {path}")
    return week


def prepare(model_root: Path, model_version: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for industry in INDUSTRIES:
        selected = pd.concat(
            [
                pd.read_csv(model_root / industry / series / "summary.csv", encoding="utf-8-sig")
                for series in SERIES
            ],
            ignore_index=True,
        ).set_index("scenario")
        if set(selected.index) != {"IF", "IG"}:
            raise ValueError(f"Missing IF/IG summary for {industry}")
        baseline_peak = float(selected.loc["IF", "per_host_existing_grid_capacity_mw"])
        for series in SERIES:
            path = model_root / industry / series / "hourly.csv"
            week = read_week(path)
            increment = float(selected.loc[series, "per_host_incremental_grid_expansion_mw"])
            for row in week.itertuples(index=False):
                rows.append(
                    {
                        "model_version": model_version,
                        "industry_code": industry,
                        "industry_name": INDUSTRY_LABELS[industry],
                        "series": series,
                        "series_label": SERIES_LABELS[series],
                        "hour": int(row.hour),
                        "day": int(row.day),
                        "hour_of_day": int(row.hour_of_day),
                        "original_load_mw": float(row.base_load_mw),
                        "ai_server_load_mw": float(row.ai_facility_power_mw),
                        "rooftop_pv_output_mw": float(row.rooftop_pv_output_mw),
                        "battery_discharge_positive_mw": float(row.battery_mw_positive_discharge),
                        "grid_import_mw": float(row.grid_import_mw),
                        "no_ai_baseline_peak_mw": baseline_peak,
                        "incremental_grid_capacity_mw": increment,
                        "triggers_capacity_addition": bool(increment > 1e-6),
                        "source": path.as_posix(),
                    }
                )
    return pd.DataFrame(rows)


def has_repeated_typical_day(values: pd.Series) -> bool:
    profile = values.to_numpy(dtype=float)
    return bool(
        len(profile) == 168
        and np.allclose(profile, np.tile(profile[:24], 7), rtol=0.0, atol=1e-10)
    )


def plot(data: pd.DataFrame, svg: Path, png: Path | None) -> None:
    configure()
    fig, axes = plt.subplots(3, 2, figsize=(13.2, 10.0), sharex=True)
    x = np.arange(168)
    component_handles = None
    for row_index, industry in enumerate(INDUSTRIES):
        industry_data = data[data["industry_code"].eq(industry)]
        baseline_peak = float(industry_data["no_ai_baseline_peak_mw"].iloc[0])
        for column_index, series in enumerate(SERIES):
            ax = axes[row_index, column_index]
            curve = industry_data[industry_data["series"].eq(series)].sort_values("hour")
            original = curve["original_load_mw"].to_numpy(float)
            ai = curve["ai_server_load_mw"].to_numpy(float)
            total = original + ai
            original_fill = ax.fill_between(
                x, 0, original, label="企业原始负荷",
                color=COLORS["original"], alpha=0.72,
            )
            ai_fill = ax.fill_between(
                x, original, total, label="新增AI服务器负荷",
                color=COLORS["ai"], alpha=0.86,
            )
            total_line = ax.plot(
                x, total, label="企业总负荷",
                color=COLORS["grid"], linewidth=1.7,
            )[0]
            ax.axhline(
                baseline_peak, color="#9A9A9A", linewidth=0.9,
                linestyle=":", alpha=0.8,
            )
            for boundary in range(24, 168, 24):
                ax.axvline(boundary - 0.5, color="#FFFFFF", linewidth=0.8, alpha=0.85)
            increment = float(curve["incremental_grid_capacity_mw"].iloc[0])
            status = "无新增容量" if increment <= 1e-6 else f"新增容量 {increment:.2f} MW"
            ax.text(
                0.98, 0.90, status, transform=ax.transAxes, ha="right", va="top",
                fontsize=9.0, fontweight="bold",
                color="#2F6B45" if increment <= 1e-6 else "#A14C32",
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            )
            title = SERIES_LABELS[series]
            if column_index == 0:
                title = f"{industry} {INDUSTRY_LABELS[industry]}｜{title}"
            ax.set_title(title, loc="left", fontsize=10.8, fontweight="bold")
            ax.set_ylabel("负荷（MW）")
            ax.grid(axis="both", alpha=0.18)
            ax.set_xlim(0, 167)
            if component_handles is None:
                component_handles = [original_fill, ai_fill, total_line]
    for ax in axes[-1, :]:
        ax.set_xlabel("连续一周")
        ax.set_xticks([12, 36, 60, 84, 108, 132, 156])
        ax.set_xticklabels(["周一", "周二", "周三", "周四", "周五", "周六", "周日"])
    if component_handles is not None:
        fig.legend(
            component_handles,
            [handle.get_label() for handle in component_handles],
            frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.955), fontsize=9,
        )
    fig.suptitle("AI服务器负荷与制造业原始负荷的周内匹配", fontsize=15, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.015,
        "注：各行业采用EWELD华南企业实测数据中筛选的完整连续周，并缩放至当前行业负荷规模；不同产业的代表周不要求同一日历周。基准情景不配置光伏和储能。",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.075, right=0.925, top=0.90, bottom=0.08, hspace=0.36, wspace=0.30)
    svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg, bbox_inches="tight")
    if png is not None:
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, bbox_inches="tight", dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--data-output", type=Path, required=True)
    parser.add_argument("--svg-output", type=Path, required=True)
    parser.add_argument("--png-output", type=Path)
    parser.add_argument("--validation-output", type=Path, required=True)
    args = parser.parse_args()
    data = prepare(args.model_root, args.model_version)
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.data_output, index=False, encoding="utf-8-sig")
    plot(data, args.svg_output, args.png_output)
    additions = (
        data[data["series"].isin(["IF", "IG"])]
        .groupby(["industry_code", "series"])["triggers_capacity_addition"]
        .first()
    )
    checks = {
        "three_industries": data["industry_code"].nunique() == 3,
        "two_architectures_per_industry": len(data) == 3 * 2 * 168,
        "complete_168_hour_week": bool(
            data.groupby(["industry_code", "series"])["hour"].nunique().eq(168).all()
        ),
        "measured_week_not_repeated_typical_day": bool(
            data.groupby(["industry_code", "series"])["original_load_mw"]
            .apply(lambda values: not has_repeated_typical_day(values))
            .all()
        ),
        "contains_addition_case": bool(additions.any()),
        "contains_if_and_ig": set(data["series"]) == set(SERIES),
        "retains_der_audit_columns": {
            "original_load_mw",
            "ai_server_load_mw",
            "rooftop_pv_output_mw",
            "battery_discharge_positive_mw",
            "grid_import_mw",
        }.issubset(data.columns),
        "no_der_in_core_figure": bool(
            (data["rooftop_pv_output_mw"].abs() < 1e-8).all()
            and (data["battery_discharge_positive_mw"].abs() < 1e-8).all()
        ),
        "svg_exists": args.svg_output.is_file(),
        "png_exists": args.png_output is None or args.png_output.is_file(),
    }
    if not all(checks.values()):
        raise ValueError(checks)
    args.validation_output.write_text(
        json.dumps(
            {
                "status": "validated",
                "model_version": args.model_version,
                "figure_scope": "no_der_measured_continuous_168h_original_plus_ai_stacked_load_IF_vs_IG",
                "industries": list(INDUSTRIES),
                "checks": checks,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
