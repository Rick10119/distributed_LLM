#!/usr/bin/env python3
"""Write a human-readable interpretation from validated scenario outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    row = pd.read_csv(args.summary, encoding="utf-8-sig").iloc[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(
            [
                f"# {row['industry_code']} {row['scenario']} 核心场景结果",
                "",
                f"- 模型版本：{row['model_version']}。",
                f"- 行业等价年度增量总成本：{float(row['industry_equivalent_incremental_total_cost_rmb']) / 1e8:.3f} 亿元/年。",
                f"- 单个承载节点服务器组数：{float(row['per_host_installed_server_groups']):.0f}。",
                f"- 行业等价 AI 设施用电：{float(row['industry_equivalent_annual_ai_facility_energy_twh']):.4f} TWh/年。",
                f"- 行业等价新增电网容量：{float(row['industry_equivalent_incremental_grid_expansion_mw']):.3f} MW。",
                "",
                "该结果已通过当前版本的物理与核算一致性校验，只代表当前配置下的物理—成本原型，不应解释为行业预测。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
