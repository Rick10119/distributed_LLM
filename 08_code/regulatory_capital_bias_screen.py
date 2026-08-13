"""Screen whether asymmetric cost recovery can favor a local AI asset.

This is a deterministic regulatory-incentive screen, not an estimate of an
observed utility's behavior.  It keeps market direct cost, customer revenue
requirement, shareholder accounting earnings, and shareholder economic value
as separate outputs.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CASE_INPUT = ROOT / "05_results" / "commercial_group_office_screen.csv"
PARAM_INPUT = ROOT / "04_cases" / "regulatory_capital_bias_parameters.csv"
SUMMARY_OUTPUT = ROOT / "05_results" / "regulatory_capital_bias_summary.csv"
GRID_OUTPUT = ROOT / "05_results" / "regulatory_capital_bias_grid.csv"
THRESHOLD_OUTPUT = ROOT / "05_results" / "regulatory_capital_bias_thresholds.csv"
FIGURE_DIR = ROOT / "05_results" / "figures"


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_vertical_text(image: Image.Image, text_value: str, x: int, center_y: int, text_font: ImageFont.ImageFont) -> None:
    box = text_font.getbbox(text_value)
    layer = Image.new("RGBA", (box[2] - box[0] + 24, box[3] - box[1] + 24), (255, 255, 255, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((12, 12), text_value, fill="black", font=text_font, anchor="la")
    rotated = layer.rotate(90, expand=True)
    image.paste(rotated, (x - rotated.width // 2, center_y - rotated.height // 2), rotated)


def save_grouped_bar(path: Path, local_values: list[float], cloud_values: list[float]) -> None:
    image = Image.new("RGB", (1500, 900), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, small_font = font(42), font(30), font(25)
    draw.text((750, 35), "Same AI service: two cost perspectives (cloud price = 75%)", fill="black", font=title_font, anchor="ma")
    left, top, right, bottom = 150, 130, 1420, 750
    draw.line((left, top, left, bottom), fill="black", width=3)
    draw.line((left, bottom, right, bottom), fill="black", width=3)
    maximum = max(local_values + cloud_values) * 1.12
    colors = ((54, 113, 183), (238, 113, 61))
    centers = (500, 1050)
    labels = ("Market direct cost", "Customer revenue requirement")
    for center, label, local_value, cloud_value in zip(centers, labels, local_values, cloud_values):
        for offset, value, color, name in ((-95, local_value, colors[0], "Local"), (95, cloud_value, colors[1], "Cloud")):
            height = int((value / maximum) * (bottom - top))
            x0, x1 = center + offset - 70, center + offset + 70
            draw.rectangle((x0, bottom - height, x1, bottom), fill=color)
            draw.text((center + offset, bottom - height - 12), f"{value / 1e6:.2f}", fill="black", font=small_font, anchor="ms")
            draw.text((center + offset, bottom + 12), name, fill="black", font=small_font, anchor="ma")
        draw.text((center, bottom + 72), label, fill="black", font=label_font, anchor="ma")
    for value in range(0, 15, 3):
        y = bottom - int((value * 1e6 / maximum) * (bottom - top))
        draw.line((left - 8, y, left, y), fill="black", width=2)
        draw.text((left - 18, y), str(value), fill="black", font=small_font, anchor="rm")
    draw_vertical_text(image, "Equivalent annual cost (million RMB/year)", 42, 440, label_font)
    image.save(path)


def save_heatmap(
    path: Path,
    values: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    title: str,
    x_label: str,
    y_label: str,
    binary: bool = False,
) -> None:
    width, height = 1500, 1000
    left, top, right, bottom = 190, 120, 1390, 840
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    rows, cols = values.shape
    scale = max(abs(float(values.min())), abs(float(values.max())), 1e-12)
    for row in range(rows):
        for col in range(cols):
            value = float(values[row, col])
            if binary:
                color = (73, 134, 184) if value >= 0.5 else (215, 80, 65)
            else:
                normalized = max(-1.0, min(1.0, value / scale))
                if normalized >= 0:
                    color = (int(245 - 175 * normalized), int(245 - 105 * normalized), int(245 - 55 * normalized))
                else:
                    strength = -normalized
                    color = (int(245 - 40 * strength), int(245 - 115 * strength), int(245 - 170 * strength))
            x0 = left + col * (right - left) / cols
            x1 = left + (col + 1) * (right - left) / cols
            y0 = bottom - (row + 1) * (bottom - top) / rows
            y1 = bottom - row * (bottom - top) / rows
            draw.rectangle((x0, y0, x1 + 1, y1 + 1), fill=color)
    draw.rectangle((left, top, right, bottom), outline="black", width=3)
    title_font, label_font, tick_font = font(42), font(30), font(24)
    draw.text((width / 2, 35), title, fill="black", font=title_font, anchor="ma")
    for fraction in np.linspace(0, 1, 6):
        x = left + fraction * (right - left)
        y = bottom - fraction * (bottom - top)
        draw.line((x, bottom, x, bottom + 8), fill="black", width=2)
        draw.text((x, bottom + 14), f"{x_min + fraction * (x_max - x_min):.2f}", fill="black", font=tick_font, anchor="ma")
        draw.line((left - 8, y, left, y), fill="black", width=2)
        draw.text((left - 16, y), f"{y_min + fraction * (y_max - y_min):.2f}", fill="black", font=tick_font, anchor="rm")
    draw.text(((left + right) / 2, 920), x_label, fill="black", font=label_font, anchor="ma")
    draw_vertical_text(image, y_label, 48, int((top + bottom) / 2), label_font)
    image.save(path)


def save_choice_map(path: Path, values: np.ndarray) -> None:
    width, height = 1500, 1080
    left, top, right, bottom = 190, 120, 1390, 800
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    colors = {
        0: (215, 80, 65),    # market cloud, shareholder cloud
        1: (244, 178, 76),   # market cloud, shareholder local: distortion
        2: (73, 134, 184),   # market local, shareholder local
        3: (145, 105, 170),  # market local, shareholder cloud
    }
    rows, cols = values.shape
    for row in range(rows):
        for col in range(cols):
            x0 = left + col * (right - left) / cols
            x1 = left + (col + 1) * (right - left) / cols
            y0 = bottom - (row + 1) * (bottom - top) / rows
            y1 = bottom - row * (bottom - top) / rows
            draw.rectangle((x0, y0, x1 + 1, y1 + 1), fill=colors[int(values[row, col])])
    draw.rectangle((left, top, right, bottom), outline="black", width=3)
    title_font, label_font, tick_font, legend_font = font(42), font(30), font(24), font(22)
    draw.text((750, 35), "When regulation and market cost favor different deployments", fill="black", font=title_font, anchor="ma")
    for fraction in np.linspace(0, 1, 6):
        x = left + fraction * (right - left)
        y = bottom - fraction * (bottom - top)
        draw.line((x, bottom, x, bottom + 8), fill="black", width=2)
        draw.text((x, bottom + 14), f"{0.60 + fraction * 0.50:.2f}", fill="black", font=tick_font, anchor="ma")
        draw.line((left - 8, y, left, y), fill="black", width=2)
        draw.text((left - 16, y), f"{-1.0 + fraction * 3.0:.2f}", fill="black", font=tick_font, anchor="rm")
    draw.text(((left + right) / 2, 875), "Cloud price multiplier", fill="black", font=label_font, anchor="ma")
    draw_vertical_text(image, "ROE minus equity cost (percentage points)", 48, int((top + bottom) / 2), label_font)
    legend = (
        (0, "Both favor cloud"),
        (1, "Market cloud / shareholder local"),
        (2, "Both favor local"),
        (3, "Market local / shareholder cloud"),
    )
    for index, (code, label) in enumerate(legend):
        x = 220 + (index % 2) * 610
        y = 948 + (index // 2) * 48
        draw.rectangle((x, y - 12, x + 24, y + 12), fill=colors[code])
        draw.text((x + 34, y), label, fill="black", font=legend_font, anchor="lm")
    image.save(path)


def read_parameters() -> dict[str, float]:
    with PARAM_INPUT.open(encoding="utf-8-sig", newline="") as handle:
        return {row["parameter_id"]: float(row["base_value"]) for row in csv.DictReader(handle)}


def read_case() -> tuple[dict[str, float], dict[str, float]]:
    with CASE_INPUT.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    local = next(row for row in rows if row["scenario"] == "group_center_at_office_nplus1")
    cloud = next(row for row in rows if row["scenario"] == "group_public_cloud")
    return (
        {key: float(value) for key, value in local.items() if value not in ("", None) and key != "scenario"},
        {key: float(value) for key, value in cloud.items() if value not in ("", None) and key != "scenario"},
    )


def annuity_factor(rate: float, years: int) -> float:
    return sum(1.0 / (1.0 + rate) ** year for year in range(1, years + 1))


def capital_schedule(
    purchases: list[tuple[int, float]],
    horizon: int,
    regulatory_life: int,
    technical_life: int | None = None,
    continuation_fraction: float = 1.0,
) -> tuple[list[float], list[float], list[float], float]:
    """Return depreciation, average balance, stranded loss, and terminal balance."""
    depreciation = [0.0] * horizon
    average_balance = [0.0] * horizon
    stranded = [0.0] * horizon
    terminal_balance = 0.0
    for purchase_year, amount in purchases:
        for year in range(purchase_year, horizon):
            age = year - purchase_year
            if age >= regulatory_life:
                continue
            post_retirement = technical_life is not None and age >= technical_life
            recovery_share = continuation_fraction if post_retirement else 1.0
            opening_raw = amount * max(0.0, 1.0 - age / regulatory_life)
            closing_raw = amount * max(0.0, 1.0 - (age + 1) / regulatory_life)
            opening = opening_raw * recovery_share
            closing = closing_raw * recovery_share
            depreciation[year] += opening - closing
            average_balance[year] += (opening + closing) / 2.0
            if technical_life is not None and age == technical_life:
                stranded[year] += opening_raw * (1.0 - continuation_fraction)
        terminal_age = horizon - purchase_year
        if 0 <= terminal_age < regulatory_life:
            post_retirement = technical_life is not None and terminal_age >= technical_life
            recovery_share = continuation_fraction if post_retirement else 1.0
            terminal_balance += amount * (1.0 - terminal_age / regulatory_life) * recovery_share
    return depreciation, average_balance, stranded, terminal_balance


def evaluate(
    local: dict[str, float],
    cloud: dict[str, float],
    p: dict[str, float],
    *,
    cloud_multiplier: float,
    local_investment_multiplier: float,
    local_eligible_share: float,
    cloud_capitalized_share: float,
    allowed_equity_return: float,
    equity_cost: float,
    disallowance_share: float,
    lag_years: int,
    continuation_fraction: float,
) -> dict[str, float | str]:
    horizon = int(p["R20"])
    discount = p["R19"]
    equity_share = p["R10"]
    debt_share = 1.0 - equity_share
    allowed_wacc = equity_share * allowed_equity_return + debt_share * p["R17"]

    local_capex = local["upfront_local_investment_rmb"] * local_investment_multiplier
    local_annualized_capex = local["annualized_local_capex_rmb"] * local_investment_multiplier
    local_maintenance = local["annual_local_maintenance_rmb"] * local_investment_multiplier
    local_noncapital = (
        local_maintenance
        + local["annual_local_energy_rmb"]
        + local["annual_local_max_demand_rmb"]
    )
    local_market_cost = local_annualized_capex + local_noncapital
    cloud_market_cost = cloud["annual_cloud_bill_rmb"] * cloud_multiplier

    recognized_fraction = local_eligible_share * (1.0 - disallowance_share)
    local_purchases = [
        (year, local_capex * recognized_fraction)
        for year in range(0, horizon, int(p["R12"]))
    ]
    local_dep, local_rb, local_stranded, local_terminal = capital_schedule(
        local_purchases,
        horizon,
        int(p["R11"]),
        int(p["R12"]),
        continuation_fraction,
    )
    expense_recovery_fraction = (1.0 - local_eligible_share) * (1.0 - disallowance_share)
    expense_purchases = [
        (year, local_capex * expense_recovery_fraction)
        for year in range(0, horizon, int(p["R12"]))
    ]
    expense_dep, _, _, expense_terminal = capital_schedule(
        expense_purchases, horizon, int(p["R12"])
    )

    cloud_capital_amount = cloud_market_cost * cloud_capitalized_share
    cloud_purchases = [(year, cloud_capital_amount) for year in range(horizon)]
    cloud_dep, cloud_rb, _, cloud_terminal = capital_schedule(
        cloud_purchases, horizon, int(p["R21"])
    )

    local_rr = [
        local_noncapital + local_dep[y] + expense_dep[y] + local_rb[y] * allowed_wacc
        for y in range(horizon)
    ]
    cloud_rr = [
        cloud_market_cost * (1.0 - cloud_capitalized_share) * p["R14"]
        + cloud_dep[y]
        + cloud_rb[y] * allowed_wacc
        for y in range(horizon)
    ]
    pv_factor = annuity_factor(discount, horizon)
    local_rr_pv = sum(local_rr[y] / (1.0 + discount) ** (y + 1) for y in range(horizon))
    cloud_rr_pv = sum(cloud_rr[y] / (1.0 + discount) ** (y + 1) for y in range(horizon))
    local_rr_pv += (local_terminal + expense_terminal) / (1.0 + discount) ** horizon
    cloud_rr_pv += cloud_terminal / (1.0 + discount) ** horizon

    local_accounting_pv = sum(
        local_rb[y] * equity_share * allowed_equity_return / (1.0 + discount) ** (y + 1)
        for y in range(horizon)
    )
    cloud_accounting_pv = sum(
        cloud_rb[y] * equity_share * allowed_equity_return / (1.0 + discount) ** (y + 1)
        for y in range(horizon)
    )

    local_spread_pv = sum(
        local_rb[y] * equity_share * (allowed_equity_return - equity_cost)
        / (1.0 + discount) ** (y + 1)
        for y in range(horizon)
    )
    cloud_spread_pv = sum(
        cloud_rb[y] * equity_share * (allowed_equity_return - equity_cost)
        / (1.0 + discount) ** (y + 1)
        for y in range(horizon)
    )
    # R06 changes the accounting treatment; R13 alone represents a true
    # prudence disallowance that shareholders cannot recover.
    unrecognized_each_purchase = local_capex * disallowance_share * equity_share
    unrecognized_pv = sum(
        unrecognized_each_purchase / (1.0 + discount) ** year
        for year in range(0, horizon, int(p["R12"]))
    )
    stranded_pv = sum(
        local_stranded[y] * equity_share / (1.0 + discount) ** (y + 1)
        for y in range(horizon)
    )
    local_lag_pv = sum(
        amount * equity_share * ((1.0 + equity_cost) ** lag_years - 1.0)
        / (1.0 + discount) ** year
        for year, amount in local_purchases
    )
    cloud_lag_pv = sum(
        amount * equity_share * ((1.0 + equity_cost) ** lag_years - 1.0)
        / (1.0 + discount) ** year
        for year, amount in cloud_purchases
    )
    local_economic_pv = local_spread_pv - unrecognized_pv - stranded_pv - local_lag_pv
    cloud_economic_pv = cloud_spread_pv - cloud_lag_pv

    market_choice = "local" if local_market_cost <= cloud_market_cost else "cloud"
    customer_choice = "local" if local_rr_pv <= cloud_rr_pv else "cloud"
    accounting_choice = "local" if local_accounting_pv > cloud_accounting_pv else "cloud"
    economic_choice = "local" if local_economic_pv > cloud_economic_pv else "cloud"
    return {
        "cloud_price_multiplier": cloud_multiplier,
        "local_investment_multiplier": local_investment_multiplier,
        "local_eligible_share": local_eligible_share,
        "cloud_capitalized_share": cloud_capitalized_share,
        "allowed_equity_return": allowed_equity_return,
        "equity_cost": equity_cost,
        "return_spread": allowed_equity_return - equity_cost,
        "disallowance_share": disallowance_share,
        "lag_years": lag_years,
        "continuation_fraction": continuation_fraction,
        "local_market_eac_rmb": local_market_cost,
        "cloud_market_eac_rmb": cloud_market_cost,
        "local_customer_rr_eac_rmb": local_rr_pv / pv_factor,
        "cloud_customer_rr_eac_rmb": cloud_rr_pv / pv_factor,
        "local_accounting_equity_earnings_pv_rmb": local_accounting_pv,
        "cloud_accounting_equity_earnings_pv_rmb": cloud_accounting_pv,
        "local_shareholder_economic_value_pv_rmb": local_economic_pv,
        "cloud_shareholder_economic_value_pv_rmb": cloud_economic_pv,
        "local_unrecognized_investment_pv_rmb": unrecognized_pv,
        "local_stranded_loss_pv_rmb": stranded_pv,
        "market_cost_choice": market_choice,
        "customer_rr_choice": customer_choice,
        "accounting_earnings_choice": accounting_choice,
        "shareholder_economic_choice": economic_choice,
        "accounting_distortion": int(market_choice == "cloud" and accounting_choice == "local"),
        "economic_distortion": int(market_choice == "cloud" and economic_choice == "local"),
    }


def save_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = read_parameters()
    local, cloud = read_case()
    common = dict(
        local=local,
        cloud=cloud,
        p=p,
        local_investment_multiplier=1.0,
        equity_cost=p["R09"],
    )
    scenarios = [
        ("R0_current_market", dict(cloud_multiplier=1.0, local_eligible_share=0.0, cloud_capitalized_share=0.0, allowed_equity_return=p["R08"], disallowance_share=0.0, lag_years=0, continuation_fraction=1.0)),
        ("R0_cloud_75pct", dict(cloud_multiplier=0.75, local_eligible_share=0.0, cloud_capitalized_share=0.0, allowed_equity_return=p["R08"], disallowance_share=0.0, lag_years=0, continuation_fraction=1.0)),
        ("R2_accounting_neutral_spread", dict(cloud_multiplier=0.75, local_eligible_share=1.0, cloud_capitalized_share=0.0, allowed_equity_return=p["R09"], disallowance_share=0.0, lag_years=0, continuation_fraction=1.0)),
        ("R2_positive_1ppt_clean", dict(cloud_multiplier=0.75, local_eligible_share=1.0, cloud_capitalized_share=0.0, allowed_equity_return=p["R09"] + 0.01, disallowance_share=0.0, lag_years=0, continuation_fraction=1.0)),
        ("R2_positive_1ppt_lag1", dict(cloud_multiplier=0.75, local_eligible_share=1.0, cloud_capitalized_share=0.0, allowed_equity_return=p["R09"] + 0.01, disallowance_share=0.0, lag_years=1, continuation_fraction=1.0)),
        ("R3_cloud_symmetric", dict(cloud_multiplier=0.75, local_eligible_share=1.0, cloud_capitalized_share=1.0, allowed_equity_return=p["R09"] + 0.01, disallowance_share=0.0, lag_years=0, continuation_fraction=1.0)),
        ("R4_prudence_50pct", dict(cloud_multiplier=0.75, local_eligible_share=0.5, cloud_capitalized_share=0.0, allowed_equity_return=p["R09"] + 0.01, disallowance_share=0.1, lag_years=1, continuation_fraction=1.0)),
        ("R5_obsolescence_no_lag", dict(cloud_multiplier=0.75, local_eligible_share=1.0, cloud_capitalized_share=0.0, allowed_equity_return=p["R09"] + 0.01, disallowance_share=0.0, lag_years=0, continuation_fraction=0.5)),
        ("R5_obsolescence_and_lag", dict(cloud_multiplier=0.75, local_eligible_share=1.0, cloud_capitalized_share=0.0, allowed_equity_return=p["R09"] + 0.01, disallowance_share=0.0, lag_years=1, continuation_fraction=0.5)),
    ]
    summary: list[dict[str, object]] = []
    for name, settings in scenarios:
        row = evaluate(**common, **settings)
        summary.append({"scenario": name, **row})
    save_rows(SUMMARY_OUTPUT, summary)
    by_name = {str(row["scenario"]): row for row in summary}
    clean = by_name["R2_positive_1ppt_clean"]
    value_per_unit_spread = float(clean["local_shareholder_economic_value_pv_rmb"]) / 0.01
    threshold_rows: list[dict[str, object]] = [
        {
            "threshold": "market_cloud_price_multiplier",
            "value": local["annual_enterprise_direct_rmb"] / cloud["annual_cloud_bill_rmb"],
            "unit": "multiplier",
            "interpretation": "Below this current-price fraction, cloud has lower market direct cost at base local investment.",
        }
    ]
    for scenario_name, threshold_name in (
        ("R2_positive_1ppt_lag1", "break_even_return_spread_lag1"),
        ("R5_obsolescence_no_lag", "break_even_return_spread_obsolescence"),
        ("R5_obsolescence_and_lag", "break_even_return_spread_obsolescence_and_lag"),
    ):
        friction = float(clean["local_shareholder_economic_value_pv_rmb"]) - float(by_name[scenario_name]["local_shareholder_economic_value_pv_rmb"])
        threshold_rows.append(
            {
                "threshold": threshold_name,
                "value": friction / value_per_unit_spread,
                "unit": "fraction/year",
                "interpretation": "Allowed ROE minus equity cost required for zero local shareholder economic value under the stated friction.",
            }
        )
    symmetric = by_name["R3_cloud_symmetric"]
    threshold_rows.append(
        {
            "threshold": "cloud_capitalized_share_for_equal_shareholder_value",
            "value": float(clean["local_shareholder_economic_value_pv_rmb"]) / float(symmetric["cloud_shareholder_economic_value_pv_rmb"]),
            "unit": "fraction_of_annual_cloud_bill",
            "interpretation": "Stylized capitalization fraction needed to match local shareholder value; values above one are infeasible under this rule.",
        }
    )
    save_rows(THRESHOLD_OUTPUT, threshold_rows)

    cloud_grid = np.linspace(0.60, 1.10, 51)
    capex_grid = np.linspace(0.80, 1.30, 51)
    eligibility_grid = np.linspace(0.0, 1.0, 51)
    spread_grid = np.linspace(-0.01, 0.02, 61)
    grid_rows: list[dict[str, object]] = []
    market_map = np.zeros((len(capex_grid), len(cloud_grid)))
    economic_map = np.zeros((len(spread_grid), len(eligibility_grid)))
    distortion_map = np.zeros((len(spread_grid), len(cloud_grid)))
    for i, capex_mult in enumerate(capex_grid):
        for j, cloud_mult in enumerate(cloud_grid):
            row = evaluate(
                local, cloud, p,
                cloud_multiplier=float(cloud_mult),
                local_investment_multiplier=float(capex_mult),
                local_eligible_share=0.0,
                cloud_capitalized_share=0.0,
                allowed_equity_return=p["R09"],
                equity_cost=p["R09"],
                disallowance_share=0.0,
                lag_years=0,
                continuation_fraction=1.0,
            )
            market_map[i, j] = 1 if row["market_cost_choice"] == "local" else 0
    for i, spread in enumerate(spread_grid):
        for j, eligible in enumerate(eligibility_grid):
            row = evaluate(
                local, cloud, p,
                cloud_multiplier=0.75,
                local_investment_multiplier=1.0,
                local_eligible_share=float(eligible),
                cloud_capitalized_share=0.0,
                allowed_equity_return=p["R09"] + float(spread),
                equity_cost=p["R09"],
                disallowance_share=0.0,
                lag_years=0,
                continuation_fraction=1.0,
            )
            economic_map[i, j] = (float(row["local_shareholder_economic_value_pv_rmb"]) / 1e6)
            grid_rows.append({
                "grid": "eligible_share_return_spread",
                "local_eligible_share": eligible,
                "return_spread": spread,
                "local_shareholder_economic_value_pv_rmb": row["local_shareholder_economic_value_pv_rmb"],
                "cloud_shareholder_economic_value_pv_rmb": row["cloud_shareholder_economic_value_pv_rmb"],
                "market_cost_choice": row["market_cost_choice"],
                "shareholder_economic_choice": row["shareholder_economic_choice"],
                "economic_distortion": row["economic_distortion"],
            })
    for i, spread in enumerate(spread_grid):
        for j, cloud_mult in enumerate(cloud_grid):
            row = evaluate(
                local, cloud, p,
                cloud_multiplier=float(cloud_mult),
                local_investment_multiplier=1.0,
                local_eligible_share=1.0,
                cloud_capitalized_share=0.0,
                allowed_equity_return=p["R09"] + float(spread),
                equity_cost=p["R09"],
                disallowance_share=0.0,
                lag_years=0,
                continuation_fraction=1.0,
            )
            market_local = row["market_cost_choice"] == "local"
            shareholder_local = row["shareholder_economic_choice"] == "local"
            distortion_map[i, j] = 2 if market_local and shareholder_local else 3 if market_local else 1 if shareholder_local else 0
            grid_rows.append({
                "grid": "cloud_price_return_spread",
                "local_eligible_share": 1.0,
                "return_spread": spread,
                "local_shareholder_economic_value_pv_rmb": row["local_shareholder_economic_value_pv_rmb"],
                "cloud_shareholder_economic_value_pv_rmb": row["cloud_shareholder_economic_value_pv_rmb"],
                "market_cost_choice": row["market_cost_choice"],
                "shareholder_economic_choice": row["shareholder_economic_choice"],
                "economic_distortion": row["economic_distortion"],
            })
    save_rows(GRID_OUTPUT, grid_rows)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    selected = summary[3]
    local_values = [selected["local_market_eac_rmb"], selected["local_customer_rr_eac_rmb"]]
    cloud_values = [selected["cloud_market_eac_rmb"], selected["cloud_customer_rr_eac_rmb"]]
    save_grouped_bar(FIGURE_DIR / "regulatory_cost_perspectives.png", local_values, cloud_values)
    save_heatmap(
        FIGURE_DIR / "regulatory_market_choice_map.png", market_map,
        0.60, 1.10, 0.80, 1.30,
        "Market-cost deployment choice (blue: local; red: cloud)",
        "Cloud price multiplier", "Local investment multiplier", binary=True,
    )
    save_heatmap(
        FIGURE_DIR / "regulatory_shareholder_value_map.png", economic_map,
        0.0, 1.0, -1.0, 2.0,
        "Local shareholder economic value relative to cloud",
        "Share admitted to rate base", "ROE minus equity cost (percentage points)",
    )
    save_choice_map(FIGURE_DIR / "regulatory_distortion_region.png", distortion_map)

    print(f"market_switch_cloud_multiplier={local['annual_enterprise_direct_rmb'] / cloud['annual_cloud_bill_rmb']:.6f}")
    for row in summary:
        print(
            row["scenario"],
            row["market_cost_choice"],
            row["customer_rr_choice"],
            row["accounting_earnings_choice"],
            row["shareholder_economic_choice"],
            round(float(row["local_shareholder_economic_value_pv_rmb"]) / 1e6, 3),
        )


if __name__ == "__main__":
    main()
