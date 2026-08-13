"""Industry input assembly for equal-service core scenarios."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from core.capacity import local_installed_capacity_floor
from core.config import rooted


@dataclass(frozen=True)
class FlexibleJob:
    release_hour: int
    deadline_hours: int
    amount_service_units: float
    task_id: str
    flexibility_class: str


@dataclass(frozen=True)
class IndustryInputs:
    industry_code: str
    industry_name: str
    base_load_mw: np.ndarray
    rigid_service_units: np.ndarray
    rigid_service_units_by_task: dict[str, np.ndarray]
    flexible_jobs: tuple[FlexibleJob, ...]
    daily_effective_service_units: float
    accelerator_h_per_service_unit: float
    reference_daily_accelerator_h: float
    external_energy_low_twh: float
    external_energy_central_twh: float
    external_energy_high_twh: float
    derived_reference_energy_twh: float
    external_energy_alignment_ratio: float
    reference_energy_server_groups: int
    pv_capacity_factor: np.ndarray
    roof_area_proxy_m2: float
    roof_area_case: str
    roof_source_naics: str
    roof_mapping_type: str
    roof_evidence_grade: str


@dataclass(frozen=True)
class BatteryCostParameters:
    energy_capex_rmb_per_kwh: float
    power_capex_rmb_per_kw: float
    energy_lifetime_years: float
    power_lifetime_years: float
    power_fom_fraction_per_year: float
    roundtrip_efficiency: float

    def annualized_cost_rmb_per_mw_year(
        self, duration_h: float, discount_rate: float
    ) -> float:
        def crf(years: float) -> float:
            return discount_rate * (1.0 + discount_rate) ** years / (
                (1.0 + discount_rate) ** years - 1.0
            )

        return (
            self.energy_capex_rmb_per_kwh
            * 1000.0
            * duration_h
            * crf(self.energy_lifetime_years)
            + self.power_capex_rmb_per_kw
            * 1000.0
            * crf(self.power_lifetime_years)
            + self.power_capex_rmb_per_kw
            * 1000.0
            * self.power_fom_fraction_per_year
        )


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    maximum = float(np.max(values))
    if values.shape != (24,) or maximum <= 0:
        raise ValueError("A workload shape must contain 24 positive-capacity hours")
    return values / maximum


def parse_office_agent_shapes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = frame[
        (frame["record_type"] == "hourly_curve")
        & (frame["scenario"] == "2030_integrated")
    ].sort_values("hour")
    if len(frame) != 24:
        raise ValueError("Workload-shape source must provide 24 hourly rows")

    def extract(name: str) -> np.ndarray:
        values: list[float] = []
        pattern = re.compile(rf"(?:^|; )?{re.escape(name)}=([0-9.]+)")
        for text in frame["derivation_or_definition"].astype(str):
            match = pattern.search(text)
            values.append(float(match.group(1)) if match else 0.0)
        return np.asarray(values)

    office = normalize(extract("human_tasks") + 0.05)
    agent = normalize(extract("transaction_tasks") + extract("batch_tasks") + 0.10)
    return office, agent


def task_shapes(
    base_normalized: np.ndarray,
    office: np.ndarray,
    agent: np.ndarray,
) -> dict[str, np.ndarray]:
    scheduling = np.full(24, 0.20)
    for hour, value in {
        5: 1.2,
        6: 1.6,
        7: 1.0,
        13: 1.0,
        14: 1.5,
        15: 1.0,
        21: 1.0,
        22: 1.5,
        23: 1.2,
    }.items():
        scheduling[hour] = value
    simulation = np.full(24, 0.10)
    simulation[[0, 1, 2, 3, 4, 5, 22, 23]] = 1.0
    return {
        "office": normalize(office),
        "agent": normalize(agent),
        "vision": normalize(base_normalized),
        "maintenance": normalize(0.75 * normalize(base_normalized) + 0.25),
        "scheduling": normalize(scheduling),
        "simulation": normalize(simulation),
    }


def reference_energy_mwh(
    execution: np.ndarray,
    calibration: dict[str, float],
    installed_reserve_fraction: float,
    n_plus_spare_server_groups: int,
) -> tuple[float, int]:
    accelerators = float(calibration["accelerators_per_server"])
    utilization = float(calibration["installed_utilization"])
    reserve = float(installed_reserve_fraction)
    maximum = float(calibration["maximum_power_kw"])
    idle = float(calibration["online_idle_power_kw"])
    required_groups = float(np.max(execution)) / (accelerators * utilization)
    groups = math.ceil(
        local_installed_capacity_floor(
            required_groups,
            reserve,
            n_plus_spare_server_groups,
        )
    )
    dynamic_kw_per_accelerator = (maximum - idle) / accelerators
    facility_mw = float(calibration["pue"]) * (
        groups * idle + execution * dynamic_kw_per_accelerator
    ) / 1000.0
    return float(np.sum(facility_mw)), groups


def read_pv_capacity_factor(config: dict) -> np.ndarray:
    frame = pd.read_csv(rooted(config, config["paths"]["pv_profile_source"]), encoding="utf-8-sig")
    case = config["factory"]["pv_profile_case"]
    values = frame[frame["case"] == case].sort_values("hour")["pv_kw"].to_numpy(dtype=float)
    return normalize(values)


def read_industry_rooftop_parameters(config: dict, industry_code: str) -> dict[str, object]:
    frame = pd.read_csv(
        rooted(config, config["paths"]["model_ready_industry_rooftop"]),
        encoding="utf-8-sig",
    )
    selected = frame[frame["industry_code"] == industry_code]
    if len(selected) != 1:
        raise ValueError(f"Expected one rooftop-parameter row for {industry_code}")
    row = selected.iloc[0]
    area = float(row["roof_area_proxy_m2"])
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError(f"Invalid rooftop-area proxy for {industry_code}")
    return {
        "roof_area_proxy_m2": area,
        "roof_area_case": str(row["roof_area_case"]),
        "roof_source_naics": str(row["us_naics_3"]),
        "roof_mapping_type": str(row["mapping_type"]),
        "roof_evidence_grade": str(row["evidence_grade"]),
    }


def spot_retail_adder_rmb_per_kwh(config: dict) -> float:
    """Return non-energy volumetric charges added to wholesale spot prices."""
    components = config["energy"]["spot_retail_volumetric_adders_rmb_per_kwh"]
    required = {
        "line_loss",
        "transmission_and_distribution",
        "system_operation",
        "government_funds_and_surcharges",
    }
    if set(components) != required:
        raise ValueError(f"Spot retail adder components must be exactly {sorted(required)}")
    values = np.asarray([float(components[key]) for key in sorted(required)], dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("Spot retail adder components must be finite and non-negative")
    return float(values.sum())


def read_core_grid_energy_prices(config: dict) -> np.ndarray:
    """Read the configured hourly grid-energy price boundary for the core model."""
    energy = config["energy"]
    mode = str(energy["grid_energy_price_mode"])
    horizon_hours = int(config["model"]["horizon_hours"])
    if mode == "flat_tariff":
        return np.full(
            horizon_hours,
            float(energy["flat_grid_energy_rmb_per_kwh"]) * 1000.0,
            dtype=float,
        )
    if mode != "guangdong_spot_retail_representative_week":
        raise ValueError(f"Unsupported grid-energy price mode: {mode}")

    settings = energy.get("spot_period", energy["spot_representative_week"])
    frame = pd.read_csv(
        rooted(config, config["paths"]["spot_price_source"]),
        encoding="utf-8-sig",
    )
    frame["date"] = pd.to_datetime(frame["business_date"])
    start = pd.Timestamp(str(settings["start_date"]))
    end = pd.Timestamp(str(settings["end_date"]))
    inclusive_days = (end - start).days + 1
    if inclusive_days * 24 != horizon_hours:
        raise ValueError(
            "The selected inclusive spot-price date range must match model.horizon_hours"
        )
    selected = frame[
        (frame["province_code"] == str(settings["province_code"]))
        & (frame["settlement_type"] == str(settings["settlement_type"]))
        & frame["date"].between(start, end)
    ].copy()
    counts = selected.groupby(["date", "business_hour"]).size()
    if len(counts) != horizon_hours or not (counts == 4).all():
        raise ValueError(
            "The selected Guangdong day-ahead period must contain four 15-minute prices per hour"
        )
    wholesale = (
        selected.groupby(["date", "business_hour"], as_index=False)["price_rmb_mwh"]
        .mean()
        .sort_values(["date", "business_hour"])["price_rmb_mwh"]
        .to_numpy(dtype=float)
    )
    retail = wholesale + spot_retail_adder_rmb_per_kwh(config) * 1000.0
    if retail.shape != (horizon_hours,) or not np.isfinite(retail).all():
        raise ValueError("Spot-price aggregation did not match the configured horizon")
    return retail


def read_battery_cost_parameters(config: dict) -> BatteryCostParameters:
    """Read the split battery-energy and inverter-power cost boundary."""
    settings = config["energy"]["battery_cost"]
    year = int(settings["source_year"])
    frame = pd.read_csv(
        rooted(config, config["paths"]["battery_cost_source"]),
        encoding="utf-8-sig",
    )

    def value(technology: str, parameter: str, unit: str) -> float:
        selected = frame[
            (frame["technology"] == technology)
            & (frame["year"] == year)
            & (frame["parameter"] == parameter)
            & (frame["unit"] == unit)
        ]
        if len(selected) != 1:
            raise ValueError(
                f"Expected one {technology}/{parameter}/{unit} row for {year}"
            )
        result = float(selected["value"].iloc[0])
        if not np.isfinite(result) or result < 0.0:
            raise ValueError(f"Invalid battery cost value for {technology}/{parameter}")
        return result

    energy_technology = str(settings["energy_technology"])
    power_technology = str(settings["power_technology"])
    eur_to_rmb = float(settings["eur_to_rmb"])
    if eur_to_rmb <= 0.0:
        raise ValueError("Battery-cost EUR-to-RMB conversion must be positive")
    efficiency = value(power_technology, "efficiency", "per unit")
    configured_efficiency = float(config["energy"]["battery_roundtrip_efficiency"])
    if abs(efficiency - configured_efficiency) > 1e-12:
        raise ValueError("Configured battery efficiency does not match costs_2025.csv")
    return BatteryCostParameters(
        energy_capex_rmb_per_kwh=value(
            energy_technology, "investment", "EUR/kWh"
        )
        * eur_to_rmb,
        power_capex_rmb_per_kw=value(
            power_technology, "investment", "EUR/kW"
        )
        * eur_to_rmb,
        energy_lifetime_years=value(energy_technology, "lifetime", "years"),
        power_lifetime_years=value(power_technology, "lifetime", "years"),
        power_fom_fraction_per_year=value(power_technology, "FOM", "%/year")
        / 100.0,
        roundtrip_efficiency=efficiency,
    )


def load_industry_inputs(
    config: dict,
    industry_code: str,
    *,
    enforce_external_alignment: bool = True,
) -> IndustryInputs:
    demand = config["demand"]
    horizon_hours = int(config["model"]["horizon_hours"])
    represented_days = horizon_hours // 24
    hourly = pd.read_csv(
        rooted(config, config["paths"]["hourly_industry_profiles"]),
        encoding="utf-8-sig",
    )
    selected = hourly[
        (hourly["industry_code"] == industry_code)
        & (hourly["temporal_scenario"] == demand["temporal_scenario"])
    ].sort_values("hour")
    if len(selected) != horizon_hours:
        raise ValueError(
            f"{industry_code} must provide exactly {horizon_hours} continuous load hours; "
            "the core workflow no longer repeats a 24-hour fallback profile"
        )
    if list(selected["hour"]) != list(range(horizon_hours)):
        raise ValueError(f"{industry_code} load profile must cover hours 0..{horizon_hours - 1}")
    industry_name = str(selected["industry_name_cn"].iloc[0])
    base_load_source = selected["baseline_load_mw"].to_numpy(dtype=float)
    normalized_source = selected["base_normalized_load"].to_numpy(dtype=float)
    base_load_profile = base_load_source
    if str(config["model"].get("load_profile_mode", "measured_continuous")) == "measured_continuous":
        required_provenance = {"timestamp", "source_user", "load_profile_kind"}
        if not required_provenance.issubset(selected.columns):
            raise ValueError(
                f"{industry_code} measured continuous load lacks provenance columns: "
                f"{sorted(required_provenance - set(selected.columns))}"
            )
        if horizon_hours == 168 and np.allclose(
            base_load_profile, np.tile(base_load_profile[:24], 7), rtol=0.0, atol=1e-10
        ):
            raise ValueError(f"{industry_code} load is a repeated typical day, not a measured week")
    # Task templates remain daily.  For a measured week/year, use its mean
    # clock-hour shape so service demand keeps the existing annual boundary.
    base_normalized = np.vstack(
        [normalized_source[offset::24] for offset in range(24)]
    ).mean(axis=1)

    office, agent = parse_office_agent_shapes(
        rooted(config, config["paths"]["workload_shape_source"])
    )
    shapes = task_shapes(base_normalized, office, agent)

    task_frame = pd.read_csv(
        rooted(config, config["paths"]["model_ready_task_service"]),
        encoding="utf-8-sig",
    )
    service = demand["effective_service"]
    task_rows = task_frame[
        (task_frame["industry_code"] == industry_code)
        & (task_frame["year"] == int(demand["year"]))
        & (task_frame["parameter_case"] == service["parameter_case"])
    ]
    if set(task_rows["task_id"]) != set(shapes):
        raise ValueError(f"Task set mismatch for {industry_code}")
    service_source_column = str(service["source_column"])
    accelerator_h_per_service_unit = float(service["accelerator_h_per_service_unit"])
    daily_service_by_task: dict[str, np.ndarray] = {}
    for row in task_rows.itertuples():
        shape = shapes[row.task_id]
        daily_service_by_task[row.task_id] = (
            float(getattr(row, service_source_column)) * shape / float(np.sum(shape))
        )
    daily_total_service_profile = sum(daily_service_by_task.values(), np.zeros(24))
    reference_compute_profile = daily_total_service_profile * accelerator_h_per_service_unit
    service_by_task = {
        task_id: np.tile(profile, represented_days)
        for task_id, profile in daily_service_by_task.items()
    }

    allocation = pd.read_csv(
        rooted(config, config["paths"]["topdown_allocation"]),
        encoding="utf-8-sig",
    )
    row = allocation[allocation["industry_code"] == industry_code]
    if len(row) != 1:
        raise ValueError(f"Expected one top-down allocation row for {industry_code}")
    alignment = demand["external_energy_alignment"]
    external_low_twh = float(row[alignment["lower_allocation_column"]].iloc[0])
    external_central_twh = float(row[alignment["central_allocation_column"]].iloc[0])
    external_high_twh = float(row[alignment["upper_allocation_column"]].iloc[0])
    reference_daily_mwh, reference_groups = reference_energy_mwh(
        reference_compute_profile,
        demand["reference_energy_check"],
        float(config["server"]["installed_reserve_fraction"]),
        int(config["server"]["n_plus_spare_server_groups"]),
    )
    derived_reference_twh = (
        reference_daily_mwh * float(config["model"]["annualization_days"]) / 1e6
    )
    alignment_ratio = derived_reference_twh / external_central_twh
    hard_industry_check = alignment.get("per_industry_check", "hard") == "hard"
    if enforce_external_alignment and hard_industry_check and not (
        float(alignment["warning_ratio_low"])
        <= alignment_ratio
        <= float(alignment["warning_ratio_high"])
    ):
        raise ValueError(
            "Effective-service parameters imply reference electricity outside the declared "
            f"external-alignment band: ratio={alignment_ratio:.3f}"
        )

    flexibility_frame = pd.read_csv(
        rooted(config, config["paths"]["flexibility_mapping"]),
        encoding="utf-8-sig",
    )
    flexibility_frame = flexibility_frame[
        flexibility_frame["scenario"] == demand["flexibility_scenario"]
    ]
    if set(flexibility_frame["task_id"]) != set(shapes):
        raise ValueError("Flexibility mapping does not cover all six tasks")
    flexibility = {row.task_id: row for row in flexibility_frame.itertuples()}

    rigid = np.zeros(horizon_hours)
    rigid_by_task: dict[str, np.ndarray] = {}
    jobs: list[FlexibleJob] = []
    for task_id, profile in service_by_task.items():
        setting = flexibility[task_id]
        shares = float(setting.rho_rigid) + float(setting.rho_intraday) + float(setting.rho_batch)
        if abs(shares - 1.0) > 1e-9:
            raise ValueError(f"Flexibility shares do not sum to one for {task_id}")
        task_rigid = profile * float(setting.rho_rigid)
        rigid_by_task[task_id] = task_rigid
        rigid += task_rigid
        for class_name, share, deadline in (
            ("F_day", float(setting.rho_intraday), int(setting.intraday_deadline_h)),
            ("F_batch", float(setting.rho_batch), int(setting.batch_deadline_h)),
        ):
            arrivals = profile * share
            for hour, amount in enumerate(arrivals):
                if amount > 0:
                    jobs.append(
                        FlexibleJob(
                            release_hour=hour,
                            deadline_hours=deadline,
                            amount_service_units=float(amount),
                            task_id=task_id,
                            flexibility_class=class_name,
                        )
                    )
    total_service = float(np.sum(rigid)) + sum(job.amount_service_units for job in jobs)
    expected_service = float(np.sum(daily_total_service_profile)) * represented_days
    if abs(total_service - expected_service) > max(1e-4, expected_service * 1e-10):
        raise ValueError("Rigid and flexible service do not reconstruct total service")
    rooftop = read_industry_rooftop_parameters(config, industry_code)
    return IndustryInputs(
        industry_code=industry_code,
        industry_name=industry_name,
        base_load_mw=base_load_profile,
        rigid_service_units=rigid,
        rigid_service_units_by_task=rigid_by_task,
        flexible_jobs=tuple(jobs),
        daily_effective_service_units=total_service / represented_days,
        accelerator_h_per_service_unit=accelerator_h_per_service_unit,
        reference_daily_accelerator_h=float(np.sum(reference_compute_profile)),
        external_energy_low_twh=external_low_twh,
        external_energy_central_twh=external_central_twh,
        external_energy_high_twh=external_high_twh,
        derived_reference_energy_twh=derived_reference_twh,
        external_energy_alignment_ratio=alignment_ratio,
        reference_energy_server_groups=reference_groups,
        pv_capacity_factor=np.tile(read_pv_capacity_factor(config), represented_days),
        roof_area_proxy_m2=float(rooftop["roof_area_proxy_m2"]),
        roof_area_case=str(rooftop["roof_area_case"]),
        roof_source_naics=str(rooftop["roof_source_naics"]),
        roof_mapping_type=str(rooftop["roof_mapping_type"]),
        roof_evidence_grade=str(rooftop["roof_evidence_grade"]),
    )


def scale_workload(
    inputs: IndustryInputs,
    ai_scale: float,
) -> tuple[np.ndarray, tuple[FlexibleJob, ...]]:
    rigid = inputs.rigid_service_units * ai_scale
    jobs = tuple(
        FlexibleJob(
            release_hour=job.release_hour,
            deadline_hours=job.deadline_hours,
            amount_service_units=job.amount_service_units * ai_scale,
            task_id=job.task_id,
            flexibility_class=job.flexibility_class,
        )
        for job in inputs.flexible_jobs
    )
    return rigid, jobs


def scale_task_workload(
    inputs: IndustryInputs,
    ai_scale: float,
) -> tuple[dict[str, np.ndarray], tuple[FlexibleJob, ...]]:
    """Scale task-resolved rigid demand and flexible jobs for heterogeneous routing."""
    rigid_by_task = {
        task_id: np.asarray(profile, dtype=float) * ai_scale
        for task_id, profile in inputs.rigid_service_units_by_task.items()
    }
    jobs = tuple(
        FlexibleJob(
            release_hour=job.release_hour,
            deadline_hours=job.deadline_hours,
            amount_service_units=job.amount_service_units * ai_scale,
            task_id=job.task_id,
            flexibility_class=job.flexibility_class,
        )
        for job in inputs.flexible_jobs
    )
    return rigid_by_task, jobs
