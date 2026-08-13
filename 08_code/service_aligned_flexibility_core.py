"""Core routines for equal-service local/cloud flexibility prototypes.

The model uses aggregate accelerator-hours, common task-class flexibility
parameters, fixed installed server groups, and a cyclic 24-hour linear program
solved by HiGHS.  It intentionally avoids individual inference-request
variables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


HOURS_PER_DAY = 24


@dataclass(frozen=True)
class ServerTechnology:
    accelerators_per_group: float
    idle_power_kw: float
    maximum_power_kw: float

    @property
    def dynamic_power_kw(self) -> float:
        return self.maximum_power_kw - self.idle_power_kw


@dataclass(frozen=True)
class Architecture:
    name: str
    pue: float
    target_installed_utilization: float
    reserve_fraction: float


@dataclass(frozen=True)
class FlexibleArrival:
    release_hour: int
    deadline_hours: int
    amount_accelerator_h: float
    task_id: str
    flexibility_class: str


@dataclass
class ScheduleResult:
    feasible: bool
    daily_execution_accelerator_h: np.ndarray
    daily_execution_intraday_accelerator_h: np.ndarray
    daily_execution_batch_accelerator_h: np.ndarray
    target_peak_mw: float
    target_facility_peak_mw: float
    solver_name: str
    solver_status: str


def provision_server_groups(
    peak_accelerator_h_per_hour: float,
    technology: ServerTechnology,
    architecture: Architecture,
) -> int:
    """Provision installed groups from unshifted peak, target use, and reserve."""
    if peak_accelerator_h_per_hour <= 0:
        return 0
    effective_accelerators = (
        technology.accelerators_per_group
        * architecture.target_installed_utilization
    )
    return math.ceil(
        peak_accelerator_h_per_hour
        * (1.0 + architecture.reserve_fraction)
        / effective_accelerators
    )


def facility_power_mw(
    execution_accelerator_h: np.ndarray,
    installed_groups: int,
    technology: ServerTechnology,
    pue: float,
) -> np.ndarray:
    """Map one-hour aggregate execution to facility power with all groups online."""
    execution = np.asarray(execution_accelerator_h, dtype=float)
    equivalent_full_groups = execution / technology.accelerators_per_group
    it_kw = (
        installed_groups * technology.idle_power_kw
        + equivalent_full_groups * technology.dynamic_power_kw
    )
    return pue * it_kw / 1000.0


def capacity_from_peak_target(
    target_peak_mw: float,
    base_load_mw: np.ndarray,
    installed_groups: int,
    technology: ServerTechnology,
    pue: float,
) -> np.ndarray:
    """Convert a combined-load peak target into hourly accelerator-hour limits."""
    base = np.asarray(base_load_mw, dtype=float)
    available_facility_mw = np.maximum(0.0, target_peak_mw - base)
    idle_it_mw = installed_groups * technology.idle_power_kw / 1000.0
    dynamic_it_mw = available_facility_mw / pue - idle_it_mw
    equivalent_full_groups = np.maximum(
        0.0,
        dynamic_it_mw / (technology.dynamic_power_kw / 1000.0),
    )
    accelerator_capacity = (
        equivalent_full_groups * technology.accelerators_per_group
    )
    nameplate_capacity = installed_groups * technology.accelerators_per_group
    return np.minimum(accelerator_capacity, nameplate_capacity)


def optimize_peak_with_highs(
    rigid_daily: np.ndarray,
    flexible_arrivals: list[FlexibleArrival],
    base_load_mw: np.ndarray,
    installed_groups: int,
    technology: ServerTechnology,
    architecture: Architecture,
    lexicographic_tolerance_mw: float = 1e-6,
) -> ScheduleResult:
    """Solve cyclic aggregate scheduling with two lexicographic HiGHS LPs.

    The primary objective minimizes the peak of base load plus AI facility
    power.  The secondary objective minimizes the AI facility peak while
    allowing at most ``lexicographic_tolerance_mw`` deterioration in the first
    objective.  Electricity prices are deliberately absent from both objectives.
    """
    rigid = np.asarray(rigid_daily, dtype=float)
    base = np.asarray(base_load_mw, dtype=float)
    if rigid.shape != (HOURS_PER_DAY,) or base.shape != (HOURS_PER_DAY,):
        raise ValueError("rigid and base profiles must contain 24 hours")
    nameplate_accelerators = installed_groups * technology.accelerators_per_group
    if float(np.max(rigid)) > nameplate_accelerators + 1e-7:
        raise ValueError("rigid service exceeds installed accelerator capacity")

    assignments: list[tuple[int, int, str]] = []
    job_amounts: list[float] = []
    for job_index, job in enumerate(flexible_arrivals):
        if not 0 <= job.release_hour < HOURS_PER_DAY:
            raise ValueError("release hour must lie in 0..23")
        if job.deadline_hours < 0:
            raise ValueError("deadline must be nonnegative")
        if job.amount_accelerator_h < -1e-9:
            raise ValueError("flexible service amount cannot be negative")
        if job.flexibility_class not in {"F_day", "F_batch"}:
            raise ValueError(f"unknown flexibility class: {job.flexibility_class}")
        job_amounts.append(max(0.0, float(job.amount_accelerator_h)))
        admissible_hours = sorted(
            {
                (job.release_hour + offset) % HOURS_PER_DAY
                for offset in range(job.deadline_hours + 1)
            }
        )
        assignments.extend(
            (job_index, hour, job.flexibility_class) for hour in admissible_hours
        )

    assignment_count = len(assignments)
    combined_peak_index = assignment_count
    facility_peak_index = assignment_count + 1
    variable_count = assignment_count + 2
    hour_assignment_indices: list[list[int]] = [[] for _ in range(HOURS_PER_DAY)]
    job_assignment_indices: list[list[int]] = [
        [] for _ in range(len(flexible_arrivals))
    ]
    for variable_index, (job_index, hour, _) in enumerate(assignments):
        hour_assignment_indices[hour].append(variable_index)
        job_assignment_indices[job_index].append(variable_index)

    equality_rows: list[int] = []
    equality_cols: list[int] = []
    equality_data: list[float] = []
    for job_index, variable_indices in enumerate(job_assignment_indices):
        for variable_index in variable_indices:
            equality_rows.append(job_index)
            equality_cols.append(variable_index)
            equality_data.append(1.0)
    equality_matrix = coo_matrix(
        (equality_data, (equality_rows, equality_cols)),
        shape=(len(flexible_arrivals), variable_count),
    ).tocsr()

    idle_facility_mw = (
        architecture.pue * installed_groups * technology.idle_power_kw / 1000.0
    )
    dynamic_facility_mw_per_accelerator = (
        architecture.pue
        * technology.dynamic_power_kw
        / technology.accelerators_per_group
        / 1000.0
    )
    inequality_rows: list[int] = []
    inequality_cols: list[int] = []
    inequality_data: list[float] = []
    inequality_rhs: list[float] = []

    for hour in range(HOURS_PER_DAY):
        # Explicit accelerator nameplate constraint.
        capacity_row = len(inequality_rhs)
        for variable_index in hour_assignment_indices[hour]:
            inequality_rows.append(capacity_row)
            inequality_cols.append(variable_index)
            inequality_data.append(1.0)
        inequality_rhs.append(nameplate_accelerators - float(rigid[hour]))

        # Base load plus AI facility power cannot exceed combined peak.
        combined_row = len(inequality_rhs)
        for variable_index in hour_assignment_indices[hour]:
            inequality_rows.append(combined_row)
            inequality_cols.append(variable_index)
            inequality_data.append(dynamic_facility_mw_per_accelerator)
        inequality_rows.append(combined_row)
        inequality_cols.append(combined_peak_index)
        inequality_data.append(-1.0)
        inequality_rhs.append(
            -float(base[hour])
            - idle_facility_mw
            - dynamic_facility_mw_per_accelerator * float(rigid[hour])
        )

        # AI facility power cannot exceed the secondary peak variable.
        facility_row = len(inequality_rhs)
        for variable_index in hour_assignment_indices[hour]:
            inequality_rows.append(facility_row)
            inequality_cols.append(variable_index)
            inequality_data.append(dynamic_facility_mw_per_accelerator)
        inequality_rows.append(facility_row)
        inequality_cols.append(facility_peak_index)
        inequality_data.append(-1.0)
        inequality_rhs.append(
            -idle_facility_mw
            - dynamic_facility_mw_per_accelerator * float(rigid[hour])
        )

    inequality_matrix = coo_matrix(
        (inequality_data, (inequality_rows, inequality_cols)),
        shape=(len(inequality_rhs), variable_count),
    ).tocsr()
    base_bounds: list[tuple[float | None, float | None]] = [
        (0.0, None) for _ in range(variable_count)
    ]

    primary_objective = np.zeros(variable_count)
    primary_objective[combined_peak_index] = 1.0
    primary = linprog(
        primary_objective,
        A_ub=inequality_matrix,
        b_ub=np.asarray(inequality_rhs),
        A_eq=equality_matrix,
        b_eq=np.asarray(job_amounts),
        bounds=base_bounds,
        method="highs",
        options={"presolve": True},
    )
    if not primary.success:
        raise RuntimeError(
            f"HiGHS primary peak optimization failed for {architecture.name}: "
            f"{primary.message}"
        )

    secondary_bounds = list(base_bounds)
    secondary_bounds[combined_peak_index] = (
        0.0,
        float(primary.x[combined_peak_index]) + lexicographic_tolerance_mw,
    )
    secondary_objective = np.zeros(variable_count)
    secondary_objective[facility_peak_index] = 1.0
    secondary = linprog(
        secondary_objective,
        A_ub=inequality_matrix,
        b_ub=np.asarray(inequality_rhs),
        A_eq=equality_matrix,
        b_eq=np.asarray(job_amounts),
        bounds=secondary_bounds,
        method="highs",
        options={"presolve": True},
    )
    if not secondary.success:
        raise RuntimeError(
            f"HiGHS secondary peak optimization failed for {architecture.name}: "
            f"{secondary.message}"
        )

    intraday = np.zeros(HOURS_PER_DAY)
    batch = np.zeros(HOURS_PER_DAY)
    for variable_index, (_, hour, flexibility_class) in enumerate(assignments):
        amount = max(0.0, float(secondary.x[variable_index]))
        if flexibility_class == "F_day":
            intraday[hour] += amount
        else:
            batch[hour] += amount
    total = rigid + intraday + batch
    return ScheduleResult(
        feasible=True,
        daily_execution_accelerator_h=total,
        daily_execution_intraday_accelerator_h=intraday,
        daily_execution_batch_accelerator_h=batch,
        target_peak_mw=float(primary.x[combined_peak_index]),
        target_facility_peak_mw=float(secondary.x[facility_peak_index]),
        solver_name="HiGHS via scipy.optimize.linprog",
        solver_status=f"primary={primary.message}; secondary={secondary.message}",
    )


def calibrate_service_scale_to_reference_energy(
    unscaled_execution: np.ndarray,
    target_daily_facility_energy_mwh: float,
    technology: ServerTechnology,
    reference_architecture: Architecture,
) -> tuple[float, int, float]:
    """Find kappa so the reference cloud's daily facility energy hits its anchor."""
    shape = np.asarray(unscaled_execution, dtype=float)

    def evaluate(scale: float) -> tuple[float, int]:
        execution = shape * scale
        groups = provision_server_groups(
            float(np.max(execution)), technology, reference_architecture
        )
        energy = float(
            np.sum(
                facility_power_mw(
                    execution,
                    groups,
                    technology,
                    reference_architecture.pue,
                )
            )
        )
        return energy, groups

    low, high = 0.0, 1.0
    energy, _ = evaluate(high)
    while energy < target_daily_facility_energy_mwh:
        high *= 2.0
        energy, _ = evaluate(high)
        if high > 1e12:
            raise RuntimeError("Could not bracket service calibration scale")

    for _ in range(100):
        midpoint = (low + high) / 2.0
        energy, _ = evaluate(midpoint)
        if energy < target_daily_facility_energy_mwh:
            low = midpoint
        else:
            high = midpoint

    scale = (low + high) / 2.0
    energy, groups = evaluate(scale)
    return scale, groups, energy
