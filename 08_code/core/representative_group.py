"""Parse representative-group parameters and construct core scenario scales."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class RepresentativeGroup:
    industry_code: str
    industry_name: str
    share_low: float
    share_base: float
    share_high: float
    factories_low: int
    factories_base: int
    factories_high: int
    evidence_grade: str

    def share(self, case: str) -> float:
        return getattr(self, f"share_{case}")

    def factories(self, case: str) -> int:
        return getattr(self, f"factories_{case}")


@dataclass(frozen=True)
class ScenarioScale:
    scenario: str
    ai_service_scale_per_host: float
    equivalent_host_multiplier: float
    physical_host_count: int
    group_share: float
    group_factory_count: int

    @property
    def industry_service_reconstruction(self) -> float:
        return self.ai_service_scale_per_host * self.equivalent_host_multiplier


def _clean(cell: str) -> str:
    return cell.strip().replace("**", "")


def read_representative_groups(path: Path) -> dict[str, RepresentativeGroup]:
    rows: dict[str, RepresentativeGroup] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not re.match(r"^\| C(?:1[3-9]|[234][0-9]) \|", line):
            continue
        cells = [_clean(cell) for cell in line.strip().strip("|").split("|")]
        if len(cells) < 15:
            raise ValueError(f"Malformed representative-group table row: {line}")
        row = RepresentativeGroup(
            industry_code=cells[0],
            industry_name=cells[1],
            share_low=float(cells[4]),
            share_base=float(cells[5]),
            share_high=float(cells[6]),
            factories_low=int(cells[7]),
            factories_base=int(cells[8]),
            factories_high=int(cells[9]),
            evidence_grade=cells[13],
        )
        if not 0 < row.share_low <= row.share_base <= row.share_high <= 1:
            raise ValueError(f"Invalid group-share ordering for {row.industry_code}")
        if not 0 < row.factories_low <= row.factories_base <= row.factories_high:
            raise ValueError(f"Invalid group-factory ordering for {row.industry_code}")
        reported_multiplier = float(cells[10])
        if abs(reported_multiplier - 1.0 / row.share_base) > 0.11:
            raise ValueError(f"Incorrect M_group for {row.industry_code}")
        rows[row.industry_code] = row
    if len(rows) != 31:
        raise ValueError(f"Expected 31 manufacturing industries, found {len(rows)}")
    return rows


def scenario_scale(
    group: RepresentativeGroup,
    parameter_case: str,
    scenario: str,
    industry_host_count: int = 1,
) -> ScenarioScale:
    share = group.share(parameter_case)
    factories = group.factories(parameter_case)
    ai_service_scale_per_factory = share / factories
    if scenario == "IF":
        ai_scale = ai_service_scale_per_factory
        multiplier = factories / share
        physical_hosts = factories
    elif scenario in {"IG", "IG_1host"}:
        ai_scale = share
        multiplier = 1.0 / share
        physical_hosts = 1
    elif scenario == "II_1host":
        ai_scale = 1.0
        multiplier = 1.0
        physical_hosts = 1
    elif scenario == "II_multihost":
        if industry_host_count < 2:
            raise ValueError("II_multihost requires industry_host_count >= 2")
        ai_scale = 1.0 / industry_host_count
        multiplier = float(industry_host_count)
        physical_hosts = industry_host_count
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    scale = ScenarioScale(
        scenario=scenario,
        ai_service_scale_per_host=ai_scale,
        equivalent_host_multiplier=multiplier,
        physical_host_count=physical_hosts,
        group_share=share,
        group_factory_count=factories,
    )
    if abs(scale.industry_service_reconstruction - 1.0) > 1e-10:
        raise AssertionError(f"Scenario {scenario} does not reconstruct industry service")
    return scale
