"""Planning utilities to determine minimal filings needed for historical coverage."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple


class PeriodType(Enum):
    FY = "FY"
    QUARTER = "Q"


@dataclass(frozen=True)
class QuarterPeriod:
    year: int
    quarter: int

    def index(self) -> int:
        return self.year * 4 + (self.quarter - 1)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return f"Q{self.quarter} {self.year}"


@dataclass(frozen=True)
class FiscalYearPeriod:
    year: int

    def __str__(self) -> str:  # pragma: no cover - convenience
        return f"FY {self.year}"


@dataclass
class FilingCandidate:
    filing_type: str  # "10-Q" or "10-K"
    year: int
    quarter: Optional[int] = None
    covers_quarters: Set[QuarterPeriod] = field(default_factory=set)
    covers_fys: Set[FiscalYearPeriod] = field(default_factory=set)

    def coverage_score(
        self,
        uncovered_quarters: Set[QuarterPeriod],
        uncovered_fys: Set[FiscalYearPeriod],
    ) -> int:
        return len(self.covers_quarters & uncovered_quarters) + len(
            self.covers_fys & uncovered_fys
        )


class PeriodParseError(ValueError):
    pass


_PERIOD_RE = re.compile(
    r"^(?P<prefix>FY|Q[1-4])[-_\s]?(?P<year>\d{4})$",
    re.IGNORECASE,
)


def parse_period(spec: str) -> Tuple[PeriodType, int, Optional[int]]:
    match = _PERIOD_RE.match(spec.strip())
    if not match:
        raise PeriodParseError(f"Could not parse period specifier: {spec!r}")
    prefix = match.group("prefix").upper()
    year = int(match.group("year"))
    if prefix.startswith("FY"):
        return PeriodType.FY, year, None
    quarter = int(prefix[1])
    return PeriodType.QUARTER, year, quarter


def quarter_range(latest: QuarterPeriod, earliest_fy_year: int) -> List[QuarterPeriod]:
    start_index = QuarterPeriod(earliest_fy_year, 1).index()
    latest_index = latest.index()
    if start_index > latest_index:
        return []
    periods: List[QuarterPeriod] = []
    for idx in range(start_index, latest_index + 1):
        year, q_index = divmod(idx, 4)
        periods.append(QuarterPeriod(year, q_index + 1))
    return periods


def fiscal_year_range(latest_year: int, years: int) -> List[FiscalYearPeriod]:
    return [FiscalYearPeriod(year) for year in range(latest_year - years + 1, latest_year + 1)]


def _select_10k_years(years: Sequence[int]) -> List[int]:
    remaining_bs = set(years)
    remaining_is = set(years)
    selected: List[int] = []

    while remaining_bs:
        year = max(remaining_bs)
        selected.append(year)
        for delta in (0, -1):
            remaining_bs.discard(year + delta)
        for delta in (0, -1, -2):
            remaining_is.discard(year + delta)

    while remaining_is:
        year = max(remaining_is)
        if year not in selected:
            selected.append(year)
        for delta in (0, -1, -2):
            remaining_is.discard(year + delta)

    return sorted(selected, reverse=True)


def _tenk_candidate(
    year: int,
    remaining_quarters: Set[QuarterPeriod],
    remaining_fys: Set[FiscalYearPeriod],
) -> FilingCandidate:
    quarter_coverage = {
        qp
        for qp in remaining_quarters
        if qp.quarter == 4 and qp.year in {year, year - 1}
    }
    fy_coverage = {
        fy for fy in remaining_fys if fy.year in {year, year - 1, year - 2}
    }
    return FilingCandidate(
        filing_type="10-K",
        year=year,
        covers_quarters=quarter_coverage,
        covers_fys=fy_coverage,
    )


def _q2_candidate(year: int, remaining_quarters: Set[QuarterPeriod]) -> FilingCandidate:
    coverage = {
        qp
        for qp in remaining_quarters
        if qp.quarter in {1, 2} and qp.year in {year, year - 1}
    }
    return FilingCandidate(
        filing_type="10-Q",
        year=year,
        quarter=2,
        covers_quarters=coverage,
    )


def _q3_candidate(year: int, remaining_quarters: Set[QuarterPeriod]) -> FilingCandidate:
    coverage = {
        qp for qp in remaining_quarters if qp.quarter == 3 and qp.year in {year, year - 1}
    }
    return FilingCandidate(
        filing_type="10-Q",
        year=year,
        quarter=3,
        covers_quarters=coverage,
    )


def _select_q2_filings(
    remaining_quarters: Set[QuarterPeriod],
    max_available_year: int,
) -> List[FilingCandidate]:
    selections: List[FilingCandidate] = []
    while True:
        needed = {qp for qp in remaining_quarters if qp.quarter in {1, 2}}
        if not needed:
            break
        highest_year = max(qp.year for qp in needed)
        if highest_year > max_available_year:
            raise RuntimeError("Unable to cover required Q1/Q2 periods with available filings")
        candidate = _q2_candidate(highest_year, remaining_quarters)
        coverage = candidate.covers_quarters
        if not coverage:
            raise RuntimeError("Unable to cover required Q1/Q2 periods with available filings")
        selections.append(candidate)
        remaining_quarters -= coverage
    return selections


def _select_q3_filings(
    remaining_quarters: Set[QuarterPeriod],
    max_available_year: int,
) -> List[FilingCandidate]:
    selections: List[FilingCandidate] = []
    while True:
        needed = {qp for qp in remaining_quarters if qp.quarter == 3}
        if not needed:
            break
        highest_year = max(qp.year for qp in needed)
        if highest_year > max_available_year:
            raise RuntimeError("Unable to cover required Q3 periods with available filings")
        candidate = _q3_candidate(highest_year, remaining_quarters)
        coverage = candidate.covers_quarters
        if not coverage:
            raise RuntimeError("Unable to cover required Q3 periods with available filings")
        selections.append(candidate)
        remaining_quarters -= coverage
    return selections


@dataclass
class CoveragePlan:
    latest_period: str
    years: int
    required_quarters: List[QuarterPeriod]
    required_fys: List[FiscalYearPeriod]
    filings: List[FilingCandidate]

    def to_json(self) -> Dict[str, object]:
        return {
            "latest": self.latest_period,
            "years": self.years,
            "required_quarters": [str(qp) for qp in self.required_quarters],
            "required_fiscal_years": [str(fy) for fy in self.required_fys],
            "filings": [
                {
                    "type": filing.filing_type,
                    "year": filing.year,
                    **(
                        {"quarter": filing.quarter}
                        if filing.quarter is not None
                        else {}
                    ),
                    "covers_quarters": [str(qp) for qp in sorted(filing.covers_quarters, key=lambda x: (x.year, x.quarter))],
                    "covers_fiscal_years": [str(fy) for fy in sorted(filing.covers_fys, key=lambda x: x.year)],
                }
                for filing in sorted(
                    self.filings,
                    key=lambda f: (0 if f.filing_type == "10-K" else 1, f.year, f.quarter or 0),
                )
            ],
        }


def plan_minimal_filings(latest_spec: str, years: int) -> CoveragePlan:
    if years <= 0:
        raise ValueError("years must be positive")

    period_type, latest_year, latest_quarter = parse_period(latest_spec)

    if period_type == PeriodType.FY:
        latest_quarter_period = QuarterPeriod(latest_year, 4)
        latest_fy = latest_year
    else:
        latest_quarter_period = QuarterPeriod(latest_year, latest_quarter or 1)
        latest_fy = latest_year - (1 if (latest_quarter or 1) < 4 else 0)

    required_fys = fiscal_year_range(latest_fy, years)
    earliest_fy_year = required_fys[0].year if required_fys else latest_fy
    required_quarters = quarter_range(latest_quarter_period, earliest_fy_year)
    remaining_quarters = set(required_quarters)
    remaining_fys = set(required_fys)
    selected: List[FilingCandidate] = []

    tenk_years_input = sorted(
        {fy.year for fy in remaining_fys}
        | {qp.year for qp in remaining_quarters if qp.quarter == 4}
    )
    tenk_years = _select_10k_years(tenk_years_input)
    for year in tenk_years:
        candidate = _tenk_candidate(year, remaining_quarters, remaining_fys)
        if not candidate.covers_quarters and not candidate.covers_fys:
            continue
        selected.append(candidate)
        remaining_quarters -= candidate.covers_quarters
        remaining_fys -= candidate.covers_fys

    if remaining_fys:
        raise RuntimeError("Unable to cover required fiscal years with available 10-K filings")

    if latest_quarter_period.quarter == 1:
        q1_coverage = {
            qp
            for qp in remaining_quarters
            if qp.quarter == 1 and qp.year == latest_quarter_period.year
        }
        if q1_coverage:
            q1_candidate = FilingCandidate(
                filing_type="10-Q",
                year=latest_quarter_period.year,
                quarter=1,
                covers_quarters=q1_coverage,
            )
            selected.append(q1_candidate)
            remaining_quarters -= q1_candidate.covers_quarters

    max_q2_year = (
        latest_quarter_period.year
        if latest_quarter_period.quarter >= 2
        else latest_quarter_period.year - 1
    )
    selected.extend(_select_q2_filings(remaining_quarters, max_q2_year))

    max_q3_year = (
        latest_quarter_period.year
        if latest_quarter_period.quarter >= 3
        else latest_quarter_period.year - 1
    )
    selected.extend(_select_q3_filings(remaining_quarters, max_q3_year))

    remaining_quarters -= {qp for qp in remaining_quarters if qp.quarter == 4}

    if remaining_quarters:
        raise RuntimeError("Unable to cover all required quarter periods with available filings")

    return CoveragePlan(
        latest_period=latest_spec,
        years=years,
        required_quarters=list(required_quarters),
        required_fys=list(required_fys),
        filings=selected,
    )
