from __future__ import annotations

from collections import Counter
import calendar
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from selector.models import StatementType

from .aliases import canonicalize, load_aliases, normalize_label
from .models import CarbonCopyResult, StatementBlock


@dataclass(slots=True)
class Measurement:
    period_end: date
    duration: str
    months: Optional[int]
    value: float
    source_pdf: Path
    derived: bool = False


def _duration_key(duration: str, months: Optional[int]) -> str:
    duration_norm = (duration or "").lower()
    if duration_norm == "ytd" and months:
        return f"ytd{months}".lower()
    if duration_norm in {"quarter", "annual", "point"}:
        return duration_norm
    if duration_norm:
        return duration_norm
    if months:
        return f"ytd{months}".lower()
    return "unknown"


def _months_between(a: date, b: date) -> int:
    return (a.year - b.year) * 12 + (a.month - b.month)


@dataclass(frozen=True)
class PeriodKey:
    period_end: date
    duration: str
    months: Optional[int]


@dataclass
class ColumnDescriptor:
    display_label: str
    raw_label: str
    duration: str
    period_end: date
    months: Optional[int]


@dataclass
class PeriodValue:
    value: Optional[float]
    source_pdf: Path
    raw_label: str
    duration: str
    months: Optional[int]


@dataclass
class LineAccumulator:
    canonical_label: str
    display_label: str
    level: int
    order_index: int
    latest_period_end: Optional[date]
    period_values: Dict[PeriodKey, PeriodValue] = field(default_factory=dict)

    def update_presentation(self, label: str, level: int, order_index: int, period_end: Optional[date]) -> None:
        if period_end is None:
            return
        if self.latest_period_end is None or period_end >= self.latest_period_end:
            self.display_label = label
            self.level = level
            self.order_index = order_index
            self.latest_period_end = period_end

    def record_value(
        self,
        key: PeriodKey,
        value: Optional[float],
        source: Path,
        raw_label: str,
        duration: str,
        months: Optional[int],
    ) -> None:
        existing = self.period_values.get(key)
        if existing is None or value is not None:
            self.period_values[key] = PeriodValue(value, source, raw_label, duration, months)


@dataclass
class StatementDataset:
    columns: Dict[PeriodKey, ColumnDescriptor] = field(default_factory=dict)
    lines: Dict[str, LineAccumulator] = field(default_factory=dict)

    def ensure_line(
        self,
        canonical_label: str,
        display_label: str,
        level: int,
        order_index: int,
        period_end: Optional[date],
    ) -> LineAccumulator:
        line = self.lines.get(canonical_label)
        if line is None:
            line = LineAccumulator(canonical_label, display_label, level, order_index, period_end)
            self.lines[canonical_label] = line
        else:
            line.update_presentation(display_label, level, order_index, period_end)
        return line

    def register_column(
        self,
        period_end: date,
        duration: str,
        months: Optional[int],
        display_label: str,
        raw_label: str,
    ) -> PeriodKey:
        key = PeriodKey(period_end, duration, months)
        descriptor = ColumnDescriptor(display_label, raw_label, duration, period_end, months)
        self.columns.setdefault(key, descriptor)
        return key


@dataclass
class AggregationDiagnostics:
    entries: List[Dict[str, str]] = field(default_factory=list)

    def add(self, statement: StatementType, label: str, issue: str, period: str, detail: str = "") -> None:
        self.entries.append(
            {
                "Statement": statement.value,
                "Line Item": label,
                "Issue": issue,
                "Period": period,
                "Detail": detail,
            }
        )

    def extend(self, records: Iterable[Dict[str, str]]) -> None:
        self.entries.extend(records)

    def to_frame(self) -> pd.DataFrame:
        if not self.entries:
            return pd.DataFrame(columns=["Statement", "Line Item", "Issue", "Period", "Detail"])
        return pd.DataFrame(self.entries)


@dataclass
class FiscalYearInfo:
    statement: StatementType
    fy_end: date
    annual_key: Optional[PeriodKey] = None
    quarter_keys: Dict[int, PeriodKey] = field(default_factory=dict)
    ytd_keys: Dict[int, PeriodKey] = field(default_factory=dict)

    @property
    def fy_year(self) -> int:
        return self.fy_end.year

    def quarter_date(self, quarter: int) -> Optional[date]:
        key = self.quarter_keys.get(quarter) or self.ytd_keys.get(quarter)
        if key:
            return key.period_end
        offsets = {1: -9, 2: -6, 3: -3, 4: 0}
        if quarter not in offsets:
            return None
        if offsets[quarter] == 0:
            return self.fy_end
        return shift_months(self.fy_end, offsets[quarter])

    def quarter_label(self, quarter: int) -> str:
        period = self.quarter_date(quarter)
        if period is None:
            return ""
        return f"3M ended {format_date(period)}"

    def fy_label(self) -> str:
        return f"FY {self.fy_year}"

    def describe_quarter(self, quarter: int) -> str:
        return f"FY{self.fy_year} Q{quarter}"


class SeriesAggregator:
    def __init__(self, alias_path: Path | None = None) -> None:
        self.alias_map = load_aliases(alias_path)
        self.datasets: Dict[StatementType, StatementDataset] = {
            statement: StatementDataset() for statement in StatementType
        }
        self.diagnostics = AggregationDiagnostics()

    def add_result(self, result: CarbonCopyResult) -> None:
        for statement, block in result.blocks.items():
            self._ingest_block(statement, block, result.source_pdf)

    def _ingest_block(self, statement: StatementType, block: StatementBlock, source_pdf: Path) -> None:
        dataset = self.datasets[statement]
        metadata = block.column_metadata
        for _, row in block.wide_table.iterrows():
            order_index = int(row["Order Index"])
            level = int(row["Level"])
            raw_label = str(row["Line Item (as printed)"])
            canonical = normalize_label(statement, raw_label, self.alias_map)
            line = dataset.ensure_line(canonical, canonicalize(raw_label), level, order_index, None)
            for column in block.wide_table.columns[3:]:
                value = row[column]
                info = metadata.get(column)
                if info is None:
                    continue
                period_end = parse_period_end(info.get("period_end") or column or info.get("raw_label"))
                if period_end is None:
                    self.diagnostics.add(
                        statement,
                        canonical,
                        "missing-period",
                        column,
                        "Could not parse period end date",
                    )
                    continue
                duration = info.get("duration", "unknown")
                months = infer_months(info.get("raw_label", ""), duration)
                key = dataset.register_column(period_end, duration, months, column, info.get("raw_label", column))
                numeric = coerce_numeric(value)
                line.update_presentation(canonicalize(raw_label), level, order_index, period_end)
                line.record_value(key, numeric, source_pdf, info.get("raw_label", column), duration, months)

    def build_frames(self) -> Tuple[Dict[StatementType, pd.DataFrame], pd.DataFrame]:
        tables: Dict[StatementType, pd.DataFrame] = {}
        calendar, fy_month, fy_day = derive_fiscal_calendar(self.datasets.values())
        for statement, dataset in self.datasets.items():
            frame, extra_diags = build_statement_frame(statement, dataset, calendar, fy_month, fy_day)
            tables[statement] = frame
            self.diagnostics.extend(extra_diags)
        diagnostics_df = self.diagnostics.to_frame()
        return tables, diagnostics_df

    def build_measurement_store(self) -> Dict[StatementType, Dict[str, Dict[str, Dict[date, Measurement]]]]:
        calendar, fy_month, fy_day = derive_fiscal_calendar(self.datasets.values())
        store: Dict[StatementType, Dict[str, Dict[str, Dict[date, Measurement]]]] = {}
        for statement, dataset in self.datasets.items():
            fiscal_years = build_fiscal_years(statement, dataset, calendar, [], fy_month, fy_day)
            line_store: Dict[str, Dict[str, Dict[date, Measurement]]] = {}
            for line in dataset.lines.values():
                measurements = compute_line_measurements(statement, line, fiscal_years)
                if measurements:
                    line_store[line.canonical_label.lower()] = measurements
            store[statement] = line_store
        return store


def derive_fiscal_calendar(datasets: Iterable[StatementDataset]) -> Tuple[List[date], int, int]:
    fy_month, fy_day = infer_fiscal_year_end(datasets)
    calendar: set[date] = set()
    for dataset in datasets:
        for key in dataset.columns:
            if key.period_end.year < 1900:
                continue
            if key.duration == "annual":
                calendar.add(date(key.period_end.year, fy_month, fy_day))
            elif key.duration == "point" and key.period_end.month == fy_month:
                calendar.add(date(key.period_end.year, key.period_end.month, key.period_end.day))
    if not calendar:
        # Fallback to using raw annual periods if nothing else is available
        for dataset in datasets:
            for key in dataset.columns:
                if key.duration == "annual" and key.period_end.year >= 1900:
                    calendar.add(key.period_end)
    return sorted(calendar), fy_month, fy_day


def infer_fiscal_year_end(datasets: Iterable[StatementDataset]) -> Tuple[int, int]:
    month_counts: Counter[int] = Counter()
    point_days: Counter[Tuple[int, int]] = Counter()
    for dataset in datasets:
        for key in dataset.columns:
            if key.period_end.year < 1900:
                continue
            if key.duration == "ytd" and key.months:
                start_month = ((key.period_end.month - key.months) % 12) + 1
                fy_month = (start_month - 2) % 12 + 1
                month_counts[fy_month] += 1
            elif key.duration == "annual" and key.months == 12 and key.period_end.month >= 1:
                month_counts[key.period_end.month] += 1
            if key.duration == "point":
                point_days[(key.period_end.month, key.period_end.day)] += 1
    if month_counts:
        fy_month = month_counts.most_common(1)[0][0]
    elif point_days:
        month_counts = Counter()
        for (month, _day), count in point_days.items():
            month_counts[month] += count
        fy_month = month_counts.most_common(1)[0][0]
    else:
        fy_month = 12

    if point_days:
        day_counts = Counter({day: count for (month, day), count in point_days.items() if month == fy_month})
        if day_counts:
            fy_day = day_counts.most_common(1)[0][0]
        else:
            fy_day = 31
    else:
        fy_day = 31
    return fy_month, fy_day


def build_statement_frame(
    statement: StatementType,
    dataset: StatementDataset,
    fiscal_calendar: List[date],
    fy_month: int,
    fy_day: int,
) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    diagnostics: List[Dict[str, str]] = []
    if not dataset.lines:
        return pd.DataFrame(columns=["Level", "Line Item"]), diagnostics

    fiscal_years = build_fiscal_years(statement, dataset, fiscal_calendar, diagnostics, fy_month, fy_day)
    if not fiscal_years:
        return pd.DataFrame(columns=["Level", "Line Item"]), diagnostics

    columns: List[str] = []
    column_map: Dict[str, Tuple[str, int, Optional[int]]] = {}
    for fy_idx, fy in enumerate(fiscal_years):
        for quarter in range(1, 5):
            label = fy.quarter_label(quarter)
            if not label:
                continue
            columns.append(label)
            column_map[label] = ("quarter", fy_idx, quarter)
        fy_label = fy.fy_label()
        columns.append(fy_label)
        column_map[fy_label] = ("annual", fy_idx, None)

    rows: List[Dict[str, object]] = []
    for line in sorted(dataset.lines.values(), key=lambda item: (item.order_index, item.level, item.canonical_label)):
        values = compute_line_series(line, fiscal_years, statement, diagnostics)
        row: Dict[str, object] = {
            "Level": line.level,
            "Line Item": line.display_label,
        }
        for column in columns:
            spec = column_map[column]
            row[column] = values.get(spec)
        rows.append(row)

    frame = pd.DataFrame(rows)
    return frame, diagnostics


def build_fiscal_years(
    statement: StatementType,
    dataset: StatementDataset,
    fiscal_calendar: List[date],
    diagnostics: List[Dict[str, str]],
    fy_month: int,
    fy_day: int,
) -> List[FiscalYearInfo]:
    fiscal_years: Dict[date, FiscalYearInfo] = {}
    for fy_end in fiscal_calendar:
        fiscal_years[fy_end] = FiscalYearInfo(statement, fy_end)

    for key in sorted(dataset.columns, key=lambda item: item.period_end):
        if key.period_end.year < 1900:
            continue
        if key.duration == "annual":
            fy_end = project_fy_end(key.period_end, fy_month, fy_day)
            info = fiscal_years.setdefault(fy_end, FiscalYearInfo(statement, fy_end))
            info.annual_key = key

    for key in sorted(dataset.columns, key=lambda item: item.period_end):
        if key.period_end.year < 1900:
            continue
        if key.duration == "annual":
            continue
        target = select_fiscal_year(key, fiscal_years, fy_month, fy_day)
        if target is None:
            fy_end = project_fy_end(key.period_end, fy_month, fy_day)
            target = fiscal_years.setdefault(fy_end, FiscalYearInfo(statement, fy_end))
        quarter_idx = infer_quarter_index(key, target.fy_end)
        if quarter_idx is None:
            diagnostics.append(
                {
                    "Statement": statement.value,
                    "Line Item": "",
                    "Issue": "quarter-index",
                    "Period": format_date(key.period_end),
                    "Detail": f"Unable to map duration {key.duration}",
                }
            )
            continue
        if key.duration == "ytd":
            target.ytd_keys.setdefault(quarter_idx, key)
        else:
            target.quarter_keys.setdefault(quarter_idx, key)

    ordered = [fiscal_years[fy_end] for fy_end in sorted(fiscal_years)]
    return ordered


def select_fiscal_year(
    key: PeriodKey,
    fiscal_years: Dict[date, FiscalYearInfo],
    fy_month: int,
    fy_day: int,
) -> Optional[FiscalYearInfo]:
    if key.period_end.year < 1900:
        return None
    projected_end = project_fy_end(key.period_end, fy_month, fy_day)
    candidates = [fy for fy in fiscal_years.values() if projected_end <= fy.fy_end]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.fy_end)
    chosen = candidates[0]
    if (chosen.fy_end - key.period_end).days > 400:
        return None
    return chosen


def infer_quarter_index(key: PeriodKey, fy_end: date) -> Optional[int]:
    if key.period_end.year < 1900:
        return None
    if key.duration == "annual":
        return None
    if key.duration == "ytd" and key.months:
        quarter = key.months // 3
        return quarter if 1 <= quarter <= 4 else None
    if key.months == 12:
        return 4
    delta_months = (fy_end.year - key.period_end.year) * 12 + (fy_end.month - key.period_end.month)
    if delta_months < 0:
        return None
    quarter = 4 - (delta_months // 3)
    if 1 <= quarter <= 4:
        return quarter
    return None


def project_fy_end(period_end: date, fy_month: int, fy_day: int) -> date:
    year = period_end.year
    if (period_end.month, period_end.day) > (fy_month, fy_day):
        year += 1
    last_day = calendar.monthrange(year, fy_month)[1]
    day = min(fy_day, last_day)
    return date(year, fy_month, day)


def shift_months(value: date, months: int) -> date:
    total_months = (value.year * 12 + value.month - 1) + months
    year = total_months // 12
    month = total_months % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    day = min(value.day, last_day)
    return date(year, month, day)


def compute_line_series(
    line: LineAccumulator,
    fiscal_years: List[FiscalYearInfo],
    statement: StatementType,
    diagnostics: List[Dict[str, str]],
) -> Dict[Tuple[str, int, Optional[int]], Optional[float]]:
    results: Dict[Tuple[str, int, Optional[int]], Optional[float]] = {}
    for fy_idx, fy in enumerate(fiscal_years):
        quarter_values: Dict[int, float] = {}
        ytd_values: Dict[int, float] = {}
        for quarter, key in fy.quarter_keys.items():
            entry = line.period_values.get(key)
            if entry and entry.value is not None:
                quarter_values[quarter] = entry.value
        for quarter, key in fy.ytd_keys.items():
            entry = line.period_values.get(key)
            if entry and entry.value is not None:
                ytd_values[quarter] = entry.value

        # Use YTD figures to back-fill quarters
        for quarter in range(1, 4):
            if quarter in quarter_values:
                continue
            ytd_val = ytd_values.get(quarter)
            if ytd_val is None:
                continue
            if quarter == 1:
                quarter_values[quarter] = ytd_val
            else:
                if all(idx in quarter_values for idx in range(1, quarter)):
                    subtotal = sum(quarter_values[idx] for idx in range(1, quarter))
                    quarter_values[quarter] = ytd_val - subtotal
                else:
                    diagnostics.append(
                        {
                            "Statement": statement.value,
                            "Line Item": line.display_label,
                            "Issue": "ytd-backfill",
                            "Period": fy.describe_quarter(quarter),
                            "Detail": "Missing prior quarters prevents YTD backfill",
                        }
                    )

        # Derive Q4 from YTD 12M or FY
        if 4 not in quarter_values:
            ytd_four = ytd_values.get(4)
            if ytd_four is not None and all(idx in quarter_values for idx in (1, 2, 3)):
                quarter_values[4] = ytd_four - sum(quarter_values[idx] for idx in (1, 2, 3))
        if 4 not in quarter_values and fy.annual_key:
            annual_entry = line.period_values.get(fy.annual_key)
            if annual_entry and annual_entry.value is not None and all(idx in quarter_values for idx in (1, 2, 3)):
                quarter_values[4] = annual_entry.value - sum(quarter_values[idx] for idx in (1, 2, 3))
        if 4 not in quarter_values and statement == StatementType.BALANCE:
            direct = fy.quarter_keys.get(4)
            entry = line.period_values.get(direct) if direct else None
            if entry and entry.value is not None:
                quarter_values[4] = entry.value

        annual_value: Optional[float] = None
        if fy.annual_key:
            annual_entry = line.period_values.get(fy.annual_key)
            if annual_entry and annual_entry.value is not None:
                annual_value = annual_entry.value
        if annual_value is None:
            ytd_total = ytd_values.get(4)
            if ytd_total is not None:
                annual_value = ytd_total
            elif all(idx in quarter_values for idx in (1, 2, 3, 4)):
                annual_value = sum(quarter_values[idx] for idx in (1, 2, 3, 4))
            elif statement == StatementType.BALANCE and 4 in quarter_values:
                annual_value = quarter_values[4]

        for quarter in range(1, 5):
            results[("quarter", fy_idx, quarter)] = quarter_values.get(quarter)
        results[("annual", fy_idx, None)] = annual_value

    return results


def compute_line_measurements(
    statement: StatementType,
    line: LineAccumulator,
    fiscal_years: List[FiscalYearInfo],
) -> Dict[str, Dict[date, Measurement]]:
    buckets: Dict[str, Dict[date, Measurement]] = {}

    def assign(measure: Measurement, overwrite: bool = False) -> None:
        key = _duration_key(measure.duration, measure.months)
        dest = buckets.setdefault(key, {})
        existing = dest.get(measure.period_end)
        if existing is None or overwrite:
            dest[measure.period_end] = measure

    for key, entry in line.period_values.items():
        if entry.value is None or key.period_end.year < 1900:
            continue
        duration_norm = (key.duration or "").lower()
        months_norm = key.months
        if statement == StatementType.BALANCE:
            duration_norm = "point"
        measure = Measurement(
            period_end=key.period_end,
            duration=duration_norm or "unknown",
            months=months_norm,
            value=float(entry.value),
            source_pdf=entry.source_pdf,
            derived=False,
        )
        assign(measure)

    if statement == StatementType.BALANCE:
        return buckets

    def can_diff(current: Measurement, previous: Measurement) -> bool:
        if current.period_end <= previous.period_end:
            return False
        if current.months is not None and previous.months is not None:
            return current.months - previous.months == 3
        return _months_between(current.period_end, previous.period_end) == 3

    for fy in fiscal_years:
        quarter_values: Dict[int, Measurement] = {}
        ytd_values: Dict[int, Measurement] = {}

        for quarter, key in fy.quarter_keys.items():
            entry = line.period_values.get(key)
            if entry and entry.value is not None and key.period_end.year >= 1900:
                meas = Measurement(
                    period_end=key.period_end,
                    duration="quarter",
                    months=key.months or 3,
                    value=float(entry.value),
                    source_pdf=entry.source_pdf,
                    derived=False,
                )
                quarter_values[quarter] = meas
                assign(meas, overwrite=True)

        for quarter, key in fy.ytd_keys.items():
            entry = line.period_values.get(key)
            if entry and entry.value is not None and key.period_end.year >= 1900:
                meas = Measurement(
                    period_end=key.period_end,
                    duration="ytd",
                    months=key.months,
                    value=float(entry.value),
                    source_pdf=entry.source_pdf,
                    derived=False,
                )
                ytd_values[quarter] = meas
                assign(meas)

        for quarter in range(1, 4):
            if quarter in quarter_values:
                continue
            ytd_measure = ytd_values.get(quarter)
            if ytd_measure is None:
                continue
            period_end = fy.quarter_date(quarter) or ytd_measure.period_end
            derived_value: Optional[float] = None
            if quarter == 1:
                if ytd_measure.months in (None, 3):
                    derived_value = ytd_measure.value
            else:
                prev_ytd = ytd_values.get(quarter - 1)
                if prev_ytd and can_diff(ytd_measure, prev_ytd):
                    derived_value = ytd_measure.value - prev_ytd.value
            if derived_value is None and quarter > 1 and all(
                q in quarter_values for q in range(1, quarter)
            ):
                derived_value = ytd_measure.value - sum(
                    quarter_values[q].value for q in range(1, quarter)
                )
            if derived_value is None:
                continue
            derived_measure = Measurement(
                period_end=period_end,
                duration="quarter",
                months=3,
                value=derived_value,
                source_pdf=ytd_measure.source_pdf,
                derived=True,
            )
            quarter_values[quarter] = derived_measure
            assign(derived_measure, overwrite=True)

        if 4 not in quarter_values:
            derived_measure: Optional[Measurement] = None
            ytd_four = ytd_values.get(4)
            prev_ytd = ytd_values.get(3)
            if ytd_four and prev_ytd and can_diff(ytd_four, prev_ytd):
                period_end = fy.quarter_date(4) or ytd_four.period_end
                derived_measure = Measurement(
                    period_end=period_end,
                    duration="quarter",
                    months=3,
                    value=ytd_four.value - prev_ytd.value,
                    source_pdf=ytd_four.source_pdf,
                    derived=True,
                )
            if derived_measure is None and fy.annual_key:
                entry = line.period_values.get(fy.annual_key)
                if entry and entry.value is not None:
                    period_end = fy.annual_key.period_end
                    if prev_ytd:
                        derived_value = float(entry.value) - prev_ytd.value
                    elif all(q in quarter_values for q in (1, 2, 3)):
                        derived_value = float(entry.value) - sum(
                            quarter_values[q].value for q in (1, 2, 3)
                        )
                    else:
                        derived_value = None
                    if derived_value is not None:
                        derived_measure = Measurement(
                            period_end=period_end,
                            duration="quarter",
                            months=3,
                            value=derived_value,
                            source_pdf=entry.source_pdf,
                            derived=True,
                        )
            if (
                derived_measure is None
                and ytd_four
                and all(q in quarter_values for q in (1, 2, 3))
            ):
                period_end = fy.quarter_date(4) or ytd_four.period_end
                derived_measure = Measurement(
                    period_end=period_end,
                    duration="quarter",
                    months=3,
                    value=ytd_four.value - sum(
                        quarter_values[q].value for q in (1, 2, 3)
                    ),
                    source_pdf=ytd_four.source_pdf,
                    derived=True,
                )
            if derived_measure is not None:
                quarter_values[4] = derived_measure
                assign(derived_measure, overwrite=True)

        annual_measure: Optional[Measurement] = None
        if fy.annual_key:
            entry = line.period_values.get(fy.annual_key)
            if entry and entry.value is not None and fy.annual_key.period_end.year >= 1900:
                annual_measure = Measurement(
                    period_end=fy.annual_key.period_end,
                    duration="annual",
                    months=fy.annual_key.months or 12,
                    value=float(entry.value),
                    source_pdf=entry.source_pdf,
                    derived=False,
                )
        if annual_measure is None:
            ytd_four = ytd_values.get(4)
            if ytd_four and ytd_four.months:
                annual_measure = Measurement(
                    period_end=ytd_four.period_end,
                    duration="annual",
                    months=12,
                    value=ytd_four.value,
                    source_pdf=ytd_four.source_pdf,
                    derived=True,
                )
            elif all(q in quarter_values for q in (1, 2, 3, 4)):
                annual_value = sum(quarter_values[q].value for q in range(1, 5))
                annual_measure = Measurement(
                    period_end=fy.fy_end,
                    duration="annual",
                    months=12,
                    value=annual_value,
                    source_pdf=quarter_values[4].source_pdf,
                    derived=True,
                )
        if annual_measure is not None:
            assign(annual_measure, overwrite=True)

    return buckets
def parse_period_end(candidate: str) -> Optional[date]:
    if not candidate:
        return None
    candidate = candidate.strip()
    if not candidate:
        return None
    try:
        return date.fromisoformat(candidate)
    except ValueError:
        pass
    normalized = " ".join(candidate.replace(",", ", ").split())
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%B %Y", "%b %Y"):
        try:
            return datetime.strptime(normalized, fmt).date()
        except ValueError:
            continue
    if candidate.isdigit() and len(candidate) == 4:
        return date(int(candidate), 12, 31)
    return None


def infer_months(raw_label: str, duration: str) -> Optional[int]:
    lowered = (raw_label or "").lower()
    if duration == "quarter":
        return 3
    if duration == "annual":
        return 12
    if "three months" in lowered:
        return 3
    if "six months" in lowered:
        return 6
    if "nine months" in lowered:
        return 9
    if "twelve months" in lowered or "twelve-month" in lowered:
        return 12
    return None


def format_date(value: date) -> str:
    return value.strftime("%B %d, %Y")


def coerce_numeric(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (value != value):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    cleaned = text.replace(",", "")
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = f"-{cleaned[1:-1]}"
    try:
        return float(cleaned)
    except ValueError:
        return None
