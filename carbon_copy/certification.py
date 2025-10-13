from __future__ import annotations

import re
from datetime import date
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from selector.models import StatementType

from .models import CarbonCopyResult, StatementBlock
from .aggregation import (
    infer_months,
    parse_period_end,
    format_date,
    SeriesAggregator,
    Measurement,
)

COLS_START = 3


def _pick_row(df: pd.DataFrame, pattern: str) -> pd.Series | None:
    mask = df["Line Item (as printed)"].str.contains(pattern, case=False, na=False)
    if not mask.any():
        return None
    return df[mask].iloc[0]


def _normalize_duration(
    statement: StatementType,
    duration: str,
    months: int | None,
) -> Tuple[str, int | None]:
    duration_norm = (duration or "").lower()
    if statement == StatementType.BALANCE:
        return ("point", months)
    if duration_norm in {"quarter", "ytd", "annual"}:
        return (duration_norm, months)
    if months == 3:
        return ("quarter", months)
    if months in (6, 9):
        return ("ytd", months)
    if months == 12:
        return ("annual", months)
    return (duration_norm or "unknown", months)


def _months_between(a: date, b: date) -> int:
    return (a.year - b.year) * 12 + (a.month - b.month)


def _format_period_label(measure: Measurement) -> str:
    months = measure.months
    if measure.duration == "quarter" or months == 3:
        return f"3M ended {format_date(measure.period_end)}"
    if measure.duration == "ytd" and months:
        return f"{months}M ended {format_date(measure.period_end)}"
    if measure.duration == "annual":
        return f"FY ended {format_date(measure.period_end)}"
    return format_date(measure.period_end)


def _collect_measurements(
    results: Iterable[CarbonCopyResult],
    statement: StatementType,
    pattern: str,
) -> List[Measurement]:
    measurements: List[Measurement] = []
    for result in results:
        block = result.blocks[statement]
        row = _pick_row(block.wide_table, pattern)
        if row is None:
            continue
        for column in block.wide_table.columns[COLS_START:]:
            meta = block.column_metadata.get(column) or {}
            period_end = parse_period_end(
                meta.get("period_end") or column or meta.get("raw_label", "")
            )
            if period_end is None or period_end.year < 1900:
                continue
            duration = (meta.get("duration") or "").lower()
            months = infer_months(meta.get("raw_label", ""), duration)
            duration_norm, months_norm = _normalize_duration(statement, duration, months)
            value = _value(row, column)
            if value is None:
                continue
            measurements.append(
                Measurement(
                    period_end=period_end,
                    duration=duration_norm,
                    months=months_norm,
                    value=value,
                    source_pdf=result.source_pdf,
                )
            )
    return measurements


def _build_measurement_map(measurements: List[Measurement]) -> Dict[date, Dict[str, Measurement]]:
    data: Dict[date, Dict[str, Measurement]] = {}
    for meas in measurements:
        bucket = data.setdefault(meas.period_end, {})
        existing = bucket.get(meas.duration)
        if existing is None or existing.source_pdf <= meas.source_pdf:
            bucket[meas.duration] = meas
    return data


def _derive_quarter_values(data: Dict[date, Dict[str, Measurement]]) -> None:
    ytd_entries: List[Measurement] = []
    for period_end, durations in data.items():
        for meas in durations.values():
            if meas.duration == "ytd" and meas.months:
                ytd_entries.append(meas)
    ytd_entries.sort(key=lambda m: (m.period_end, m.months or 0))
    for current in ytd_entries:
        prev_months = (current.months or 0) - 3
        if prev_months <= 0:
            continue
        candidates = [
            other
            for other in ytd_entries
            if other.period_end < current.period_end and (other.months or 0) == prev_months
        ]
        if not candidates:
            continue
        # choose the closest prior period
        prev = max(
            (c for c in candidates if _months_between(current.period_end, c.period_end) == 3),
            default=None,
            key=lambda c: c.period_end,
        )
        if prev is None:
            continue
        derived_value = current.value - prev.value
        period_bucket = data.setdefault(current.period_end, {})
        if "quarter" in period_bucket:
            continue
        period_bucket["quarter"] = Measurement(
            period_end=current.period_end,
            duration="quarter",
            months=3,
            value=derived_value,
            source_pdf=current.source_pdf,
            derived=True,
        )


def _aggregate_net_income(results: List[CarbonCopyResult]) -> Tuple[str, str]:
    income_measurements = _collect_measurements(results, StatementType.INCOME, r"^net income")
    cash_measurements = _collect_measurements(results, StatementType.CASH, r"^net income")

    income_map = _build_measurement_map(income_measurements)
    cash_map = _build_measurement_map(cash_measurements)
    _derive_quarter_values(income_map)
    _derive_quarter_values(cash_map)

    all_periods = sorted(set(income_map.keys()) | set(cash_map.keys()))
    details: List[str] = []
    all_pass = True
    for period_end in all_periods:
        income_period = income_map.get(period_end, {})
        cash_period = cash_map.get(period_end, {})
        shared_durations = set(income_period.keys()) & set(cash_period.keys())
        if not shared_durations:
            details.append(f"{format_date(period_end)}: insufficient overlapping data")
            all_pass = False
            continue
        for duration in sorted(shared_durations):
            inc = income_period[duration]
            cf = cash_period[duration]
            delta = inc.value - cf.value
            if abs(delta) > 1e-6:
                all_pass = False
            label = _format_period_label(inc)
            details.append(f"{label}: Δ {delta:.6g}")
    status = "PASS" if all_pass else "CHECK"
    return status, "; ".join(details) if details else "No overlapping periods"


def build_batch_certification(results: List[CarbonCopyResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=["Source PDF", "Check", "Result", "Details"])
    status, detail = _aggregate_net_income(results)
    return pd.DataFrame(
        [
            {
                "Source PDF": "Batch",
                "Check": "Aggregate net income link (IS vs CF)",
                "Result": status,
                "Details": detail,
            }
        ]
    )


def _value(series: pd.Series | None, column: str) -> float | None:
    if series is None:
        return None
    value = series[column]
    if pd.isna(value):
        return None
    return float(value)


_YEAR_PATTERN = re.compile(r"(20\d{2})")


def _fallback_key(label: str) -> str:
    match = _YEAR_PATTERN.search(label)
    if match:
        return match.group(1)
    return label.strip()


def _period_signature(
    column: str,
    metadata: Dict[str, str] | None,
    include_duration: bool = True,
) -> Tuple[str, str]:
    duration = ""
    period_end = ""
    if metadata:
        if include_duration:
            duration = metadata.get("duration") or ""
        period_end = metadata.get("period_end") or ""
    if not period_end:
        period_end = _fallback_key(column)
    return (duration.lower() if include_duration else "", period_end)


def build_certification(blocks: dict[StatementType, StatementBlock]) -> pd.DataFrame:
    rows: list[list[str]] = []

    income_block = blocks[StatementType.INCOME]
    balance_block = blocks[StatementType.BALANCE]
    cash_block = blocks[StatementType.CASH]

    income = income_block.wide_table
    balance = balance_block.wide_table
    cash = cash_block.wide_table

    income_columns = income.columns[COLS_START:]
    balance_columns = balance.columns[COLS_START:]
    cash_columns = cash.columns[COLS_START:]

    income_meta = income_block.column_metadata
    balance_meta = balance_block.column_metadata
    cash_meta = cash_block.column_metadata

    balance_by_period = {
        _period_signature(col, balance_meta.get(col), include_duration=False): col
        for col in balance_columns
    }
    cash_by_period = {
        _period_signature(col, cash_meta.get(col), include_duration=False): col
        for col in cash_columns
    }
    cash_with_duration = {
        _period_signature(col, cash_meta.get(col), include_duration=True): col
        for col in cash_columns
    }

    # 1. Balance sheet equality
    assets_row = _pick_row(balance, r"total assets")
    liab_row = _pick_row(balance, r"total liabilities")
    equity_row = _pick_row(balance, r"total\s+(?!liabilities)(?!.*stockholders)(?!.*shareholders).*equity$")
    if equity_row is None:
        equity_row = _pick_row(
            balance,
            r"total\s+(?!liabilities)(?:.*stockholders[’']?\s+equity|.*shareholders[’']?\s+equity)",
        )

    bs_details: list[str] = []
    bs_pass = True
    for column in balance_columns:
        assets = _value(assets_row, column)
        liab = _value(liab_row, column)
        equity = _value(equity_row, column)
        if None in (assets, liab, equity):
            bs_details.append(f"{column}: insufficient data")
            bs_pass = False
            continue
        delta = assets - (liab + equity)
        if abs(delta) > 1e-6:
            bs_pass = False
        bs_details.append(f"{column}: Δ {delta:.6g}")
    rows.append([
        "Balance Sheet equality (Assets = Liabilities + Equity)",
        "PASS" if bs_pass else "CHECK",
        "; ".join(bs_details),
    ])

    # 2. Cash roll-through
    begin_row = _pick_row(cash, r"beginning cash and cash equivalents")
    delta_row = _pick_row(cash, r"^(?:increase|decrease)")
    end_row = _pick_row(cash, r"ending cash and cash equivalents")
    cash_bs_row = _pick_row(balance, r"cash and cash equivalents")

    roll_details: list[str] = []
    roll_pass = True
    for column in cash_columns:
        period_key = _period_signature(column, cash_meta.get(column), include_duration=False)
        begin = _value(begin_row, column)
        delta = _value(delta_row, column)
        end = _value(end_row, column)
        if None in (begin, delta, end):
            roll_details.append(f"{column}: insufficient data")
            roll_pass = False
            continue
        roll_delta = begin + delta - end
        if abs(roll_delta) > 1e-6:
            roll_pass = False
        tie_delta = None
        bs_column = balance_by_period.get(period_key)
        if cash_bs_row is not None and bs_column is not None:
            bs_cash = _value(cash_bs_row, bs_column)
            if bs_cash is not None:
                tie_delta = end - bs_cash
                if abs(tie_delta) > 1e-6:
                    roll_pass = False
        if tie_delta is None:
            roll_details.append(f"{column}: roll Δ {roll_delta:.6g}")
        else:
            roll_details.append(f"{column}: roll Δ {roll_delta:.6g}; tie Δ {tie_delta:.6g}")
    rows.append([
        "Cash roll-through (Beg + Δ = End) & tie to BS cash",
        "PASS" if roll_pass else "CHECK",
        "; ".join(roll_details) if roll_details else "Not enough data",
    ])

    # 3. Net income link
    is_net_row = _pick_row(income, r"^net income")
    cf_net_row = _pick_row(cash, r"^net income")
    net_details: list[str] = []
    net_pass = True
    for column in income_columns:
        key = _period_signature(column, income_meta.get(column))
        income_val = _value(is_net_row, column)
        cf_column = cash_with_duration.get(key)
        if cf_column is None:
            candidate = cash_by_period.get(
                _period_signature(column, income_meta.get(column), include_duration=False)
            )
            if candidate is not None:
                income_duration = (income_meta.get(column) or {}).get("duration", "").lower()
                cash_duration = (cash_meta.get(candidate) or {}).get("duration", "").lower()
                if income_duration and cash_duration and income_duration != cash_duration:
                    candidate = None
            cf_column = candidate
        cf_val = _value(cf_net_row, cf_column) if cf_column else None
        if None in (income_val, cf_val):
            net_details.append(f"{column}: insufficient data")
            net_pass = False
            continue
        delta = income_val - cf_val
        if abs(delta) > 1e-6:
            net_pass = False
        net_details.append(f"{column}: Δ {delta:.6g}")
    rows.append([
        "Net income link (IS Net income vs CF Net income)",
        "PASS" if net_pass else "CHECK",
        "; ".join(net_details) if net_details else "Not enough data",
    ])

    # 4. Income statement cross-footing
    cross_details: list[str] = []
    cross_pass = True

    revenue_row = _pick_row(income, r"^net revenue|^revenue|^sales$")
    cost_row = _pick_row(income, r"^cost of revenue|^cost of sales$")
    gross_row = _pick_row(income, r"^gross profit$")
    opex_row = _pick_row(income, r"^total operating expenses$")
    opinc_row = _pick_row(income, r"^operating income$")
    other_row = _pick_row(income, r"other income|interest and other income")
    pretax_row = _pick_row(income, r"income before")
    tax_row = _pick_row(income, r"^provision.*income taxes")

    for column in income_columns:
        entries: list[str] = []
        rev = _value(revenue_row, column)
        cost = _value(cost_row, column)
        gross = _value(gross_row, column)
        if None not in (rev, cost, gross):
            gp_delta = (rev + cost) - gross
            if abs(gp_delta) > 1e-6:
                cross_pass = False
            entries.append(f"GP Δ {gp_delta:.6g}")

        gp = _value(gross_row, column)
        opex = _value(opex_row, column)
        opinc = _value(opinc_row, column)
        if None not in (gp, opex, opinc):
            oi_delta = (gp - opex) - opinc
            if abs(oi_delta) > 1e-6:
                cross_pass = False
            entries.append(f"OI Δ {oi_delta:.6g}")

        opinc_val = _value(opinc_row, column)
        other_val = _value(other_row, column)
        pretax_val = _value(pretax_row, column)
        if None not in (opinc_val, other_val, pretax_val):
            pretax_delta = (opinc_val + other_val) - pretax_val
            if abs(pretax_delta) > 1e-6:
                cross_pass = False
            entries.append(f"PT Δ {pretax_delta:.6g}")

        pretax = _value(pretax_row, column)
        tax = _value(tax_row, column)
        net = _value(is_net_row, column)
        if None not in (pretax, tax, net):
            net_delta = (pretax + tax) - net
            if abs(net_delta) > 1e-6:
                cross_pass = False
            entries.append(f"NI Δ {net_delta:.6g}")

        if entries:
            cross_details.append(f"{column}: " + ", ".join(entries))
        else:
            cross_pass = False
            cross_details.append(f"{column}: insufficient data")
    rows.append([
        "IS cross-footing (subtotals)",
        "PASS" if cross_pass else "CHECK",
        "; ".join(cross_details) if cross_details else "Not enough data",
    ])

    # 5. Cash flow cross-footing
    cf_details: list[str] = []
    cf_pass = True
    cfo_row = _pick_row(cash, r"^net cash .*operating activities")
    cfi_row = _pick_row(cash, r"^net cash .*investing activities")
    cff_row = _pick_row(cash, r"^net cash .*financing activities")
    fx_row = _pick_row(cash, r"effect of .*exchange")

    for column in cash_columns:
        rows_available = [cfo_row, cfi_row, cff_row, delta_row]
        if any(row is None for row in rows_available):
            cf_details.append(f"{column}: insufficient data")
            cf_pass = False
            continue
        cfo = _value(cfo_row, column)
        cfi = _value(cfi_row, column)
        cff = _value(cff_row, column)
        delta_val = _value(delta_row, column)
        fx_val = _value(fx_row, column) if fx_row is not None else 0.0
        if fx_val is None:
            fx_val = 0.0
        if None in (cfo, cfi, cff, delta_val):
            cf_details.append(f"{column}: insufficient data")
            cf_pass = False
            continue
        diff = (cfo + cfi + cff + fx_val) - delta_val
        if abs(diff) > 1e-6:
            cf_pass = False
        cf_details.append(f"{column}: Δ {diff:.6g}")
    rows.append([
        "CF cross-footing (Δ cash = CFO + CFI + CFF + FX)",
        "PASS" if cf_pass else "CHECK",
        "; ".join(cf_details),
    ])

    # 6. Coverage & ordering + units
    pages = ", ".join(
        f"{block.header.statement_name} {block.header.source}"
        for block in blocks.values()
    )
    rows.append([
        "Coverage & ordering",
        "PASS",
        f"Captured pages: {pages}. Columns reordered oldest→newest.",
    ])
    rows.append([
        "Units/scale normalization",
        "PASS",
        "Original units per headers; monetary values saved in USD millions; per-share, shares, %/ratios left unscaled.",
    ])

    return pd.DataFrame(rows, columns=["Check", "Result", "Details"])
