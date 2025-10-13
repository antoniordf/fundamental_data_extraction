from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from selector.models import StatementType

from .aggregation import Measurement, SeriesAggregator, shift_months, format_date
from .models import CarbonCopyResult


def build_batch_certification(results: List[CarbonCopyResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=["Source PDF", "Check", "Result", "Details"])

    aggregator = SeriesAggregator()
    for result in results:
        aggregator.add_result(result)

    measurement_store = aggregator.build_measurement_store()

    rows: List[Dict[str, str]] = []
    rows.append(_batch_balance_sheet_check(measurement_store))
    rows.append(_batch_cash_roll_check(measurement_store))
    rows.append(_batch_net_income_check(measurement_store))
    rows.append(_batch_is_cross_check(measurement_store))
    rows.append(_batch_cf_cross_check(measurement_store))

    rows = [row for row in rows if row]
    return pd.DataFrame(rows, columns=["Source PDF", "Check", "Result", "Details"])


# --- helpers -----------------------------------------------------------------

DurationMap = Dict[str, Dict[date, Measurement]]
StatementStore = Dict[StatementType, Dict[str, DurationMap]]


def _line_score(durations: DurationMap) -> int:
    quarter_count = len(durations.get("quarter", {}))
    total = sum(len(values) for values in durations.values())
    return quarter_count * 10 + total


def _find_line(store: Dict[str, DurationMap], patterns: Iterable[str]) -> DurationMap:
    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    best: Tuple[int, DurationMap] | None = None
    for label, durations in store.items():
        if any(pattern.search(label) for pattern in compiled):
            score = _line_score(durations)
            if best is None or score > best[0]:
                best = (score, durations)
    return best[1] if best else {}


def _extract_duration(
    durations: DurationMap,
    keys: Iterable[str],
) -> Dict[date, Measurement]:
    collector: Dict[date, Measurement] = {}
    for key in keys:
        data = durations.get(key, {})
        for period_end, measure in data.items():
            existing = collector.get(period_end)
            if existing is None or (existing.derived and not measure.derived):
                collector[period_end] = measure
    return collector


def _format_period(period_end: date) -> str:
    return period_end.strftime("%B %d, %Y")


def _collect_durations(
    store: Dict[str, DurationMap],
    patterns: Iterable[str],
) -> Dict[str, Dict[date, Measurement]]:
    durations = _find_line(store, patterns)
    result: Dict[str, Dict[date, Measurement]] = {}
    for key, values in durations.items():
        if not values or key == "unknown":
            continue
        result[key] = dict(values)
    return result


def _describe_measure(measure: Measurement) -> str:
    if measure.duration == "quarter" or measure.months == 3:
        return f"3M ended {format_date(measure.period_end)}"
    if measure.duration == "annual" or measure.months == 12:
        return f"FY ended {format_date(measure.period_end)}"
    if measure.months:
        return f"{measure.months}M ended {format_date(measure.period_end)}"
    return format_date(measure.period_end)


def _format_delta(label: str, delta: float, derived_flags: Iterable[bool]) -> str:
    suffix = " (derived)" if any(derived_flags) else ""
    prefix = f"{label} " if label else ""
    return f"{prefix}Δ {delta:.6g}{suffix}"


def _lookup_point(points: Dict[date, Measurement], target: date) -> Measurement | None:
    if target in points:
        return points[target]
    for offset in (-1, 1, -2, 2, -3, 3):
        candidate = target + timedelta(days=offset)
        if candidate in points:
            return points[candidate]
    return None


def _describe_period(duration_key: str, period_end: date, sample: Measurement | None = None) -> str:
    if sample is not None:
        return _describe_measure(sample)
    if duration_key == "quarter":
        return f"3M ended {format_date(period_end)}"
    if duration_key == "annual":
        return f"FY ended {format_date(period_end)}"
    if duration_key.startswith("ytd"):
        try:
            months = int("".join(ch for ch in duration_key if ch.isdigit()))
        except ValueError:
            months = None
        if months:
            return f"{months}M ended {format_date(period_end)}"
    return format_date(period_end)


def _batch_balance_sheet_check(store: StatementStore) -> Dict[str, str] | None:
    balance_store = store.get(StatementType.BALANCE)
    if not balance_store:
        return _batch_row("Balance Sheet equality (Assets = Liabilities + Equity)", "CHECK", "No balance sheet data")

    assets = _extract_duration(_find_line(balance_store, [r"total assets"]), ["point"])
    liabilities = _extract_duration(_find_line(balance_store, [r"total liabilities"]), ["point"])
    equity = _extract_duration(
        _find_line(balance_store, [r"total stockholders", r"total shareholders"]),
        ["point"],
    )

    all_periods = sorted(set(assets.keys()) | set(liabilities.keys()) | set(equity.keys()))
    if not all_periods:
        return _batch_row("Balance Sheet equality (Assets = Liabilities + Equity)", "CHECK", "No balance sheet data")

    details: List[str] = []
    passed = True
    for period_end in all_periods:
        asset_val = assets.get(period_end)
        liab_val = liabilities.get(period_end)
        equity_val = equity.get(period_end)
        if None in (asset_val, liab_val, equity_val):
            details.append(f"{_format_period(period_end)}: insufficient data")
            passed = False
            continue
        delta = asset_val.value - (liab_val.value + equity_val.value)
        if abs(delta) > 1e-6:
            passed = False
        details.append(f"{_format_period(period_end)}: Δ {delta:.6g}")

    return _batch_row(
        "Balance Sheet equality (Assets = Liabilities + Equity)",
        "PASS" if passed else "CHECK",
        "; ".join(details),
    )


def _batch_cash_roll_check(store: StatementStore) -> Dict[str, str] | None:
    cash_store = store.get(StatementType.CASH)
    balance_store = store.get(StatementType.BALANCE)
    if not cash_store or not balance_store:
        return _batch_row("Cash roll-through (Beg + Δ = End) & tie to BS cash", "CHECK", "Missing statements")

    begin_map = _collect_durations(
        cash_store,
        [r"beginning cash and cash equivalents"],
    )
    delta_map = _collect_durations(
        cash_store,
        [r"^(?:increase|decrease).*cash and cash equivalents"],
    )
    end_map = _collect_durations(
        cash_store,
        [r"ending cash and cash equivalents"],
    )
    cash_points = _extract_duration(
        _find_line(balance_store, [r"cash and cash equivalents"]),
        ["point"],
    )

    if not begin_map or not delta_map or not end_map or not cash_points:
        return _batch_row(
            "Cash roll-through (Beg + Δ = End) & tie to BS cash",
            "CHECK",
            "Insufficient data",
        )

    duration_keys = sorted(set(begin_map.keys()) & set(delta_map.keys()) & set(end_map.keys()))
    details: List[str] = []
    passed = True
    comparisons = 0
    for key in duration_keys:
        begin_periods = begin_map.get(key, {})
        delta_periods = delta_map.get(key, {})
        end_periods = end_map.get(key, {})
        shared_periods = sorted(
            set(begin_periods.keys()) & set(delta_periods.keys()) & set(end_periods.keys())
        )
        if not shared_periods:
            label = key.upper() if key != "quarter" else "Quarter"
            details.append(f"{label}: insufficient overlapping data")
            passed = False
            continue
        for period_end in shared_periods:
            begin = begin_periods[period_end]
            delta = delta_periods[period_end]
            end = end_periods[period_end]
            roll_delta = begin.value + delta.value - end.value
            comparisons += 1
            messages: List[str] = [_format_delta("roll", roll_delta, [begin.derived, delta.derived, end.derived])]
            if abs(roll_delta) > 1e-6:
                passed = False

            end_cash = cash_points.get(period_end)
            if end_cash is None:
                passed = False
                messages.append("missing BS cash at end")
            else:
                tie_delta = end.value - end_cash.value
                messages.append(_format_delta("end tie", tie_delta, [end.derived, end_cash.derived]))
                if abs(tie_delta) > 1e-6:
                    passed = False

            if begin.months:
                start_period = shift_months(period_end, -(begin.months or 0))
                start_cash = _lookup_point(cash_points, start_period)
                if start_cash is None:
                    passed = False
                    messages.append("missing BS cash at start")
                else:
                    start_delta = begin.value - start_cash.value
                    messages.append(_format_delta("start tie", start_delta, [begin.derived, start_cash.derived]))
                    if abs(start_delta) > 1e-6:
                        passed = False

            details.append(f"{_describe_measure(end)}: " + "; ".join(messages))

    if comparisons == 0:
        return _batch_row(
            "Cash roll-through (Beg + Δ = End) & tie to BS cash",
            "CHECK",
            "No overlapping duration data",
        )

    return _batch_row(
        "Cash roll-through (Beg + Δ = End) & tie to BS cash",
        "PASS" if passed else "CHECK",
        "; ".join(details),
    )


def _batch_net_income_check(store: StatementStore) -> Dict[str, str] | None:
    income_store = store.get(StatementType.INCOME)
    cash_store = store.get(StatementType.CASH)
    if not income_store or not cash_store:
        return _batch_row("Net income link (IS Net income vs CF Net income)", "CHECK", "Missing statements")

    income_map = _collect_durations(income_store, [r"^net income"])
    cash_map = _collect_durations(cash_store, [r"^net income"])

    if not income_map or not cash_map:
        return _batch_row("Net income link (IS Net income vs CF Net income)", "CHECK", "No overlapping duration data")

    relevant_keys = [
        key
        for key in sorted(set(income_map.keys()) & set(cash_map.keys()))
        if key == "quarter" or key == "annual" or key.startswith("ytd")
    ]
    if not relevant_keys:
        return _batch_row(
            "Net income link (IS Net income vs CF Net income)",
            "CHECK",
            "No overlapping duration data",
        )

    details: List[str] = []
    passed = True
    comparisons = 0
    for key in relevant_keys:
        income_periods = income_map.get(key, {})
        cash_periods = cash_map.get(key, {})
        shared_periods = sorted(set(income_periods.keys()) & set(cash_periods.keys()))
        if not shared_periods:
            label = key.upper() if key != "quarter" else "Quarter"
            details.append(f"{label}: insufficient overlapping data")
            passed = False
            continue
        for period_end in shared_periods:
            inc = income_periods[period_end]
            cf = cash_periods[period_end]
            delta = inc.value - cf.value
            comparisons += 1
            if abs(delta) > 1e-6:
                passed = False
            details.append(f"{_describe_measure(inc)}: {_format_delta('', delta, [inc.derived, cf.derived])}")

    if comparisons == 0:
        return _batch_row(
            "Net income link (IS Net income vs CF Net income)",
            "CHECK",
            "No overlapping duration data",
        )

    return _batch_row(
        "Net income link (IS Net income vs CF Net income)",
        "PASS" if passed else "CHECK",
        "; ".join(details),
    )


def _batch_is_cross_check(store: StatementStore) -> Dict[str, str] | None:
    income_store = store.get(StatementType.INCOME)
    if not income_store:
        return _batch_row("IS cross-footing (subtotals)", "CHECK", "Missing income statement")

    revenue_map = _collect_durations(income_store, [r"^net revenue", r"^revenue", r"^sales$"])
    cost_map = _collect_durations(income_store, [r"^cost of revenue", r"^cost of sales"])
    gross_map = _collect_durations(income_store, [r"^gross profit"])
    opex_map = _collect_durations(income_store, [r"^total operating expenses"])
    opinc_map = _collect_durations(income_store, [r"^operating income"])
    other_map = _collect_durations(income_store, [r"other income", r"interest and other income"])
    pretax_map = _collect_durations(income_store, [r"income before"])
    tax_map = _collect_durations(income_store, [r"^provision", r"^benefit", r"income taxes"])
    net_map = _collect_durations(income_store, [r"^net income"])

    if not any(
        mapping
        for mapping in (
            revenue_map,
            cost_map,
            gross_map,
            opex_map,
            opinc_map,
            other_map,
            pretax_map,
            tax_map,
            net_map,
        )
    ):
        return _batch_row("IS cross-footing (subtotals)", "CHECK", "Insufficient income statement data")

    entries: Dict[Tuple[str, date], List[str]] = {}
    samples: Dict[Tuple[str, date], Measurement] = {}
    passed = True

    def process(tag: str, mappings: List[Dict[str, Dict[date, Measurement]]], calculator) -> None:
        nonlocal passed
        duration_keys = sorted(set().union(*(mapping.keys() for mapping in mappings)))
        for duration in duration_keys:
            period_set: set[date] = set()
            for mapping in mappings:
                period_set |= set(mapping.get(duration, {}).keys())
            for period_end in sorted(period_set):
                key = (duration, period_end)
                measures = [mapping.get(duration, {}).get(period_end) for mapping in mappings]
                sample = next((measure for measure in measures if measure is not None), None)
                if sample is not None and key not in samples:
                    samples[key] = sample
                bucket = entries.setdefault(key, [])
                if any(measure is None for measure in measures):
                    bucket.append(f"insufficient data ({tag})")
                    passed = False
                    continue
                delta = calculator(*measures)
                if abs(delta) > 1e-6:
                    passed = False
                bucket.append(_format_delta(tag, delta, [m.derived for m in measures]))

    process(
        "GP",
        [revenue_map, cost_map, gross_map],
        lambda rev, cst, grp: (rev.value + cst.value) - grp.value,
    )
    process(
        "OI",
        [gross_map, opex_map, opinc_map],
        lambda grp, opex, op_inc: (grp.value - opex.value) - op_inc.value,
    )
    process(
        "PT",
        [opinc_map, other_map, pretax_map],
        lambda op_inc, other, pre: (op_inc.value + other.value) - pre.value,
    )
    process(
        "NI",
        [pretax_map, tax_map, net_map],
        lambda pre, tax, net_meas: (pre.value + tax.value) - net_meas.value,
    )

    if not entries:
        return _batch_row("IS cross-footing (subtotals)", "CHECK", "Insufficient income statement data")

    details = []
    for duration, period_end in sorted(entries.keys(), key=lambda item: (item[1], item[0])):
        sample = samples.get((duration, period_end))
        label = _describe_period(duration, period_end, sample)
        details.append(f"{label}: " + ", ".join(entries[(duration, period_end)]))

    return _batch_row(
        "IS cross-footing (subtotals)",
        "PASS" if passed else "CHECK",
        "; ".join(details),
    )


def _batch_cf_cross_check(store: StatementStore) -> Dict[str, str] | None:
    cash_store = store.get(StatementType.CASH)
    if not cash_store:
        return _batch_row("CF cross-footing (Δ cash = CFO + CFI + CFF + FX)", "CHECK", "Missing cash flow statement")

    cfo_map = _collect_durations(cash_store, [r"^net cash .*operating activities"])
    cfi_map = _collect_durations(cash_store, [r"^net cash .*investing activities"])
    cff_map = _collect_durations(cash_store, [r"^net cash .*financing activities"])
    fx_map = _collect_durations(cash_store, [r"effect of .*exchange"])
    delta_map = _collect_durations(
        cash_store,
        [r"increase", r"decrease in cash and cash equivalents"],
    )

    if not delta_map or not cfo_map or not cfi_map or not cff_map:
        return _batch_row("CF cross-footing (Δ cash = CFO + CFI + CFF + FX)", "CHECK", "Insufficient cash flow data")

    entries: Dict[Tuple[str, date], List[str]] = {}
    samples: Dict[Tuple[str, date], Measurement] = {}
    passed = True

    duration_keys = sorted(
        set(delta_map.keys())
        | set(cfo_map.keys())
        | set(cfi_map.keys())
        | set(cff_map.keys())
    )
    for duration in duration_keys:
        period_set: set[date] = set()
        for mapping in (delta_map, cfo_map, cfi_map, cff_map, fx_map):
            period_set |= set(mapping.get(duration, {}).keys())
        for period_end in sorted(period_set):
            key = (duration, period_end)
            delta = delta_map.get(duration, {}).get(period_end)
            cfo = cfo_map.get(duration, {}).get(period_end)
            cfi = cfi_map.get(duration, {}).get(period_end)
            cff = cff_map.get(duration, {}).get(period_end)
            fx = fx_map.get(duration, {}).get(period_end)
            sample = next(
                (measure for measure in (delta, cfo, cfi, cff) if measure is not None),
                None,
            )
            if sample is not None and key not in samples:
                samples[key] = sample
            bucket = entries.setdefault(key, [])
            if any(measure is None for measure in (delta, cfo, cfi, cff)):
                bucket.append("insufficient data")
                passed = False
                continue
            fx_value = fx.value if fx is not None else 0.0
            diff = (cfo.value + cfi.value + cff.value + fx_value) - delta.value
            if abs(diff) > 1e-6:
                passed = False
            bucket.append(_format_delta("", diff, [delta.derived, cfo.derived, cfi.derived, cff.derived, fx.derived if fx else False]))

    if not entries:
        return _batch_row("CF cross-footing (Δ cash = CFO + CFI + CFF + FX)", "CHECK", "Insufficient cash flow data")

    details = []
    for duration, period_end in sorted(entries.keys(), key=lambda item: (item[1], item[0])):
        label = _describe_period(duration, period_end, samples.get((duration, period_end)))
        details.append(f"{label}: " + ", ".join(entries[(duration, period_end)]))

    return _batch_row(
        "CF cross-footing (Δ cash = CFO + CFI + CFF + FX)",
        "PASS" if passed else "CHECK",
        "; ".join(details),
    )


def _batch_row(check: str, result: str, detail: str) -> Dict[str, str]:
    return {
        "Source PDF": "Batch",
        "Check": check,
        "Result": result,
        "Details": detail,
    }
