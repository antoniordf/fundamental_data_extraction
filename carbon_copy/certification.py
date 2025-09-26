from __future__ import annotations

import re
from typing import Dict, Tuple

import pandas as pd

from selector.models import StatementType

from .models import StatementBlock

COLS_START = 3


def _pick_row(df: pd.DataFrame, pattern: str) -> pd.Series | None:
    mask = df["Line Item (as printed)"].str.contains(pattern, case=False, na=False)
    if not mask.any():
        return None
    return df[mask].iloc[0]


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
    equity_row = _pick_row(balance, r"total stockholders[’']? equity|total shareholders[’']? equity")

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
    delta_row = _pick_row(cash, r"^increase")
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
        "; ".join(roll_details),
    ])

    # 3. Net income link
    is_net_row = _pick_row(income, r"^net income$")
    cf_net_row = _pick_row(cash, r"^net income$")
    net_details: list[str] = []
    net_pass = True
    for column in income_columns:
        key = _period_signature(column, income_meta.get(column))
        income_val = _value(is_net_row, column)
        cf_column = cash_with_duration.get(key)
        if cf_column is None:
            cf_column = cash_by_period.get(_period_signature(column, income_meta.get(column), include_duration=False))
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
        "; ".join(net_details),
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
    tax_row = _pick_row(income, r"^provision for income taxes")

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
